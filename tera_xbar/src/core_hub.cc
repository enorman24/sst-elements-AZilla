
#include <sst/core/sst_config.h>

#include <sst/core/unitAlgebra.h>

#include "core_hub.h"

using namespace SST;
using namespace SST::Interfaces;
using namespace SST::TeraNoC;

CoreHub::CoreHub(ComponentId_t id, Params& params)
    : Component(id)
    , issued(0)
    , completed(0)
    , next_req_id(0)
    , first_issue_cycle(0)
    , done(false)
{
    const int verbosity = params.find<int>("verbose", 0);
    char prefix[256];
    sprintf(prefix, "[t=@t][%s]: ", getName().c_str());
    output = new Output(prefix, verbosity, 0, Output::STDOUT);

    core_id         = params.find<int>("core_id", 0);
    num_cores       = params.find<int>("num_cores", 4);
    num_banks       = params.find<int>("num_banks", 16);
    bank_port_base  = params.find<int>("bank_port_base", -1);
    if (bank_port_base < 0) {
        bank_port_base = num_cores;
    }
    max_outstanding = params.find<uint32_t>("max_outstanding", 8);
    max_requests    = params.find<uint64_t>("max_requests", 1000);
    pattern         = params.find<std::string>("pattern", "stride");
    target_bank     = params.find<int>("target_bank", 0);
    store_fraction  = params.find<double>("store_fraction", 0.0);
    vec_len         = params.find<uint64_t>("vec_len", 1024);
    mat_dim         = params.find<uint64_t>("mat_dim", 16);

    const uint64_t seed = params.find<uint64_t>("seed", 0);
    rng_state = 0x9E3779B97F4A7C15ULL ^ (seed + 0x1000193ULL * (core_id + 1));

    // Kernel patterns run to their natural length; ping-pong is a dependent
    // access chain (each access waits for the previous response).
    if (pattern == "axpy") {
        max_requests = 3 * vec_len;
    } else if (pattern == "matmul") {
        if (mat_dim % num_cores != 0) {
            output->fatal(CALL_INFO, -1, "Error: mat_dim %" PRIu64 " not divisible by num_cores %d\n",
                mat_dim, num_cores);
        }
        const uint64_t rows_per_core = mat_dim / num_cores;
        max_requests = rows_per_core * mat_dim * (2 * mat_dim + 1);
    } else if (pattern == "pingpong") {
        max_outstanding = 1;
    } else if (pattern != "stride" && pattern != "linear" && pattern != "uniform"
               && pattern != "conflict" && pattern != "neighbor") {
        output->fatal(CALL_INFO, -1, "Error: unknown pattern \"%s\"\n", pattern.c_str());
    }

    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();

    Params nicParams;
    nic = dynamic_cast<SimpleNetwork*>(loadSubComponent("shogun.ShogunNIC", this, nicParams));
    if (nullptr == nic) {
        output->fatal(CALL_INFO, -1, "Error: unable to load shogun.ShogunNIC\n");
    }
    nic->initialize("port", UnitAlgebra("1GB/s"), 1, UnitAlgebra("8B"), UnitAlgebra("8B"));

    const std::string clock_rate = params.find<std::string>("clock", "1GHz");
    registerClock(clock_rate, new Clock::Handler<CoreHub>(this, &CoreHub::tick));

    statReqLatency    = registerStatistic<uint64_t>("req_latency");
    statIssued        = registerStatistic<uint64_t>("requests_issued");
    statLoads         = registerStatistic<uint64_t>("loads_issued");
    statStores        = registerStatistic<uint64_t>("stores_issued");
    statStallLsuFull  = registerStatistic<uint64_t>("stall_lsu_full");
    statStallNoCredit = registerStatistic<uint64_t>("stall_no_credit");
    statOutstanding   = registerStatistic<uint64_t>("outstanding_reqs");
    statActiveCycles  = registerStatistic<uint64_t>("active_cycles");
}

CoreHub::~CoreHub()
{
    delete output;
}

void CoreHub::init(unsigned int phase)
{
    nic->init(phase);
}

void CoreHub::setup()
{
    nic->setup();
}

void CoreHub::finish()
{
    nic->finish();
}

uint64_t CoreHub::lcgNext()
{
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return rng_state >> 33;
}

// Produce the idx-th access of this core's pattern as a word address (bank =
// word % num_banks, mirroring the word-interleaved mempool_addr_scrambler).
void CoreHub::nextAccess(uint64_t idx, uint64_t& word, bool& is_store)
{
    is_store = false;

    if (pattern == "stride") {
        // conflict-free rotation: each core hits a distinct bank each cycle
        word = idx * num_banks + (core_id + idx) % num_banks;
    } else if (pattern == "linear") {
        // all cores sweep in lockstep: conflict wave on every bank
        word = idx * num_banks + idx % num_banks;
    } else if (pattern == "uniform") {
        word = idx * num_banks + lcgNext() % num_banks;
    } else if (pattern == "conflict") {
        word = idx * num_banks + target_bank;
    } else if (pattern == "pingpong") {
        // dependent load chain bouncing between this core's two home banks
        const uint64_t bank = 2 * core_id + (idx % 2);
        word = idx * num_banks + bank;
    } else if (pattern == "neighbor") {
        // stream through the next core's home region (halo-exchange shape);
        // bank sets are disjoint across cores
        const uint64_t region = num_banks / num_cores;
        const uint64_t bank = ((core_id + 1) % num_cores) * region + idx % region;
        word = idx * num_banks + bank;
    } else if (pattern == "axpy") {
        // z[g] = a*x[g] + y[g]: load x, load y, store z per element
        const uint64_t e  = idx / 3;
        const uint64_t ph = idx % 3;
        const uint64_t g  = core_id * vec_len + e;   // this core's chunk
        const uint64_t total = num_cores * vec_len;  // words per array
        if (ph == 0) {
            word = g;                     // x[g]
        } else if (ph == 1) {
            word = total + g;             // y[g]
        } else {
            word = 2 * total + g;         // z[g]
            is_store = true;
        }
    } else { // matmul: C = A*B, row-major, cores split rows of C
        const uint64_t D = mat_dim;
        const uint64_t rows_per_core = D / num_cores;
        const uint64_t per_elem = 2 * D + 1;
        const uint64_t b = idx / per_elem;           // output element index
        const uint64_t r = idx % per_elem;
        const uint64_t i = core_id * rows_per_core + b / D;
        const uint64_t j = b % D;
        if (r < 2 * D) {
            const uint64_t k = r / 2;
            if (r % 2 == 0) {
                word = i * D + k;                    // A[i][k]
            } else {
                word = D * D + k * D + j;            // B[k][j]
            }
        } else {
            word = 2 * D * D + i * D + j;            // C[i][j]
            is_store = true;
        }
    }
}

bool CoreHub::tick(SST::Cycle_t cycle)
{
    // Retire any responses delivered since the last cycle
    SimpleNetwork::Request* rsp;
    while ((rsp = nic->recv(0)) != nullptr) {
        TeraNoCMsg* msg = dynamic_cast<TeraNoCMsg*>(rsp->takePayload());
        if (nullptr == msg) {
            output->fatal(CALL_INFO, -1, "Error: response payload is not a TeraNoCMsg\n");
        }

        // Routing correctness: this response must belong to this core
        if (msg->init_core != (uint16_t)core_id) {
            output->fatal(CALL_INFO, -1,
                "Error: core %d received a response initiated by core %" PRIu16 " (misrouted)\n",
                core_id, msg->init_core);
        }
        if (msg->isRequest()) {
            output->fatal(CALL_INFO, -1, "Error: core %d received a request-type message\n", core_id);
        }

        std::map<uint64_t, uint64_t>::iterator it = inflight.find(msg->req_id);
        if (it == inflight.end()) {
            output->fatal(CALL_INFO, -1, "Error: response for unknown req_id %" PRIu64 "\n", msg->req_id);
        }

        statReqLatency->addData(cycle - it->second);
        inflight.erase(it);
        completed++;

        output->verbose(CALL_INFO, 2, 0, "retired req %" PRIu64 " (%" PRIu64 "/%" PRIu64 " complete)\n",
            msg->req_id, completed, max_requests);

        delete msg;
        delete rsp;
    }

    // Issue at most one request per cycle (single-issue Snitch LSU)
    if (issued < max_requests) {
        if (inflight.size() >= max_outstanding) {
            statStallLsuFull->addData(1);
        } else if (!nic->spaceToSend(0, 32)) {
            statStallNoCredit->addData(1);
        } else {
            uint64_t word;
            bool is_store;
            nextAccess(issued, word, is_store);

            // Synthetic patterns can mix in stores via store_fraction
            if (!is_store && store_fraction > 0.0
                && pattern != "axpy" && pattern != "matmul") {
                is_store = (lcgNext() % 1000000) < (uint64_t)(store_fraction * 1000000.0);
            }

            const uint64_t addr = word << 2;
            const uint32_t bank = word % num_banks;

            TeraNoCMsg* msg = new TeraNoCMsg(
                is_store ? TeraNoCMsg::STORE_REQ : TeraNoCMsg::LOAD_REQ,
                addr, 4, next_req_id, (uint16_t)core_id);

            SimpleNetwork::Request* req = new SimpleNetwork::Request(
                bank_port_base + bank, nic->getEndpointID(), 32, true, true, msg);

            if (!nic->send(req, 0)) {
                output->fatal(CALL_INFO, -1, "Error: send failed after spaceToSend returned true\n");
            }

            if (0 == issued) {
                first_issue_cycle = cycle;
            }

            inflight[next_req_id] = cycle;
            next_req_id++;
            issued++;

            statIssued->addData(1);
            if (is_store) {
                statStores->addData(1);
            } else {
                statLoads->addData(1);
            }
        }
    }

    if (issued > 0 && !done) {
        statOutstanding->addData(inflight.size());
    }

    if (!done && completed >= max_requests) {
        if (!inflight.empty()) {
            output->fatal(CALL_INFO, -1, "Error: all requests retired but %zu still in flight\n",
                inflight.size());
        }
        done = true;
        statActiveCycles->addData(cycle - first_issue_cycle);
        output->verbose(CALL_INFO, 1, 0, "core %d done: %" PRIu64 " requests in %" PRIu64 " cycles\n",
            core_id, completed, cycle - first_issue_cycle);
        primaryComponentOKToEndSim();
    }

    return done; // deregister the clock once all requests have retired
}
