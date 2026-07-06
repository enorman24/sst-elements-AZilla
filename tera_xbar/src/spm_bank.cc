
#include <sst/core/sst_config.h>

#include <sst/core/unitAlgebra.h>

#include "spm_bank.h"

using namespace SST;
using namespace SST::Interfaces;
using namespace SST::TeraNoC;

SPMBank::SPMBank(ComponentId_t id, Params& params)
    : Component(id)
{
    const int verbosity = params.find<int>("verbose", 0);
    char prefix[256];
    sprintf(prefix, "[t=@t][%s]: ", getName().c_str());
    output = new Output(prefix, verbosity, 0, Output::STDOUT);

    access_time = params.find<uint64_t>("access_time", 1);
    queue_depth = params.find<uint32_t>("queue_depth", 2);
    bank_id     = params.find<int>("bank_id", -1);
    num_banks   = params.find<int>("num_banks", 16);

    Params nicParams;
    nic = dynamic_cast<SimpleNetwork*>(loadSubComponent("shogun.ShogunNIC", this, nicParams));
    if (nullptr == nic) {
        output->fatal(CALL_INFO, -1, "Error: unable to load shogun.ShogunNIC\n");
    }
    nic->initialize("port", UnitAlgebra("1GB/s"), 1, UnitAlgebra("8B"), UnitAlgebra("8B"));

    const std::string clock_rate = params.find<std::string>("clock", "1GHz");
    registerClock(clock_rate, new Clock::Handler<SPMBank>(this, &SPMBank::tick));

    statReqs       = registerStatistic<uint64_t>("requests_received");
    statResps      = registerStatistic<uint64_t>("responses_sent");
    statRespStalls = registerStatistic<uint64_t>("response_stalls");
    statOccupancy  = registerStatistic<uint64_t>("queue_occupancy");
}

SPMBank::~SPMBank()
{
    delete output;
}

void SPMBank::init(unsigned int phase)
{
    nic->init(phase);
}

void SPMBank::setup()
{
    nic->setup();
}

void SPMBank::finish()
{
    nic->finish();
}

bool SPMBank::tick(SST::Cycle_t cycle)
{
    // Emit at most one ready response (single response port, like the RTL's
    // one rvalid per bank per cycle)
    if (!pipe.empty() && pipe.front().ready_cycle <= cycle) {
        if (nic->spaceToSend(0, 32)) {
            SimpleNetwork::Request* orig = pipe.front().req;
            TeraNoCMsg* msg = dynamic_cast<TeraNoCMsg*>(orig->takePayload());
            if (nullptr == msg) {
                output->fatal(CALL_INFO, -1, "Error: request payload is not a TeraNoCMsg\n");
            }

            msg->type = (msg->type == TeraNoCMsg::STORE_REQ) ? TeraNoCMsg::STORE_ACK
                                                             : TeraNoCMsg::LOAD_RSP;

            SimpleNetwork::Request* rsp = new SimpleNetwork::Request(
                orig->src, nic->getEndpointID(), 32, true, true, msg);

            if (!nic->send(rsp, 0)) {
                output->fatal(CALL_INFO, -1, "Error: send failed after spaceToSend returned true\n");
            }

            delete orig;
            pipe.pop_front();
            statResps->addData(1);
        } else {
            statRespStalls->addData(1);
        }
    }

    // Accept at most one new request (single-ported SRAM); withholding the
    // recv (and thus the credit) when the pipeline is full backpressures the
    // crossbar output.
    if (pipe.size() < queue_depth) {
        SimpleNetwork::Request* req = nic->recv(0);
        if (nullptr != req) {
            // Routing correctness: the word-interleaved decode of the request
            // address must select this bank.
            if (bank_id >= 0) {
                TeraNoCMsg* msg = dynamic_cast<TeraNoCMsg*>(req->inspectPayload());
                if (nullptr == msg) {
                    output->fatal(CALL_INFO, -1, "Error: request payload is not a TeraNoCMsg\n");
                }
                if (!msg->isRequest()) {
                    output->fatal(CALL_INFO, -1, "Error: bank %d received a response-type message\n", bank_id);
                }
                const int decoded = (int)((msg->addr >> 2) % (uint64_t)num_banks);
                if (decoded != bank_id) {
                    output->fatal(CALL_INFO, -1,
                        "Error: bank %d received addr 0x%" PRIx64 " which decodes to bank %d (misrouted)\n",
                        bank_id, msg->addr, decoded);
                }
            }

            PipeEntry entry;
            entry.ready_cycle = cycle + access_time;
            entry.req = req;
            pipe.push_back(entry);
            statReqs->addData(1);
        }
    }

    statOccupancy->addData(pipe.size());

    return false;
}
