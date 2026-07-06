// TeraNoC SST model: CoreHub - the core-side endpoint of the Hier-L0 tile
// crossbar. Models what the L0 interconnect sees of a Snitch core: the
// latency-tolerant LSU issuing at most one word request per cycle with up to
// max_outstanding (8) requests in flight, fed by a configurable address
// pattern (snitch_addr_demux + LSU of the RTL).
//
// Patterns:
//   synthetic:  stride | linear | uniform | conflict
//   kernels:    pingpong (dependent alternating loads, outstanding=1)
//               neighbor (stream the next core's home banks)
//               axpy     (z[i] = a*x[i] + y[i]: load,load,store per element)
//               matmul   (C = A*B row-major: 2*D loads + 1 store per element)
// Kernel patterns derive their own request count (vec_len / mat_dim params).

#ifndef _H_TERANOC_CORE_HUB
#define _H_TERANOC_CORE_HUB

#include <map>

#include <sst/core/component.h>
#include <sst/core/interfaces/simpleNetwork.h>
#include <sst/core/output.h>

#include "teranoc_event.h"

namespace SST {
namespace TeraNoC {

    class CoreHub : public SST::Component {

    public:
        SST_ELI_REGISTER_COMPONENT(
            CoreHub,
            "teranoc",
            "CoreHub",
            SST_ELI_ELEMENT_VERSION(1, 0, 0),
            "TeraNoC core-side endpoint: Snitch-LSU-like traffic generator for the Hier-L0 crossbar",
            COMPONENT_CATEGORY_PROCESSOR)

        SST_ELI_DOCUMENT_PARAMS(
            { "verbose",         "Output verbosity, 0 = none", "0" },
            { "clock",           "Core clock frequency", "1GHz" },
            { "core_id",         "Index of this core within the tile (0..num_cores-1)", "0" },
            { "num_cores",       "Cores per tile (M)", "4" },
            { "num_banks",       "SPM banks per tile (N)", "16" },
            { "bank_port_base",  "Crossbar port index of bank 0 (defaults to num_cores)", "-1" },
            { "max_outstanding", "LSU outstanding-request slots (Snitch: 8)", "8" },
            { "max_requests",    "Requests to issue (synthetic patterns; kernels derive their own)", "1000" },
            { "pattern",         "stride | linear | uniform | conflict | pingpong | neighbor | axpy | matmul", "stride" },
            { "target_bank",     "Bank targeted by the 'conflict' pattern", "0" },
            { "store_fraction",  "Fraction of stores for synthetic patterns [0.0-1.0]", "0.0" },
            { "vec_len",         "Elements per core for the axpy kernel", "1024" },
            { "mat_dim",         "Matrix dimension D for the matmul kernel (D %% num_cores == 0)", "16" },
            { "seed",            "Per-core RNG seed offset", "0" })

        SST_ELI_DOCUMENT_PORTS(
            { "port", "Link into the tile crossbar (shogun.ShogunXBar port)", { "shogun.ShogunEvent" } })

        SST_ELI_DOCUMENT_STATISTICS(
            { "req_latency",       "Round-trip latency of each retired request (core cycles)", "cycles", 1 },
            { "requests_issued",   "Requests injected into the crossbar", "requests", 1 },
            { "loads_issued",      "Load requests injected", "requests", 1 },
            { "stores_issued",     "Store requests injected", "requests", 1 },
            { "stall_lsu_full",    "Cycles a ready request stalled because all outstanding slots were busy", "cycles", 1 },
            { "stall_no_credit",   "Cycles a ready request stalled on crossbar input backpressure", "cycles", 1 },
            { "outstanding_reqs",  "In-flight requests, sampled every active cycle (mean = Sum/Count)", "requests", 1 },
            { "active_cycles",     "Cycles from first issue until last retire", "cycles", 1 })

        CoreHub(SST::ComponentId_t id, SST::Params& params);
        ~CoreHub();

        void init(unsigned int phase) override;
        void setup() override;
        void finish() override;

    private:
        bool tick(SST::Cycle_t cycle);
        void nextAccess(uint64_t idx, uint64_t& word, bool& is_store);
        uint64_t lcgNext();

        SST::Output* output;
        SST::Interfaces::SimpleNetwork* nic;

        int      core_id;
        int      num_cores;
        int      num_banks;
        int      bank_port_base;
        uint32_t max_outstanding;
        uint64_t max_requests;
        std::string pattern;
        int      target_bank;
        double   store_fraction;
        uint64_t vec_len;
        uint64_t mat_dim;
        uint64_t rng_state;

        uint64_t issued;
        uint64_t completed;
        uint64_t next_req_id;
        uint64_t first_issue_cycle;
        bool     done;

        std::map<uint64_t, uint64_t> inflight; // req_id -> issue cycle

        Statistic<uint64_t>* statReqLatency;
        Statistic<uint64_t>* statIssued;
        Statistic<uint64_t>* statLoads;
        Statistic<uint64_t>* statStores;
        Statistic<uint64_t>* statStallLsuFull;
        Statistic<uint64_t>* statStallNoCredit;
        Statistic<uint64_t>* statOutstanding;
        Statistic<uint64_t>* statActiveCycles;
    };

}
}

#endif
