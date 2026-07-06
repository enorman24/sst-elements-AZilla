// TeraNoC SST model: SPMBank - one 1 KiB scratchpad bank behind the Hier-L0
// crossbar (tcdm_adapter + SRAM macro in the RTL). Single-ported: accepts at
// most one request and emits at most one response per cycle; the SRAM read
// costs access_time cycles. A stalled response path backpressures into the
// crossbar by withholding the NIC credit.

#ifndef _H_TERANOC_SPM_BANK
#define _H_TERANOC_SPM_BANK

#include <deque>

#include <sst/core/component.h>
#include <sst/core/interfaces/simpleNetwork.h>
#include <sst/core/output.h>

#include "teranoc_event.h"

namespace SST {
namespace TeraNoC {

    class SPMBank : public SST::Component {

    public:
        SST_ELI_REGISTER_COMPONENT(
            SPMBank,
            "teranoc",
            "SPMBank",
            SST_ELI_ELEMENT_VERSION(1, 0, 0),
            "TeraNoC 1 KiB scratchpad memory bank (single-ported, 1-cycle access)",
            COMPONENT_CATEGORY_MEMORY)

        SST_ELI_DOCUMENT_PARAMS(
            { "verbose",     "Output verbosity, 0 = none", "0" },
            { "clock",       "Bank clock frequency", "1GHz" },
            { "access_time", "SRAM access latency in cycles", "1" },
            { "queue_depth", "Requests held inside the bank pipeline before backpressuring", "2" },
            { "bank_id",     "This bank's index; arriving requests are checked against the address decode (-1 disables)", "-1" },
            { "num_banks",   "Banks per tile, for the address-decode check", "16" })

        SST_ELI_DOCUMENT_PORTS(
            { "port", "Link into the tile crossbar (shogun.ShogunXBar port)", { "shogun.ShogunEvent" } })

        SST_ELI_DOCUMENT_STATISTICS(
            { "requests_received", "Requests accepted by this bank", "requests", 1 },
            { "responses_sent",    "Responses returned by this bank", "responses", 1 },
            { "response_stalls",   "Cycles a ready response stalled on crossbar backpressure", "cycles", 1 },
            { "queue_occupancy",   "Bank pipeline depth, sampled every cycle (mean = Sum/Count)", "requests", 1 })

        SPMBank(SST::ComponentId_t id, SST::Params& params);
        ~SPMBank();

        void init(unsigned int phase) override;
        void setup() override;
        void finish() override;

    private:
        struct PipeEntry {
            uint64_t ready_cycle;
            SST::Interfaces::SimpleNetwork::Request* req;
        };

        bool tick(SST::Cycle_t cycle);

        SST::Output* output;
        SST::Interfaces::SimpleNetwork* nic;

        uint64_t access_time;
        uint32_t queue_depth;
        int      bank_id;
        int      num_banks;

        std::deque<PipeEntry> pipe;

        Statistic<uint64_t>* statReqs;
        Statistic<uint64_t>* statResps;
        Statistic<uint64_t>* statRespStalls;
        Statistic<uint64_t>* statOccupancy;
    };

}
}

#endif
