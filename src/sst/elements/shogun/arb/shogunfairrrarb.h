
#ifndef _H_SHOGUN_FAIR_RR_ARB_H
#define _H_SHOGUN_FAIR_RR_ARB_H

#include <vector>

#include "shogun_event.h"
#include "shogunarb.h"

namespace SST {
namespace Shogun {

    // Per-output round-robin arbitration matching the PULP common_cells
    // rr_arb_tree (FairArb=1, as instantiated by stream_xbar in the TeraNoC
    // tile interconnect):
    //  - one priority pointer rr[o] per OUTPUT port
    //  - winner = first input requesting output o at-or-after rr[o] (cyclic)
    //  - on a grant, rr[o] <- lowest requester strictly above rr[o],
    //    wrapping to the lowest requester overall if none above
    //  - no grant (output busy / no requester) leaves rr[o] unchanged, which
    //    also gives the LockIn=1 behavior: the same head request wins again
    //    once the output frees up.
    // Each input's head-of-queue targets exactly one output and each output
    // grants at most once per cycle, as in the RTL.
    class ShogunFairRoundRobinArbitrator : public ShogunArbitrator {

    public:
        ShogunFairRoundRobinArbitrator() {}
        ~ShogunFairRoundRobinArbitrator() {}

        void moveEvents(const int num_events,
                        const int port_count,
                        ShogunQueue<ShogunEvent*>** inputQueues,
                        int32_t output_slots,
                        ShogunEvent*** outputEvents,
                        uint64_t cycle ) override
        {
            if (static_cast<int>(rr.size()) != port_count) {
                rr.assign(port_count, 0);
            }

            // Head-of-line destination per input (-1 = no request)
            std::vector<int> headDest(port_count, -1);
            for (int i = 0; i < port_count; ++i) {
                if (!inputQueues[i]->empty()) {
                    headDest[i] = inputQueues[i]->peek()->getDestination();
                }
            }

            int32_t moved_count = 0;

            for (int o = 0; o < port_count; ++o) {
                // Winner: first requester at-or-after rr[o], cyclic scan
                int winner = -1;
                for (int off = 0; off < port_count; ++off) {
                    const int idx = (rr[o] + off) % port_count;
                    if (headDest[idx] == o) {
                        winner = idx;
                        break;
                    }
                }

                if (winner < 0) {
                    continue;
                }

                // Find a free output slot; if none, the output cannot grant
                // this cycle (gnt=0 in the RTL) and rr[o] must not advance.
                int freeSlot = -1;
                for (int32_t k = 0; k < output_slots; ++k) {
                    if (nullptr == outputEvents[o][k]) {
                        freeSlot = k;
                        break;
                    }
                }

                if (freeSlot < 0) {
                    output->verbose(CALL_INFO, 4, 0,
                        "-> out: %" PRIi32 " winner: %" PRIi32 " but output slots full, no grant\n", o, winner);
                    continue;
                }

                outputEvents[o][freeSlot] = inputQueues[winner]->pop();
                moved_count++;

                output->verbose(CALL_INFO, 4, 0,
                    "-> out: %" PRIi32 " granted to in: %" PRIi32 " (rr was %" PRIi32 ")\n", o, winner, rr[o]);

                // rr_arb_tree FairArb pointer update: next requester strictly
                // above rr[o]; if none, lowest requester (<= rr[o]).
                int next = -1;
                for (int idx = rr[o] + 1; idx < port_count; ++idx) {
                    if (headDest[idx] == o) {
                        next = idx;
                        break;
                    }
                }
                if (next < 0) {
                    for (int idx = 0; idx <= rr[o]; ++idx) {
                        if (headDest[idx] == o) {
                            next = idx;
                            break;
                        }
                    }
                }
                rr[o] = (next < 0) ? rr[o] : next;

                // This input's head has been consumed; it cannot win another
                // output this cycle.
                headDest[winner] = -1;
            }

            bundle->getPacketsMoved()->addData(moved_count);
        }

    private:
        std::vector<int> rr;
    };

}
}

#endif
