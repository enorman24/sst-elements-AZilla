# TeraNoC Hier-L0 tile model (Phase 1 of the roadmap):
#   M=4 CoreHubs + N=16 SPMBanks on one 20-port shogun.ShogunXBar.
#
# Matches the RTL configuration:
#   - 2-deep boundary FIFOs           -> queue_slots = 2
#   - one 32-bit word per port/cycle  -> in/out_msg_per_cycle = 1
#   - per-output rr_arb_tree (FairArb)-> arbitration = output_roundrobin
#   - 1-cycle SRAM banks              -> SPMBank access_time = 1
#   - Snitch LSU, 8 outstanding       -> CoreHub max_outstanding = 8
#
# Knobs (environment variables):
#   TERA_PATTERN = stride | linear | uniform | conflict
#                | pingpong | neighbor | axpy | matmul     (default stride)
#   TERA_REQS    = requests per core (synthetic patterns)  (default 10000)
#   TERA_VECLEN  = axpy elements per core                  (default 1024)
#   TERA_MATDIM  = matmul dimension D                      (default 16)
#   TERA_ARB     = output_roundrobin | roundrobin          (default output_roundrobin)
#   TERA_CLOCK   = clock frequency                         (default 1GHz)
#   TERA_STATS   = stats CSV filename                      (default stats.csv)

import os
import sst

M       = 4        # cores per tile
N       = 16       # banks per tile
QDEPTH  = 2        # boundary FIFO depth
CLOCK   = os.environ.get("TERA_CLOCK", "1GHz")
PATTERN = os.environ.get("TERA_PATTERN", "stride")
REQS    = int(os.environ.get("TERA_REQS", "10000"))
VECLEN  = int(os.environ.get("TERA_VECLEN", "1024"))
MATDIM  = int(os.environ.get("TERA_MATDIM", "16"))
ARB     = os.environ.get("TERA_ARB", "output_roundrobin")
STATS   = os.environ.get("TERA_STATS", "stats.csv")

xbar = sst.Component("tile_xbar", "shogun.ShogunXBar")
xbar.addParams({
    "verbose":          0,
    "clock":            CLOCK,
    "port_count":       M + N,
    "queue_slots":      QDEPTH,
    "in_msg_per_cycle":  1,
    "out_msg_per_cycle": 1,
    "arbitration":      ARB,
})

for c in range(M):
    core = sst.Component("core%d" % c, "teranoc.CoreHub")
    core.addParams({
        "verbose":         0,
        "clock":           CLOCK,
        "core_id":         c,
        "num_cores":       M,
        "num_banks":       N,
        "max_outstanding": 8,
        "max_requests":    REQS,
        "pattern":         PATTERN,
        "target_bank":     0,
        "vec_len":         VECLEN,
        "mat_dim":         MATDIM,
    })
    link = sst.Link("link_core%d" % c)
    link.connect((core, "port", "1ps"), (xbar, "port%d" % c, "1ps"))

for b in range(N):
    bank = sst.Component("bank%d" % b, "teranoc.SPMBank")
    bank.addParams({
        "verbose":     0,
        "clock":       CLOCK,
        "access_time": 1,
        "queue_depth": 2,
        "bank_id":     b,
        "num_banks":   N,
    })
    link = sst.Link("link_bank%d" % b)
    link.connect((bank, "port", "1ps"), (xbar, "port%d" % (M + b), "1ps"))

sst.setStatisticLoadLevel(7)
sst.setStatisticOutput("sst.statOutputCSV", {"filepath": STATS, "separator": ";"})
sst.enableAllStatisticsForAllComponents()
