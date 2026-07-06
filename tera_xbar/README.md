# TeraNoC Hier-L0 Crossbar in SST (Phase 1)

SST 9.0.0 model of one TeraNoC tile (arXiv 2508.02446): **M=4 cores + N=16 SPM banks
on a single-cycle, round-robin-arbitrated crossbar**, following the mapping in the
TeraNoC-in-SST reference doc (Hier-L0 → `shogun.ShogunXBar`).

## What's here

| File | Role |
|------|------|
| `src/teranoc_event.h` | `TeraNoCMsg` — the word request/response riding inside `SimpleNetwork::Request` payloads |
| `src/core_hub.{h,cc}` | `teranoc.CoreHub` — core-side endpoint: Snitch-like latency-tolerant LSU (1 issue/cycle, 8 outstanding) + traffic patterns |
| `src/spm_bank.{h,cc}` | `teranoc.SPMBank` — single-ported 1-cycle SRAM bank with credit backpressure |
| `Makefile` | builds `libteranoc.so` out-of-tree and registers it via `sst-register` |
| `teranoc_l0_tile.py` | the tile topology (20-port shogun, 4 CoreHubs, 16 banks) |

Both endpoints attach to the crossbar through the stock `shogun.ShogunNIC`
(`SimpleNetwork` interface) — no shogun datapath code was touched.

## RTL correspondence

Verified against `TeraNoC/hardware/src`:

- The tile's core→bank and bank→core paths are `stream_xbar`s
  (`mempool_tcdm_bank_interco.sv`), which put one `rr_arb_tree`
  (**FairArb=1, LockIn=1**) on every output: per-output pointer, winner = first
  requester at-or-after the pointer, pointer ← next requester strictly above it
  on a grant (wrapping), unchanged when the output can't grant.
- Shogun's stock arbiter instead rotates a single global scan pointer with ties
  falling through in port-index order → up to ~P-fold grant bias under sustained
  conflicts (reference doc §7).

**Two patches to sst-elements shogun** (in
`/homes/blin24/projects/sst9/sst-elements-library-9.0.0/src/sst/elements/shogun/`):

1. `shogun.cc` now honors the (previously ignored) `arbitration` parameter.
2. New `arb/shogunfairrrarb.h` — `output_roundrobin`, a cycle-accurate model of
   the `rr_arb_tree` FairArb policy per output port.

Config mirrors the RTL: `queue_slots=2` (2-deep boundary FIFOs),
`in/out_msg_per_cycle=1` (one 32-bit word per port per cycle), 1-cycle banks,
8 outstanding requests per core, word-interleaved addresses
(`bank = (addr>>2) % 16`, mirroring `mempool_addr_scrambler`).

## Build & run

```sh
make && make install         # builds libteranoc.so, registers with sst 9.0.0
sst teranoc_l0_tile.py       # sst = /homes/blin24/projects/sst9/install/sstcore-9.0.0/bin/sst
```

Knobs via environment: `TERA_PATTERN` (stride|linear|uniform|conflict),
`TERA_REQS`, `TERA_ARB` (output_roundrobin|roundrobin), `TERA_CLOCK`, `TERA_STATS`.

## Validation results (10 000 requests/core, 1 GHz)

| Run | Result | Target |
|-----|--------|--------|
| `stride` (conflict-free) | 0.9995 req/core/cycle, latency exactly 5 cycles, zero stalls | 1 req/core/cycle ✓ |
| `uniform` random | 0.904 req/core/cycle | analytic 16·(1−(15/16)⁴)/4 = 0.905 ✓ |
| `conflict` + `output_roundrobin` | all cores: mean latency 11.0, finish 40001–40004 cy — strict 4-way rotation | 25/25/25/25 ✓ |
| `conflict` + legacy `roundrobin` | core0 finishes in 11 771 cy, core3 in 40 004; max latency 43 vs 11 | reproduces the §7 index bias (artifact the RTL doesn't have) |

Aggregate throughput under conflict is identical for both arbiters (bank serves
1/cycle either way) — only fairness differs, as expected.

**Latency offset:** the RTL round trip is 1 cycle (combinational xbar + 1-cycle
bank); SST charges ≥1 cycle per clocked component hop, giving a deterministic
5-cycle zero-load RT. Treat as a constant offset — cores are latency-tolerant,
so throughput results are unaffected (reference doc §18).

## Next (Phase 2+)

Group level: shared tile→group port (5-port shogun mux), 16×16 H0-to-H0 xbar,
`xbar_adapter` boundary re-addressing with the 1-cycle spill register.
