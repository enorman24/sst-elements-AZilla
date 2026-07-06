/*
 Copyright (c) 2025 imec v.z.w.
 All Rights Reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are
 met: redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer;
 redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution;
 neither the name of the copyright holders nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 Author: Erwan Lenormand (imec/CSA)
*/
#ifndef EMBERLLMDataParallelism_H
#define EMBERLLMDataParallelism_H

#include "mpi/embermpigen.h"

namespace SST {
namespace Ember {

class EmberLLMDataParallelismGenerator : public EmberMessagePassingGenerator {
public:
   SST_ELI_REGISTER_SUBCOMPONENT(
         EmberLLMDataParallelismGenerator,
         "ember",
         "LLMDataParallelismMotif",
         SST_ELI_ELEMENT_VERSION(1,0,0),
         "Performs a Full Shared Data Parallel communication Motif",
         SST::Ember::EmberLLMDataParallelismGenerator
         )

   SST_ELI_DOCUMENT_PARAMS(
         {  "arg.batch_size",      "Number of sequence processed in parallel", "1" },
         {  "arg.sequence_len",    "Number of token per sequence", "8192" },
         {  "arg.n_batch",          "Number of batches to process", "128" },

         { "args.dram_bw",         "DRAM bandwidth in byte/second", "1555000000000"},
         { "args.peak_flop",       "Peak compute throughput @ targeted data type", "78000000000000"},

         {  "arg.llm_config",      "Configuration file of the Large Language Model", NULL },
         {  "arg.verbose",         "Enable debug prints", "0" }
         )

   SST_ELI_DOCUMENT_STATISTICS(
         { "time-Init", "Time spent in Init event",          "ns",  0},
         { "time-Finalize", "Time spent in Finalize event",  "ns", 0},
         { "time-Rank", "Time spent in Rank event",          "ns", 0},
         { "time-Size", "Time spent in Size event",          "ns", 0},
         { "time-Send", "Time spent in Recv event",          "ns", 0},
         { "time-Recv", "Time spent in Recv event",          "ns", 0},
         { "time-Irecv", "Time spent in Irecv event",        "ns", 0},
         { "time-Isend", "Time spent in Isend event",        "ns", 0},
         { "time-Wait", "Time spent in Wait event",          "ns", 0},
         { "time-Waitall", "Time spent in Waitall event",    "ns", 0},
         { "time-Waitany", "Time spent in Waitany event",    "ns", 0},
         { "time-Compute", "Time spent in Compute event",    "ns", 0},
         { "time-Barrier", "Time spent in Barrier event",    "ns", 0},
         { "time-Alltoallv", "Time spent in Alltoallv event", "ns", 0},
         { "time-Alltoall", "Time spent in Alltoall event",  "ns", 0},
         { "time-Allreduce", "Time spent in Allreduce event", "ns", 0},
         { "time-Reduce", "Time spent in Reduce event",      "ns", 0},
         { "time-Bcast", "Time spent in Bcast event",        "ns", 0},
         { "time-Gettime", "Time spent in Gettime event",    "ns", 0},
         { "time-Commsplit", "Time spent in Commsplit event", "ns", 0},
         { "time-Commcreate", "Time spent in Commcreate event", "ns", 0})

public:
   EmberLLMDataParallelismGenerator(SST::ComponentId_t id, Params& params);
   bool generate(std::queue<EmberEvent*>& evQ);

private:
   enum exec_stage_e : uint8_t {INIT, EMBD, TB_FWD, OUT, TB_BCK, UPDATE, UNKNOWN};

   void allgather(std::queue<EmberEvent*>& evQ, uint64_t count, PayloadDataType dtype, Communicator comm);
   void reduce(std::queue<EmberEvent*>& evQ, uint64_t count, PayloadDataType dtype, ReductionOperation op, int root, Communicator comm);

   uint64_t batch_size;
   uint64_t seq_len;
   uint64_t n_batch;

   uint64_t hidden_size;
   uint64_t intermediate_size;
   uint64_t num_hidden_layers;
   uint64_t vocab_size;
   uint64_t num_key_value_heads;

   PayloadDataType dtype;

   exec_stage_e current_stage;
   uint32_t layer_index;
   uint32_t batch_index;

   std::vector<uint64_t> compute_time;
   std::vector<uint64_t> num_params;
   double dram_bw; //byte/ns
   double peak_flop; // flop/ns

   int my_rank;
   uint32_t n_ranks;

   Output * out;
   int verbose;
};

}
}


#endif /* end of #ifndef EMBERLLMDataParallelism_H scope */
