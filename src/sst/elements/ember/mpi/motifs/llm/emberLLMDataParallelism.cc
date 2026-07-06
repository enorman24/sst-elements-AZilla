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
#include <fstream>
#include <sst/core/sst_config.h>
#include "../../../../mercury/external/json.hpp"
#include "emberLLMDataParallelism.h"

#include "analytical_model.hpp"

using namespace SST::Ember;

const static std::vector<std::string> exec_stage_s = {
   "INIT",
   "EMBD",
   "TB_FWD",
   "OUT",
   "TB_BCK",
   "UPDATE",
   "UNKNOWN"
};

#define TOKENIZE_DURATION 1e6

const static int max_buffer_capacity = 1*1024*1024*1024;

EmberLLMDataParallelismGenerator::EmberLLMDataParallelismGenerator(SST::ComponentId_t id, Params& params) :
   EmberMessagePassingGenerator(id, params, "LLMDataParallelism" ),
   current_stage(INIT),
   layer_index(0),
   batch_index(0) {

      my_rank = rank();
      n_ranks = size();

      verbose = params.find("arg.verbose", 0);
      std::ostringstream prefix;
      prefix << "[@t] rank:" << my_rank << " EmberEngine:" << getMotifName() << ":@p:@l: ";
      out = new Output(prefix.str().c_str(), verbose, 0, Output::STDOUT);

      batch_size = params.find<uint64_t>("arg.batch_size", 1);
      seq_len = params.find<uint64_t>("arg.sequence_len", 8192);
      n_batch = params.find<uint64_t>("arg.n_batch", 128);

      dram_bw = (double)params.find<uint64_t>("arg.dram_bw", 1555000000000UL) / 1e9;
      peak_flop = (double)params.find<uint64_t>("args.peak_flop", 78000000000000UL) / 1e9;

      bool found = false;
      std::string config_path = params.find<std::string>("arg.llm_config", found);

      if(!found) {
         out->fatal(CALL_INFO_LONG, -1, "LLM config file not set\n");
      }

      std::ifstream config_file(config_path);
      nlohmann::json config = nlohmann::json::parse(config_file);

      uint64_t max_ctx = config["max_position_embeddings"];

      if(max_ctx < seq_len) {
         out->fatal(CALL_INFO_LONG, -1, "Sequence length (%lu) is greater than context length(%lu)\n", seq_len, max_ctx);
      }

      hidden_size = config["hidden_size"];
      intermediate_size = config["intermediate_size"];
      num_hidden_layers = config["num_hidden_layers"];
      num_key_value_heads = config["num_key_value_heads"];
      uint64_t num_key_value_heads = config["num_key_value_heads"];
      uint64_t num_attention_heads = config["num_attention_heads"];


      std::string dtype_s = config["torch_dtype"];
      if(dtype_s == "float32" || dtype_s == "float") {
         dtype = INT32_T;
      } else if (dtype_s == "float16" || dtype_s == "half" || dtype_s == "bfloat16") {
         dtype = INT16_T;
      } else if (dtype_s == "float8") {
         dtype = INT8_T;
      } else {
         out->fatal(CALL_INFO_LONG, -1, "Unknown dataype: %s\n", dtype_s.c_str());
      }
      vocab_size = config["vocab_size"];

      const uint64_t dtype_size = sizeofDataType(dtype);

      compute_time.resize(exec_stage_e::UNKNOWN,0);


      std::map<compute_step_e, uint64_t> compute_map = get_compute_time(batch_size,
                                                                        seq_len,
                                                                        hidden_size,
                                                                        num_hidden_layers,
                                                                        num_key_value_heads,
                                                                        num_attention_heads,
                                                                        intermediate_size,
                                                                        vocab_size,
                                                                        dtype_size,
                                                                        dram_bw,
                                                                        peak_flop,
                                                                        1);


      compute_time.resize(exec_stage_e::UNKNOWN,0);
      compute_time[exec_stage_e::EMBD] = (my_rank == 0 ? compute_map[compute_step_e::TOKENIZE] : 0) +
                                         compute_map[compute_step_e::EMBD];

      compute_time[exec_stage_e::TB_FWD] = compute_map[compute_step_e::ATT_FWD] + compute_map[compute_step_e::MLP_FWD];

      compute_time[exec_stage_e::OUT] =   compute_map[compute_step_e::LOGITS] +
                                          compute_map[compute_step_e::FWD_MAX] +
                                          compute_map[compute_step_e::FWD_SOFTMAX] +
                                          compute_map[compute_step_e::LOSS] +
                                          compute_map[compute_step_e::BCK_LINEAR] +
                                          compute_map[compute_step_e::BCK_NORM];
      compute_time[exec_stage_e::TB_BCK] = compute_map[compute_step_e::MLP_BCK] +
                                           compute_map[compute_step_e::ATT_BCK];
      compute_time[exec_stage_e::UPDATE] = compute_map[compute_step_e::UPDATE] / n_ranks;

      num_params.resize(exec_stage_e::UPDATE,0);
      num_params[exec_stage_e::EMBD] = vocab_size * hidden_size;
      num_params[exec_stage_e::TB_FWD] = hidden_size * (2 + 2*hidden_size + 2*hidden_size/num_key_value_heads + 3*intermediate_size);
      num_params[exec_stage_e::OUT] = 2 * hidden_size * (1 + vocab_size);
      num_params[exec_stage_e::TB_BCK] = hidden_size * (2 + 2*hidden_size + 2*hidden_size/num_key_value_heads + 3*intermediate_size);

      if(my_rank == 0) {
         for(int i = 0; i < exec_stage_e::UNKNOWN; i++)
            out->verbose(CALL_INFO, 6, 0, "Compute time[%s]: %lu\n",
               exec_stage_s[i].c_str(), compute_time[i]);
      }
   }


void EmberLLMDataParallelismGenerator::allgather(std::queue<EmberEvent*>& evQ, uint64_t count, PayloadDataType dtype, Communicator comm)  {
   uint64_t sent = 0;
   while(sent < count) {
      int to_send = (count-sent) < max_buffer_capacity ? (count-sent) : max_buffer_capacity;
      enQ_allgather(evQ, NULL, to_send, dtype, NULL, to_send, dtype, comm);
      sent += to_send;
   }
}

void EmberLLMDataParallelismGenerator::reduce(std::queue<EmberEvent*>& evQ, uint64_t count, PayloadDataType dtype, ReductionOperation op, int root, Communicator comm)  {
   uint64_t sent = 0;
   while(sent < count) {
      int to_send = (count-sent) < max_buffer_capacity ? (count-sent) : max_buffer_capacity;
      enQ_reduce(evQ, NULL, NULL, to_send, dtype, op, root, comm);
      sent += to_send;
   }
}

bool EmberLLMDataParallelismGenerator::generate(std::queue<EmberEvent*>& evQ) {
   bool finish = false;
   const uint64_t msg_size = num_params[current_stage]/n_ranks;

   exec_stage_e next_stage = exec_stage_e::UNKNOWN;

   switch (current_stage) {
      case exec_stage_e::INIT:
         next_stage =  exec_stage_e::EMBD;
         batch_index = 0;
         layer_index = 0;
         break;
      case exec_stage_e::EMBD:
         allgather(evQ, msg_size, dtype, GroupWorld);
         next_stage = exec_stage_e::TB_FWD;
         enQ_compute(evQ, compute_time[current_stage]);
         break;
      case exec_stage_e::TB_FWD: {
         allgather(evQ, msg_size, dtype, GroupWorld);
         layer_index++;
         next_stage = (layer_index == num_hidden_layers) ? exec_stage_e::OUT : exec_stage_e::TB_FWD;
         enQ_compute(evQ, compute_time[current_stage]);
         break;
      }
      case exec_stage_e::OUT: {
         allgather(evQ, msg_size, dtype, GroupWorld);
         next_stage =  exec_stage_e::TB_BCK;
         layer_index = 0;

         enQ_compute(evQ, compute_time[current_stage]);

         for(int r = 0; r < n_ranks; r++) {
            reduce(evQ, num_params[current_stage]/n_ranks, dtype, MP::SUM, r, GroupWorld);
         }
         break;
      }
      case exec_stage_e::TB_BCK: {
         allgather(evQ, msg_size, dtype, GroupWorld);
         layer_index++;
         next_stage = (layer_index == num_hidden_layers) ? exec_stage_e::UPDATE : exec_stage_e::TB_BCK;

         enQ_compute(evQ, compute_time[current_stage]);

         for(int r = 0; r < n_ranks; r++) {
            reduce(evQ, num_params[current_stage]/n_ranks, dtype, MP::SUM, r, GroupWorld);
         }
         break;
      }
      case exec_stage_e::UPDATE:
         layer_index = 0;
         batch_index+= n_ranks;
         next_stage = exec_stage_e::EMBD;
         if(batch_index >= n_batch)
            finish = true;

         enQ_compute(evQ, compute_time[current_stage]);
         break;
      default:
         out->fatal(CALL_INFO_LONG, -1, "Unknown execution stage: %d\n", current_stage);
         break;
   }

   if(verbose && my_rank == 0)
      out->verbose(CALL_INFO, 8, 0, "Current stage: %s\tnext stage: %s\tlayer index:%u\tbatch index:%u\n",
               exec_stage_s[current_stage].c_str(), exec_stage_s[next_stage].c_str(), layer_index, batch_index);

   current_stage = next_stage;

   return finish;
}
