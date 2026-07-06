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
#include <sstream>
#include <sst/core/sst_config.h>
#include "../../../../mercury/external/json.hpp"
#include "emberLLMTensorParallelism.h"

#include "analytical_model.hpp"

using namespace SST::Ember;

const static std::vector<std::string> exec_stage_s = {
   "INIT",
   "TOKENIZE",
   "EMBD",
   "ATT_FWD",
   "MLP_FWD",
   "LOGITS",
   "FWD_MAX",
   "FWD_SOFTMAX",
   "LOSS",
   "BCK_LINEAR",
   "BCK_NORM",
   "MLP_BCK",
   "ATT_BCK",
   "UPDATE",
   "UNKNOWN"
};

const static uint64_t max_buffer_capacity = 1*1024*1024*1024;

EmberLLMTensorParallelismGenerator::EmberLLMTensorParallelismGenerator(SST::ComponentId_t id, Params& params) :
   EmberMessagePassingGenerator(id, params, "LLMTensorParallelism" ),
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
      // Read the whole file, then parse from the string: nlohmann 3.3's
      // istream input adapter mis-reads the stream in this build (garbage
      // bytes -> parse_error.101), while string parsing is reliable.
      std::stringstream config_buf;
      config_buf << config_file.rdbuf();
      nlohmann::json config = nlohmann::json::parse(config_buf.str());

      uint64_t max_ctx = config["max_position_embeddings"];

      if(max_ctx < seq_len) {
         out->fatal(CALL_INFO_LONG, -1, "Sequence length (%lu) is greater than context length(%lu)\n", seq_len, max_ctx);
      }

      hidden_size = config["hidden_size"];
      intermediate_size = config["intermediate_size"];
      num_hidden_layers = config["num_hidden_layers"];
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
                                                                        n_ranks);


      compute_time.resize(exec_stage_e::UNKNOWN,0);
      compute_time[exec_stage_e::TOKENIZE] = my_rank == 0 ? compute_map[compute_step_e::TOKENIZE] : 0;
      compute_time[exec_stage_e::EMBD] = compute_map[compute_step_e::EMBD];
      compute_time[exec_stage_e::ATT_FWD] = compute_map[compute_step_e::ATT_FWD];
      compute_time[exec_stage_e::MLP_FWD] = compute_map[compute_step_e::MLP_FWD];
      compute_time[exec_stage_e::LOGITS] = compute_map[compute_step_e::LOGITS];
      compute_time[exec_stage_e::FWD_MAX] = compute_map[compute_step_e::FWD_MAX];
      compute_time[exec_stage_e::FWD_SOFTMAX] = compute_map[compute_step_e::FWD_SOFTMAX];
      compute_time[exec_stage_e::LOSS] =  compute_map[compute_step_e::LOSS];
      compute_time[exec_stage_e::BCK_LINEAR] = compute_map[compute_step_e::BCK_LINEAR];
      compute_time[exec_stage_e::BCK_NORM] = compute_map[compute_step_e::BCK_NORM];
      compute_time[exec_stage_e::MLP_BCK] = compute_map[compute_step_e::MLP_BCK];
      compute_time[exec_stage_e::ATT_BCK] = compute_map[compute_step_e::ATT_BCK];
      compute_time[exec_stage_e::UPDATE] = compute_map[compute_step_e::UPDATE];

      if(my_rank == 0) {
         for(int i = 0; i < exec_stage_e::UNKNOWN; i++)
            out->verbose(CALL_INFO, 6, 0, "Compute time[%s]: %lu\n",
               exec_stage_s[i].c_str(), compute_time[i]);
      }
   }


void EmberLLMTensorParallelismGenerator::bcast(std::queue<EmberEvent*>& evQ, uint64_t count, PayloadDataType dtype, int root, Communicator comm) {
   uint64_t sent = 0;
   while(sent < count) {
      int to_send = (count-sent) < max_buffer_capacity ? (count-sent) : max_buffer_capacity;
      enQ_bcast(evQ, NULL, to_send, dtype, root, comm);
      sent += to_send;
   }
}

void EmberLLMTensorParallelismGenerator::allgather(std::queue<EmberEvent*>& evQ, uint64_t count, PayloadDataType dtype, Communicator comm)  {
   uint64_t sent = 0;
   while(sent < count) {
      int to_send = (count-sent) < max_buffer_capacity ? (count-sent) : max_buffer_capacity;
      enQ_allgather(evQ, NULL, to_send, dtype, NULL, to_send, dtype, comm);
      sent += to_send;
   }
}

void EmberLLMTensorParallelismGenerator::allreduce(std::queue<EmberEvent*>& evQ, uint64_t count, PayloadDataType dtype,ReductionOperation op, Communicator comm)  {
   uint64_t sent = 0;
   while(sent < count) {
      int to_send = (count-sent) < max_buffer_capacity ? (count-sent) : max_buffer_capacity;
      enQ_allreduce(evQ, NULL, NULL, to_send, dtype, op, comm);
      sent += to_send;
   }
}


bool EmberLLMTensorParallelismGenerator::generate(std::queue<EmberEvent*>& evQ) {
   bool finish = false;
   uint64_t msg_size = batch_size*seq_len*hidden_size/n_ranks;

   exec_stage_e next_stage = exec_stage_e::UNKNOWN;

   enQ_compute(evQ, compute_time[current_stage]);
   switch (current_stage) {
      case INIT:
         next_stage = exec_stage_e::TOKENIZE;
         batch_index = 0;
         break;
      case TOKENIZE: {
         layer_index = 0;
         bcast(evQ, batch_size*seq_len, UINT32_T, 0, GroupWorld);
         next_stage = exec_stage_e::EMBD;
         break;
      }
      case exec_stage_e::EMBD:
         allgather(evQ, msg_size, dtype, GroupWorld);
         next_stage = exec_stage_e::ATT_FWD;
         break;
      case exec_stage_e::ATT_FWD:
         allreduce(evQ, msg_size, dtype, Hermes::MP::SUM, GroupWorld);
         next_stage = exec_stage_e::MLP_FWD;
         break;
      case exec_stage_e::MLP_FWD:
         layer_index++;
         allreduce(evQ, msg_size, dtype, Hermes::MP::SUM, GroupWorld);
         next_stage = (layer_index == num_hidden_layers) ? exec_stage_e::LOGITS : exec_stage_e::ATT_FWD;
         break;
      case exec_stage_e::LOGITS:
         allgather(evQ, msg_size, dtype, GroupWorld);
         next_stage = exec_stage_e::FWD_MAX;
         break;
      case exec_stage_e::FWD_MAX:
         msg_size =  batch_size*seq_len/n_ranks;
         allreduce(evQ, msg_size, INT32_T, Hermes::MP::MAX, GroupWorld);
         next_stage = exec_stage_e::FWD_SOFTMAX;
         break;
      case exec_stage_e::FWD_SOFTMAX:
         msg_size = 2*batch_size*seq_len/n_ranks;
         allreduce(evQ, msg_size, INT32_T, Hermes::MP::SUM, GroupWorld);
         next_stage = exec_stage_e::LOSS;
         break;
      case exec_stage_e::LOSS:
         msg_size = batch_size*seq_len*vocab_size/n_ranks;
         allreduce(evQ, msg_size, dtype, Hermes::MP::SUM, GroupWorld);
         next_stage = exec_stage_e::BCK_LINEAR;
         layer_index = 0;
         break;
      case exec_stage_e::BCK_LINEAR:
         allreduce(evQ, msg_size, dtype, Hermes::MP::SUM, GroupWorld);
         next_stage = exec_stage_e::BCK_NORM;
         break;
      case exec_stage_e::BCK_NORM:
         allgather(evQ, msg_size, dtype, GroupWorld);
         next_stage = exec_stage_e::MLP_BCK;
         break;

      case exec_stage_e::MLP_BCK:
         next_stage = exec_stage_e::ATT_BCK;
         allreduce(evQ, msg_size, dtype, Hermes::MP::SUM, GroupWorld);
         break;
      case exec_stage_e::ATT_BCK:
         layer_index++;
         allreduce(evQ, msg_size, dtype, Hermes::MP::SUM, GroupWorld);
         next_stage = (layer_index == num_hidden_layers) ? exec_stage_e::UPDATE : exec_stage_e::MLP_BCK;
         break;

      case exec_stage_e::UPDATE:
         layer_index = 0;
         next_stage = exec_stage_e::TOKENIZE;
         batch_index++;
         if(batch_index == n_batch) {
            finish = true;
         }
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
