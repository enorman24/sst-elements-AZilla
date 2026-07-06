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
#include "emberLLMPipelineParallelism.h"

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

EmberLLMPipelineParallelismGenerator::EmberLLMPipelineParallelismGenerator(SST::ComponentId_t id, Params& params) :
   EmberMessagePassingGenerator(id, params, "LLMPipelineParallelism" ),
   current_stage(INIT),
   batch_idx_fwd(0),
   batch_idx_bck(0){

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

      if(n_ranks < 3){
         out->fatal(CALL_INFO_LONG, -1, "Number of ranks must be at least 3! n_ranks = %u\n", n_ranks);
      }

      // First rank performs the embeddings layer and updates the weights
      // Last rank performs the cross entropy loss
      if(my_rank == 0 || my_rank == n_ranks-1) {
         n_layer_to_execute = 0;
      } else {
         n_layer_to_execute = 1 + ((num_hidden_layers - 1) / (n_ranks - 2));
         n_layer_to_execute = ((my_rank * n_layer_to_execute) > num_hidden_layers) ? (num_hidden_layers - (n_layer_to_execute * (my_rank - 1))) : n_layer_to_execute;
      }

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
      compute_time[exec_stage_e::UPDATE] = compute_map[compute_step_e::UPDATE];



      compute_time[exec_stage_e::TB_FWD] *= n_layer_to_execute;
      compute_time[exec_stage_e::TB_BCK] *= n_layer_to_execute;
      for(int i = 0; i < exec_stage_e::UNKNOWN; i++)
         out->verbose(CALL_INFO, 6, 0, "Compute time[%s]: %lu\n",
               exec_stage_s[i].c_str(), compute_time[i]);
      }

#define FWD_PATH 0
#define BCK_PATH 1

static uint32_t create_tag(const uint32_t batch, const uint32_t path) {
   uint32_t tag = path&0x1;
   tag |= batch << 1;
   return tag;
}

void EmberLLMPipelineParallelismGenerator::isend(std::queue<EmberEvent*>& evQ, uint64_t count, PayloadDataType dtype, int dst, uint32_t tag, Communicator comm, MessageRequest &req)  {
   uint64_t sent = 0;
   uint32_t buf_idx = 0;
   while(sent < count) {
      tag = tag << 8 | (buf_idx & 0xff);
      int to_send = (count-sent) < max_buffer_capacity ? (count-sent) : max_buffer_capacity;
      enQ_isend(evQ, NULL, to_send, dtype, dst, tag, comm, &req);
      sent += to_send;
   }
}

void EmberLLMPipelineParallelismGenerator::recv(std::queue<EmberEvent*>& evQ, uint64_t count, PayloadDataType dtype, int src, uint32_t tag, Communicator comm)  {
   uint64_t received = 0;
   uint32_t buf_idx = 0;
   while(received < count) {
      int to_receive = (count-received) < max_buffer_capacity ? (count-received) : max_buffer_capacity;
      tag = tag << 8 | (buf_idx & 0xff);
      enQ_recv(evQ, NULL, to_receive, dtype, src, tag, comm);
      received += to_receive;
   }
}

bool EmberLLMPipelineParallelismGenerator::generate(std::queue<EmberEvent*>& evQ) {
   bool finish = false;
   const uint64_t msg_size = batch_size*seq_len*hidden_size;

   exec_stage_e next_stage = exec_stage_e::UNKNOWN;

   enQ_compute(evQ, compute_time[current_stage]);

   switch (current_stage) {
      case exec_stage_e::INIT:
         if(my_rank == 0) {
            next_stage = exec_stage_e::EMBD;
         } else if(my_rank == n_ranks - 1) {
            next_stage = exec_stage_e::OUT;
         } else {
            next_stage = exec_stage_e::TB_FWD;
         }
         break;
      case exec_stage_e::EMBD: {
         uint32_t tag = create_tag(batch_idx_fwd, FWD_PATH);
         isend(evQ, msg_size, dtype, my_rank+1, tag, GroupWorld, req);

         batch_idx_fwd++;
         if(batch_idx_fwd < n_ranks) {
            next_stage = current_stage;
         } else {
            next_stage = exec_stage_e::UPDATE;
         }
         break;
      }
      case exec_stage_e::TB_FWD: {
         uint32_t tag = create_tag(batch_idx_fwd, FWD_PATH);
         recv(evQ, msg_size, dtype, my_rank-1, tag, GroupWorld);

         tag = create_tag(batch_idx_fwd, FWD_PATH);
         isend(evQ, msg_size, dtype, my_rank+1, tag, GroupWorld, req);
         batch_idx_fwd++;

         if(batch_idx_fwd < (n_ranks-my_rank) && batch_idx_fwd < n_batch) {
            next_stage = current_stage;
         } else {
            next_stage = exec_stage_e::TB_BCK;
         }
         break;
      }
      case exec_stage_e::OUT: {
         uint32_t tag = create_tag(batch_idx_fwd, FWD_PATH);
         recv(evQ, msg_size, dtype, my_rank-1, tag, GroupWorld);

         tag = create_tag(batch_idx_bck, BCK_PATH);
         isend(evQ, msg_size, dtype, my_rank-1, tag, GroupWorld, req);

         batch_idx_fwd++;
         batch_idx_bck++;

         if(batch_idx_bck == n_batch) {
            finish = true;
         }

         next_stage = current_stage;
         break;
      }
      case exec_stage_e::TB_BCK: {
         uint32_t tag = create_tag(batch_idx_bck, BCK_PATH);
         recv(evQ, msg_size, dtype, my_rank+1, tag, GroupWorld);

         tag = create_tag(batch_idx_bck, BCK_PATH);
         isend(evQ, msg_size, dtype, my_rank-1, tag, GroupWorld, req);

         batch_idx_bck++;

         int64_t limit = (n_batch - (n_ranks - my_rank));
         limit = (limit > 0) ? limit : 0;
         if(batch_idx_bck > limit) {
            next_stage = current_stage;
         } else {
            next_stage = exec_stage_e::TB_FWD;
         }

         if(batch_idx_bck == n_batch)
            finish = true;
         break;
      }
      case exec_stage_e::UPDATE: {
         uint32_t tag = create_tag(batch_idx_bck, BCK_PATH);
         recv(evQ, msg_size, dtype, my_rank+1, tag, GroupWorld);
         batch_idx_bck++;

         int64_t limit = (n_batch - (n_ranks - my_rank));
         limit = (limit > 0) ? limit : 0;
         if(batch_idx_bck > limit) {
            next_stage = current_stage;
         } else {
            next_stage = exec_stage_e::EMBD;
         }

         if(batch_idx_bck == n_batch)
            finish = true;
         break;
      }
      default:
         out->fatal(CALL_INFO_LONG, -1, "Unknown execution stage: %d\n", current_stage);
         break;
   }

   if(verbose)
      out->verbose(CALL_INFO, 8, 0, "Current stage: %s\tnext stage: %s\tbatch index:%u/%u\n",
               exec_stage_s[current_stage].c_str(), exec_stage_s[next_stage].c_str(), batch_idx_fwd, batch_idx_bck);

   current_stage = next_stage;

   return finish;
}
