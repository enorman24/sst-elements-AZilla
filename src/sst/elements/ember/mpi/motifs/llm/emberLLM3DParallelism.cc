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
#include <algorithm>
#include <fstream>
#include <sstream>
#include <sst/core/sst_config.h>
#include "../../../../mercury/external/json.hpp"
#include "emberLLM3DParallelism.h"

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
#define TOKENIZE_DURATION 1e6

EmberLLM3DParallelismGenerator::EmberLLM3DParallelismGenerator(SST::ComponentId_t id, Params& params) :
   EmberMessagePassingGenerator(id, params, "LLM3DParallelism" ),
   current_stage(INIT),
   layer_index(0),
   batch_idx_fwd(0),
   batch_idx_bck(0),
   tp(params.find<uint32_t>("arg.tp", 1)),
   pp(params.find<uint32_t>("arg.pp", 3)),
   dp(params.find<uint32_t>("arg.dp", 1)) {
      my_rank = rank();
      n_ranks = size();

      verbose = params.find("arg.verbose", 0);
      std::ostringstream prefix;
      prefix << "[@t] rank:" << my_rank << " EmberEngine:" << getMotifName() << ":@p:@l: ";
      out = new Output(prefix.str().c_str(), verbose, 0, Output::STDOUT);

      tp_idx = my_rank%tp;
      pp_idx = (my_rank/tp)%pp;
      dp_idx = (my_rank/(tp*pp))%dp;

      out->verbose(CALL_INFO, 5, 0, "TP index:%u PP index:%u DP index:%u\n", tp_idx, pp_idx, dp_idx);

      batch_size = params.find<uint64_t>("arg.batch_size", 1);
      seq_len = params.find<uint64_t>("arg.sequence_len", 8192);
      n_batch = params.find<uint64_t>("arg.n_batch", 128);


      n_batch = 1 + (((int64_t)n_batch) - 1)/dp;


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
                                                                        tp);


      compute_time.resize(exec_stage_e::UNKNOWN,0);
      compute_time[exec_stage_e::TOKENIZE] = tp_idx == 0 ? compute_map[compute_step_e::TOKENIZE] : 0;
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
      compute_time[exec_stage_e::UPDATE] = compute_map[compute_step_e::UPDATE] / dp;



      for(int i = 0; i < exec_stage_e::UNKNOWN; i++)
         out->verbose(CALL_INFO, 6, 0, "Compute time[%s]: %lu\n", exec_stage_s[i].c_str(), compute_time[i]);

      num_params.resize(exec_stage_e::UPDATE,0);
      num_params[exec_stage_e::EMBD] = vocab_size * hidden_size;
      num_params[exec_stage_e::ATT_FWD] = hidden_size * (1 + 2*hidden_size + 2*hidden_size/num_key_value_heads);
      num_params[exec_stage_e::MLP_FWD] = hidden_size * (1  + 3*intermediate_size);
      num_params[exec_stage_e::LOGITS] = hidden_size * (1 + vocab_size);
      num_params[exec_stage_e::BCK_LINEAR] = hidden_size * (1 + vocab_size);
      num_params[exec_stage_e::MLP_BCK] = hidden_size * (1 + 3*intermediate_size);
      num_params[exec_stage_e::ATT_BCK] = hidden_size * (1 + 2*hidden_size + 2*hidden_size/num_key_value_heads);


      if(pp < 3){
         out->fatal(CALL_INFO_LONG, -1, "Number of pp ranks must be at least 3! pp = %u\n", pp);
      }

      // First rank performs the embeddings layer and updates the weights
      // Last rank performs the cross entropy loss
      if(pp_idx == 0 || pp_idx == pp-1) {
         n_layer_to_execute = 0;
      } else {
         n_layer_to_execute = 1 + ((num_hidden_layers - 1) / (pp - 2));
         n_layer_to_execute = ((pp_idx * n_layer_to_execute) > num_hidden_layers) ? (num_hidden_layers - (n_layer_to_execute * (pp_idx - 1))) : n_layer_to_execute;
      }
   }

void EmberLLM3DParallelismGenerator::configure(std::queue<EmberEvent*>& evQ) {
   tp_grp_idx = my_rank / tp;
   tp_group.resize(dp*pp);
   tp_ranks.resize(dp*pp);

   for(int d = 0; d < dp; d++) {
      for(int p = 0; p < pp; p++) {
         tp_ranks[d*pp+p].resize(tp);
         for(int t = 0; t < tp; t++) {
            int rank = t + p*tp + d*pp*tp;
            assert(rank < n_ranks);
            tp_ranks[d*pp+p][t] = rank;
         }

         enQ_commCreate(evQ, GroupWorld, tp_ranks[d*pp+p], &tp_group[d*pp+p]);
      }
   }

   dp_group.resize(tp*pp);
   dp_ranks.resize(tp*pp);
   dp_grp_idx = (my_rank % tp) + pp_idx * tp;

   for(int p = 0; p < pp; p++) {
      for(int t = 0; t < tp; t++) {
         dp_ranks[p*tp+t].resize(dp);
         for(int d = 0; d < dp; d++) {
            dp_ranks[p*tp+t][d] = t + p*tp + d*pp*tp;
         }
         enQ_commCreate(evQ, GroupWorld, dp_ranks[p*tp+t], &dp_group[p*tp+t]);
      }
   }
}

#define FWD_PATH 0
#define BCK_PATH 1

static uint32_t create_tag(const uint32_t batch, const uint32_t path) {
   uint32_t tag = path&0x1;
   tag |= batch << 1;
   return tag;
}

void EmberLLM3DParallelismGenerator::isend(std::queue<EmberEvent*>& evQ, uint64_t count, PayloadDataType dtype, int dst, uint32_t tag, Communicator comm, MessageRequest &req)  {
   assert(dst < n_ranks);
   uint64_t sent = 0;
   uint32_t buf_idx = 0;
   while(sent < count) {
      tag = tag << 8 | (buf_idx & 0xff);
      int to_send = (count-sent) < max_buffer_capacity ? (count-sent) : max_buffer_capacity;
      enQ_isend(evQ, NULL, to_send, dtype, dst, tag, comm, &req);
      sent += to_send;
   }
}

void EmberLLM3DParallelismGenerator::recv(std::queue<EmberEvent*>& evQ, uint64_t count, PayloadDataType dtype, int src, uint32_t tag, Communicator comm)  {
   assert(src < n_ranks);
   uint64_t received = 0;
   uint32_t buf_idx = 0;
   while(received < count) {
      int to_receive = (count-received) < max_buffer_capacity ? (count-received) : max_buffer_capacity;
      tag = tag << 8 | (buf_idx & 0xff);
      enQ_recv(evQ, NULL, to_receive, dtype, src, tag, comm);
      received += to_receive;
   }
}
void EmberLLM3DParallelismGenerator::bcast(std::queue<EmberEvent*>& evQ, uint64_t count, PayloadDataType dtype, int root, Communicator comm) {
   uint64_t sent = 0;
   while(sent < count) {
      int to_send = (count-sent) < max_buffer_capacity ? (count-sent) : max_buffer_capacity;
      enQ_bcast(evQ, NULL, to_send, dtype, root, comm);
      sent += to_send;
   }
}

void EmberLLM3DParallelismGenerator::allgather(std::queue<EmberEvent*>& evQ, uint64_t count, PayloadDataType dtype, Communicator comm)  {
   uint64_t sent = 0;
   while(sent < count) {
      int to_send = (count-sent) < max_buffer_capacity ? (count-sent) : max_buffer_capacity;
      enQ_allgather(evQ, NULL, to_send, dtype, NULL, to_send, dtype, comm);
      sent += to_send;
   }
}

void EmberLLM3DParallelismGenerator::allreduce(std::queue<EmberEvent*>& evQ, uint64_t count, PayloadDataType dtype,ReductionOperation op, Communicator comm)  {
   uint64_t sent = 0;
   while(sent < count) {
      int to_send = (count-sent) < max_buffer_capacity ? (count-sent) : max_buffer_capacity;
      enQ_allreduce(evQ, NULL, NULL, to_send, dtype, op, comm);
      sent += to_send;
   }
}


bool EmberLLM3DParallelismGenerator::generate(std::queue<EmberEvent*>& evQ) {
   bool finish = false;
   exec_stage_e next_stage = exec_stage_e::UNKNOWN;

   switch(current_stage) {
      case INIT: {
         configure(evQ);
         if(pp_idx == 0) {
            next_stage = exec_stage_e::TOKENIZE;
         } else if(pp_idx == (pp - 1)) {
            next_stage = exec_stage_e::LOGITS;
         } else {
            next_stage = exec_stage_e::ATT_FWD;
         }
         break;
      }
      case TOKENIZE: {
         enQ_compute(evQ, compute_time[current_stage]);

         layer_index = 0;
         uint64_t msg_size = batch_size*seq_len/tp;
         allgather(evQ, msg_size, INT32_T, tp_group[tp_grp_idx]);
         next_stage = exec_stage_e::EMBD;
         break;
      }
      case EMBD: {
         uint64_t msg_size = num_params[current_stage]/(dp*tp);
         allgather(evQ, msg_size, dtype, dp_group[dp_grp_idx]);

         enQ_compute(evQ, compute_time[current_stage]);


         msg_size = batch_size*seq_len*hidden_size/tp;
         uint32_t tag = create_tag(batch_idx_fwd, FWD_PATH);
         isend(evQ, msg_size, dtype, my_rank + tp, tag, GroupWorld, req);

         batch_idx_fwd++;
         if(batch_idx_fwd < pp && batch_idx_fwd < n_batch) {
            next_stage = current_stage;
         } else {
            next_stage = exec_stage_e::UPDATE;
         }

         msg_size = batch_size*seq_len*hidden_size/tp;
         allgather(evQ, msg_size, dtype, tp_group[tp_grp_idx]);
         break;
      }
      case ATT_FWD: {
         uint64_t msg_size = num_params[current_stage]/(dp*tp);
         allgather(evQ, msg_size, dtype, dp_group[dp_grp_idx]);

         if(layer_index == 0) {
            uint32_t tag = create_tag(batch_idx_fwd, FWD_PATH);
            msg_size = batch_size*seq_len*hidden_size/tp;
            recv(evQ, msg_size, dtype, my_rank-tp, tag, GroupWorld);
         }

         enQ_compute(evQ, compute_time[current_stage]);

         msg_size = batch_size*seq_len*hidden_size/tp;
         allreduce(evQ, msg_size, dtype, Hermes::MP::SUM, tp_group[tp_grp_idx]);

         next_stage = exec_stage_e::MLP_FWD;
         break;
      }
      case MLP_FWD: {
         uint64_t msg_size = num_params[current_stage]/(dp*tp);
         allgather(evQ, msg_size, dtype, dp_group[dp_grp_idx]);

         enQ_compute(evQ, compute_time[current_stage]);

         layer_index++;
         msg_size = batch_size*seq_len*hidden_size/tp;
         allreduce(evQ, msg_size, dtype, Hermes::MP::SUM, tp_group[tp_grp_idx]);

         if(layer_index == n_layer_to_execute) {
            layer_index = 0;
            uint32_t tag = create_tag(batch_idx_fwd, FWD_PATH);
            uint64_t msg_size = (batch_size*seq_len*hidden_size)/tp;
            isend(evQ, msg_size, dtype, my_rank + tp, tag, GroupWorld, req);

            batch_idx_fwd++;

            if(batch_idx_fwd < (pp-pp_idx) && batch_idx_fwd < n_batch) {
               next_stage = exec_stage_e::ATT_FWD;
            } else {
               next_stage = exec_stage_e::MLP_BCK;
            }
         } else {
            next_stage = exec_stage_e::ATT_FWD;
         }
         break;
      }
      case LOGITS: {
         uint64_t msg_size = num_params[current_stage]/(dp*tp);
         allgather(evQ, msg_size, dtype, dp_group[dp_grp_idx]);

         uint32_t tag = create_tag(batch_idx_fwd, FWD_PATH);
         msg_size = (batch_size*seq_len*hidden_size)/tp;
         recv(evQ, msg_size, dtype, my_rank-tp, tag, GroupWorld);

         enQ_compute(evQ, compute_time[current_stage]);

         msg_size = batch_size*seq_len*hidden_size/tp;
         allgather(evQ, msg_size, dtype, tp_group[tp_grp_idx]);
         next_stage = exec_stage_e::FWD_MAX;
         break;
      }
      case FWD_MAX: {
         enQ_compute(evQ, compute_time[current_stage]);

         uint64_t msg_size =  batch_size*seq_len/tp;
         allreduce(evQ, msg_size, INT32_T, Hermes::MP::MAX, tp_group[tp_grp_idx]);

         next_stage = exec_stage_e::FWD_SOFTMAX;
         break;
      }
      case FWD_SOFTMAX: {
         enQ_compute(evQ, compute_time[current_stage]);

         uint64_t msg_size = 2*batch_size*seq_len/tp;
         allreduce(evQ, msg_size, INT32_T, Hermes::MP::SUM, tp_group[tp_grp_idx]);

         next_stage = exec_stage_e::LOSS;
         break;
      }
      case LOSS: {
         enQ_compute(evQ, compute_time[current_stage]);

         uint64_t msg_size = batch_size*seq_len*vocab_size/tp;
         allreduce(evQ, msg_size, dtype, Hermes::MP::SUM, tp_group[tp_grp_idx]);

         next_stage = exec_stage_e::BCK_LINEAR;
         batch_idx_fwd++;
         break;
      }
      case BCK_LINEAR: {
         uint64_t msg_size = num_params[current_stage]/(dp*tp);
         allgather(evQ, msg_size, dtype, dp_group[dp_grp_idx]);

         enQ_compute(evQ, compute_time[current_stage]);

         msg_size = batch_size*seq_len*hidden_size/tp;
         allreduce(evQ, msg_size, dtype, Hermes::MP::SUM, tp_group[tp_grp_idx]);

         next_stage = exec_stage_e::BCK_NORM;
         break;
      }
      case BCK_NORM: {
         enQ_compute(evQ, compute_time[current_stage]);

         uint64_t msg_size = batch_size*seq_len*hidden_size/tp;
         allgather(evQ, msg_size, dtype, tp_group[tp_grp_idx]);

         uint32_t tag = create_tag(batch_idx_bck, BCK_PATH);
         msg_size = batch_size*seq_len*hidden_size/tp;
         isend(evQ, msg_size, dtype, my_rank-tp, tag, GroupWorld, req);

         batch_idx_bck++;
         if(batch_idx_bck >= n_batch) {
            finish = true;
         }
         next_stage = exec_stage_e::LOGITS;
         break;
      }
      case MLP_BCK: {
         uint64_t msg_size = num_params[current_stage]/(dp*tp);
         allgather(evQ, msg_size, dtype, dp_group[dp_grp_idx]);

         if(layer_index == 0) {
            uint32_t tag = create_tag(batch_idx_bck, BCK_PATH);
            msg_size = batch_size*seq_len*hidden_size/tp;
            recv(evQ, msg_size, dtype, my_rank+tp, tag, GroupWorld);
         }

         enQ_compute(evQ, compute_time[current_stage]);

         msg_size = batch_size*seq_len*hidden_size/tp;
         allreduce(evQ, msg_size, dtype, Hermes::MP::SUM, tp_group[tp_grp_idx]);

         next_stage = exec_stage_e::ATT_BCK;
         break;
      }
      case ATT_BCK: {
         uint64_t msg_size = num_params[current_stage]/(dp*tp);
         allgather(evQ, msg_size, dtype, dp_group[dp_grp_idx]);

         enQ_compute(evQ, compute_time[current_stage]);

         layer_index++;
         msg_size = batch_size*seq_len*hidden_size/tp;
         allreduce(evQ, msg_size, dtype, Hermes::MP::SUM, tp_group[tp_grp_idx]);

         if(layer_index == n_layer_to_execute) {
            uint32_t tag = create_tag(batch_idx_bck, BCK_PATH);
            msg_size = batch_size*seq_len*hidden_size/tp;
            isend(evQ, msg_size, dtype, my_rank-tp, tag, GroupWorld, req);

            layer_index = 0;
            batch_idx_bck++;


            int64_t limit = (n_batch - (pp - pp_idx));
            limit = (limit > 0) ? limit : 0;
            if(batch_idx_bck > limit) {
               next_stage = exec_stage_e::MLP_BCK;
            } else {
               next_stage = exec_stage_e::ATT_FWD;
            }

            if(batch_idx_bck >= n_batch) {
               finish = true;
            }
         } else {
            next_stage = exec_stage_e::MLP_BCK;
         }
         assert(layer_index < n_layer_to_execute);

         break;
      }
      case UPDATE: {
         uint32_t tag = create_tag(batch_idx_bck, BCK_PATH);
         uint64_t msg_size = batch_size*seq_len*hidden_size/tp;
         recv(evQ, msg_size, dtype, my_rank+tp, tag, GroupWorld);

         enQ_compute(evQ, compute_time[current_stage]);

         next_stage = exec_stage_e::TOKENIZE;
         batch_idx_bck++;

         int64_t limit = (n_batch - (pp - pp_idx));
         limit = (limit > 0) ? limit : 0;
         if(batch_idx_bck > limit) {
            next_stage = current_stage;
         } else {
            next_stage = exec_stage_e::TOKENIZE;
         }

         if(batch_idx_bck >= n_batch) {
            finish = true;
         }
         break;
      }
      default:
         out->fatal(CALL_INFO_LONG, -1, "Unknown execution stage: %d\n", current_stage);
         break;
   }

   if(dp_idx == 1)
      out->verbose(CALL_INFO, 8, 0, "Current stage: %s\tnext stage: %s\tlayer index:%u\tbatch index:%u/%u\n",
         exec_stage_s[current_stage].c_str(), exec_stage_s[next_stage].c_str(), layer_index, batch_idx_fwd*dp, batch_idx_bck*dp);

   current_stage = next_stage;
   return finish;
}
