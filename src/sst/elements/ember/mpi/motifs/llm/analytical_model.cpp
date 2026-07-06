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

#include "analytical_model.hpp"

const uint64_t get_params_size(const uint64_t hidden_size,
                              const uint64_t vocab_size,
                              const uint64_t num_hidden_layers,
                              const uint64_t num_key_value_heads,
                              const uint64_t intermediate_size,
                              const uint64_t dtype_size) {

   const uint64_t size =
      vocab_size * hidden_size * dtype_size +
      num_hidden_layers * (
            2 * hidden_size * sizeof(float) +
            hidden_size*(2*hidden_size + 2*hidden_size/num_key_value_heads) * dtype_size +
            3 * hidden_size * intermediate_size * dtype_size) +
         hidden_size * sizeof(float) +
         hidden_size * vocab_size * dtype_size;

   return size;

}

std::map<compute_step_e, uint64_t> get_compute_time(const uint64_t batch_size,
                                                    const uint64_t seq_len,
                                                    const uint64_t hidden_size,
                                                    const uint64_t num_hidden_layers,
                                                    const uint64_t num_key_value_heads,
                                                    const uint64_t num_attention_heads,
                                                    const uint64_t intermediate_size,
                                                    const uint64_t vocab_size,
                                                    const int dtype_size,
                                                    const double dram_bw,
                                                    const double peak_flop,
                                                    const int tp) {

      std::map<compute_step_e, uint64_t> compute_time;

      const uint64_t head_him = hidden_size/num_attention_heads;
      const uint64_t params_size = get_params_size(hidden_size, vocab_size, num_hidden_layers, num_key_value_heads, intermediate_size, dtype_size);

      compute_time[compute_step_e::TOKENIZE] = TOKENIZE_DURATION;

      compute_time[compute_step_e::EMBD] = (batch_size*seq_len*sizeof(float) + hidden_size/tp * dtype_size * seq_len * 2)/dram_bw;

      compute_time[compute_step_e::ATT_FWD] =
         // RMS NORM
         (hidden_size * sizeof(float) + 2 * batch_size * seq_len * hidden_size * dtype_size) / dram_bw +
         // Q proj
         (2 * batch_size * seq_len * hidden_size * hidden_size)/(peak_flop * tp) +
         // K & V Proj
         (2 * 2 * batch_size * seq_len * hidden_size * hidden_size/num_key_value_heads)/(peak_flop * tp) +
         // K & V RoPE
         (batch_size * seq_len * dtype_size * (hidden_size + hidden_size/num_key_value_heads))/(tp * dram_bw) +
         // KQ
         (2 * batch_size * num_attention_heads * seq_len * seq_len * head_him)/(peak_flop * tp) +
         // Softmax
         (2 * batch_size * num_attention_heads * seq_len * seq_len * dtype_size) / (dram_bw * tp) +
         // QKV
         (2 * batch_size * num_attention_heads * seq_len * head_him * seq_len) / (peak_flop * tp) +
         // ATT OUT
         (2 * batch_size * seq_len * hidden_size * hidden_size) / (peak_flop * tp) +
         // Residual
         (3 * batch_size * seq_len * hidden_size * dtype_size) / (dram_bw * tp);


      compute_time[compute_step_e::MLP_FWD] =
         // RMS NORM
         (hidden_size * sizeof(float) + 2 * batch_size * seq_len * hidden_size * dtype_size) / dram_bw +
         // UP & Gate
         (2 * 2 * batch_size * seq_len * hidden_size * intermediate_size) / (peak_flop * tp) +
         // Silu
         (2 * batch_size * seq_len * intermediate_size * dtype_size) / (dram_bw * tp) +
         // Mul
         (3 * batch_size * seq_len * intermediate_size * dtype_size) / (dram_bw * tp) +
         // FFN Out
         (2 * batch_size * seq_len * hidden_size * intermediate_size) / (peak_flop * tp) +
         // Residual
         (3 * batch_size * seq_len * hidden_size * dtype_size) / (dram_bw * tp);

      compute_time[compute_step_e::LOGITS] = (hidden_size * sizeof(float) + 2 * batch_size * seq_len * hidden_size / tp * dtype_size) / dram_bw +
         (2 * batch_size * seq_len * vocab_size * hidden_size) / (peak_flop * tp);
      compute_time[compute_step_e::FWD_MAX] = 2*batch_size*seq_len * vocab_size * sizeof(float) / (dram_bw * tp);
      compute_time[compute_step_e::FWD_SOFTMAX] =  2*batch_size*seq_len * vocab_size * sizeof(float) / (dram_bw * tp);

      compute_time[compute_step_e::LOSS] = 2*batch_size*seq_len * vocab_size * sizeof(float) / (dram_bw * tp); // FIXME

      compute_time[compute_step_e::BCK_LINEAR] = (2 * 2 * batch_size * seq_len * vocab_size * hidden_size) / (peak_flop * tp);
      compute_time[compute_step_e::BCK_NORM] = ((4 * batch_size * seq_len * hidden_size * dtype_size)/tp + 3 * hidden_size * sizeof(float))/ dram_bw ;
      compute_time[compute_step_e::MLP_BCK] =
         // RMS NORM
         (3 * hidden_size * sizeof(float) + 4 * batch_size * seq_len * hidden_size * dtype_size) / dram_bw +
         // UP & Gate
         (2 * 2 * 2 * batch_size * seq_len * hidden_size * intermediate_size) / (peak_flop * tp) +
         // Silu
         (4 * batch_size * seq_len * intermediate_size * dtype_size) / (dram_bw * tp) +
         // Mul
         (5 * batch_size * seq_len * intermediate_size * dtype_size) / (dram_bw * tp) +
         // FFN Out
         (2 * 2 * batch_size * seq_len * hidden_size * intermediate_size) / (peak_flop * tp) +
         // Residual
         (5 * batch_size * seq_len * hidden_size * dtype_size) / (dram_bw * tp);



      compute_time[compute_step_e::ATT_BCK] =
         // RMS NORM
         (3 * hidden_size * sizeof(float) + 4 * batch_size * seq_len * hidden_size * dtype_size) / dram_bw +
         // Q proj
         (2 * 2 * batch_size * seq_len * hidden_size * hidden_size)/(peak_flop * tp) +
         // K & V Proj
         (2 * 2 * 2 * batch_size * seq_len * hidden_size * hidden_size/num_key_value_heads)/(peak_flop * tp) +
         // K & V RoPE
         (batch_size * seq_len * dtype_size * (hidden_size + hidden_size/num_key_value_heads))/(tp * dram_bw) + //Fixme
         // KQ
         (2 * 2 * batch_size * num_attention_heads * seq_len * seq_len * head_him)/(peak_flop * tp) +
         // Softmax
         (3 * batch_size * num_attention_heads * seq_len * seq_len * dtype_size) / (dram_bw * tp) +
         // QKV
         (2 * 2 * batch_size * num_attention_heads * seq_len * head_him * seq_len) / (peak_flop * tp) +
         // ATT OUT
         (2 * 2 * batch_size * seq_len * hidden_size * hidden_size) / (peak_flop * tp) +
         // Residual
         (5 * batch_size * seq_len * hidden_size * dtype_size) / (dram_bw * tp);



      compute_time[compute_step_e::UPDATE] = 7 * params_size / (dram_bw * tp);


      return compute_time;
}
