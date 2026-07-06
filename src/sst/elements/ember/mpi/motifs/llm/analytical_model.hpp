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
#ifndef ANALYTICAL_MODEL_H
#define ANALYTICAL_MODEL_H
#include <cstdint>
#include <map>

#define TOKENIZE_DURATION 1e6

enum compute_step_e : uint8_t {
   TOKENIZE,
   EMBD,
   ATT_FWD,
   MLP_FWD,
   LOGITS,
   FWD_MAX,
   FWD_SOFTMAX,
   LOSS,
   BCK_LINEAR,
   BCK_NORM,
   MLP_BCK,
   ATT_BCK,
   UPDATE
};

const uint64_t get_params_size(const uint64_t hidden_size,
                              const uint64_t vocab_size,
                              const uint64_t num_hidden_layers,
                              const uint64_t num_key_value_heads,
                              const uint64_t intermediate_size,
                              const uint64_t dtype_size);

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
                                                    const int tp);

#endif /* end of #ifndef ANALYTICAL_MODEL_H scope */
