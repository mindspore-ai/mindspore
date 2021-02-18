/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MINDSPORE_LITE_NNACL_FP16_GRU_H_
#define MINDSPORE_LITE_NNACL_FP16_GRU_H_
#include "nnacl/gru_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
void GruFp16(float16_t *output, const float16_t *input, const float16_t *weight_g, const float16_t *weight_r,
             const float16_t *bias, float16_t *hidden_state, float16_t *gate_buffer, float16_t *matmul_buffer[2],
             int check_seq_len, const GruParameter *gru_param);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP16_GRU_H_
