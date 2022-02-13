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
#ifndef MINDSPORE_NNACL_FP16_PAD_FP16_H_
#define MINDSPORE_NNACL_FP16_PAD_FP16_H_

#include "nnacl/fp32/pad_fp32.h"

#ifdef __cplusplus
extern "C" {
#endif
void PadFp16(const float16_t *input_data, float16_t *output_data, const int *input_shape, const int *output_shape,
             const int *paddings, int tid, int thread_num);
void MirrorPadFp16(const float16_t *input_data, float16_t *output_data, const int *input_shape,
                   const PadParameter *pad_param, int begin, int end);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP16_PAD_FP16_H_
