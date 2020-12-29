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

#ifndef MINDSPORE_LITE_NNACL_FP16_POOLING_FP16_H_
#define MINDSPORE_LITE_NNACL_FP16_POOLING_FP16_H_

#include <math.h>
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/pooling_parameter.h"
#ifdef __cplusplus
extern "C" {
#endif
int AvgPoolingFp16(const float16_t *input_ptr, float16_t *output_ptr, PoolingParameter *pooling_param, int task_id,
                   float16_t min, float16_t max);

void MaxPoolingFp16(const float16_t *input_ptr, float16_t *output_ptr, PoolingParameter *pooling_param, int task_id,
                    float16_t min, float16_t max);
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_NNACL_FP16_POOLING_FP16_H_
