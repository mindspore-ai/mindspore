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

#ifndef MINDSPORE_LITE_NNACL_FP16_WINOGRAD_UTILS_H_
#define MINDSPORE_LITE_NNACL_FP16_WINOGRAD_UTILS_H_

#include <arm_neon.h>
#include "nnacl/conv_parameter.h"
#include "nnacl/op_base.h"

#define MAX_LEN 256

#ifdef __cplusplus
extern "C" {
#endif
void GeneralInputTransformUnitFp16(const float16_t *src_data, float16_t *dst_data, float16_t *matrix_b,
                                   float16_t *matrix_bt, int src_step, int dst_step, int in_unit);

void GeneralOutputTransformUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    float16_t *matrix_a, float16_t *matrix_at, int src_step, int dst_step, int in_unit,
                                    int out_unit);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP16_WINOGRAD_UTILS_H_
