/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_NNACL_FP32_SPLICE_FP32_H_
#define MINDSPORE_LITE_NNACL_FP32_SPLICE_FP32_H_
#include <string.h>
#include "nnacl/splice_parameter.h"
#ifdef __cplusplus
extern "C" {
#endif

void SpliceFp32(const float *src_data, int src_row, int src_col, const SpliceParameter *splice_parameter,
                float *dst_data, int dst_row, int dst_col);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_NNACL_FP32_SPLICE_FP32_H_
