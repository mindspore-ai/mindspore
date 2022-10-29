/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_FP32_ONLINE_FUSION_FP32_SPLIT_REDUCE_CONCAT_F32_ACTIVATION_H_
#define MINDSPORE_NNACL_FP32_ONLINE_FUSION_FP32_SPLIT_REDUCE_CONCAT_F32_ACTIVATION_H_

#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int64_t Fp32SplitReduceSumConcatFusion(const float *src, float *dst, int64_t inner_size, int64_t mid_size,
                                       int *mid_split, int64_t mid_len, int64_t out_size);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_ONLINE_FUSION_FP32_SPLIT_REDUCE_CONCAT_F32_ACTIVATION_H_
