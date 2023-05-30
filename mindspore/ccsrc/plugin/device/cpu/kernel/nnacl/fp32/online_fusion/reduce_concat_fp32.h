/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_FP32_ONLINE_FUSION_FP32_REDUCE_CONCAT_F32_ACTIVATION_H_
#define MINDSPORE_NNACL_FP32_ONLINE_FUSION_FP32_REDUCE_CONCAT_F32_ACTIVATION_H_

#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int64_t Fp32ReduceSumConcatFusion(float *output_data, float **input_datas, const int64_t *reduce_axis_size,
                                  int64_t input_nums, int64_t batch, int64_t batch_tile_size, int64_t inner_tile,
                                  int64_t thread_num, int64_t task_id);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_ONLINE_FUSION_FP32_REDUCE_CONCAT_F32_ACTIVATION_H_
