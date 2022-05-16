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
#ifndef MINDSPORE_NNACL_FP32_BROADCAST_TO_H_
#define MINDSPORE_NNACL_FP32_BROADCAST_TO_H_

#include "nnacl/broadcast_to_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
#define BROADCAST_TO(type, input, shape_info, output) broadcast_to_##type(input, shape_info, output)
int broadcast_to_bool(const bool *input, BroadcastShapeInfo *shape_info, bool *output);
int broadcast_to_int8_t(const int8_t *input, BroadcastShapeInfo *shape_info, int8_t *output);
int broadcast_to_int16_t(const int16_t *input, BroadcastShapeInfo *shape_info, int16_t *output);
int broadcast_to_int32_t(const int32_t *input, BroadcastShapeInfo *shape_info, int32_t *output);
int broadcast_to_int64_t(const int64_t *input, BroadcastShapeInfo *shape_info, int64_t *output);
int broadcast_to_int(const int *input, BroadcastShapeInfo *shape_info, int *output);
int broadcast_to_float(const float *input, BroadcastShapeInfo *shape_info, float *output);
#ifdef ENABLE_FP16
int broadcast_to_float16_t(const float16_t *input, BroadcastShapeInfo *shape_info, float16_t *output);
#endif
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_BROADCAST_TO_H_
