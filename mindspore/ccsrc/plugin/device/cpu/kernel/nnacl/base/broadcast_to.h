/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef NNACL_FP32_BROADCAST_TO_H_
#define NNACL_FP32_BROADCAST_TO_H_

#include "nnacl/broadcast_to_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
#define BYTE_SIZE 8
int BroadcastToSize8(const void *input, BroadcastShapeInfo *shape_info, void *output);
int BroadcastToSize16(const void *input, BroadcastShapeInfo *shape_info, void *output);
int BroadcastToSize32(const void *input, BroadcastShapeInfo *shape_info, void *output);
int BroadcastToSize64(const void *input, BroadcastShapeInfo *shape_info, void *output);
int BroadcastToSize128(const void *input, BroadcastShapeInfo *shape_info, void *output);
#ifdef __cplusplus
}
#endif

#endif  //  NNACL_FP32_BROADCAST_TO_H_
