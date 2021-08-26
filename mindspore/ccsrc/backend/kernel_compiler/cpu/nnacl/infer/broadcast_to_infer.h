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
#ifndef MINDSPORE_NNACL_BROADCAST_TO_INFER_H
#define MINDSPORE_NNACL_BROADCAST_TO_INFER_H

#include "nnacl/infer/common_infer.h"
#include "nnacl/base/broadcast_to.h"

#ifdef __cplusplus
extern "C" {
#endif

int BroadcastToInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outpus_size,
                          OpParameter *parameter);
void MakeUpInputShapes(const int input_shape0_size, const int input_shape1_size, const int *input_shape0,
                       const int *input_shape1, int *ndim, int *in_shape0, int *in_shape1);
int BroadCastOutputShape(const int *in_shape0, const int *in_shape1, const int ndim, int *out_shape,
                         bool *has_broad_cast);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_BROADCAST_TO_INFER_H
