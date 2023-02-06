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
#ifndef MINDSPORE_NNACL_FP32_SPARSE_RESHAPE_FP32_H_
#define MINDSPORE_NNACL_FP32_SPARSE_RESHAPE_FP32_H_

#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif
int SparseReshapeInferOutputShapeFp32(int32_t *in_inshape_ptr, int32_t *in_outshape_ptr, int32_t *out_outshape_ptr,
                                      size_t input_rank, size_t output_rank);

int SparseReshapeInOutCoordTrans(int32_t *in_indices_ptr, int32_t *in_inshape_ptr, int32_t *out_outshape_ptr,
                                 int32_t in_indices_num, int32_t *out_indices_ptr, int32_t *in_stride,
                                 int32_t *out_stride, size_t input_rank, size_t output_rank);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_SPARSE_RESHAPE_FP32_H_
