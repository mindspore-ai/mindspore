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
#ifndef NNACL_BASE_SLICE_BASE_H_
#define NNACL_BASE_SLICE_BASE_H_

#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"
#include "nnacl/slice_parameter.h"
#include "nnacl/kernel/slice.h"

#ifdef __cplusplus
extern "C" {
#endif
void InitSliceStruct(SliceStruct *slice, TensorC *in_tensor, TensorC *begin_tensor, TensorC *size_tensor);
void PadSliceParameterTo8D(SliceStruct *param);

void DoSlice(const void *input, void *output, const SliceStruct *param, int thread_id, int thread_num, int data_size);
void DoSliceNoParallel(const void *input, void *output, const SliceStruct *param, int data_size);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_BASE_SLICE_BASE_H_
