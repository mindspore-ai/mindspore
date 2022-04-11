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

#ifndef MINDSPORE_NNACL_TENSORLISTC_UTILS_H_
#define MINDSPORE_NNACL_TENSORLISTC_UTILS_H_

#include <stddef.h>
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/infer/common_infer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vvector {
  int **shape_;      // value of shapes
  int *shape_size_;  // size of shape
  size_t size_;      // number of shapes
} vvector;

typedef struct TensorListC {
  bool is_ready_;
  int data_type_;
  int format_;
  int shape_value_;
  int tensors_data_type_;  // element_data_type_, keep same as c++
  int max_elements_num_;
  int element_shape_[8];
  size_t element_num_;
  size_t element_shape_size_;
  TensorC *tensors_;
} TensorListC;

int MallocTensorListData(TensorListC *tensor_list, TypeIdC dtype, const vvector *tensor_shape);
int TensorListMergeShape(int *element_shape, size_t *element_shape_size, const int *tmp, size_t tmp_size);
bool TensorListIsFullyDefined(const int *shape, size_t shape_size);
bool InferFlagTensorList(TensorC *tensor_list);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_TENSORLISTC_UTILS_H_
