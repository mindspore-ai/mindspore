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
#ifndef MINDSPORE_NNACL_TENSOR_ARRAY_PARAMETER_H_
#define MINDSPORE_NNACL_TENSOR_ARRAY_PARAMETER_H_
#include "nnacl/op_base.h"

typedef struct TensorArrayParameter {
  OpParameter op_parameter_;
  bool dynamic_size_;
  bool identical_element_shapes_;
  int *element_shape_;
  int element_shape_size_;
  int data_type_;
} TensorArrayParameter;

#endif  // MINDSPORE_NNACL_TENSOR_ARRAY_PARAMETER_H_
