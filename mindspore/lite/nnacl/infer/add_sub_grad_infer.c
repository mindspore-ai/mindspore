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

#include "nnacl/infer/add_sub_grad_infer.h"
#include "nnacl/infer/arithmetic_grad_infer.h"

int AddSubGradInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                         OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 3, 2);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *dy = inputs[0];
  const TensorC *x1 = inputs[1];
  const TensorC *x2 = inputs[2];
  TensorC *dx1 = outputs[0];
  TensorC *dx2 = outputs[1];

  ArithmeticGradParameter *param = (ArithmeticGradParameter *)parameter;

  int in_shape0[MAX_SHAPE_SIZE];
  size_t in_shape0_size = 0;
  ShapeSet(in_shape0, &in_shape0_size, x1->shape_, x1->shape_size_);
  int in_shape1[MAX_SHAPE_SIZE];
  size_t in_shape1_size = 0;
  ShapeSet(in_shape1, &in_shape1_size, x2->shape_, x2->shape_size_);
  int outShape[MAX_SHAPE_SIZE];
  size_t outShape_size = 0;
  ShapeSet(outShape, &outShape_size, dy->shape_, dy->shape_size_);

  param->ndim_ = outShape_size;
  param->x1_shape_size_ = param->ndim_;
  param->x2_shape_size_ = param->ndim_;
  param->dy_shape_size_ = param->ndim_;
  int fill_dim_num0 = outShape_size - in_shape0_size;
  int fill_dim_num1 = outShape_size - in_shape1_size;
  int j0 = 0;
  int j1 = 0;
  for (unsigned int i = 0; i < outShape_size; i++) {
    param->x1_shape_[i] = (i < fill_dim_num0) ? 1 : in_shape0[j0++];
    param->x2_shape_[i] = (i < fill_dim_num1) ? 1 : in_shape1[j1++];
    param->dy_shape_[i] = outShape[i];
  }

  SetShapeTensor(dx1, x1);
  SetShapeTensor(dx2, x2);
  dx1->data_type_ = dy->data_type_;
  dx2->data_type_ = dy->data_type_;
  return NNACL_OK;
}
