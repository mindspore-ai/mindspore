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
#include "nnacl/arithmetic.h"
#include "nnacl/infer/arithmetic_grad_infer.h"
#include "nnacl/infer/infer_register.h"

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

  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  ArithmeticParameter *param = (ArithmeticParameter *)parameter;

  param->ndim_ = dy->shape_size_;
  param->in_elements_num0_ = param->ndim_;
  param->in_elements_num1_ = param->ndim_;
  param->out_elements_num_ = param->ndim_;
  int fillDimNum0 = dy->shape_size_ - x1->shape_size_;
  int fillDimNum1 = dy->shape_size_ - x2->shape_size_;
  int j0 = 0;
  int j1 = 0;
  for (unsigned int i = 0; i < dy->shape_size_; i++) {
    param->in_shape0_[i] = (i < fillDimNum0) ? 1 : x1->shape_[j0++];
    param->in_shape1_[i] = (i < fillDimNum1) ? 1 : x2->shape_[j1++];
    param->out_shape_[i] = dy->shape_[i];
  }

  SetShapeTensor(dx1, x1);
  SetShapeTensor(dx2, x2);
  SetDataTypeFormat(dx1, dy);
  SetDataTypeFormat(dx2, dy);
  return NNACL_OK;
}

REG_INFER(AddGrad, PrimType_AddGrad, AddSubGradInferShape)
REG_INFER(SubGrad, PrimType_SubGrad, AddSubGradInferShape)
