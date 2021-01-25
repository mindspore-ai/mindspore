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

#include "nnacl/infer/arithmetic_grad_infer.h"

int ArithmeticGradInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                             OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 3, 2);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  const TensorC *dy = inputs[0];
  const TensorC *x1 = inputs[1];
  const TensorC *x2 = inputs[2];
  TensorC *dx1 = outputs[0];
  TensorC *dx2 = outputs[1];

  ArithmeticGradParameter *param = (ArithmeticGradParameter *)parameter;
  if ((param->type_ == PrimitiveType_MaximumGrad) || (param->type_ == PrimitiveType_MinimumGrad)) {
    x1 = inputs[0];
    x2 = inputs[1];
    dy = inputs[2];
  }

  int inShape0[MAX_SHAPE_SIZE];
  size_t inShape0_size = 0;
  ShapeSet(inShape0, &inShape0_size, x1->shape_, x1->shape_size_);
  int inShape1[MAX_SHAPE_SIZE];
  size_t inShape1_size = 0;
  ShapeSet(inShape1, &inShape1_size, x2->shape_, x2->shape_size_);
  int outShape[MAX_SHAPE_SIZE];
  size_t outShape_size = 0;
  ShapeSet(outShape, &outShape_size, dy->shape_, dy->shape_size_);

  if ((param->type_ == PrimitiveType_AddGrad) || (param->type_ == PrimitiveType_SubGrad) ||
      (param->type_ == PrimitiveType_MaximumGrad) || (param->type_ == PrimitiveType_MinimumGrad)) {
    param->ndim_ = outShape_size;
    param->x1_shape_size_ = param->ndim_;
    param->x2_shape_size_ = param->ndim_;
    param->dy_shape_size_ = param->ndim_;
    int fillDimNum0 = outShape_size - inShape0_size;
    int fillDimNum1 = outShape_size - inShape1_size;
    int j0 = 0;
    int j1 = 0;
    for (unsigned int i = 0; i < outShape_size; i++) {
      param->x1_shape_[i] = (i < fillDimNum0) ? 1 : inShape0[j0++];
      param->x2_shape_[i] = (i < fillDimNum1) ? 1 : inShape1[j1++];
      param->dy_shape_[i] = outShape[i];
    }
  } else {
    if (GetElementNum(dx1) < GetElementNum(dx2)) {
      param->ndim_ = inShape1_size;
      param->x1_shape_size_ = param->ndim_;
      param->x2_shape_size_ = param->ndim_;
      param->dy_shape_size_ = param->ndim_;
      int fillDimNum = inShape1_size - inShape0_size;  // This will not work for batch!
      int j = 0;
      for (unsigned int i = 0; i < inShape1_size; i++) {
        if (i < fillDimNum) {
          param->x2_shape_[i] = 1;
        } else {
          param->x2_shape_[i] = inShape0[j++];
        }
        param->x1_shape_[i] = inShape1[i];
        param->dy_shape_[i] = outShape[i];
      }
    } else if (GetElementNum(dx2) < GetElementNum(dx1)) {  // if (inShape0.size() > inShape1.size())
      param->ndim_ = inShape0_size;
      param->x1_shape_size_ = param->ndim_;
      param->x2_shape_size_ = param->ndim_;
      param->dy_shape_size_ = param->ndim_;
      param->broadcasting_ = true;
      int j = 0;
      int fillDimNum = inShape0_size - inShape1_size;
      for (unsigned int i = 0; i < inShape0_size; i++) {
        if (i < fillDimNum) {
          param->x2_shape_[i] = 1;
        } else {
          param->x2_shape_[i] = inShape1[j++];
        }
        param->x1_shape_[i] = inShape0[i];
        param->dy_shape_[i] = outShape[i];
      }
    } else {
      param->broadcasting_ = false;
      for (unsigned int i = 0; i < inShape0_size; i++) {
        param->x2_shape_[i] = inShape1[i];
        param->x1_shape_[i] = inShape0[i];
        param->dy_shape_[i] = outShape[i];
      }
    }
  }

  SetShapeTensor(dx1, x1);
  SetShapeTensor(dx2, x2);
  dx1->data_type_ = dy->data_type_;
  dx2->data_type_ = dy->data_type_;
  return NNACL_OK;
}
