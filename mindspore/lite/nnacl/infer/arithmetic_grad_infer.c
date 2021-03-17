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
#include "nnacl/arithmetic.h"
#include "nnacl/infer/infer_register.h"

/*
 * the Arithmetic Grad op include AddGrad, SubGrad, MulGrad, DivGrad, MaximumGrad, MinimumGrad
 * according to the arithmetic_fp32.h now
 * the MaximumGrad, MinimumGrad run through MaximumGradInfershape
 * the AddGrad, SubGrad run through AddSubGradInfershape
 * the others run through this function
 * */
int ArithmeticGradInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
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

  int in_shape0[MAX_SHAPE_SIZE];
  size_t in_shape0_size = 0;
  ShapeSet(in_shape0, &in_shape0_size, x1->shape_, x1->shape_size_);
  int in_shape1[MAX_SHAPE_SIZE];
  size_t in_shape1_size = 0;
  ShapeSet(in_shape1, &in_shape1_size, x2->shape_, x2->shape_size_);
  int out_shape[MAX_SHAPE_SIZE];
  size_t out_shape_size = 0;
  ShapeSet(out_shape, &out_shape_size, dy->shape_, dy->shape_size_);

  ArithmeticParameter *param = (ArithmeticParameter *)parameter;

  if (GetElementNum(dx1) < GetElementNum(dx2)) {
    param->ndim_ = in_shape1_size;
    param->in_elements_num0_ = param->ndim_;
    param->in_elements_num1_ = param->ndim_;
    param->out_elements_num_ = param->ndim_;
    int fill_dim_num = in_shape1_size - in_shape0_size;  // This will not work for batch!
    int j = 0;
    for (unsigned int i = 0; i < in_shape1_size; i++) {
      if (i < fill_dim_num) {
        param->in_shape1_[i] = 1;
      } else {
        param->in_shape1_[i] = in_shape0[j++];
      }
      param->in_shape0_[i] = in_shape1[i];
      param->out_shape_[i] = out_shape[i];
    }
  } else if (GetElementNum(dx2) < GetElementNum(dx1)) {
    param->ndim_ = in_shape0_size;
    param->in_elements_num0_ = param->ndim_;
    param->in_elements_num1_ = param->ndim_;
    param->out_elements_num_ = param->ndim_;
    param->broadcasting_ = true;
    int j = 0;
    int fill_dim_num = in_shape0_size - in_shape1_size;
    for (unsigned int i = 0; i < in_shape0_size; i++) {
      if (i < fill_dim_num) {
        param->in_shape1_[i] = 1;
      } else {
        param->in_shape1_[i] = in_shape1[j++];
      }
      param->in_shape0_[i] = in_shape0[i];
      param->out_shape_[i] = out_shape[i];
    }
  } else {
    param->broadcasting_ = false;
    for (unsigned int i = 0; i < in_shape0_size; i++) {
      param->in_shape1_[i] = in_shape1[i];
      param->in_shape0_[i] = in_shape0[i];
      param->out_shape_[i] = out_shape[i];
    }
  }

  SetShapeTensor(dx1, x1);
  SetShapeTensor(dx2, x2);
  dx1->data_type_ = dy->data_type_;
  dx2->data_type_ = dy->data_type_;
  return NNACL_OK;
}

REG_INFER(DivGrad, PrimType_DivGrad, ArithmeticGradInferShape)
REG_INFER(MulGrad, PrimType_MulGrad, ArithmeticGradInferShape)
REG_INFER(MinimumGrad, PrimType_MinimumGrad, ArithmeticGradInferShape)
