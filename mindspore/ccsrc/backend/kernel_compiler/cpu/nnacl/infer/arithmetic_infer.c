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

#include "nnacl/infer/arithmetic_infer.h"
#include "nnacl/infer/infer_register.h"

void UpdateInputShape(const int input_shape0_size, const int input_shape1_size, int *ndim, const int *input_shape0,
                      const int *input_shape1, int *in_shape0, int *in_shape1) {
  if (input_shape0_size < input_shape1_size) {
    *ndim = input_shape1_size;
    int fill_dim_num = input_shape1_size - input_shape0_size;
    int j = 0;
    for (size_t i = 0; i < input_shape1_size; i++) {
      if (i < fill_dim_num) {
        in_shape0[i] = 1;
      } else {
        in_shape0[i] = input_shape0[j++];
      }
      in_shape1[i] = input_shape1[i];
    }
  } else if (input_shape0_size > input_shape1_size) {
    *ndim = input_shape0_size;
    int fill_dim_num = input_shape0_size - input_shape1_size;
    int j = 0;
    for (size_t i = 0; i < input_shape0_size; i++) {
      if (i < fill_dim_num) {
        in_shape1[i] = 1;
      } else {
        in_shape1[i] = input_shape1[j++];
      }
      in_shape0[i] = input_shape0[i];
    }
  } else {
    for (size_t i = 0; i < input_shape0_size; i++) {
      in_shape1[i] = input_shape1[i];
      in_shape0[i] = input_shape0[i];
    }
  }
}

int ArithmeticInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                         OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  ArithmeticParameter *param = (ArithmeticParameter *)parameter;
  param->broadcasting_ = false;

  const TensorC *input0 = inputs[0];
  const TensorC *input1 = inputs[1];
  TensorC *output = outputs[0];

  const int *input_shape0 = input0->shape_;
  size_t input_shape0_size = input0->shape_size_;
  const int *input_shape1 = input1->shape_;
  size_t input_shape1_size = input1->shape_size_;
  output->format_ = input0->format_;
  output->data_type_ = input0->data_type_;
  if ((input0->data_type_ == kNumberTypeInt8) && (input1->data_type_ == kNumberTypeFloat32)) {
    output->data_type_ = input1->data_type_;
  }

  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  if (input_shape0_size >= MAX_SHAPE_SIZE || input_shape1_size >= MAX_SHAPE_SIZE) {
    return NNACL_ERR;
  }
  int in_shape0[10];
  int in_shape1[10];
  int out_shape[10];
  int ndim = input_shape0_size;
  UpdateInputShape(input_shape0_size, input_shape1_size, &ndim, input_shape0, input_shape1, in_shape0, in_shape1);

  int output_shape[MAX_SHAPE_SIZE] = {0};
  size_t output_shape_size = 0;
  if (ndim >= MAX_SHAPE_SIZE) {
    return NNACL_INFER_INVALID;
  }
  for (int i = 0; i < ndim; i++) {
    if (in_shape0[i] != in_shape1[i]) {
      if (in_shape0[i] == 1) {
        out_shape[i] = in_shape1[i];
      } else if (in_shape1[i] == 1) {
        out_shape[i] = in_shape0[i];
      } else {
        return NNACL_ERR;
      }
      param->broadcasting_ = true;
    } else {
      out_shape[i] = in_shape0[i];
    }
    output_shape[output_shape_size] = out_shape[i];
    output_shape_size++;
  }

  SetShapeArray(output, output_shape, output_shape_size);

  param->ndim_ = ndim;
  memcpy(param->in_shape0_, in_shape0, ndim * sizeof(int));
  memcpy(param->in_shape1_, in_shape1, ndim * sizeof(int));
  memcpy(param->out_shape_, out_shape, ndim * sizeof(int));

  param->in_elements_num0_ = 1;
  param->in_elements_num1_ = 1;
  param->out_elements_num_ = 1;
  for (int i = 0; i < ndim; i++) {
    param->in_elements_num0_ *= param->in_shape0_[i];
    param->in_elements_num1_ *= param->in_shape1_[i];
    param->out_elements_num_ *= param->out_shape_[i];
  }
  return NNACL_OK;
}

REG_INFER(Add, PrimType_AddFusion, ArithmeticInferShape)
REG_INFER(Div, PrimType_DivFusion, ArithmeticInferShape)
REG_INFER(Eltwise, PrimType_Eltwise, ArithmeticInferShape)
REG_INFER(FloorDiv, PrimType_FloorDiv, ArithmeticInferShape)
REG_INFER(FloorMod, PrimType_FloorMod, ArithmeticInferShape)
REG_INFER(LogicalAnd, PrimType_LogicalAnd, ArithmeticInferShape)
REG_INFER(LogicalOr, PrimType_LogicalOr, ArithmeticInferShape)
REG_INFER(Maximum, PrimType_Maximum, ArithmeticInferShape)
REG_INFER(Minimum, PrimType_Minimum, ArithmeticInferShape)
REG_INFER(Mod, PrimType_Mod, ArithmeticInferShape)
REG_INFER(Mul, PrimType_MulFusion, ArithmeticInferShape)
REG_INFER(RealDiv, PrimType_RealDiv, ArithmeticInferShape)
REG_INFER(Sub, PrimType_SubFusion, ArithmeticInferShape)
REG_INFER(SquaredDifference, PrimType_SquaredDifference, ArithmeticInferShape)
