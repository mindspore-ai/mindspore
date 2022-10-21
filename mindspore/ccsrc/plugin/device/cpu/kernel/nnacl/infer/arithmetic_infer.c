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
#include "nnacl/infer/broadcast_to_infer.h"

void SetOutputDtypeFormat(const TensorC *input0, const TensorC *input1, TensorC *output) {
  output->format_ = input0->format_;
  output->data_type_ = input0->data_type_;
  // e.g. input0's shape is 1 and input1's shape is 1 15 15 1
  // only regard larger shape size input as the right format input currently
  // legacy problem: if input0 infer failed before, its shape is [-1], and input1's shape is [1,2] which need to
  // be broadcasted. In this case our program will use input1's format, that's wrong and need to be solved later.
  if (input0->data_ != NULL || input0->shape_size_ < input1->shape_size_) {
    output->format_ = input1->format_;
  }
  // when input0 is const, it is quanted before insert quant trans op, so use input1 data type instead
  if (((input0->data_ != NULL) && (input1->data_type_ != kTypeUnknown)) ||
      ((input0->data_type_ == kNumberTypeInt8) && (input1->data_type_ == kNumberTypeFloat32))) {
    output->data_type_ = input1->data_type_;
  }
}

int BroadCastInferShape(const int input_shape0_size, const int input_shape1_size, const int *input_shape0,
                        const int *input_shape1, int *ndim, int *in_shape0, int *in_shape1, int *out_shape,
                        bool *has_broad_cast) {
  if (input_shape0_size > MAX_SHAPE_SIZE || input_shape1_size > MAX_SHAPE_SIZE) {
    return NNACL_ERR;
  }
  MakeUpInputShapes(input_shape0_size, input_shape1_size, input_shape0, input_shape1, ndim, in_shape0, in_shape1);
  if (*ndim >= MAX_SHAPE_SIZE) {
    return NNACL_INFER_INVALID;
  }

  return BroadCastOutputShape(in_shape0, in_shape1, *ndim, out_shape, has_broad_cast);
}

int ArithmeticInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                         OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  ArithmeticParameter *param = (ArithmeticParameter *)parameter;
  NNACL_CHECK_NULL_RETURN_ERR(param);
  param->broadcasting_ = false;

  const TensorC *input0 = inputs[0];
  const TensorC *input1 = inputs[1];
  TensorC *output = outputs[0];

  const int *input_shape0 = input0->shape_;
  size_t input_shape0_size = input0->shape_size_;
  const int *input_shape1 = input1->shape_;
  size_t input_shape1_size = input1->shape_size_;
  SetOutputDtypeFormat(input0, input1, output);

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  int in_shape0[MAX_SHAPE_SIZE] = {0};
  int in_shape1[MAX_SHAPE_SIZE] = {0};
  int output_shape[MAX_SHAPE_SIZE] = {0};
  int ndim = (int)input_shape0_size;
  bool has_broad_cast = false;
  if (BroadCastInferShape(input_shape0_size, input_shape1_size, input_shape0, input_shape1, &ndim, in_shape0, in_shape1,
                          output_shape, &has_broad_cast) != NNACL_OK) {
    return NNACL_ERR;
  }

  SetShapeArray(output, output_shape, ndim);

  param->broadcasting_ = has_broad_cast;
  param->ndim_ = (size_t)ndim;
  if (ndim > MAX_SHAPE_SIZE) {
    return NNACL_ERR;
  }
  memcpy(param->in_shape0_, in_shape0, ndim * sizeof(int));
  memcpy(param->in_shape1_, in_shape1, ndim * sizeof(int));
  memcpy(param->out_shape_, output_shape, ndim * sizeof(int));

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
REG_INFER(BiasAdd, PrimType_BiasAdd, ArithmeticInferShape)
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
