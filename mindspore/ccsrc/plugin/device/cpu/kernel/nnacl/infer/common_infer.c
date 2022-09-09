/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use tensor file except in compliance with the License.
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

#include "nnacl/infer/common_infer.h"
#include <stdlib.h>
#include <string.h>
#include "nnacl/infer/infer_register.h"
#include "nnacl/op_base.h"

void ShapeSet(int *dst_shape, size_t *dst_shape_size, const int *src_shape, size_t src_shape_size) {
  size_t i = 0;
  for (; i < src_shape_size && i < MAX_SHAPE_SIZE; i++) {
    dst_shape[i] = src_shape[i];
  }
  *dst_shape_size = i;
}

void ShapePush(int *shape, size_t *shape_size, int value) {
  if (*shape_size >= MAX_SHAPE_SIZE) {
    return;
  }
  shape[*shape_size] = value;
  *shape_size = *shape_size + 1;
}

int ShapeInsert(int *shape, size_t *shape_size, int index, int value) {
  if (index < 0 || index > *shape_size) {
    return NNACL_ERR;
  }
  if (*shape_size >= MAX_SHAPE_SIZE) {
    return NNACL_ERR;
  }
  for (int i = *shape_size; i > index; i--) {
    shape[i] = shape[i - 1];
  }
  shape[index] = value;
  *shape_size = *shape_size + 1;
  return NNACL_OK;
}

int ShapeErase(int *shape, size_t *shape_size, int index) {
  if (index < 0 || index >= *shape_size) {
    return NNACL_ERR;
  }

  for (int i = index; i < *shape_size - 1; i++) {
    shape[i] = shape[i + 1];
  }
  *shape_size = *shape_size - 1;
  return NNACL_OK;
}

bool ShapeEqual(const int *shape0, size_t shape0_size, const int *shape1, size_t shape1_size) {
  if (shape0_size != shape1_size) {
    return false;
  }
  for (size_t i = 0; i < shape0_size; i++) {
    if (shape0[i] != shape1[i]) {
      return false;
    }
  }
  return true;
}

void iswap(int *a, int *b) {
  int tmp = *a;
  *a = *b;
  *b = tmp;
}

int imin(int a, int b) { return a > b ? b : a; }

int imax(int a, int b) { return a < b ? b : a; }

// input == output completely refer to
// 1. zeros_like
int CommonInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
  if (parameter == NULL || inputs[0] == NULL || outputs[0] == NULL) {
    return NNACL_NULL_PTR;
  }
  SetDataTypeFormat(outputs[0], inputs[0]);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  SetShapeTensor(outputs[0], inputs[0]);
  return NNACL_OK;
}

int CommonGradInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                         OpParameter *parameter) {
  int ret = CheckAugmentNullInputSize(inputs, inputs_size, outputs, outputs_size, parameter, 2);
  if (ret != NNACL_OK) {
    return ret;
  }
  SetDataTypeFormat(outputs[0], inputs[0]);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  MS_CHECK_TRUE_RET(inputs[0]->shape_size_ == inputs[1]->shape_size_, NNACL_ERR);
  for (int i = 0; i < inputs[0]->shape_size_; i++) {
    if (inputs[0]->shape_[i] != inputs[1]->shape_[i]) {
      return NNACL_ERR;
    }
  }
  SetShapeTensor(outputs[0], inputs[0]);
  return NNACL_OK;
}

int CommonInferShapeWithOneInput(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                 size_t outputs_size, OpParameter *parameter) {
  int ret = CheckAugmentNullInputSize(inputs, inputs_size, outputs, outputs_size, parameter, 1);
  if (ret != NNACL_OK) {
    return ret;
  }
  SetDataTypeFormat(outputs[0], inputs[0]);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  SetShapeTensor(outputs[0], inputs[0]);
  return NNACL_OK;
}

int CommonInferShapeWithTwoInput(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                 size_t outputs_size, OpParameter *parameter) {
  int ret = CheckAugmentNullInputSize(inputs, inputs_size, outputs, outputs_size, parameter, 2);
  if (ret != NNACL_OK) {
    return ret;
  }
  SetDataTypeFormat(outputs[0], inputs[0]);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  SetShapeTensor(outputs[0], inputs[0]);
  return NNACL_OK;
}

int CommonInferShapeWithNHWC(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                             OpParameter *parameter) {
  if (parameter == NULL || inputs[0] == NULL || outputs[0] == NULL) {
    return NNACL_NULL_PTR;
  }
  if (inputs[0]->format_ != Format_NHWC) {
    return NNACL_FORMAT_ERROR;
  }
  SetDataTypeFormat(outputs[0], inputs[0]);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  SetShapeTensor(outputs[0], inputs[0]);
  return NNACL_OK;
}

int FftInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                  const OpParameter *parameter) {
  int ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (ret != NNACL_OK) {
    return ret;
  }
  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  output->data_type_ = kNumberTypeFloat32;
  output->format_ = input->format_;
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  if (input->shape_size_ > MAX_SHAPE_SIZE) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  int input_shape[MAX_SHAPE_SIZE] = {0};
  size_t input_shape_size = 0;
  ShapeSet(input_shape, &input_shape_size, input->shape_, input->shape_size_);
  if (input_shape_size == 0) {
    return NNACL_ERR;
  }
  input_shape_size--;
  SetShapeArray(output, input_shape, input_shape_size);
  return NNACL_OK;
}

bool InferFlag(const TensorC *const *inputs, size_t inputs_size) {
  if (inputs == NULL) {
    return false;
  }
  for (size_t i = 0; i < inputs_size; i++) {
    if (inputs[i] == NULL) {
      return false;
    }
    if (inputs[i]->data_type_ == kObjectTypeTensorType) {
      if (InferFlagTensorList((TensorC *)inputs[i]) == false) {
        return false;
      }
    } else {
      for (size_t j = 0; j < inputs[i]->shape_size_; ++j) {
        if (inputs[i]->shape_[j] == -1) {
          return false;
        }
      }
    }
  }
  return true;
}

REG_INFER(Abs, PrimType_Abs, CommonInferShape)
REG_INFER(AbsGrad, PrimType_AbsGrad, CommonGradInferShape)
REG_INFER(Activation, PrimType_Activation, CommonInferShape)
REG_INFER(BatchNorm, PrimType_BatchNorm, CommonInferShape)
REG_INFER(BinaryCrossEntropyGrad, PrimType_BinaryCrossEntropyGrad, CommonInferShape)
REG_INFER(Ceil, PrimType_Ceil, CommonInferShape)
REG_INFER(Clip, PrimType_Clip, CommonInferShape)
REG_INFER(Cos, PrimType_Cos, CommonInferShape)
REG_INFER(Depend, PrimType_Depend, CommonInferShape)
REG_INFER(Elu, PrimType_Elu, CommonInferShape)
REG_INFER(Erf, PrimType_Erf, CommonInferShape)
REG_INFER(Exp, PrimType_ExpFusion, CommonInferShape)
REG_INFER(FakeQuantWithMinMaxVars, PrimType_FakeQuantWithMinMaxVars, CommonInferShape)
REG_INFER(Floor, PrimType_Floor, CommonInferShapeWithOneInput)
REG_INFER(LeakyRelu, PrimType_LeakyRelu, CommonInferShape)
REG_INFER(Log, PrimType_Log, CommonInferShape)
REG_INFER(Log1p, PrimType_Log1p, CommonInferShape)
REG_INFER(LogGrad, PrimType_LogGrad, CommonGradInferShape)
REG_INFER(LogicalNot, PrimType_LogicalNot, CommonInferShape)
REG_INFER(LRN, PrimType_LRN, CommonInferShapeWithNHWC)
REG_INFER(L2Normalize, PrimType_L2NormalizeFusion, CommonInferShape)
REG_INFER(Neg, PrimType_Neg, CommonInferShape)
REG_INFER(NegGrad, PrimType_NegGrad, CommonGradInferShape)
REG_INFER(OnesLike, PrimType_OnesLike, CommonInferShape)
REG_INFER(PowerGrad, PrimType_PowerGrad, CommonGradInferShape)
REG_INFER(PReLU, PrimType_PReLUFusion, CommonInferShape)
REG_INFER(Reciprocal, PrimType_Reciprocal, CommonInferShape)
REG_INFER(ReverseSequence, PrimType_ReverseSequence, CommonInferShape)
REG_INFER(Reverse, PrimType_ReverseV2, CommonInferShape)
REG_INFER(Round, PrimType_Round, CommonInferShape)
REG_INFER(Rsqrt, PrimType_Rsqrt, CommonInferShape)
REG_INFER(Scale, PrimType_ScaleFusion, CommonInferShape)
REG_INFER(SigmoidCrossEntropyWithLogits, PrimType_SigmoidCrossEntropyWithLogits, CommonInferShape)
REG_INFER(SigmoidCrossEntropyWithLogitsGrad, PrimType_SigmoidCrossEntropyWithLogitsGrad, CommonInferShape)
REG_INFER(Sin, PrimType_Sin, CommonInferShape)
REG_INFER(SmoothL1Loss, PrimType_SmoothL1Loss, CommonInferShape)
REG_INFER(SmoothL1LossGrad, PrimType_SmoothL1LossGrad, CommonInferShape)
REG_INFER(Sqrt, PrimType_Sqrt, CommonInferShape)
REG_INFER(SqrtGrad, PrimType_SqrtGrad, CommonInferShape)
REG_INFER(Square, PrimType_Square, CommonInferShape)
REG_INFER(ZerosLike, PrimType_ZerosLike, CommonInferShape)
