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

#include "nnacl/infer/matmul_infer.h"
#include <math.h>
#include "nnacl/infer/infer_register.h"

#define MIN_SHAPE_SIZE 2

int CheckMatmulInputShape(int *a_shape, size_t a_shape_size, int *b_shape, size_t b_shape_size, const int *bias_shape,
                          size_t bias_shape_size, const MatMulParameter *param) {
  if (a_shape_size < MIN_SHAPE_SIZE || b_shape_size < MIN_SHAPE_SIZE) {
    return NNACL_PARAM_INVALID;
  }
  for (size_t i = 0; i < (a_shape_size - 2) && i < (b_shape_size - 2); ++i) {
    int min_value = MSMIN(a_shape[i], b_shape[i]);
    int max_value = MSMAX(a_shape[i], b_shape[i]);
    if (min_value == 0) {
      return NNACL_ERR;
    }
    if (max_value % min_value != 0) {
      return NNACL_INPUT_TENSOR_ERROR;
    }
  }
  if (param->a_transpose_) {
    iswap(&a_shape[a_shape_size - 1], &a_shape[a_shape_size - DIMENSION_2D]);
  }
  if (param->b_transpose_) {
    iswap(&b_shape[b_shape_size - 1], &b_shape[b_shape_size - 2]);
  }
  if (bias_shape_size == DIMENSION_1D && bias_shape[0] != b_shape[b_shape_size - 1]) {
    return NNACL_ERR;
  }
  if (a_shape[a_shape_size - 1] != b_shape[b_shape_size - 2]) {
    return NNACL_ERR;
  }
  return NNACL_OK;
}

int CheckMatMulBias(int *shape, size_t dim_size) {
  if (dim_size > 1) {
    for (size_t i = 0; i < dim_size - 1; i++) {
      if (shape[i] != DIMENSION_1D) {
        return NNACL_ERR;
      }
    }
  }
  return NNACL_OK;
}

int SetShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
             OpParameter *parameter) {
  TensorC *input0 = (TensorC *)inputs[0];
  TensorC *input1 = (TensorC *)inputs[1];
  TensorC *output = outputs[0];
  MatMulParameter *param = (MatMulParameter *)parameter;
  int a_shape[MAX_SHAPE_SIZE] = {0};
  size_t a_shape_size = 0;
  ShapeSet(a_shape, &a_shape_size, input0->shape_, input0->shape_size_);
  int b_shape[MAX_SHAPE_SIZE] = {0};
  size_t b_shape_size = 0;
  ShapeSet(b_shape, &b_shape_size, input1->shape_, input1->shape_size_);
  int bias_shape[MAX_AXIS_SIZE] = {0};
  size_t bias_shape_size = 0;
  if (inputs_size == kInputSize2) {
    TensorC *bias = (TensorC *)inputs[2];
    ShapeSet(bias_shape, &bias_shape_size, bias->shape_, bias->shape_size_);
    MS_CHECK_TRUE_RET(CheckMatMulBias(bias_shape, bias_shape_size) == NNACL_OK, NNACL_ERR);
  }

  if (a_shape_size == COMM_SHAPE_SIZE && a_shape[THIRD_INPUT] == 1 && a_shape[FOURTH_INPUT] == 1) {
    a_shape_size = 2;
    SetShapeArray(input0, a_shape, a_shape_size);
  }

  bool del_start = false;
  bool del_end = false;
  if (a_shape_size == 1) {
    int insert_ret = ShapeInsert(a_shape, &a_shape_size, 0, 1);
    if (insert_ret != NNACL_OK) {
      return NNACL_ERR;
    }
    SetShapeArray(input0, a_shape, a_shape_size);
    del_start = true;
  }
  if (b_shape_size == 1) {
    ShapePush(b_shape, &b_shape_size, 1);
    SetShapeArray(input1, b_shape, b_shape_size);
    del_end = true;
  }
  int ret = CheckMatmulInputShape(a_shape, a_shape_size, b_shape, b_shape_size, bias_shape, bias_shape_size, param);
  if (ret != NNACL_OK) {
    return NNACL_ERR;
  }
  int c_shape[MAX_SHAPE_SIZE];
  size_t c_shape_size = 0;
  ShapeSet(c_shape, &c_shape_size, a_shape, a_shape_size);
  c_shape[c_shape_size - 1] = b_shape[b_shape_size - 1];
  if (del_start) {
    int erase_ret = ShapeErase(c_shape, &c_shape_size, 0);
    if (erase_ret != NNACL_OK) {
      return NNACL_ERR;
    }
  }
  if (del_end) {
    c_shape_size--;
  }

  for (size_t i = 0; i < (a_shape_size - 2) && i < (b_shape_size - 2); ++i) {
    c_shape[i] = MSMAX(a_shape[i], b_shape[i]);
  }

  SetShapeArray(output, c_shape, c_shape_size);
  return NNACL_OK;
}

int MatmulInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
  int check_ret = CheckAugmentNullSizeInputTwo(inputs, inputs_size, outputs, outputs_size, parameter, 2, 3, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  TensorC *input0 = (TensorC *)inputs[0];
  TensorC *input1 = (TensorC *)inputs[1];
  TensorC *output = outputs[0];

  int diff = abs((int)input0->shape_size_ - (int)input1->shape_size_);
  TensorC *in = input0->shape_size_ > input1->shape_size_ ? input1 : input0;
  for (int i = 0; i < diff; ++i) {
    ShapeInsert(in->shape_, &in->shape_size_, 0, 1);
  }
  TensorC *input = input1->data_ == NULL ? input1 : input0;  // transfer the input which comes from the other node.
  SetDataTypeFormat(output, input);
  if (parameter->quant_type_ == QuantType_QUANT_DYNAMIC || parameter->quant_type_ == QuantType_QUANT_WEIGHT) {
    output->data_type_ = kNumberTypeFloat32;
  }
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  return SetShape(inputs, inputs_size, outputs, outputs_size, parameter);
}

REG_INFER(MatMul, PrimType_MatMulFusion, MatmulInferShape)
