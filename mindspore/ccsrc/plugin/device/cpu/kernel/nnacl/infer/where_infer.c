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

#include "nnacl/infer/where_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/infer/broadcast_to_infer.h"

int WhereBroadCastInferShape(const int input_shape0_size, const int input_shape1_size, const int *input_shape0,
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

int WhereInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                    OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];

  // Need to dynamically allocate at runtime.
  if (inputs_size == 1) {
    output->data_type_ = kNumberTypeInt32;
    output->format_ = input->format_;
    return NNACL_INFER_INVALID;
  }

  if (inputs_size < 3 || outputs_size != 1) {
    return NNACL_INPUT_TENSOR_ERROR;
  }

  const TensorC *input0 = inputs[0];
  const TensorC *input1 = inputs[1];
  const TensorC *input2 = inputs[2];
  SetDataTypeFormat(output, input1);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  int in_shape0[MAX_SHAPE_SIZE] = {0};
  int in_shape1[MAX_SHAPE_SIZE] = {0};
  int in_shape2[MAX_SHAPE_SIZE] = {0};
  int output_shape[MAX_SHAPE_SIZE] = {0};
  size_t input_shape0_size = input0->shape_size_;
  size_t input_shape1_size = input1->shape_size_;
  size_t input_shape2_size = input2->shape_size_;
  const int *input_shape0 = input0->shape_;
  const int *input_shape1 = input1->shape_;
  const int *input_shape2 = input2->shape_;
  int ndim = (int)input_shape0_size;
  bool has_broad_cast_1 = false;
  bool has_broad_cast_2 = false;
  if (WhereBroadCastInferShape(input_shape0_size, input_shape1_size, input_shape0, input_shape1, &ndim, in_shape0,
                               in_shape1, output_shape, &has_broad_cast_1) != NNACL_OK) {
    return NNACL_ERR;
  }
  if (WhereBroadCastInferShape(ndim, input_shape2_size, output_shape, input_shape2, &ndim, in_shape0, in_shape2,
                               output_shape, &has_broad_cast_2) != NNACL_OK) {
    return NNACL_ERR;
  }
  ShapeSet(output->shape_, &output->shape_size_, output_shape, ndim);
  return NNACL_OK;
}

REG_INFER(Where, PrimType_Where, WhereInferShape)
