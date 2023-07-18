/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/infer/format_transpose_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/format_transpose_parameter.h"
#include "nnacl/tensor_c_utils.h"

int FormatTransposeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                              OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  FormatTransposeParameter *param = (FormatTransposeParameter *)parameter;
  output->format_ = (int)(param->dst_format_);
  output->data_type_ = input->data_type_;

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  if (input->shape_size_ != DIMENSION_4D) {
    SetShapeArray(output, input->shape_, input->shape_size_);
    return NNACL_OK;
  }

  int input_b = GetBatch(input);
  int input_h = GetHeight(input);
  int input_w = GetWidth(input);
  int input_c = GetChannel(input);

  // set output shape
  int out_shape[MAX_SHAPE_SIZE] = {0};
  out_shape[DIMENSION_0D] = input_b;
  if (param->dst_format_ == Format_NCHW || param->dst_format_ == Format_NC4HW4 || param->dst_format_ == Format_NC8HW8) {
    out_shape[DIMENSION_1D] = input_c;
    out_shape[DIMENSION_2D] = input_h;
    out_shape[DIMENSION_3D] = input_w;
  } else if (param->dst_format_ == Format_NHWC) {
    out_shape[DIMENSION_1D] = input_h;
    out_shape[DIMENSION_2D] = input_w;
    out_shape[DIMENSION_3D] = input_c;
  } else {
    return NNACL_ERR;
  }

  SetShapeArray(output, out_shape, input->shape_size_);
  return NNACL_OK;
}

REG_INFER(FormatTranspose, PrimType_FormatTranspose, FormatTransposeInferShape)
