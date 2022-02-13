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

#include "nnacl/infer/splice_infer.h"
#include "nnacl/infer/infer_register.h"

int SpliceInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  if (input->shape_size_ != DIMENSION_3D) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  SpliceParameter *param = (SpliceParameter *)parameter;
  if (param == NULL) {
    return NNACL_NULL_PTR;
  }
  int out_dim = param->output_dim_;
  ShapeSet(output->shape_, &output->shape_size_, input->shape_, input->shape_size_);

  if (param->context_dim_ == 0) {
    return NNACL_ERRCODE_DIVISOR_ZERO;
  }
  if (param->forward_indexes_dim_ % param->context_dim_ != 0) {
    return NNACL_PARAM_INVALID;
  }
  int out_size = param->forward_indexes_dim_ / param->context_dim_;
  output->shape_[DIMENSION_1D] = out_size;
  output->shape_[DIMENSION_2D] = out_dim;
  return NNACL_OK;
}

REG_INFER(Splice, PrimType_Splice, SpliceInferShape)
