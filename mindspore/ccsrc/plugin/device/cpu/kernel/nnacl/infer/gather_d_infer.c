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

#include "nnacl/infer/gather_d_infer.h"
#include "nnacl/infer/infer_register.h"

int GatherDInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                      OpParameter *parameter) {
  int ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (ret != NNACL_OK) {
    return ret;
  }
  const int input_size_limit = 3;
  const int output_size_limit = 1;
  if (inputs_size != input_size_limit || outputs_size != output_size_limit) {
    return NNACL_ERR;
  }
  const TensorC *input = inputs[0];
  const TensorC *index = inputs[2];
  TensorC *output = outputs[0];
  output->data_type_ = input->data_type_;
  if (parameter->quant_type_ == Quant_QuantWeight) {
    output->data_type_ = kNumberTypeFloat32;
  }
  output->format_ = input->format_;
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  SetShapeTensor(output, index);
  return NNACL_OK;
}

REG_INFER(GatherD, PrimType_GatherD, GatherDInferShape)
