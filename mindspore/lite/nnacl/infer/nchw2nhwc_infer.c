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

#include "nnacl/infer/nchw2nhwc_infer.h"

int Nchw2NhwcInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                        OpParameter *parameter) {
  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  if (parameter == NULL || input == NULL || output == NULL) {
    return NNACL_NULL_PTR;
  }
  output->format_ = Format_NHWC;
  output->data_type_ = input->data_type_;
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  if (input->shape_size_ != 4) {
    SetShapeTensor(output, input);
  } else {
    output->shape_[kNHWC_N] = input->shape_[kNCHW_N];
    output->shape_[kNHWC_H] = input->shape_[kNCHW_H];
    output->shape_[kNHWC_W] = input->shape_[kNCHW_W];
    output->shape_[kNHWC_C] = input->shape_[kNCHW_C];
    output->shape_size_ = 4;
  }
  return NNACL_OK;
}
