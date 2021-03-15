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

#include "nnacl/infer/rfft_infer.h"
#include "nnacl/infer/infer_register.h"

int RfftInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                   OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  output->data_type_ = kNumberTypeComplex64;
  output->format_ = input->format_;
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  ShapeSet(output->shape_, &(output->shape_size_), input->shape_, input->shape_size_);
  RfftParameter *param = (RfftParameter *)parameter;
  output->shape_[input->shape_size_ - 1] = param->fft_length_ / 2 + 1;
  ShapePush(output->shape_, &(output->shape_size_), 2);
  return NNACL_OK;
}

REG_INFER(Rfft, PrimType_Rfft, RfftInferShape)
