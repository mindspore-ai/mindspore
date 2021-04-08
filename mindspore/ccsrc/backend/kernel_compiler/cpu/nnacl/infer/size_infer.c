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

#include "nnacl/infer/size_infer.h"
#include "nnacl/infer/infer_register.h"

int SizeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                   OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *in_tensor = inputs[0];
  TensorC *out_tensor = outputs[0];
  out_tensor->data_type_ = kNumberTypeInt32;
  out_tensor->format_ = in_tensor->format_;
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  out_tensor->shape_size_ = 1;
  out_tensor->shape_[0] = 1;

  return NNACL_OK;
}

REG_INFER(SizeOp, PrimType_Size, SizeInferShape)
