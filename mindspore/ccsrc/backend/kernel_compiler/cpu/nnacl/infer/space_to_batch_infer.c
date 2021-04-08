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

#include "nnacl/infer/space_to_batch_infer.h"
#include "nnacl/infer/infer_register.h"

int SpaceToBatchInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                           OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  if (input->format_ != Format_NHWC) {
    return NNACL_ERR;
  }
  SetDataTypeFormat(outputs[0], input);
  SpaceToBatchParameter *param = (SpaceToBatchParameter *)parameter;
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  if (input->shape_size_ != 4) {
    return NNACL_ERR;
  }

  int *block_shape = param->block_sizes_;
  size_t block_shape_size = param->m_;
  int *paddings = param->paddings_;
  int padding_left = 0;
  int padding_right = 0;
  int block_w = 1;
  if (block_shape_size == 2) {
    padding_left = paddings[2];
    padding_right = paddings[3];
    block_w = block_shape[1];
  }

  outputs[0]->shape_[kNHWC_N] = input->shape_[kNHWC_N] * (block_shape[0] * block_w);
  outputs[0]->shape_[kNHWC_H] = (input->shape_[kNHWC_H] + paddings[0] + paddings[1]) / block_shape[0];
  outputs[0]->shape_[kNHWC_W] = (input->shape_[kNHWC_W] + padding_left + padding_right) / block_w;
  outputs[0]->shape_[kNHWC_C] = input->shape_[kNHWC_C];
  outputs[0]->shape_size_ = input->shape_size_;
  return NNACL_OK;
}

REG_INFER(SpaceToBatch, PrimType_SpaceToBatch, SpaceToBatchInferShape)
