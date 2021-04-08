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

#include "nnacl/infer/space_to_depth_infer.h"
#include <limits.h>
#include "nnacl/infer/infer_register.h"

int SpaceToDepthInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
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
  SpaceToDepthParameter *param = (SpaceToDepthParameter *)parameter;
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  if (input->shape_size_ != 4) {
    return NNACL_ERR;
  }

  int32_t block_size = param->block_size_;
  if (block_size == 0) {
    return NNACL_ERR;
  }
  if (input->shape_[kNHWC_H] % block_size != 0 || input->shape_[kNHWC_H] == 0 ||
      input->shape_[kNHWC_W] % block_size != 0 || input->shape_[kNHWC_W] == 0) {
    return NNACL_ERR;
  }
  outputs[0]->shape_[kNHWC_N] = input->shape_[kNHWC_N];
  outputs[0]->shape_[kNHWC_H] = input->shape_[kNHWC_H] / block_size;
  outputs[0]->shape_[kNHWC_W] = input->shape_[kNHWC_W] / block_size;
  if (block_size * block_size > INT_MAX / input->shape_[kNHWC_C]) {
    return NNACL_ERR;
  }
  outputs[0]->shape_[kNHWC_C] = input->shape_[kNHWC_C] * (block_size * block_size);
  outputs[0]->shape_size_ = input->shape_size_;
  return NNACL_OK;
}

REG_INFER(SpaceToDepth, PrimType_SpaceToDepth, SpaceToDepthInferShape)
