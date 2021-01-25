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

#include "nnacl/infer/batch_to_space_infer.h"

int BatchToSpaceInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                           OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  if (input->format_ != Format_NHWC) {
    return NNACL_ERR;
  }
  SetDataTypeFormat(outputs[0], input);
  BatchToSpaceParameter *param = (BatchToSpaceParameter *)parameter;
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  int input_shape[MAX_SHAPE_SIZE];
  size_t input_shape_size = 0;
  ShapeSet(input_shape, &input_shape_size, input->shape_, input->shape_size_);
  if (input_shape_size != 4) {
    return NNACL_PARAM_INVALID;
  }

  int32_t *block_shape = param->block_shape_;
  // if (block_shape.size() != kBlockShapeSize) {
  // MS_LOG(ERROR) << "Block shape size should be " << kBlockShapeSize;
  // return RET_PARAM_INVALID;
  //  return NNACL_PARAM_INVALID;
  //}
  int32_t *crops = param->crops_;
  // if (crops.size() != kCropsSize) {
  // MS_LOG(ERROR) << "Crops size should be " << kCropsSize;
  // return RET_PARAM_INVALID;
  //  return NNACL_PARAM_INVALID;
  //}
  int mul_block_shape = 1;

  for (size_t i = 0; i < 2; ++i) {
    if (block_shape[i] <= 0) {
      return NNACL_PARAM_INVALID;
    }
    if (input_shape[kNHWC_N] % block_shape[i]) {
      return NNACL_ERR;
    }
    mul_block_shape *= block_shape[i];
  }

  if (input_shape[kNHWC_N] < mul_block_shape) {
    return NNACL_PARAM_INVALID;
  }
  for (size_t i = 0; i < 4; ++i) {
    if (crops[i] < 0) {
      return NNACL_PARAM_INVALID;
    }
  }
  int32_t output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = input_shape_size;
  output_shape[kNHWC_N] = input_shape[kNHWC_N] / mul_block_shape;
  output_shape[kNHWC_H] = input_shape[kNHWC_H] * block_shape[0] - crops[0] - crops[1];
  output_shape[kNHWC_W] = input_shape[kNHWC_W] * block_shape[1] - crops[2] - crops[3];
  output_shape[kNHWC_C] = input_shape[kNHWC_C];

  SetShapeArray(outputs[0], output_shape, output_shape_size);
  return NNACL_OK;
}
