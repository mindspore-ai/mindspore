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

#include "nnacl/infer/depth_to_space_infer.h"
#include "nnacl/infer/infer_register.h"

int DepthToSpaceInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
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
  DepthToSpaceParameter *param = (DepthToSpaceParameter *)parameter;
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  int input_shape[MAX_SHAPE_SIZE];
  size_t input_shape_size = 0;
  ShapeSet(input_shape, &input_shape_size, input->shape_, input->shape_size_);
  if (input_shape_size != 4) {
    return NNACL_PARAM_INVALID;
  }

  int32_t block_size = param->block_size_;
  if (input_shape[kNHWC_C] % (block_size * block_size) != 0 || input_shape[kNHWC_C] == 0) {
    return NNACL_PARAM_INVALID;
  }
  int32_t output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = input_shape_size;
  output_shape[kNHWC_N] = input_shape[kNHWC_N];
  output_shape[kNHWC_H] = input_shape[kNHWC_H] * block_size;
  output_shape[kNHWC_W] = input_shape[kNHWC_W] * block_size;
  output_shape[kNHWC_C] = input_shape[kNHWC_C] / (block_size * block_size);
  SetShapeArray(outputs[0], output_shape, output_shape_size);
  return NNACL_OK;
}

REG_INFER(DepthToSpace, PrimType_DepthToSpace, DepthToSpaceInferShape)
