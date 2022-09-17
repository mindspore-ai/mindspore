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

#include "nnacl/infer/squeeze_infer.h"
#include "nnacl/infer/infer_register.h"

int SqueezeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                      OpParameter *parameter) {
  int check_ret =
    CheckAugmentNullSizeInputTwo(inputs, inputs_size, outputs, outputs_size, parameter, 1, kInputSize1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  SqueezeParameter *param = (SqueezeParameter *)parameter;
  SetDataTypeFormat(outputs[0], input);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  if (input->shape_size_ > MAX_SHAPE_SIZE) {
    return NNACL_INPUT_TENSOR_ERROR;
  }

  if (inputs_size == kInputSize1) {
    MS_CHECK_TRUE_RET(inputs[1]->data_type_ == kNumberTypeInt32 || inputs[1]->data_type_ == kNumberTypeInt,
                      NNACL_PARAM_INVALID);
    int *axis_data = (int *)(inputs[1]->data_);
    MS_CHECK_TRUE_RET(axis_data != NULL, NNACL_PARAM_INVALID);
    param->axis_size_ = GetElementNum(inputs[1]);
    for (size_t i = 0; i < param->axis_size_; i++) {
      param->axis_[i] = *(axis_data + i);
    }
  }
  if (param->axis_size_ > MAX_SHAPE_SIZE) {
    return NNACL_PARAM_INVALID;
  }
  int out_shape[MAX_SHAPE_SIZE] = {0};
  size_t out_shape_size = 0;

  for (size_t i = 0; i < param->axis_size_; i++) {
    param->axis_[i] = param->axis_[i] >= 0 ? param->axis_[i] : param->axis_[i] + (int)input->shape_size_;
  }

  if (param->axis_size_ == 0) {
    for (size_t i = 0; i < input->shape_size_; i++) {
      if (input->shape_[i] != 1) {
        ShapePush(out_shape, &out_shape_size, input->shape_[i]);
      }
    }
  } else {
    size_t axisIdx = 0;
    for (size_t i = 0; i < input->shape_size_; i++) {
      if (axisIdx < param->axis_size_ && param->axis_[axisIdx] == (int)(i)) {
        if (input->shape_[i] != 1) return NNACL_PARAM_INVALID;
        axisIdx++;
        continue;
      } else {
        ShapePush(out_shape, &out_shape_size, input->shape_[i]);
      }
    }
  }
  SetShapeArray(outputs[0], out_shape, out_shape_size);
  return NNACL_OK;
}

REG_INFER(Squeeze, PrimType_Squeeze, SqueezeInferShape)
