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

#include "nnacl/infer/layer_norm_infer.h"

int LayerNormInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                        OpParameter *parameter) {
  int check_ret = CheckAugmentNullSizeInputTwo(inputs, inputs_size, outputs, outputs_size, parameter, 1, 3, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  LayerNormParameter *param = (LayerNormParameter *)parameter;
  if (param->elementwise_affine_ && inputs_size != 3) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  if (!param->elementwise_affine_ && inputs_size != 1) {
    return NNACL_INPUT_TENSOR_ERROR;
  }

  if (!param->op_parameter_.infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  int *normalized_shape = param->normalized_shape_;
  size_t normalized_shape_size = param->normalized_dims_;
  param->elementwise_mode_ = param->elementwise_affine_ ? 2 : 0;
  if (normalized_shape_size > input->shape_size_) {
    return NNACL_PARAM_INVALID;
  }
  if (normalized_shape_size == 0 && param->begin_norm_axis_ != 0) {
    size_t begin_norm_axis =
      param->begin_norm_axis_ < 0 ? param->begin_norm_axis_ + input->shape_size_ : param->begin_norm_axis_;
    for (size_t i = begin_norm_axis; i < input->shape_size_; ++i) {
      ShapePush(normalized_shape, &normalized_shape_size, input->shape_[i]);
    }
  }
  if (normalized_shape_size == 0) {
    // instance norm -> layernorm only for nchw
    if (input->format_ == Format_NCHW) {
      for (size_t i = 2; i < input->shape_size_; i++) {
        ShapeInsert(normalized_shape, &normalized_shape_size, i - 2, input->shape_[i]);
      }
      param->elementwise_mode_ = 1;
    } else {
      for (size_t i = 1; i < input->shape_size_; i++) {
        ShapeInsert(normalized_shape, &normalized_shape_size, i - 1, input->shape_[i]);
      }
    }
  }
  param->normalized_dims_ = normalized_shape_size;
  size_t first_index = input->shape_size_ - normalized_shape_size;
  for (size_t i = first_index; i < input->shape_size_; ++i) {
    if (input->shape_[i] != normalized_shape[i - first_index]) {
      return NNACL_PARAM_INVALID;
    }
  }

  SetShapeTensor(output, input);
  return NNACL_OK;
}
