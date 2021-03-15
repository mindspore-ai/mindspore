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

#include "nnacl/infer/mean_infer.h"
#include "nnacl/infer/infer_register.h"

int MeanInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                   OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  ReduceParameter *param = (ReduceParameter *)parameter;
  bool keep_dims = (bool)(param->keep_dims_);
  int out_shape[MAX_SHAPE_SIZE];
  size_t out_shape_size = 0;
  int *axes = param->axes_;
  int num_axes = param->num_axes_;
  // reduce on all axes
  if (num_axes == 0) {
    if (keep_dims) {
      for (size_t i = 0; i < input->shape_size_; i++) {
        ShapePush(out_shape, &out_shape_size, 1);
      }
    }
    SetShapeArray(output, out_shape, out_shape_size);
    output->data_type_ = input->data_type_;
    return NNACL_OK;
  }
  // reduce on selected axes
  for (size_t i = 0; i < input->shape_size_; i++) {
    bool reduce_axis = false;
    for (size_t idx = 0; idx < num_axes; ++idx) {
      if (((size_t)(axes[idx])) == i) {
        reduce_axis = true;
        break;
      }
    }
    if (reduce_axis) {
      if (keep_dims) {
        ShapePush(out_shape, &out_shape_size, 1);
      }
    } else {
      ShapePush(out_shape, &out_shape_size, input->shape_[i]);
    }
  }
  SetShapeArray(output, out_shape, out_shape_size);
  return NNACL_OK;
}
