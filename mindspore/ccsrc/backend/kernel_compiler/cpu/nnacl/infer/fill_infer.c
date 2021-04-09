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

#include "nnacl/infer/fill_infer.h"
#include "nnacl/infer/infer_register.h"

int FillInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                   OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  const TensorC *dst_shape_tensor = inputs[1];
  const int32_t *dst_shape = (int32_t *)(dst_shape_tensor->data_);
  size_t num_dims = 1;
  for (size_t i = 0; i < dst_shape_tensor->shape_size_; ++i) {
    num_dims *= dst_shape_tensor->shape_[i];
  }
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  if (num_dims != 0 && dst_shape == NULL) {
    return NNACL_INFER_INVALID;
  }
  int output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = 0;
  for (size_t i = 0; i < num_dims; i++) {
    ShapePush(output_shape, &output_shape_size, dst_shape[i]);
  }
  SetShapeArray(output, output_shape, output_shape_size);
  return NNACL_OK;
}

REG_INFER(Fill, PrimType_Fill, FillInferShape)
