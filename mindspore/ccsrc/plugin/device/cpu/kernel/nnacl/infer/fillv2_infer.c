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

#include "nnacl/infer/fillv2_infer.h"
#include "nnacl/infer/infer_register.h"

int FillV2InferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[1];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  const TensorC *dst_shape_tensor = inputs[0];
  const int32_t *dst_shape = (int32_t *)(dst_shape_tensor->data_);
  int num_dims = 1;
  if (dst_shape_tensor->shape_size_ != DIMENSION_1D) {
    return NNACL_ERR;
  }
  for (size_t i = 0; i < dst_shape_tensor->shape_size_; ++i) {
    if (INT_MUL_OVERFLOW(num_dims, dst_shape_tensor->shape_[i])) {
      return NNACL_ERRCODE_MUL_OVERFLOW;
    }
    NNACL_CHECK_FALSE(dst_shape_tensor->shape_[i] < 0, NNACL_ERR);
    num_dims *= dst_shape_tensor->shape_[i];
  }
  if (num_dims != 0 && dst_shape == NULL) {
    return NNACL_INFER_INVALID;
  }
  if (num_dims > MAX_SHAPE_SIZE) {
    return NNACL_ERR;
  }
  int output_shape[MAX_SHAPE_SIZE] = {0};
  size_t output_shape_size = 0;
  for (int i = 0; i < num_dims; i++) {
    ShapePush(output_shape, &output_shape_size, dst_shape[i]);
  }
  SetShapeArray(output, output_shape, output_shape_size);
  return NNACL_OK;
}

REG_INFER(FillV2, PrimType_FillV2, FillV2InferShape)
