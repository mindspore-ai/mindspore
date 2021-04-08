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

#include "nnacl/infer/lsh_projection_infer.h"
#include "nnacl/infer/infer_register.h"

int LshProjectionInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                            OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSizeInputTwo(inputs, inputs_size, outputs, outputs_size, parameter, 2, 3, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *in_hash = inputs[0];
  if (in_hash->shape_size_ != 2 || GetDimensionSize(in_hash, 1) > 32) {
    return NNACL_ERR;
  }
  TensorC *out_tensor = outputs[0];
  out_tensor->data_type_ = kNumberTypeInt32;
  out_tensor->format_ = Format_NHWC;

  int out_shape[MAX_SHAPE_SIZE];
  size_t out_shape_size = 0;
  LshProjectionParameter *param = (LshProjectionParameter *)parameter;
  switch (param->lsh_type_) {
    case LshProjectionType_SPARSE:
      ShapePush(out_shape, &out_shape_size, GetDimensionSize(in_hash, 0));
      break;
    case LshProjectionType_DENSE:
      ShapePush(out_shape, &out_shape_size, GetDimensionSize(in_hash, 0) * GetDimensionSize(in_hash, 1));
      break;
    default:
      return NNACL_ERR;
  }
  SetShapeArray(out_tensor, out_shape, out_shape_size);
  return NNACL_OK;
}

REG_INFER(LshProjection, PrimType_LshProjection, LshProjectionInferShape)
