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

#include "nnacl/infer/constant_of_shape_infer.h"
#include "nnacl/infer/infer_register.h"

int ConstantOfShapeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                              OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *in_tensor = inputs[0];
  TensorC *out_tensor = outputs[0];
  ConstantOfShapeParameter *param = (ConstantOfShapeParameter *)parameter;
  out_tensor->data_type_ = (TypeIdC)(param->data_type_);
  out_tensor->format_ = in_tensor->format_;
  if (!parameter->infer_flag_ || in_tensor->data_ == NULL) {
    return NNACL_INFER_INVALID;
  }
  int size = GetElementNum(in_tensor);
  int out_shape[MAX_SHAPE_SIZE];
  size_t out_shape_size = size;
  switch (in_tensor->data_type_) {
    case kNumberTypeInt32: {
      int32_t *in_data = (int32_t *)(in_tensor->data_);
      for (int i = 0; i < size; ++i) {
        out_shape[i] = in_data[i];
        if (out_shape[i] <= 0) {
          return NNACL_ERR;
        }
      }
      break;
    }
    case kNumberTypeInt64: {
      int64_t *in_data = (int64_t *)(in_tensor->data_);
      for (int i = 0; i < size; ++i) {
        out_shape[i] = in_data[i];
        if (out_shape[i] <= 0) {
          return NNACL_ERR;
        }
      }
      break;
    }
    default:
      return NNACL_INFER_INVALID;
  }

  SetShapeArray(out_tensor, out_shape, out_shape_size);
  return NNACL_OK;
}

REG_INFER(ConstantOfShape, PrimType_ConstantOfShape, ConstantOfShapeInferShape)
