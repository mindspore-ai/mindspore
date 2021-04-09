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

#include "nnacl/infer/range_infer.h"
#include <math.h>
#include "nnacl/infer/infer_register.h"

int RangeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                    OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  if (input == NULL || output == NULL) {
    return NNACL_NULL_PTR;
  }

  if (inputs_size == 3) {
    output->data_type_ = input->data_type_;
  } else {
    output->data_type_ = kNumberTypeInt32;
  }
  output->format_ = input->format_;
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  int shape_size = 0;
  if (inputs_size == 3) {
    if ((inputs[0]->data_ == NULL) || (inputs[1]->data_ == NULL) || (inputs[2]->data_ == NULL)) {
      return NNACL_INFER_INVALID;
    }
    switch (inputs[0]->data_type_) {
      case kNumberTypeInt:
      case kNumberTypeInt32: {
        int start = *(int *)(inputs[0]->data_);
        int limit = *(int *)(inputs[1]->data_);
        int delta = *(int *)(inputs[2]->data_);
        shape_size = imax((int)(ceil((float)(limit - start) / delta)), 0);
      } break;
      case kNumberTypeFloat32:
      case kNumberTypeFloat: {
        float start = *(float *)(inputs[0]->data_);
        float limit = *(float *)(inputs[1]->data_);
        float delta = *(float *)(inputs[2]->data_);
        shape_size = imax((int)(ceil((float)(limit - start) / delta)), 0);
      } break;
      default: {
        return NNACL_ERR;
      }
    }
  } else {
    RangeParameter *param = (RangeParameter *)parameter;
    shape_size = ceil((float)(param->limit_ - param->start_) / param->delta_);
  }

  output->shape_size_ = 1;
  output->shape_[0] = shape_size;
  return NNACL_OK;
}

REG_INFER(Range, PrimType_Range, RangeInferShape)
