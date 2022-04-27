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

#include <stdio.h>
#include "nnacl/infer/conv2d_grad_filter_infer.h"
#include "nnacl/infer/infer_register.h"

int Conv2dGradFilterInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                               OpParameter *parameter) {
  int ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (ret != NNACL_OK) {
    return ret;
  }
  if (inputs_size < 3 || outputs_size != 1) {
    return NNACL_ERR;
  }
  if (inputs[FIRST_INPUT]->format_ != Format_NHWC || inputs[SECOND_INPUT]->format_ != Format_NHWC) {
    return NNACL_FORMAT_ERROR;
  }
  SetDataTypeFormat(outputs[FIRST_INPUT], inputs[FIRST_INPUT]);

  if (inputs[THIRD_INPUT]->shape_size_ < DIMENSION_1D || inputs[THIRD_INPUT]->data_ == NULL) {
    return NNACL_ERR;
  }
  if (inputs[THIRD_INPUT]->shape_[kNCHW_N] < 0) {
    return NNACL_ERR;
  }
  size_t filter_shape_size = (size_t)(inputs[THIRD_INPUT]->shape_[kNCHW_N]);
  if (filter_shape_size != DIMENSION_4D) {
    return NNACL_ERR;
  }

  int filter_shape[MAX_SHAPE_SIZE];
  if (inputs[THIRD_INPUT]->format_ == Format_NCHW || inputs[THIRD_INPUT]->format_ == Format_KCHW) {
    const int nchw2nhwc[] = {kNCHW_N, kNCHW_H, kNCHW_W, kNCHW_C};
    for (size_t i = 0; i < filter_shape_size; i++) {
      filter_shape[i] = *((int *)(inputs[THIRD_INPUT]->data_) + nchw2nhwc[i]);
    }
  } else if (inputs[THIRD_INPUT]->format_ == Format_NHWC || inputs[THIRD_INPUT]->format_ == Format_KHWC) {
    memcpy(filter_shape, inputs[THIRD_INPUT]->data_, filter_shape_size * sizeof(int));
  } else {
    return NNACL_ERR;
  }
  SetShapeArray(outputs[0], filter_shape, filter_shape_size);
  return NNACL_OK;
}

REG_INFER(Conv2DBackpropFilterFusion, PrimType_Conv2DBackpropFilterFusion, Conv2dGradFilterInferShape)
