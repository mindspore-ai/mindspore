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

#include "nnacl/infer/conv2d_grad_filter_infer.h"
#include "nnacl/infer/infer_register.h"

int Conv2dGradFilterInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                               OpParameter *parameter) {
  if (inputs_size < 3 || outputs_size != 1) {
    return NNACL_ERR;
  }
  SetDataTypeFormat(outputs[0], inputs[0]);

  size_t filter_shape_size = inputs[2]->shape_[0];
  if (filter_shape_size != 4) {
    return NNACL_ERR;
  }

  int filter_shape[MAX_SHAPE_SIZE];
  const int nchw2nhwc[4] = {0, 2, 3, 1};
  for (size_t i = 0; i < filter_shape_size; i++) {
    filter_shape[i] = *((int *)(inputs[2]->data_) + nchw2nhwc[i]);
  }

  SetShapeArray(outputs[0], filter_shape, filter_shape_size);
  return NNACL_OK;
}

REG_INFER(Conv2DBackpropFilterFusion, PrimType_Conv2DBackpropFilterFusion, Conv2dGradFilterInferShape)
