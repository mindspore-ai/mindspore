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

#include "nnacl/infer/instance_norm_infer.h"
#include "nnacl/infer/crop_infer.h"
#include "nnacl/infer/infer_register.h"

int InstanceNormInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                           OpParameter *parameter) {
  if (parameter == NULL || inputs[0] == NULL || outputs[0] == NULL) {
    return NNACL_NULL_PTR;
  }
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, inputs[0]);
  if (output->format_ == Format_NC4HW4) {
    output->format_ = Format_NHWC;
  }
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  SetShapeTensor(output, inputs[0]);
  if (inputs[0]->format_ != Format_NC4HW4) {
    return NNACL_OK;
  }
  if (output->shape_size_ <= DIMENSION_2D) {
    return NNACL_OK;
  }
  int channel = output->shape_[1];
  ShapeErase(output->shape_, &output->shape_size_, 1);
  ShapePush(output->shape_, &output->shape_size_, channel);
  return NNACL_OK;
}
REG_INFER(InstanceNorm, PrimType_InstanceNorm, InstanceNormInferShape)
