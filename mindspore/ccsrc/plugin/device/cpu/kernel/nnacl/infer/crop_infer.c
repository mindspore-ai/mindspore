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

#include "nnacl/infer/crop_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/crop_parameter.h"
#include "nnacl/tensor_c_utils.h"

int CropInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                   OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  size_t input_shape_size = inputs[0]->shape_size_;
  CropParameter *param = (CropParameter *)parameter;
  int64_t axis = param->axis_ < 0 ? param->axis_ + (int64_t)input_shape_size : param->axis_;
  if (axis < 0 || axis >= (int64_t)input_shape_size) {
    return NNACL_ERR;
  }

  SetDataTypeFormat(outputs[0], inputs[0]);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  SetShapeTensor(outputs[0], inputs[1]);
  return NNACL_OK;
}

REG_INFER(Crop, PrimType_Crop, CropInferShape)
