/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/infer/bias_infer.h"
#include "nnacl/infer/infer_register.h"

int BiasInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                   OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  SetDataTypeFormat(outputs[0], inputs[0]);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  MS_CHECK_TRUE_RET(inputs[0]->shape_size_ >= 1, NNACL_ERR);
  MS_CHECK_TRUE_RET(inputs[1]->shape_size_ == 1, NNACL_ERR);
  size_t dim = inputs[0]->shape_size_ - 1;
  if (inputs[0]->format_ == Format_KCHW || inputs[0]->format_ == Format_NCHW) {
    dim = 1;
  }
  if (inputs[0]->shape_[dim] != inputs[1]->shape_[0]) {
    return NNACL_ERR;
  }
  SetShapeTensor(outputs[0], inputs[0]);

  return NNACL_OK;
}

REG_INFER(BiasAdd, PrimType_BiasAdd, BiasInferShape)
