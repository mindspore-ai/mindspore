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

#include "nnacl/infer/bn_grad_infer.h"
#include "nnacl/infer/infer_register.h"

int BnGradInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 6, 3);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *in = inputs[1];
  const TensorC *scale = inputs[2];
  if (in->shape_size_ != 4) {
    return NNACL_INPUT_TENSOR_ERROR;
  }

  SetShapeTensor(outputs[0], in);
  SetDataTypeFormat(outputs[0], in);
  SetShapeTensor(outputs[1], scale);
  SetDataTypeFormat(outputs[1], scale);
  SetShapeTensor(outputs[2], scale);
  SetDataTypeFormat(outputs[2], scale);
  return NNACL_OK;
}

REG_INFER(BatchNormGrad, PrimType_BatchNormGrad, BnGradInferShape)
