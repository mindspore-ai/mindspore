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

#include "nnacl/infer/unique_infer.h"
#include "nnacl/infer/infer_register.h"

int UniqueInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
  int ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 2);
  if (ret != NNACL_OK) {
    return ret;
  }

  const TensorC *input0 = inputs[0];
  TensorC *output0 = outputs[0];
  TensorC *output1 = outputs[1];

  SetDataTypeFormat(output0, input0);
  output1->data_type_ = kNumberTypeInt32;
  output1->format_ = input0->format_;
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  SetShapeTensor(output0, input0);
  SetShapeTensor(output1, input0);
  return NNACL_OK;
}

REG_INFER(Unique, PrimType_Unique, UniqueInferShape)
