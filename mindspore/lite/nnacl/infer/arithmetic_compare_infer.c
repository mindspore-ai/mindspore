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

#include "nnacl/infer/arithmetic_compare_infer.h"
#include "nnacl/infer/infer_register.h"

int ArithmeticCompareInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                size_t outputs_size, OpParameter *parameter) {
  int res = ArithmeticInferShape(inputs, inputs_size, outputs, outputs_size, parameter);
  TensorC *output = outputs[0];
  if (output == NULL) {
    return NNACL_NULL_PTR;
  }
  output->data_type_ = kNumberTypeBool;
  return res;
}

REG_INFER(Equal, PrimType_Equal, ArithmeticCompareInferShape)
REG_INFER(Greater, PrimType_Greater, ArithmeticCompareInferShape)
REG_INFER(GreaterEqual, PrimType_GreaterEqual, ArithmeticCompareInferShape)
REG_INFER(Less, PrimType_Less, ArithmeticCompareInferShape)
REG_INFER(LessEqual, PrimType_LessEqual, ArithmeticCompareInferShape)
REG_INFER(NotEqual, PrimType_NotEqual, ArithmeticCompareInferShape)
