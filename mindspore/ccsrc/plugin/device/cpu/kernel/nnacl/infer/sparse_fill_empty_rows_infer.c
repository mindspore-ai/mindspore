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

#include "nnacl/infer/sparse_fill_empty_rows_infer.h"
#include "nnacl/infer/infer_register.h"

int SparseFillEmptyRowsInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                  size_t outputs_size, OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, C4NUM, C3NUM);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output0 = outputs[0];
  TensorC *output1 = outputs[1];
  TensorC *output2 = outputs[C2NUM];
  SetDataTypeFormat(output0, input);
  SetDataTypeFormat(output1, input);
  SetDataTypeFormat(output2, input);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  return NNACL_INFER_INVALID;
}

REG_INFER(SparseFillEmptyRows, PrimType_SparseFillEmptyRows, SparseFillEmptyRowsInferShape)
