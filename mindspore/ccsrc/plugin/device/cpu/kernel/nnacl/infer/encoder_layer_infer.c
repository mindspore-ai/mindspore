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
#include <stdio.h>
#include "nnacl/infer/encoder_layer_infer.h"
#include "nnacl/infer/infer_register.h"

int EncoderLayerInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                           OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, C9NUM, C1NUM);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  const TensorC *input = inputs[FIRST_INPUT];
  TensorC *output0 = outputs[FIRST_INPUT];
  SetDataTypeFormat(output0, input);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  SetShapeTensor(output0, input);
  return NNACL_OK;
}

REG_INFER(EncoderLayer, PrimType_Inner_EncoderLayer, EncoderLayerInferShape)
