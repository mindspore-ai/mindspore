/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "nnacl/infer/fse_decoder_infer.h"
#include "nnacl/infer/infer_register.h"

size_t kInputSize = 7;

int FseDecoderInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                         OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, kInputSize, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  const TensorC *cen_input = inputs[4];
  TensorC *output0 = outputs[FIRST_INPUT];
  SetDataTypeFormat(output0, cen_input);

  return NNACL_OK;
}

REG_INFER(FseDecode, PrimType_Inner_FseDecode, FseDecoderInferShape)
