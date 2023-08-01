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

#include "nnacl/infer/invalid_infer.h"
#include "nnacl/infer/infer_register.h"

int InvalidInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                      OpParameter *parameter) {
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  return NNACL_INFER_INVALID;
}

REG_INFER(PartialFusion, PrimType_PartialFusion, InvalidInferShape)
REG_INFER(Switch, PrimType_Switch, InvalidInferShape)
REG_INFER(Call, PrimType_Call, InvalidInferShape)
REG_INFER(SwitchLayer, PrimType_SwitchLayer, InvalidInferShape)
