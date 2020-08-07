/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/nnacl/int8/quant_dtype_cast.h"
#include "src/runtime/kernel/arm/nnacl/errorcode.h"

int DequantizeInt8(int8_t *quant_values, float *real_values, float scale, int32_t zp, int size) {
  if (quant_values == nullptr || real_values == nullptr) {
    return NNACL_PARAM_INVALID;
  }

  for (int i = 0; i < size; ++i) {
    real_values[i] = (quant_values[i] + zp) * scale;
  }
  return NNACL_OK;
}

int QuantizeToInt8(float *real_values, int8_t *quant_values, float scale, int32_t zp, int size) {
  if (quant_values == nullptr || real_values == nullptr) {
    return NNACL_PARAM_INVALID;
  }

  for (int i = 0; i < size; ++i) {
    quant_values[i] = (int8_t)round(real_values[i] / scale + zp);
  }
  return NNACL_OK;
}
