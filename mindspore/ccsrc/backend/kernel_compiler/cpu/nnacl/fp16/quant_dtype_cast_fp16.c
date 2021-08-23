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

#include <math.h>
#include "nnacl/fp16/quant_dtype_cast_fp16.h"
#include "nnacl/errorcode.h"

int DoDequantizeInt8ToFp16(const int8_t *quant_values, float16_t *real_values, float scale, int32_t zp, int size) {
  if (quant_values == NULL || real_values == NULL) {
    return NNACL_PARAM_INVALID;
  }

  for (int i = 0; i < size; ++i) {
    real_values[i] = (quant_values[i] - zp) * scale;
  }
  return NNACL_OK;
}

int DoQuantizeFp16ToInt8(const float16_t *real_values, int8_t *quant_values, float scale, int32_t zp, int size) {
  if (quant_values == NULL || real_values == NULL) {
    return NNACL_PARAM_INVALID;
  }

  for (int i = 0; i < size; ++i) {
    if (isinf(real_values[i])) {
      quant_values[i] = 127;
      continue;
    }
    float temp = round((float)real_values[i] / scale + zp);
    if (temp > 127) {
      quant_values[i] = 127;
    } else if (temp < -128) {
      quant_values[i] = -128;
    } else {
      quant_values[i] = (int8_t)temp;
    }
  }
  return NNACL_OK;
}

int DoDequantizeUInt8ToFp16(const uint8_t *quant_values, float16_t *real_values, float scale, int32_t zp, int size) {
  uint8_t zp_ = (uint8_t)zp;
  if (quant_values == NULL || real_values == NULL) {
    return NNACL_PARAM_INVALID;
  }

  for (int i = 0; i < size; ++i) {
    real_values[i] = (quant_values[i] - zp_) * scale;
  }
  return NNACL_OK;
}

int DoQuantizeFp16ToUInt8(const float16_t *real_values, uint8_t *quant_values, float scale, int32_t zp, int size) {
  if (quant_values == NULL || real_values == NULL) {
    return NNACL_PARAM_INVALID;
  }

  for (int i = 0; i < size; ++i) {
    if (isinf(real_values[i])) {
      quant_values[i] = 255;
      continue;
    }
    float temp = round((float)real_values[i] / scale + zp);
    if (temp > 255.0f) {
      quant_values[i] = 255;
    } else if (temp < 0.0f) {
      quant_values[i] = 0;
    } else {
      quant_values[i] = (uint8_t)temp;
    }
  }
  return NNACL_OK;
}
