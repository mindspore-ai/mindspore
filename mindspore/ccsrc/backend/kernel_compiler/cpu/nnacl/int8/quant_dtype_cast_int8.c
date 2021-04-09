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
#include "nnacl/int8/quant_dtype_cast_int8.h"
#include "nnacl/errorcode.h"

int DoDequantizeInt8ToFp32(const int8_t *quant_values, float *real_values, float scale, int32_t zp, int size) {
  if (quant_values == NULL || real_values == NULL) {
    return NNACL_PARAM_INVALID;
  }

  for (int i = 0; i < size; ++i) {
    real_values[i] = (quant_values[i] - zp) * scale;
  }
  return NNACL_OK;
}

int DoQuantizeFp32ToInt8(const float *real_values, int8_t *quant_values, float scale, int32_t zp, int size,
                         bool uint8_flag) {
  if (quant_values == NULL || real_values == NULL) {
    return NNACL_PARAM_INVALID;
  }

  if (uint8_flag) {
    zp += 128;
  }
  const float inverse_scale = 1.0f / scale;
  for (int i = 0; i < size; ++i) {
    if (isinf(real_values[i])) {
      quant_values[i] = 127;
    } else {
      int temp = round(real_values[i] * inverse_scale + zp);
      if (uint8_flag) {
        temp -= 128;
      }
      temp = temp < 127 ? temp : 127;
      temp = temp > -128 ? temp : -128;
      quant_values[i] = (int8_t)temp;
    }
  }
  return NNACL_OK;
}

int DoDequantizeUInt8ToFp32(const uint8_t *quant_values, float *real_values, float scale, int32_t zp, int size) {
  if (quant_values == NULL || real_values == NULL) {
    return NNACL_PARAM_INVALID;
  }

  for (int i = 0; i < size; ++i) {
    real_values[i] = (float)((int)quant_values[i] - zp) * scale;
  }
  return NNACL_OK;
}

int DoQuantizeFp32ToUInt8(const float *real_values, uint8_t *quant_values, float scale, int32_t zp, int size) {
  if (quant_values == NULL || real_values == NULL) {
    return NNACL_PARAM_INVALID;
  }

  for (int i = 0; i < size; ++i) {
    if (isinf(real_values[i])) {
      quant_values[i] = 255;
    } else {
      float temp = (float)round(real_values[i] * 1.0 / scale + zp);
      if (temp > 255) {
        quant_values[i] = 255;
      } else if (temp < 0) {
        quant_values[i] = 0;
      } else {
        quant_values[i] = (uint8_t)temp;
      }
    }
  }
  return NNACL_OK;
}

int Int8ToUInt8(const int8_t *quant_values, uint8_t *real_values, int size) {
  if (quant_values == NULL || real_values == NULL) {
    return NNACL_PARAM_INVALID;
  }

  for (int i = 0; i < size; ++i) {
    int temp = quant_values[i] + 128;
    if (temp > 255) {
      real_values[i] = (uint8_t)255;
    } else if (temp < 0) {
      real_values[i] = 0;
    } else {
      real_values[i] = (uint8_t)temp;
    }
  }
  return NNACL_OK;
}

int UInt8ToInt8(const uint8_t *real_values, int8_t *quant_values, int size) {
  if (quant_values == NULL || real_values == NULL) {
    return NNACL_PARAM_INVALID;
  }

  for (int i = 0; i < size; ++i) {
    int temp = real_values[i] - 128;
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
