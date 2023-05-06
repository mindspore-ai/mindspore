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
#ifdef ENABLE_ARM
#include <arm_neon.h>
#endif

#ifdef ENABLE_ARM64
inline void Int8ToFp32_arm64(const int8_t *quant_values, float *dst, float scale, int32_t zp, int size) {
  asm volatile(
    "mov w8, %w[size]\n"
    "cmp w8, #0\n"
    "beq 2f\n"

    "dup v20.4s, %w[zp32]\n"
    "dup v21.4s, %w[scale]\n"

    "cmp w8, #16\n"
    "blt 1f\n"

    "0:\n"
    "subs w8, w8, #16\n"
    "ld1 {v7.16b}, [%[quant_values]], #16\n"

    "sxtl v8.8h, v7.8b\n"
    "sxtl2 v9.8h, v7.16b\n"

    "sxtl v0.4s, v8.4h\n"
    "sxtl2 v1.4s, v8.8h\n"
    "sxtl v2.4s, v9.4h\n"
    "sxtl2 v3.4s, v9.8h\n"
    "sub v0.4s, v0.4s, v20.4s\n"
    "sub v1.4s, v1.4s, v20.4s\n"
    "sub v2.4s, v2.4s, v20.4s\n"
    "sub v3.4s, v3.4s, v20.4s\n"
    "scvtf v4.4s, v0.4s\n"
    "scvtf v5.4s, v1.4s\n"
    "scvtf v6.4s, v2.4s\n"
    "scvtf v7.4s, v3.4s\n"

    "fmul v0.4s, v4.4s, v21.4s\n"
    "fmul v1.4s, v5.4s, v21.4s\n"
    "fmul v2.4s, v6.4s, v21.4s\n"
    "fmul v3.4s, v7.4s, v21.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[dst]], #64\n"
    "beq 2f\n"
    "cmp w8, #16\n"
    "bge 0b\n"

    "1:\n"
    "ldrsb w9, [%[quant_values]], #1\n"

    "subs w8, w8, #1\n"
    "sub w9, w9, %w[zp32]\n"
    "scvtf s9, w9\n"

    "fmul s9, s9, s21\n"
    "str s9, [%[dst]], #4\n"
    "bne 1b\n"

    "2:\n"

    :
    : [ quant_values ] "r"(quant_values), [ dst ] "r"(dst), [ scale ] "r"(scale), [ zp32 ] "r"(zp), [ size ] "r"(size)
    : "w8", "w9", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v20", "v21");
}
#endif

int DoDequantizeInt8ToFp32(const int8_t *quant_values, float *real_values, float scale, int32_t zp, int size) {
  if (quant_values == NULL || real_values == NULL) {
    return NNACL_PARAM_INVALID;
  }

#ifdef ENABLE_ARM64
  Int8ToFp32_arm64(quant_values, real_values, scale, zp, size);
#else
  for (int i = 0; i < size; i++) {
    real_values[i] = (quant_values[i] - zp) * scale;
  }
#endif
  return NNACL_OK;
}

#ifdef ENABLE_ARM64
inline void Fp32ToInt8_arm64(const float *real_values, int8_t *quant_values, float scale, int32_t zp, int size,
                             int32_t min_value, int32_t max_value) {
  float ivs = 1.0f / scale;

  asm volatile(
    "mov w8, %w[size]\n"
    "cmp w8, #0\n"
    "beq 2f\n"

    "dup v12.4s, %w[ivs]\n"
    "dup v13.4s, %w[min_value]\n"
    "dup v14.4s, %w[max_value]\n"
    "cmp w8, #16\n"
    "blt 1f\n"
    "0:\n"
    "subs w8, w8, #16\n"
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[real_values]], #64\n"
    "dup v8.4s, %w[zp]\n"
    "dup v9.4s, %w[zp]\n"
    "dup v10.4s, %w[zp]\n"
    "dup v11.4s, %w[zp]\n"
    "scvtf v4.4s, v8.4s\n"
    "scvtf v5.4s, v9.4s\n"
    "scvtf v6.4s, v10.4s\n"
    "scvtf v7.4s, v11.4s\n"
    "fmla v4.4s, v0.4s, v12.4s\n"
    "fmla v5.4s, v1.4s, v12.4s\n"
    "fmla v6.4s, v2.4s, v12.4s\n"
    "fmla v7.4s, v3.4s, v12.4s\n"

    "fcvtas v0.4s, v4.4s\n"
    "fcvtas v1.4s, v5.4s\n"
    "fcvtas v2.4s, v6.4s\n"
    "fcvtas v3.4s, v7.4s\n"
    "smax v0.4s, v0.4s, v13.4s\n"
    "smax v1.4s, v1.4s, v13.4s\n"
    "smax v2.4s, v2.4s, v13.4s\n"
    "smax v3.4s, v3.4s, v13.4s\n"
    "smin v0.4s, v0.4s, v14.4s\n"
    "smin v1.4s, v1.4s, v14.4s\n"
    "smin v2.4s, v2.4s, v14.4s\n"
    "smin v3.4s, v3.4s, v14.4s\n"

    "sqxtn v4.4h, v0.4s\n"
    "sqxtn2 v4.8h, v1.4s\n"
    "sqxtn v5.4h, v2.4s\n"
    "sqxtn2 v5.8h, v3.4s\n"
    "sqxtn v6.8b, v4.8h\n"
    "sqxtn2 v6.16b, v5.8h\n"
    "st1 {v6.16b}, [%[quant_values]], #16\n"

    "beq 2f\n"
    "cmp w8, #16\n"
    "bge 0b\n"

    "1:\n"
    "scvtf s0, %w[zp]\n"
    "subs w8, w8, #1\n"
    "ldr s4, [%[real_values]], #4\n"
    "fmul s4, s4, s12\n"
    "fadd s0, s0, s4\n"
    "fcvtas s0, s0\n"
    "smax v0.4s, v0.4s, v13.4s\n"
    "smin v0.4s, v0.4s, v14.4s\n"
    "sqxtn v1.4h, v0.4s\n"
    "sqxtn v0.8b, v1.8h\n"
    "st1 {v0.b}[0], [%[quant_values]], #1\n"

    "bne 1b\n"

    "2:\n"
    :
    : [ quant_values ] "r"(quant_values), [ real_values ] "r"(real_values), [ scale ] "r"(scale), [ zp ] "r"(zp),
      [ size ] "r"(size), [ ivs ] "r"(ivs), [ min_value ] "r"(min_value), [ max_value ] "r"(max_value)
    : "w8", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14");
}
#endif

int DoQuantizeFp32ToInt8(const float *real_values, int8_t *quant_values, float scale, int32_t zp, int size,
                         int32_t min_value, int32_t max_value) {
  if (quant_values == NULL || real_values == NULL) {
    return NNACL_PARAM_INVALID;
  }
#ifdef ENABLE_ARM64
  Fp32ToInt8_arm64(real_values, quant_values, scale, zp, size, min_value, max_value);
#else
  const float inverse_scale = 1.0f / scale;
  for (int i = 0; i < size; ++i) {
    if (real_values[i] == INFINITY) {
      quant_values[i] = max_value;
    } else if (real_values[i] == -INFINITY) {
      quant_values[i] = min_value;
    } else {
      int temp = round(real_values[i] * inverse_scale + zp);
      temp = temp < max_value ? temp : max_value;
      temp = temp > min_value ? temp : min_value;
      quant_values[i] = (int8_t)temp;
    }
  }
#endif
  return NNACL_OK;
}

#ifdef ENABLE_ARM64
inline void Fp32ToInt8Perchannel_arm64(const float *real_values, int8_t *quant_values, float *scales, int32_t *zps,
                                       int size, int row_length, int32_t min_value, int32_t max_value) {
  volatile float ivs[size];
  for (int i = 0; i < size; i++) {
    volatile int channel_index = i / row_length;
    ivs[i] = 1.0f / scales[channel_index];
  }
  volatile int32_t zp = zps[0];

  asm volatile(
    "mov w8, %w[size]\n"
    "cmp w8, #0\n"
    "beq 2f\n"

    "mov x4, %[ivs]\n"  // reload ivs
    "dup v13.4s, %w[min_value]\n"
    "dup v14.4s, %w[max_value]\n"
    "cmp w8, #16\n"
    "blt 1f\n"
    "0:\n"
    "subs w8, w8, #16\n"
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[real_values]], #64\n"
    "dup v8.4s, %w[zp]\n"
    "dup v9.4s, %w[zp]\n"
    "dup v10.4s, %w[zp]\n"
    "dup v11.4s, %w[zp]\n"
    "scvtf v4.4s, v8.4s\n"
    "scvtf v5.4s, v9.4s\n"
    "scvtf v6.4s, v10.4s\n"
    "scvtf v7.4s, v11.4s\n"
    "ld1 {v12.4s}, [x4], #16\n"
    "fmla v4.4s, v0.4s, v12.4s\n"
    "ld1 {v12.4s}, [x4], #16\n"
    "fmla v5.4s, v1.4s, v12.4s\n"
    "ld1 {v12.4s}, [x4], #16\n"
    "fmla v6.4s, v2.4s, v12.4s\n"
    "ld1 {v12.4s}, [x4], #16\n"
    "fmla v7.4s, v3.4s, v12.4s\n"

    "fcvtas v0.4s, v4.4s\n"
    "fcvtas v1.4s, v5.4s\n"
    "fcvtas v2.4s, v6.4s\n"
    "fcvtas v3.4s, v7.4s\n"
    "smax v0.4s, v0.4s, v13.4s\n"
    "smax v1.4s, v1.4s, v13.4s\n"
    "smax v2.4s, v2.4s, v13.4s\n"
    "smax v3.4s, v3.4s, v13.4s\n"
    "smin v0.4s, v0.4s, v14.4s\n"
    "smin v1.4s, v1.4s, v14.4s\n"
    "smin v2.4s, v2.4s, v14.4s\n"
    "smin v3.4s, v3.4s, v14.4s\n"

    "sqxtn v4.4h, v0.4s\n"
    "sqxtn2 v4.8h, v1.4s\n"
    "sqxtn v5.4h, v2.4s\n"
    "sqxtn2 v5.8h, v3.4s\n"
    "sqxtn v6.8b, v4.8h\n"
    "sqxtn2 v6.16b, v5.8h\n"
    "st1 {v6.16b}, [%[quant_values]], #16\n"

    "beq 2f\n"
    "cmp w8, #16\n"
    "bge 0b\n"

    "1:\n"
    "scvtf s0, %w[zp]\n"
    "subs w8, w8, #1\n"
    "ldr s4, [%[real_values]], #4\n"
    "fmul s4, s4, s12\n"
    "fadd s0, s0, s4\n"
    "fcvtas s0, s0\n"
    "smax v0.4s, v0.4s, v13.4s\n"
    "smin v0.4s, v0.4s, v14.4s\n"
    "sqxtn v1.4h, v0.4s\n"
    "sqxtn v0.8b, v1.8h\n"
    "st1 {v0.b}[0], [%[quant_values]], #1\n"

    "bne 1b\n"

    "2:\n"
    :
    : [ quant_values ] "r"(quant_values), [ real_values ] "r"(real_values), [ scales ] "r"(scales), [ zp ] "r"(zp),
      [ size ] "r"(size), [ row_length ] "r"(row_length), [ ivs ] "r"(ivs), [ min_value ] "r"(min_value),
      [ max_value ] "r"(max_value)
    : "w8", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "x4");
}
#endif

int DoChannelRowFp32ToInt8(const float *real_values, int8_t *quant_values, float *scale, int32_t *zp, int size,
                           int row_length, int32_t min_value, int32_t max_value) {
  if (quant_values == NULL || real_values == NULL || scale == NULL || zp == NULL || row_length == 0) {
    return NNACL_PARAM_INVALID;
  }
#ifdef ENABLE_ARM64
  Fp32ToInt8Perchannel_arm64(real_values, quant_values, scale, zp, size, row_length, min_value, max_value);
#else
  for (int i = 0; i < size; ++i) {
    int channel_index = i / row_length;
    const float inverse_scale = 1.0f / scale[channel_index];
    if (real_values[i] == INFINITY) {
      quant_values[i] = max_value;
    } else if (real_values[i] == -INFINITY) {
      quant_values[i] = min_value;
    } else {
      int temp = round(real_values[i] * inverse_scale + zp[channel_index]);
      temp = temp < max_value ? temp : max_value;
      temp = temp > min_value ? temp : min_value;
      quant_values[i] = (int8_t)temp;
    }
  }
#endif
  return NNACL_OK;
}

int DoChannelColFp32ToInt8(const float *real_values, int8_t *quant_values, float *scale, int32_t *zp, int size,
                           int row_length, int32_t min_value, int32_t max_value) {
  if (quant_values == NULL || real_values == NULL || scale == NULL || zp == NULL || row_length == 0) {
    return NNACL_PARAM_INVALID;
  }
  int row_total = size / row_length;
  for (int r = 0; r < row_total; r++) {
    const float *real_current = real_values + r * row_length;
    int8_t *quant_current = quant_values + r * row_length;
    for (int c = 0; c < row_length; c++) {
      const float inverse_scale = 1.0f / scale[c];
      if (real_current[c] == INFINITY) {
        quant_current[c] = max_value;
      } else if (real_current[c] == -INFINITY) {
        quant_current[c] = min_value;
      } else {
        int temp = round(real_current[c] * inverse_scale + zp[c]);
        temp = temp < max_value ? temp : max_value;
        temp = temp > min_value ? temp : min_value;
        quant_current[c] = (int8_t)temp;
      }
    }
  }
  return NNACL_OK;
}

int DoQuantizeFp32ToInt8FromUint8Source(const float *real_values, int8_t *quant_values, float scale, int32_t zp,
                                        int size, int32_t min_value, int32_t max_value) {
  if (quant_values == NULL || real_values == NULL) {
    return NNACL_PARAM_INVALID;
  }

  zp += 128;
  const float inverse_scale = 1.0f / scale;
  for (int i = 0; i < size; ++i) {
    if (real_values[i] == INFINITY) {
      quant_values[i] = max_value;
    } else if (real_values[i] == -INFINITY) {
      quant_values[i] = min_value;
    } else {
      int temp = round(real_values[i] * inverse_scale + zp);
      temp -= 128;
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
    int temp = (int)real_values[i] - 128;
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
