/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifdef ENABLE_ARM64
void Int8ToFp16_arm64(const int8_t *quant_values, float16_t *dst, float scale, int32_t zp, int size) {
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

    "fcvtn v4.4h, v0.4s\n"
    "fcvtn2 v4.8h, v1.4s\n"
    "fcvtn v5.4h, v2.4s\n"
    "fcvtn2 v5.8h, v3.4s\n"

    "st1 {v4.8h, v5.8h}, [%[dst]], #32\n"
    "beq 2f\n"
    "cmp w8, #16\n"
    "bge 0b\n"

    "1:\n"
    "ldrsb w9, [%[quant_values]], #1\n"

    "subs w8, w8, #1\n"
    "sub w9, w9, %w[zp32]\n"
    "scvtf s9, w9\n"

    "fmul s9, s9, s21\n"
    "fcvtn v4.4h, v9.4s\n"
    "str h4, [%[dst]], #2\n"
    "bne 1b\n"

    "2:\n"

    :
    : [ quant_values ] "r"(quant_values), [ dst ] "r"(dst), [ scale ] "r"(scale), [ zp32 ] "r"(zp), [ size ] "r"(size)
    : "w8", "w9", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v20", "v21");
}
#endif

int DoDequantizeInt8ToFp16(const int8_t *quant_values, float16_t *real_values, float scale, int32_t zp, int size) {
  if (quant_values == NULL || real_values == NULL) {
    return NNACL_PARAM_INVALID;
  }
#ifdef ENABLE_ARM64
  Int8ToFp16_arm64(quant_values, real_values, scale, zp, size);
#else
  for (int i = 0; i < size; ++i) {
    real_values[i] = (quant_values[i] - zp) * scale;
  }
#endif
  return NNACL_OK;
}

#ifdef ENABLE_ARM64
void Fp16ToInt8_arm64(const float16_t *real_values, int8_t *quant_values, float scale, int32_t zp, int size) {
  const float one = 1.0f;
  const float ivs = one / scale;
  const int32_t min_value = -128;
  const int32_t max_value = 127;
  asm volatile(
    "mov w8, %w[size]\n"
    "cmp w8, wzr\n"
    "beq 3f\n"

    "dup v28.4s, %w[ivs]\n"
    "dup v29.4s, %w[min_value]\n"
    "dup v30.4s, %w[max_value]\n"

    "cmp w8, #32\n"
    "blt 2f\n"
    "1:\n"  // loop 32
    "subs w8, w8, #32\n"
    "ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [%[real_values]], #64\n"
    "fcvtl v8.4s, v0.4h\n"
    "fcvtl2 v9.4s, v0.8h\n"
    "fcvtl v10.4s, v1.4h\n"
    "fcvtl2 v11.4s, v1.8h\n"
    "fcvtl v12.4s, v2.4h\n"
    "fcvtl2 v13.4s, v2.8h\n"
    "fcvtl v14.4s, v3.4h\n"
    "fcvtl2 v15.4s, v3.8h\n"

    "dup v16.4s, %w[zp]\n"
    "dup v17.4s, %w[zp]\n"
    "dup v18.4s, %w[zp]\n"
    "dup v19.4s, %w[zp]\n"
    "dup v20.4s, %w[zp]\n"
    "dup v21.4s, %w[zp]\n"
    "dup v22.4s, %w[zp]\n"
    "dup v23.4s, %w[zp]\n"
    "scvtf v16.4s, v16.4s\n"
    "scvtf v17.4s, v17.4s\n"
    "scvtf v18.4s, v18.4s\n"
    "scvtf v19.4s, v19.4s\n"
    "scvtf v20.4s, v20.4s\n"
    "scvtf v21.4s, v21.4s\n"
    "scvtf v22.4s, v22.4s\n"
    "scvtf v23.4s, v23.4s\n"

    "fmla v16.4s, v8.4s, v28.4s\n"
    "fmla v17.4s, v9.4s, v28.4s\n"
    "fmla v18.4s, v10.4s, v28.4s\n"
    "fmla v19.4s, v11.4s, v28.4s\n"
    "fmla v20.4s, v12.4s, v28.4s\n"
    "fmla v21.4s, v13.4s, v28.4s\n"
    "fmla v22.4s, v14.4s, v28.4s\n"
    "fmla v23.4s, v15.4s, v28.4s\n"

    "fcvtas v8.4s, v16.4s\n"
    "fcvtas v9.4s, v17.4s\n"
    "fcvtas v10.4s, v18.4s\n"
    "fcvtas v11.4s, v19.4s\n"
    "fcvtas v12.4s, v20.4s\n"
    "fcvtas v13.4s, v21.4s\n"
    "fcvtas v14.4s, v22.4s\n"
    "fcvtas v15.4s, v23.4s\n"

    "smax v8.4s, v8.4s, v29.4s\n"
    "smax v9.4s, v9.4s, v29.4s\n"
    "smax v10.4s, v10.4s, v29.4s\n"
    "smax v11.4s, v11.4s, v29.4s\n"
    "smax v12.4s, v12.4s, v29.4s\n"
    "smax v13.4s, v13.4s, v29.4s\n"
    "smax v14.4s, v14.4s, v29.4s\n"
    "smax v15.4s, v15.4s, v29.4s\n"
    "smin v8.4s, v8.4s, v30.4s\n"
    "smin v9.4s, v9.4s, v30.4s\n"
    "smin v10.4s, v10.4s, v30.4s\n"
    "smin v11.4s, v11.4s, v30.4s\n"
    "smin v12.4s, v12.4s, v30.4s\n"
    "smin v13.4s, v13.4s, v30.4s\n"
    "smin v14.4s, v14.4s, v30.4s\n"
    "smin v15.4s, v15.4s, v30.4s\n"

    "sqxtn v16.4h, v8.4s\n"
    "sqxtn2 v16.8h, v9.4s\n"
    "sqxtn v17.4h, v10.4s\n"
    "sqxtn2 v17.8h, v11.4s\n"
    "sqxtn v18.4h, v12.4s\n"
    "sqxtn2 v18.8h, v13.4s\n"
    "sqxtn v19.4h, v14.4s\n"
    "sqxtn2 v19.8h, v15.4s\n"
    "sqxtn v20.8b, v16.8h\n"
    "sqxtn2 v20.16b, v17.8h\n"
    "sqxtn v21.8b, v18.8h\n"
    "sqxtn2 v21.16b, v19.8h\n"

    "st1 {v20.16b, v21.16b}, [%[quant_values]], #32\n"

    "beq 3f\n"
    "cmp w8, #32\n"
    "bge 1b\n"

    "2:\n"  // 1 by 1
    "scvtf s10, %w[zp]\n"
    "subs w8, w8, #1\n"
    "ldr h0, [%[real_values]], #2\n"
    "fcvt s0, h0\n"
    "fmul s0, s0, s28\n"
    "fadd s0, s0, s10\n"
    "fcvtas s0, s0\n"
    "smax v0.4s, v0.4s, v29.4s\n"
    "smin v0.4s, v0.4s, v30.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn v0.8b, v0.8h\n"
    "st1 {v0.b}[0], [%[quant_values]], #1\n"
    "bne 2b\n"

    "3:\n"
    :
    : [ size ] "r"(size), [ ivs ] "r"(ivs), [ real_values ] "r"(real_values), [ quant_values ] "r"(quant_values),
      [ zp ] "r"(zp), [ min_value ] "r"(min_value), [ max_value ] "r"(max_value)
    : "w8", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
      "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30");
}
#endif

int DoQuantizeFp16ToInt8(const float16_t *real_values, int8_t *quant_values, float scale, int32_t zp, int size) {
  if (quant_values == NULL || real_values == NULL) {
    return NNACL_PARAM_INVALID;
  }
#ifdef ENABLE_ARM64
  Fp16ToInt8_arm64(real_values, quant_values, scale, zp, size);
#else
  const int8_t min_value = -128;
  const int8_t max_value = 127;
  for (int i = 0; i < size; ++i) {
    if (real_values[i] == INFINITY) {
      quant_values[i] = max_value;
      continue;
    }
    if (real_values[i] == -INFINITY) {
      quant_values[i] = min_value;
      continue;
    }
    float temp = round((float)real_values[i] / scale + zp);
    if (temp > max_value) {
      quant_values[i] = max_value;
    } else if (temp < min_value) {
      quant_values[i] = min_value;
    } else {
      quant_values[i] = (int8_t)temp;
    }
  }
#endif
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
    if (isinf((float)real_values[i])) {
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
