/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
// * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "nnacl/fp32/prelu_fp32.h"

#ifdef ENABLE_ARM64
static inline void PRelu4x16(const float *in, float *out, const float *cur_slope, size_t step) {
  asm volatile(
    "mov x10, %[in]\n"
    "mov x11, %[out]\n"
    "mov x12, %[cur_slope]\n"
    "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x12]\n"
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], %[step]\n"
    "fmul v16.4s, v0.4s, v4.4s\n"
    "fmul v17.4s, v1.4s, v5.4s\n"
    "fmul v18.4s, v2.4s, v6.4s\n"
    "fmul v19.4s, v3.4s, v7.4s\n"
    "fcmgt v20.4s, v0.4s, #0\n"
    "fcmgt v21.4s, v1.4s, #0\n"
    "fcmgt v22.4s, v2.4s, #0\n"
    "fcmgt v23.4s, v3.4s, #0\n"
    "ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x10], %[step]\n"
    "bif v0.16b, v16.16b, v20.16b\n"
    "bif v1.16b, v17.16b, v21.16b\n"
    "bif v2.16b, v18.16b, v22.16b\n"
    "bif v3.16b, v19.16b, v23.16b\n"
    "fmul v8.4s, v24.4s, v4.4s\n"
    "fmul v9.4s, v25.4s, v5.4s\n"
    "fmul v10.4s, v26.4s, v6.4s\n"
    "fmul v11.4s, v27.4s, v7.4s\n"
    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x11], %[step]\n"
    "fcmgt v12.4s, v24.4s, #0\n"
    "fcmgt v13.4s, v25.4s, #0\n"
    "fcmgt v14.4s, v26.4s, #0\n"
    "fcmgt v15.4s, v27.4s, #0\n"
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], %[step]\n"
    "bif v24.16b, v8.16b, v12.16b\n"
    "bif v25.16b, v9.16b, v13.16b\n"
    "bif v26.16b, v10.16b, v14.16b\n"
    "bif v27.16b, v11.16b, v15.16b\n"
    "fmul v16.4s, v0.4s, v4.4s\n"
    "fmul v17.4s, v1.4s, v5.4s\n"
    "fmul v18.4s, v2.4s, v6.4s\n"
    "fmul v19.4s, v3.4s, v7.4s\n"
    "st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x11], %[step]\n"
    "fcmgt v20.4s, v0.4s, #0\n"
    "fcmgt v21.4s, v1.4s, #0\n"
    "fcmgt v22.4s, v2.4s, #0\n"
    "fcmgt v23.4s, v3.4s, #0\n"
    "ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x10]\n"
    "bif v0.16b, v16.16b, v20.16b\n"
    "bif v1.16b, v17.16b, v21.16b\n"
    "bif v2.16b, v18.16b, v22.16b\n"
    "bif v3.16b, v19.16b, v23.16b\n"
    "fmul v8.4s, v24.4s, v4.4s\n"
    "fmul v9.4s, v25.4s, v5.4s\n"
    "fmul v10.4s, v26.4s, v6.4s\n"
    "fmul v11.4s, v27.4s, v7.4s\n"
    "fcmgt v12.4s, v24.4s, #0\n"
    "fcmgt v13.4s, v25.4s, #0\n"
    "fcmgt v14.4s, v26.4s, #0\n"
    "fcmgt v15.4s, v27.4s, #0\n"
    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x11], %[step]\n"
    "bif v24.16b, v8.16b, v12.16b\n"
    "bif v25.16b, v9.16b, v13.16b\n"
    "bif v26.16b, v10.16b, v14.16b\n"
    "bif v27.16b, v11.16b, v15.16b\n"
    "st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x11]\n"
    :
    : [ in ] "r"(in), [ out ] "r"(out), [ cur_slope ] "r"(cur_slope), [ step ] "r"(step)
    : "x10", "x11", "x12", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
      "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
}
#endif

void PRelu(const float *input, float *output, const float *slope, int start, int end, int channel) {
  int i = start;
#ifdef ENABLE_ARM64
  for (; i < end - 3; i += 4) {
    const float *cur_in = input + i * channel;
    float *cur_out = output + i * channel;
    int j = 0;
    for (; j < channel - 15; j += 16) {
      const float *in = cur_in + j;
      float *out = cur_out + j;
      const float *cur_slope = slope + j;
      size_t step = channel * sizeof(float);
      PRelu4x16(in, out, cur_slope, step);
    }
    for (; j < channel; j++) {
      cur_out[j] = (cur_in[j] > 0) ? cur_in[j] : (cur_in[j] * slope[j]);
      cur_out[j + channel] = (cur_in[j + channel] > 0) ? cur_in[j + channel] : cur_in[j + channel] * slope[j];
      cur_out[j + 2 * channel] =
        (cur_in[j + 2 * channel] > 0) ? cur_in[j + 2 * channel] : (cur_in[j + 2 * channel] * slope[j]);
      cur_out[j + 3 * channel] =
        (cur_in[j + 3 * channel] > 0) ? cur_in[j + 3 * channel] : (cur_in[j + 3 * channel] * slope[j]);
    }
  }
#endif
  for (; i < end; i++) {
    const float *cur_in = input + i * channel;
    float *cur_out = output + i * channel;
    int j = 0;
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
    for (; j < channel - 3; j += 4) {
      MS_FLOAT32X4 in = MS_LDQ_F32(cur_in + j);
      MS_FLOAT32X4 s = MS_LDQ_F32(slope + j);
      MS_FLOAT32X4 mul = MS_MULQ_F32(in, s);
      MS_FLOAT32X4 zero = MS_MOVQ_F32(0.0f);
      MS_FLOAT32X4 res = MS_BLENDQ_F32(mul, in, MS_CMPGTQ_F32(in, zero));
      MS_STQ_F32(cur_out + j, res);
    }
#endif
    for (; j < channel; j++) {
      if (cur_in[j] > 0) {
        cur_out[j] = cur_in[j];
      } else {
        cur_out[j] = cur_in[j] * slope[j];
      }
    }
  }
}

void PReluShareChannel(const float *input, float *output, float slope, int start, int end) {
  for (int i = start; i < end; i++) {
    if (input[i] > 0) {
      output[i] = input[i];
    } else {
      output[i] = input[i] * slope;
    }
  }
}
