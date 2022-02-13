/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "nnacl/fp16/prelu_fp16.h"

#ifdef ENABLE_ARM64
static inline void PReluFp164x32(const float16_t *in, float16_t *out, const float16_t *cur_slope, size_t step) {
  asm volatile(
    "mov x10, %[in]\n"
    "mov x11, %[out]\n"
    "mov x12, %[cur_slope]\n"
    "ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x12]\n"
    "ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x10], %[step]\n"
    "fmul v16.8h, v0.8h, v4.8h\n"
    "fmul v17.8h, v1.8h, v5.8h\n"
    "fmul v18.8h, v2.8h, v6.8h\n"
    "fmul v19.8h, v3.8h, v7.8h\n"
    "fcmgt v20.8h, v0.8h, #0\n"
    "fcmgt v21.8h, v1.8h, #0\n"
    "fcmgt v22.8h, v2.8h, #0\n"
    "fcmgt v23.8h, v3.8h, #0\n"
    "ld1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x10], %[step]\n"
    "bif v0.16b, v16.16b, v20.16b\n"
    "bif v1.16b, v17.16b, v21.16b\n"
    "bif v2.16b, v18.16b, v22.16b\n"
    "bif v3.16b, v19.16b, v23.16b\n"
    "fmul v8.8h, v24.8h, v4.8h\n"
    "fmul v9.8h, v25.8h, v5.8h\n"
    "fmul v10.8h, v26.8h, v6.8h\n"
    "fmul v11.8h, v27.8h, v7.8h\n"
    "st1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x11], %[step]\n"
    "fcmgt v12.8h, v24.8h, #0\n"
    "fcmgt v13.8h, v25.8h, #0\n"
    "fcmgt v14.8h, v26.8h, #0\n"
    "fcmgt v15.8h, v27.8h, #0\n"
    "ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x10], %[step]\n"
    "bif v24.16b, v8.16b, v12.16b\n"
    "bif v25.16b, v9.16b, v13.16b\n"
    "bif v26.16b, v10.16b, v14.16b\n"
    "bif v27.16b, v11.16b, v15.16b\n"
    "fmul v16.8h, v0.8h, v4.8h\n"
    "fmul v17.8h, v1.8h, v5.8h\n"
    "fmul v18.8h, v2.8h, v6.8h\n"
    "fmul v19.8h, v3.8h, v7.8h\n"
    "st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x11], %[step]\n"
    "fcmgt v20.8h, v0.8h, #0\n"
    "fcmgt v21.8h, v1.8h, #0\n"
    "fcmgt v22.8h, v2.8h, #0\n"
    "fcmgt v23.8h, v3.8h, #0\n"
    "ld1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x10]\n"
    "bif v0.16b, v16.16b, v20.16b\n"
    "bif v1.16b, v17.16b, v21.16b\n"
    "bif v2.16b, v18.16b, v22.16b\n"
    "bif v3.16b, v19.16b, v23.16b\n"
    "fmul v8.8h, v24.8h, v4.8h\n"
    "fmul v9.8h, v25.8h, v5.8h\n"
    "fmul v10.8h, v26.8h, v6.8h\n"
    "fmul v11.8h, v27.8h, v7.8h\n"
    "fcmgt v12.8h, v24.8h, #0\n"
    "fcmgt v13.8h, v25.8h, #0\n"
    "fcmgt v14.8h, v26.8h, #0\n"
    "fcmgt v15.8h, v27.8h, #0\n"
    "st1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x11], %[step]\n"
    "bif v24.16b, v8.16b, v12.16b\n"
    "bif v25.16b, v9.16b, v13.16b\n"
    "bif v26.16b, v10.16b, v14.16b\n"
    "bif v27.16b, v11.16b, v15.16b\n"
    "st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x11]\n"
    :
    : [ in ] "r"(in), [ out ] "r"(out), [ cur_slope ] "r"(cur_slope), [ step ] "r"(step)
    : "x10", "x11", "x12", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
      "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
}
#endif

void PReluFp16(const float16_t *input, float16_t *output, const float16_t *slope, int start, int end, int channel) {
  int i = start;
#ifdef ENABLE_ARM64
  for (; i <= end - C4NUM; i += C4NUM) {
    const float16_t *cur_in = input + i * channel;
    float16_t *cur_out = output + i * channel;
    int j = 0;
    for (; j <= channel - C32NUM; j += C32NUM) {
      const float16_t *in = cur_in + j;
      float16_t *out = cur_out + j;
      const float16_t *cur_slope = slope + j;
      size_t step = channel * sizeof(float16_t);
      PReluFp164x32(in, out, cur_slope, step);
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
    const float16_t *cur_in = input + i * channel;
    float16_t *cur_out = output + i * channel;
    int j = 0;
#ifdef ENABLE_NEON
    for (; j <= channel - C8NUM; j += C8NUM) {
      float16x8_t in = vld1q_f16(cur_in + j);
      float16x8_t s = vld1q_f16(slope + j);
      float16x8_t mul = vmulq_f16(in, s);
      uint16x8_t mask = vcleq_f16(in, vmovq_n_f16(0.0f));
      vst1q_f16(cur_out + j, vbslq_f16(mask, mul, in));
    }
#endif
    for (; j < channel; j++) {
      cur_out[j] = cur_in[j] > 0 ? cur_in[j] : cur_in[j] * slope[j];
    }
  }
}

void PReluShareChannelFp16(const float16_t *input, float16_t *output, float16_t slope, int start, int end) {
  int i = start;
#ifdef ENABLE_NEON
  float16x8_t zero_data = vdupq_n_f16(0);
  float16x8_t slope_data = vdupq_n_f16(slope);
  for (; i <= end - C8NUM; i += C8NUM) {
    float16x8_t src_tmp = vld1q_f16(input + i);
    float16x8_t mul_tmp = vmulq_f16(src_tmp, slope_data);
    uint16x8_t mask = vcleq_f16(src_tmp, zero_data);
    vst1q_f16(output + i, vbslq_f16(mask, mul_tmp, src_tmp));
  }
#endif
  for (; i < end; i++) {
    output[i] = input[i] > 0 ? input[i] : input[i] * slope;
  }
}
