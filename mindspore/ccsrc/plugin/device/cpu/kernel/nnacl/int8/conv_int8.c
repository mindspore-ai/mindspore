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

#include "nnacl/int8/conv_int8.h"

#ifdef ENABLE_ARM32
void PackInputSum16x4PerChannelArm32(const int8_t *input_value, int32_t *input_sum, const int32_t *filter_zp_ptr,
                                     size_t plane_size, size_t input_channel, size_t output_channel) {
  size_t hw4 = UP_ROUND(plane_size, C4NUM);
  size_t ic16 = UP_ROUND(input_channel, C16NUM);

#ifdef ENABLE_ARM32
  size_t oc_div2 = output_channel / C2NUM * C2NUM;
  size_t oc_res2 = output_channel - oc_div2;
  size_t inputsun_stride = hw4 * C2NUM * 4 - C4NUM * C2NUM * 4;
  PreSum4x16Int8Peroc(input_value, input_sum, filter_zp_ptr, hw4, ic16, oc_div2, oc_res2, inputsun_stride);
#else
  for (int ri = 0; ri < plane_size; ri++) {
    int ri4div = ri / C4NUM, ri4mod = ri % C4NUM;
    for (int ci = 0; ci < output_channel; ci++) {
      int32_t tmp_sum_value = 0;
      int ci2div = ci / C2NUM, ci2mod = ci % C2NUM;
      int32_t filter_zp = filter_zp_ptr[ci];
      for (int di = 0; di < input_channel; di++) {
        size_t di16div = di / C16NUM, di16mod = di % C16NUM;
        int src_index = ri4div * C4NUM * ic16 + di16div * C16NUM * C4NUM + ri4mod * C16NUM + di16mod;
        tmp_sum_value += input_value[src_index];
      }
      int dst_index = ci2div * C2NUM * hw4 + ri * C2NUM + ci2mod;
      input_sum[dst_index] = tmp_sum_value * filter_zp;
    }
  }
#endif
  return;
}
#endif

void PackInputSum16x4PerChannel(const int8_t *input_value, int32_t *input_sum, const int32_t *filter_zp_ptr,
                                size_t plane_size, size_t input_channel, size_t output_channel) {
  size_t hw4 = UP_ROUND(plane_size, C4NUM);
  size_t ic16 = UP_ROUND(input_channel, C16NUM);
#ifdef ENABLE_ARM64
  size_t oc_div4 = output_channel / C4NUM * C4NUM;
  size_t oc_res4 = output_channel - oc_div4;
  size_t inputsun_stride = hw4 * C4NUM * 4 - C4NUM * C4NUM * 4;
  PreSum4x16Int8Peroc(input_value, input_sum, filter_zp_ptr, hw4, ic16, oc_div4, oc_res4, inputsun_stride);
#else

  for (int ri = 0; ri < plane_size; ri++) {
    int ri4div = ri / C4NUM, ri4mod = ri % C4NUM;
    for (int ci = 0; ci < output_channel; ci++) {
      int32_t tmp_sum_value = 0;
      int ci4div = ci / C4NUM, ci4mod = ci % C4NUM;
      int32_t filter_zp = filter_zp_ptr[ci];
      for (int di = 0; di < input_channel; di++) {
        size_t di16div = di / C16NUM, di16mod = di % C16NUM;
        int src_index = ri4div * C4NUM * ic16 + di16div * C16NUM * C4NUM + ri4mod * C16NUM + di16mod;
        tmp_sum_value += input_value[src_index];
      }
      int dst_index = ci4div * C4NUM * hw4 + ri * C4NUM + ci4mod;
      input_sum[dst_index] = tmp_sum_value * filter_zp;
    }
  }
#endif
  return;
}

void Conv1x1PreOptPeroc(const int8_t *src_input, int8_t *packed_input, int32_t *input_sum, size_t input_channel,
                        size_t output_channel, size_t plane_size, const int32_t *filter_zp, size_t inputsum_stride) {
  int ic4 = UP_ROUND(input_channel, C4NUM);
  int oc8 = UP_ROUND(output_channel, C8NUM);
  int hw8 = UP_ROUND(plane_size, C8NUM);
  size_t hw_8div = plane_size / C8NUM * C8NUM;
  size_t oc_8div = output_channel / C8NUM * C8NUM;
  size_t oc_8res = output_channel - oc_8div;
  size_t ic_4div = input_channel / C4NUM * C4NUM;

  const int8_t *src_r = src_input;
  int8_t *pack_r = packed_input;
  int32_t *input_sum_r = input_sum;

  for (int hwi = 0; hwi < hw_8div; hwi += C8NUM) {
    const int8_t *src_ic = src_r;
    int8_t *pack_ic = pack_r;
    int32_t *input_sum_oc = input_sum_r;
#ifdef ENABLE_ARM64
    size_t src_stride = input_channel;
    size_t ic_4res = input_channel - ic_4div;
    size_t input_sum_stride = inputsum_stride * 4 - C8NUM * C8NUM * 4;
    asm volatile(
      "dup v16.4s, wzr \n"
      "dup v17.4s, wzr \n"

      "mov x10, %[src_ic] \n"
      "mov x11, %[pack_ic] \n"

      "mov x0, #0 \n"
      "1: \n"
      "cmp x0, %[ic_4div] \n"
      "add x0, x0, #4\n"
      "mov x12, x10 \n"
      "add x10, x10, #4\n"
      "blt 2f \n"
      "cmp %[ic_4res], #0\n"
      "beq 6f \n"
      "cmp %[ic_4res], #1\n"
      "beq 3f \n"
      "cmp %[ic_4res], #2\n"
      "beq 4f \n"
      "cmp %[ic_4res], #3\n"
      "beq 5f \n"

      "2: \n"
      "ld1 {v0.s}[0], [x12], %[src_stride]\n"
      "ld1 {v0.s}[1], [x12], %[src_stride]\n"
      "ld1 {v0.s}[2], [x12], %[src_stride]\n"
      "ld1 {v0.s}[3], [x12], %[src_stride]\n"
      "ld1 {v1.s}[0], [x12], %[src_stride]\n"
      "ld1 {v1.s}[1], [x12], %[src_stride]\n"
      "ld1 {v1.s}[2], [x12], %[src_stride]\n"
      "ld1 {v1.s}[3], [x12], %[src_stride]\n"

      "st1 {v0.16b}, [x11], #16\n"
      "st1 {v1.16b}, [x11], #16\n"

      "saddlp v4.8h, v0.16b \n"
      "saddlp v5.8h, v1.16b \n"
      "saddlp v0.4s, v4.8h \n"
      "saddlp v1.4s, v5.8h \n"
      "add v16.4s, v16.4s, v0.4s \n"
      "add v17.4s, v17.4s, v1.4s \n"
      "b 1b \n"

      "3: \n" /* col res 1 */
      "dup v0.4s, wzr \n"
      "dup v1.4s, wzr \n"

      "ld1 {v0.b}[0],  [x12], %[src_stride]\n"
      "ld1 {v0.b}[4],  [x12], %[src_stride]\n"
      "ld1 {v0.b}[8],  [x12], %[src_stride]\n"
      "ld1 {v0.b}[12], [x12], %[src_stride]\n"
      "ld1 {v1.b}[0],  [x12], %[src_stride]\n"
      "ld1 {v1.b}[4],  [x12], %[src_stride]\n"
      "ld1 {v1.b}[8],  [x12], %[src_stride]\n"
      "ld1 {v1.b}[12], [x12], %[src_stride]\n"

      "st1 {v0.16b}, [x11], #16\n"
      "st1 {v1.16b}, [x11], #16\n"

      "saddlp v4.8h, v0.16b \n"
      "saddlp v5.8h, v1.16b \n"
      "saddlp v0.4s, v4.8h \n"
      "saddlp v1.4s, v5.8h \n"
      "add v16.4s, v16.4s, v0.4s \n"
      "add v17.4s, v17.4s, v1.4s \n"
      "b 6f \n"

      "4: \n" /* col res 2 */
      "dup v0.4s, wzr \n"
      "dup v1.4s, wzr \n"

      "ld1 {v0.h}[0], [x12], %[src_stride]\n"
      "ld1 {v0.h}[2], [x12], %[src_stride]\n"
      "ld1 {v0.h}[4], [x12], %[src_stride]\n"
      "ld1 {v0.h}[6], [x12], %[src_stride]\n"
      "ld1 {v1.h}[0], [x12], %[src_stride]\n"
      "ld1 {v1.h}[2], [x12], %[src_stride]\n"
      "ld1 {v1.h}[4], [x12], %[src_stride]\n"
      "ld1 {v1.h}[6], [x12], %[src_stride]\n"

      "st1 {v0.16b}, [x11], #16\n"
      "st1 {v1.16b}, [x11], #16\n"

      "saddlp v4.8h, v0.16b \n"
      "saddlp v5.8h, v1.16b \n"
      "saddlp v0.4s, v4.8h \n"
      "saddlp v1.4s, v5.8h \n"
      "add v16.4s, v16.4s, v0.4s \n"
      "add v17.4s, v17.4s, v1.4s \n"
      "b 6f \n"

      "5: \n" /* col res 3 */
      "dup v0.4s, wzr \n"
      "dup v1.4s, wzr \n"
      "add x13, x12, #2 \n"

      "ld1 {v0.h}[0], [x12], %[src_stride]\n"
      "ld1 {v0.b}[2], [x13], %[src_stride]\n"
      "ld1 {v0.h}[2], [x12], %[src_stride]\n"
      "ld1 {v0.b}[6], [x13], %[src_stride]\n"
      "ld1 {v0.h}[4], [x12], %[src_stride]\n"
      "ld1 {v0.b}[10], [x13], %[src_stride]\n"
      "ld1 {v0.h}[6], [x12], %[src_stride]\n"
      "ld1 {v0.b}[14], [x13], %[src_stride]\n"
      "ld1 {v1.h}[0], [x12], %[src_stride]\n"
      "ld1 {v1.b}[2], [x13], %[src_stride]\n"
      "ld1 {v1.h}[2], [x12], %[src_stride]\n"
      "ld1 {v1.b}[6], [x13], %[src_stride]\n"
      "ld1 {v1.h}[4], [x12], %[src_stride]\n"
      "ld1 {v1.b}[10], [x13], %[src_stride]\n"
      "ld1 {v1.h}[6], [x12], %[src_stride]\n"
      "ld1 {v1.b}[14], [x13], %[src_stride]\n"

      "st1 {v0.16b}, [x11], #16\n"
      "st1 {v1.16b}, [x11], #16\n"

      "saddlp v4.8h, v0.16b \n"
      "saddlp v5.8h, v1.16b \n"
      "saddlp v0.4s, v4.8h \n"
      "saddlp v1.4s, v5.8h \n"
      "add v16.4s, v16.4s, v0.4s \n"
      "add v17.4s, v17.4s, v1.4s \n"
      "b 6f \n"

      "6: \n"
      "dup v0.4s, v16.s[0]  \n"
      "dup v1.4s, v16.s[1]  \n"
      "dup v2.4s, v16.s[2]  \n"
      "dup v3.4s, v16.s[3]  \n"
      "dup v4.4s, v17.s[0]  \n"
      "dup v5.4s, v17.s[1]  \n"
      "dup v6.4s, v17.s[2]  \n"
      "dup v7.4s, v17.s[3]  \n"
      "mov x4, #0 \n"
      "mov x10, %[filter_zp] \n"
      "mov x11, %[input_sum_oc] \n"

      "7: \n"
      "cmp x4, %[oc_8div] \n"
      "beq 8f \n"
      "add x4, x4, #8\n"
      "ld1 {v16.4s}, [x10], #16\n"
      "ld1 {v17.4s}, [x10], #16\n"

      "mul v18.4s, v16.4s, v0.4s \n"
      "mul v19.4s, v17.4s, v0.4s \n"
      "st1 {v18.4s}, [x11], #16 \n"
      "st1 {v19.4s}, [x11], #16 \n"

      "mul v20.4s, v16.4s, v1.4s \n"
      "mul v21.4s, v17.4s, v1.4s \n"
      "st1 {v20.4s}, [x11], #16 \n"
      "st1 {v21.4s}, [x11], #16 \n"

      "mul v22.4s, v16.4s, v2.4s \n"
      "mul v23.4s, v17.4s, v2.4s \n"
      "st1 {v22.4s}, [x11], #16 \n"
      "st1 {v23.4s}, [x11], #16 \n"

      "mul v24.4s, v16.4s, v3.4s \n"
      "mul v25.4s, v17.4s, v3.4s \n"
      "st1 {v24.4s}, [x11], #16 \n"
      "st1 {v25.4s}, [x11], #16 \n"

      "mul v18.4s, v16.4s, v4.4s \n"
      "mul v19.4s, v17.4s, v4.4s \n"
      "st1 {v18.4s}, [x11], #16 \n"
      "st1 {v19.4s}, [x11], #16 \n"

      "mul v20.4s, v16.4s, v5.4s \n"
      "mul v21.4s, v17.4s, v5.4s \n"
      "st1 {v20.4s}, [x11], #16 \n"
      "st1 {v21.4s}, [x11], #16 \n"

      "mul v22.4s, v16.4s, v6.4s \n"
      "mul v23.4s, v17.4s, v6.4s \n"
      "st1 {v22.4s}, [x11], #16 \n"
      "st1 {v23.4s}, [x11], #16 \n"

      "mul v24.4s, v16.4s, v7.4s \n"
      "mul v25.4s, v17.4s, v7.4s \n"
      "st1 {v24.4s}, [x11], #16 \n"
      "st1 {v25.4s}, [x11], #16 \n"

      "add x11, x11, %[input_sum_stride] \n"
      "b 7b \n"

      "8: \n"
      "cmp %[oc_8res], #0\n"
      "beq 17f \n"

      "dup v16.4s, wzr \n"
      "dup v17.4s, wzr \n"
      "cmp %[oc_8res], #1\n"
      "beq 9f \n"
      "cmp %[oc_8res], #2\n"
      "beq 10f \n"
      "cmp %[oc_8res], #3\n"
      "beq 11f \n"
      "cmp %[oc_8res], #4\n"
      "beq 12f \n"
      "cmp %[oc_8res], #5\n"
      "beq 13f \n"
      "cmp %[oc_8res], #6\n"
      "beq 14f \n"
      "cmp %[oc_8res], #7\n"
      "beq 15f \n"

      "9: \n"
      "ld1 {v16.s}[0], [x10] \n"
      "b 16f \n"

      "10: \n"
      "ld1 {v16.d}[0], [x10] \n"
      "b 16f \n"

      "11: \n"
      "ld1 {v16.d}[0], [x10] \n"
      "add x10, x10, #8 \n"
      "ld1 {v16.s}[2], [x10] \n"
      "b 16f \n"

      "12: \n"
      "ld1 {v16.4s}, [x10] \n"
      "b 16f \n"

      "13: \n"
      "ld1 {v16.4s}, [x10], #16\n"
      "ld1 {v17.s}[0], [x10] \n"
      "b 16f \n"

      "14: \n"
      "ld1 {v16.4s}, [x10], #16\n"
      "ld1 {v17.d}[0], [x10] \n"
      "b 16f \n"

      "15: \n"
      "ld1 {v16.4s}, [x10], #16\n"
      "ld1 {v17.d}[0], [x10] \n"
      "add x10, x10, #8 \n"
      "ld1 {v17.s}[2], [x10] \n"
      "b 16f \n"

      "16: \n"
      "mul v18.4s, v16.4s, v0.4s \n"
      "mul v19.4s, v17.4s, v0.4s \n"
      "mul v20.4s, v16.4s, v1.4s \n"
      "mul v21.4s, v17.4s, v1.4s \n"
      "mul v22.4s, v16.4s, v2.4s \n"
      "mul v23.4s, v17.4s, v2.4s \n"
      "mul v24.4s, v16.4s, v3.4s \n"
      "mul v25.4s, v17.4s, v3.4s \n"
      "st1 {v18.4s}, [x11], #16 \n"
      "st1 {v19.4s}, [x11], #16 \n"
      "st1 {v20.4s}, [x11], #16 \n"
      "st1 {v21.4s}, [x11], #16 \n"
      "st1 {v22.4s}, [x11], #16 \n"
      "st1 {v23.4s}, [x11], #16 \n"
      "st1 {v24.4s}, [x11], #16 \n"
      "st1 {v25.4s}, [x11], #16 \n"

      "mul v18.4s, v16.4s, v4.4s \n"
      "mul v19.4s, v17.4s, v4.4s \n"
      "mul v20.4s, v16.4s, v5.4s \n"
      "mul v21.4s, v17.4s, v5.4s \n"
      "mul v22.4s, v16.4s, v6.4s \n"
      "mul v23.4s, v17.4s, v6.4s \n"
      "mul v24.4s, v16.4s, v7.4s \n"
      "mul v25.4s, v17.4s, v7.4s \n"
      "st1 {v18.4s}, [x11], #16 \n"
      "st1 {v19.4s}, [x11], #16 \n"
      "st1 {v20.4s}, [x11], #16 \n"
      "st1 {v21.4s}, [x11], #16 \n"
      "st1 {v22.4s}, [x11], #16 \n"
      "st1 {v23.4s}, [x11], #16 \n"
      "st1 {v24.4s}, [x11], #16 \n"
      "st1 {v25.4s}, [x11], #16 \n"

      "17: \n"

      :
      : [ src_ic ] "r"(src_ic), [ pack_ic ] "r"(pack_ic), [ filter_zp ] "r"(filter_zp),
        [ input_sum_oc ] "r"(input_sum_oc), [ input_sum_stride ] "r"(input_sum_stride), [ src_stride ] "r"(src_stride),
        [ ic_4div ] "r"(ic_4div), [ ic_4res ] "r"(ic_4res), [ oc_8div ] "r"(oc_8div), [ oc_8res ] "r"(oc_8res)
      : "x0", "x1", "x4", "x9", "x10", "x11", "x12", "x13", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25");
#else
    int32_t tmp_sum_value[8] = {0};
    for (int ici = 0; ici < ic_4div; ici += C4NUM) {
      for (int i = 0; i < C8NUM; i++) {
        tmp_sum_value[i] += src_ic[0 + i * input_channel];
        tmp_sum_value[i] += src_ic[1 + i * input_channel];
        tmp_sum_value[i] += src_ic[2 + i * input_channel];
        tmp_sum_value[i] += src_ic[3 + i * input_channel];
        pack_ic[0 + i * C4NUM] = src_ic[0 + i * input_channel];
        pack_ic[1 + i * C4NUM] = src_ic[1 + i * input_channel];
        pack_ic[2 + i * C4NUM] = src_ic[2 + i * input_channel];
        pack_ic[3 + i * C4NUM] = src_ic[3 + i * input_channel];
      }
      src_ic += C4NUM;
      pack_ic += C4NUM * C8NUM;
    }
    for (int ici = ic_4div; ici < input_channel; ici += 1) {
      for (int i = 0; i < C8NUM; i++) {
        tmp_sum_value[i] += src_ic[i * input_channel];
        pack_ic[i * C4NUM] = src_ic[i * input_channel];
      }
      src_ic += 1;
      pack_ic += 1;
    }

    for (int ici = input_channel; ici < ic4; ici += 1) {
      for (int i = 0; i < C8NUM; i++) {
        pack_ic[i * C4NUM] = 0;
      }
      pack_ic += 1;
    }

    for (int oci = 0; oci < oc_8div; oci += C8NUM) {
      for (int ri = 0; ri < C8NUM; ri++) {
        input_sum_oc[ri * C8NUM + 0] = tmp_sum_value[ri] * filter_zp[oci + 0];
        input_sum_oc[ri * C8NUM + 1] = tmp_sum_value[ri] * filter_zp[oci + 1];
        input_sum_oc[ri * C8NUM + 2] = tmp_sum_value[ri] * filter_zp[oci + 2];
        input_sum_oc[ri * C8NUM + 3] = tmp_sum_value[ri] * filter_zp[oci + 3];
        input_sum_oc[ri * C8NUM + 4] = tmp_sum_value[ri] * filter_zp[oci + 4];
        input_sum_oc[ri * C8NUM + 5] = tmp_sum_value[ri] * filter_zp[oci + 5];
        input_sum_oc[ri * C8NUM + 6] = tmp_sum_value[ri] * filter_zp[oci + 6];
        input_sum_oc[ri * C8NUM + 7] = tmp_sum_value[ri] * filter_zp[oci + 7];
      }
      input_sum_oc += inputsum_stride;
    }
    if (oc_8div != output_channel) {
      for (int oci = 0; oci < oc_8res; oci += 1) {
        for (int ri = 0; ri < C8NUM; ri++) {
          input_sum_oc[ri * C8NUM + oci] = tmp_sum_value[ri] * filter_zp[oc_8div + oci];
        }
      }
      for (int oci = oc_8res; oci < C8NUM; oci += 1) {
        for (int ri = 0; ri < C8NUM; ri++) {
          input_sum_oc[ri * C8NUM + oci] = 0;
        }
      }
    } /* oc8 res done */
#endif
    src_r += input_channel * C8NUM;
    pack_r += ic4 * C8NUM;
    input_sum_r += C8NUM * C8NUM;
  }

  if (hw_8div != plane_size) {
    memset(pack_r, 0, C8NUM * ic4);
    for (int hwi = hw_8div; hwi < plane_size; hwi += 1) {
      int32_t *input_sum_oc = input_sum_r;
      int32_t tmp_sum_value = 0;
      const int8_t *src_ic = src_r;
      int8_t *pack_ic = pack_r;
      for (int ici = 0; ici < ic_4div; ici += C4NUM) {
        tmp_sum_value += src_ic[0];
        tmp_sum_value += src_ic[1];
        tmp_sum_value += src_ic[2];
        tmp_sum_value += src_ic[3];
        pack_ic[0] = src_ic[0];
        pack_ic[1] = src_ic[1];
        pack_ic[2] = src_ic[2];
        pack_ic[3] = src_ic[3];
        src_ic += C4NUM;
        pack_ic += C4NUM * C8NUM;
      }
      for (int ici = ic_4div; ici < input_channel; ici += 1) {
        tmp_sum_value += src_ic[0];
        pack_ic[0] = src_ic[0];
        src_ic += 1;
        pack_ic += 1;
      }

      for (int oci = 0; oci < oc_8div; oci += C8NUM) {
        for (int curoi = 0; curoi < C8NUM; curoi++) {
          input_sum_oc[curoi] = tmp_sum_value * filter_zp[oci + curoi];
        }
        input_sum_oc += inputsum_stride;
      }
      if (oc_8div != output_channel) {
        for (int oci = 0; oci < oc_8res; oci += 1) {
          input_sum_oc[oci] = tmp_sum_value * filter_zp[oc_8div + oci];
        }
        for (int oci = oc_8res; oci < C8NUM; oci += 1) {
          input_sum_oc[oci] = 0;
        }
      } /* oc8 res done */

      src_r += input_channel;
      pack_r += C4NUM;
      input_sum_r += C8NUM;
    }

    for (int hwi = plane_size; hwi < hw8; hwi++) {
      for (int oc = 0; oc < oc8; oc++) {
        int oc8div = oc / C8NUM, oc8res = oc % C8NUM;
        input_sum[oc8div * inputsum_stride + hwi * C8NUM + oc8res] = 0;
      }
    }
  }
  return;
}

void Conv1x1PreOptPert(const int8_t *src_input, int8_t *packed_input, int32_t *input_sum, size_t input_channel,
                       size_t plane_size, const ConvParameter *conv_param) {
  int ic4 = UP_ROUND(input_channel, C4NUM);
  size_t hw_8div = plane_size / C8NUM * C8NUM;
  size_t ic_4div = input_channel / C4NUM * C4NUM;
  int32_t filter_zp = conv_param->conv_quant_arg_.filter_quant_args_[0].zp_;

  const int8_t *src_r = src_input;
  int8_t *pack_r = packed_input;
  /* per layer */
  for (int hwi = 0; hwi < hw_8div; hwi += C8NUM) {
    const int8_t *src_ic = src_r;
    int8_t *pack_ic = pack_r;
    int32_t *input_sum_r = input_sum + hwi;
#ifdef ENABLE_ARM64
    size_t src_stride = input_channel;
    size_t ic_4res = input_channel - ic_4div;
    asm volatile(
      "dup v16.4s, wzr \n"
      "dup v17.4s, wzr \n"
      "mov x14, %[input_sum_r] \n"
      "dup v20.4s, %w[filter_zp]  \n"

      "mov x10, %[src_ic] \n"
      "mov x11, %[pack_ic] \n"

      "mov x0, #0 \n"
      "1: \n"
      "cmp x0, %[ic_4div] \n"
      "add x0, x0, #4\n"
      "mov x12, x10 \n"
      "add x10, x10, #4\n"
      "blt 2f \n"
      "cmp %[ic_4res], #0\n"
      "beq 6f \n"
      "cmp %[ic_4res], #1\n"
      "beq 3f \n"
      "cmp %[ic_4res], #2\n"
      "beq 4f \n"
      "cmp %[ic_4res], #3\n"
      "beq 5f \n"

      "2: \n"
      "ld1 {v0.s}[0], [x12], %[src_stride]\n"
      "ld1 {v0.s}[1], [x12], %[src_stride]\n"
      "ld1 {v0.s}[2], [x12], %[src_stride]\n"
      "ld1 {v0.s}[3], [x12], %[src_stride]\n"
      "ld1 {v1.s}[0], [x12], %[src_stride]\n"
      "ld1 {v1.s}[1], [x12], %[src_stride]\n"
      "ld1 {v1.s}[2], [x12], %[src_stride]\n"
      "ld1 {v1.s}[3], [x12], %[src_stride]\n"

      "st1 {v0.16b}, [x11], #16\n"
      "st1 {v1.16b}, [x11], #16\n"

      "saddlp v4.8h, v0.16b \n"
      "saddlp v5.8h, v1.16b \n"

      "saddlp v0.4s, v4.8h \n"
      "saddlp v1.4s, v5.8h \n"

      "add v16.4s, v16.4s, v0.4s \n"
      "add v17.4s, v17.4s, v1.4s \n"
      "b 1b \n"

      "3: \n" /* col res 1 */
      "dup v0.4s, wzr \n"
      "dup v1.4s, wzr \n"

      "ld1 {v0.b}[0],  [x12], %[src_stride]\n"
      "ld1 {v0.b}[4],  [x12], %[src_stride]\n"
      "ld1 {v0.b}[8],  [x12], %[src_stride]\n"
      "ld1 {v0.b}[12], [x12], %[src_stride]\n"
      "ld1 {v1.b}[0],  [x12], %[src_stride]\n"
      "ld1 {v1.b}[4],  [x12], %[src_stride]\n"
      "ld1 {v1.b}[8],  [x12], %[src_stride]\n"
      "ld1 {v1.b}[12], [x12], %[src_stride]\n"

      "st1 {v0.16b}, [x11], #16\n"
      "st1 {v1.16b}, [x11], #16\n"
      "saddlp v4.8h, v0.16b \n"
      "saddlp v5.8h, v1.16b \n"
      "saddlp v0.4s, v4.8h \n"
      "saddlp v1.4s, v5.8h \n"
      "add v16.4s, v16.4s, v0.4s \n"
      "add v17.4s, v17.4s, v1.4s \n"
      "b 6f \n"

      "4: \n" /* col res 2 */
      "dup v0.4s, wzr \n"
      "dup v1.4s, wzr \n"

      "ld1 {v0.h}[0], [x12], %[src_stride]\n"
      "ld1 {v0.h}[2], [x12], %[src_stride]\n"
      "ld1 {v0.h}[4], [x12], %[src_stride]\n"
      "ld1 {v0.h}[6], [x12], %[src_stride]\n"
      "ld1 {v1.h}[0], [x12], %[src_stride]\n"
      "ld1 {v1.h}[2], [x12], %[src_stride]\n"
      "ld1 {v1.h}[4], [x12], %[src_stride]\n"
      "ld1 {v1.h}[6], [x12], %[src_stride]\n"

      "st1 {v0.16b}, [x11], #16\n"
      "st1 {v1.16b}, [x11], #16\n"
      "saddlp v4.8h, v0.16b \n"
      "saddlp v5.8h, v1.16b \n"
      "saddlp v0.4s, v4.8h \n"
      "saddlp v1.4s, v5.8h \n"
      "add v16.4s, v16.4s, v0.4s \n"
      "add v17.4s, v17.4s, v1.4s \n"
      "b 6f \n"

      "5: \n" /* col res 3 */
      "dup v0.4s, wzr \n"
      "dup v1.4s, wzr \n"
      "add x13, x12, #2 \n"

      "ld1 {v0.h}[0], [x12], %[src_stride]\n"
      "ld1 {v0.b}[2], [x13], %[src_stride]\n"
      "ld1 {v0.h}[2], [x12], %[src_stride]\n"
      "ld1 {v0.b}[6], [x13], %[src_stride]\n"
      "ld1 {v0.h}[4], [x12], %[src_stride]\n"
      "ld1 {v0.b}[10], [x13], %[src_stride]\n"
      "ld1 {v0.h}[6], [x12], %[src_stride]\n"
      "ld1 {v0.b}[14], [x13], %[src_stride]\n"
      "ld1 {v1.h}[0], [x12], %[src_stride]\n"
      "ld1 {v1.b}[2], [x13], %[src_stride]\n"
      "ld1 {v1.h}[2], [x12], %[src_stride]\n"
      "ld1 {v1.b}[6], [x13], %[src_stride]\n"
      "ld1 {v1.h}[4], [x12], %[src_stride]\n"
      "ld1 {v1.b}[10], [x13], %[src_stride]\n"
      "ld1 {v1.h}[6], [x12], %[src_stride]\n"
      "ld1 {v1.b}[14], [x13], %[src_stride]\n"

      "st1 {v0.16b}, [x11], #16\n"
      "st1 {v1.16b}, [x11], #16\n"
      "saddlp v4.8h, v0.16b \n"
      "saddlp v5.8h, v1.16b \n"
      "saddlp v0.4s, v4.8h \n"
      "saddlp v1.4s, v5.8h \n"
      "add v16.4s, v16.4s, v0.4s \n"
      "add v17.4s, v17.4s, v1.4s \n"
      "b 6f \n"

      "6: \n"
      "mul v16.4s, v16.4s, v20.4s \n"
      "mul v17.4s, v17.4s, v20.4s \n"

      "st1 {v16.4s}, [x14], #16 \n"
      "st1 {v17.4s}, [x14], #16 \n"

      :
      : [ src_ic ] "r"(src_ic), [ pack_ic ] "r"(pack_ic), [ input_sum_r ] "r"(input_sum_r),
        [ src_stride ] "r"(src_stride), [ ic_4div ] "r"(ic_4div), [ ic_4res ] "r"(ic_4res), [ filter_zp ] "r"(filter_zp)
      : "x0", "x1", "x10", "x11", "x12", "x13", "x14", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17",
        "v20");
#else
    int32_t tmp_sum_value[8] = {0};
    for (int ici = 0; ici < ic_4div; ici += C4NUM) {
      for (int i = 0; i < C8NUM; i++) {
        tmp_sum_value[i] += src_ic[0 + i * input_channel];
        tmp_sum_value[i] += src_ic[1 + i * input_channel];
        tmp_sum_value[i] += src_ic[2 + i * input_channel];
        tmp_sum_value[i] += src_ic[3 + i * input_channel];
        pack_ic[0 + i * C4NUM] = src_ic[0 + i * input_channel];
        pack_ic[1 + i * C4NUM] = src_ic[1 + i * input_channel];
        pack_ic[2 + i * C4NUM] = src_ic[2 + i * input_channel];
        pack_ic[3 + i * C4NUM] = src_ic[3 + i * input_channel];
      }
      src_ic += C4NUM;
      pack_ic += C4NUM * C8NUM;
    }
    for (int ici = ic_4div; ici < input_channel; ici += 1) {
      for (int i = 0; i < C8NUM; i++) {
        tmp_sum_value[i] += src_ic[i * input_channel];
        pack_ic[i * C4NUM] = src_ic[i * input_channel];
      }
      src_ic += 1;
      pack_ic += 1;
    }

    for (int ici = input_channel; ici < ic4; ici += 1) {
      for (int i = 0; i < C8NUM; i++) {
        pack_ic[i * C4NUM] = 0;
      }
      pack_ic += 1;
    }

    for (int i = 0; i < C8NUM; i++) {
      input_sum_r[i] = tmp_sum_value[i] * filter_zp;
    }
#endif
    src_r += input_channel * C8NUM;
    pack_r += ic4 * C8NUM;
  }

  if (hw_8div != plane_size) {
    memset(pack_r, 0, C8NUM * ic4);
    for (int hwi = hw_8div; hwi < plane_size; hwi += 1) {
      int32_t tmp_sum_value = 0;
      const int8_t *src_ic = src_r;
      int8_t *pack_ic = pack_r;
      for (int ici = 0; ici < ic_4div; ici += C4NUM) {
        tmp_sum_value += src_ic[0];
        tmp_sum_value += src_ic[1];
        tmp_sum_value += src_ic[2];
        tmp_sum_value += src_ic[3];
        pack_ic[0] = src_ic[0];
        pack_ic[1] = src_ic[1];
        pack_ic[2] = src_ic[2];
        pack_ic[3] = src_ic[3];
        src_ic += C4NUM;
        pack_ic += C4NUM * C8NUM;
      }
      for (int ici = ic_4div; ici < input_channel; ici += 1) {
        tmp_sum_value += src_ic[0];
        pack_ic[0] = src_ic[0];
        src_ic += 1;
        pack_ic += 1;
      }
      input_sum[hwi] = tmp_sum_value * filter_zp;
      src_r += input_channel;
      pack_r += C4NUM;
    }
    for (int hwi = plane_size; hwi < UP_ROUND(plane_size, C8NUM); hwi++) {
      input_sum[hwi] = 0;
    }
  }
  return;
}

void PackInputSum16x4Int8(const int8_t *input, int32_t *input_sum, const int32_t *filter_zp,
                          const ConvParameter *conv_param) {
  size_t hw = conv_param->output_h_ * conv_param->output_w_;
  size_t hw4 = UP_ROUND(hw, C4NUM);
  size_t ic16 = UP_ROUND(conv_param->input_channel_, C16NUM);
  if (conv_param->conv_quant_arg_.filter_arg_num_ == 1) {
    PackInputSum16x4PerLayer(input, input_sum, conv_param->conv_quant_arg_.filter_quant_args_[0].zp_, hw4, ic16);
  } else {
#ifdef ENABLE_ARM32
    PackInputSum16x4PerChannelArm32(input, input_sum, filter_zp, hw, conv_param->input_channel_,
                                    conv_param->output_channel_);
#else
    PackInputSum16x4PerChannel(input, input_sum, filter_zp, hw, conv_param->input_channel_,
                               conv_param->output_channel_);
#endif
  }
  return;
}

void Im2ColPackUnitInt8Opt(const int8_t *input_data, int8_t *packed_input, int8_t *matmul_input, int real_cal_num,
                           int block_index, const int32_t *filter_zp, int32_t *input_sum,
                           const ConvParameter *conv_param, bool per_channel, bool is_optimize) {
  // input format : nhwc
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int stride_h = conv_param->stride_h_;
  int stride_w = conv_param->stride_w_;
  int pad_h = conv_param->pad_u_;
  int pad_w = conv_param->pad_l_;
  int dilation_h = conv_param->dilation_h_;
  int dilation_w = conv_param->dilation_w_;
  int in_channel = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int out_w = conv_param->output_w_;
  int kernel_plane = kernel_h * kernel_w;
  NNACL_CHECK_ZERO_RETURN(out_w);
  NNACL_CHECK_ZERO_RETURN(dilation_h);
  NNACL_CHECK_ZERO_RETURN(dilation_w);
  for (int i = 0; i < real_cal_num; i++) {
    int block_start = block_index + i;
    int input_h = block_start / out_w * stride_h - pad_h;
    int input_w = block_start % out_w * stride_w - pad_w;
    int input_stride = input_h * in_w * in_channel + input_w * in_channel;
    int kh_s = MSMAX(0, UP_DIV(-input_h, dilation_h));
    int kh_e = MSMIN(kernel_h, UP_DIV(in_h - input_h, dilation_h));
    int kw_s = MSMAX(0, UP_DIV(-input_w, dilation_w));
    int kw_e = MSMIN(kernel_w, UP_DIV(in_w - input_w, dilation_w));
    if (kw_e <= kw_s || kh_e <= kh_s) {
      continue;
    }
    if (dilation_w == 1 && dilation_h == 1) {
      for (int j = kh_s; j < kh_e; j++) {
        int input_y_stride = j * in_w * in_channel + input_stride;
        int input_x_stride = input_y_stride + kw_s * in_channel;
        int input_plane_offset = (j * kernel_w + kw_s) * in_channel + i * in_channel * kernel_plane;
        memcpy(matmul_input + input_plane_offset, input_data + input_x_stride, (kw_e - kw_s) * in_channel);
      }  // kernel_h loop
    } else {
      for (int j = kh_s; j < kh_e; j++) {
        int input_y_stride = j * dilation_h * in_w * in_channel + input_stride;
        for (int k = kw_s; k < kw_e; ++k) {
          int input_x_stride = input_y_stride + k * dilation_w * in_channel;
          int input_plane_offset = (j * kernel_w + k) * in_channel + i * in_channel * kernel_plane;
          memcpy(matmul_input + input_plane_offset, input_data + input_x_stride, in_channel);
        }
      }  // kernel_h loop
    }
  }  // tile num loop
  int deep = kernel_plane * in_channel;
  if (is_optimize) {
    if (per_channel) {
      Conv1x1PreOptPeroc(matmul_input, packed_input, input_sum, deep, conv_param->output_channel_, real_cal_num,
                         filter_zp, C8NUM * C8NUM);
    } else {
      Conv1x1PreOptPert(matmul_input, packed_input, input_sum, deep, real_cal_num, conv_param);
    }
  } else {
    RowMajor2Row16x4MajorInt8(matmul_input, packed_input, real_cal_num, deep);
    if (per_channel) {
#ifdef ENABLE_ARM32
      PackInputSum16x4PerChannelArm32(packed_input, input_sum, filter_zp, real_cal_num, deep,
                                      conv_param->output_channel_);
#else
      PackInputSum16x4PerChannel(packed_input, input_sum, filter_zp, real_cal_num, deep, conv_param->output_channel_);
#endif
    } else {
      size_t hw4 = UP_ROUND(real_cal_num, C4NUM);
      size_t ic16 = UP_ROUND(deep, C16NUM);
      PackInputSum16x4PerLayer(packed_input, input_sum, conv_param->conv_quant_arg_.filter_quant_args_[0].zp_, hw4,
                               ic16);
    }
  }
}

void ConvInt8(int8_t *input_data, int8_t *packed_input, int8_t *matmul_input, int8_t *packed_weight,
              const int32_t *bias_data, int8_t *output_data, int32_t *filter_zp, int32_t *input_sum, int task_id,
              ConvParameter *conv_param, MATMUL_OPT_R_FUNC matmul_func, bool is_optimize) {
  int in_channel = conv_param->input_channel_;
  int out_channel = conv_param->output_channel_;
  int tile_n = conv_param->tile_num_;
  int output_count = conv_param->output_h_ * conv_param->output_w_;
  NNACL_CHECK_ZERO_RETURN(tile_n);
  int output_tile_count = UP_DIV(output_count, tile_n);
  int kernel_plane = conv_param->kernel_h_ * conv_param->kernel_w_;
  int unit_size;
  int input_sum_offset;
  int up_round_oc;
#ifdef ENABLE_ARM32
  up_round_oc = UP_ROUND(out_channel, C2NUM);
  unit_size = UP_ROUND(kernel_plane * in_channel, C16NUM);
#else
  if (is_optimize) {
    up_round_oc = UP_ROUND(out_channel, C8NUM);
    unit_size = UP_ROUND(kernel_plane * in_channel, C4NUM);
  } else {
    up_round_oc = UP_ROUND(out_channel, C4NUM);
    unit_size = UP_ROUND(kernel_plane * in_channel, C16NUM);
  }
#endif
  bool per_channel = false;
  if (conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL) {
    input_sum_offset = tile_n * up_round_oc;
    per_channel = true;
  } else {
    input_sum_offset = tile_n;
    per_channel = false;
  }

  for (int b = 0; b < conv_param->input_batch_; b++) {
    int in_batch_offset = b * in_channel * conv_param->input_h_ * conv_param->input_w_;
    int out_batch_offset = b * out_channel * conv_param->output_h_ * conv_param->output_w_;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += conv_param->thread_num_) {
      int start_index = thread_id * tile_n;
      int real_cal_num = (output_count - start_index) < tile_n ? (output_count - start_index) : tile_n;
      int32_t *tmp_input_sum = input_sum + task_id * input_sum_offset;
      int8_t *gemm_input = packed_input + task_id * unit_size * tile_n;
      int8_t *matmul = matmul_input + task_id * kernel_plane * in_channel * tile_n;
      memset(matmul, conv_param->conv_quant_arg_.input_quant_args_[0].zp_, kernel_plane * in_channel * tile_n);
      Im2ColPackUnitInt8Opt(input_data + in_batch_offset, gemm_input, matmul, real_cal_num, start_index, filter_zp,
                            tmp_input_sum, conv_param, per_channel, is_optimize);

      int out_offset = thread_id * tile_n * out_channel + out_batch_offset;
      int8_t *gemm_output = output_data + out_offset;
#ifdef ENABLE_ARM32
      MatmulInt8Neon32(
        gemm_input, packed_weight, gemm_output, real_cal_num, out_channel, unit_size, tmp_input_sum, bias_data,
        conv_param->conv_quant_arg_.out_act_min_[0], conv_param->conv_quant_arg_.out_act_max_[0],
        conv_param->conv_quant_arg_.output_quant_args_[0].zp_, conv_param->conv_quant_arg_.quant_multiplier_,
        conv_param->conv_quant_arg_.left_shift_, conv_param->conv_quant_arg_.right_shift_, out_channel, per_channel);
#elif ENABLE_ARM64
      if (is_optimize) {
        matmul_func(gemm_input, packed_weight, gemm_output, real_cal_num, out_channel, unit_size, out_channel,
                    tmp_input_sum, bias_data, conv_param->conv_quant_arg_.left_shift_,
                    conv_param->conv_quant_arg_.right_shift_, conv_param->conv_quant_arg_.quant_multiplier_,
                    conv_param->conv_quant_arg_.output_quant_args_[0].zp_, conv_param->conv_quant_arg_.out_act_min_[0],
                    conv_param->conv_quant_arg_.out_act_max_[0], per_channel);
      } else {
        MatmulInt8Neon64(gemm_input, packed_weight, gemm_output, UP_ROUND(real_cal_num, C4NUM),
                         UP_ROUND(out_channel, C4NUM), unit_size, tmp_input_sum, bias_data,
                         conv_param->conv_quant_arg_.out_act_min_[0], conv_param->conv_quant_arg_.out_act_max_[0],
                         conv_param->conv_quant_arg_.output_quant_args_[0].zp_,
                         conv_param->conv_quant_arg_.quant_multiplier_, conv_param->conv_quant_arg_.left_shift_,
                         conv_param->conv_quant_arg_.right_shift_, real_cal_num, out_channel, out_channel, per_channel);
      }
#else
      MatMulInt8_8x8_r(
        gemm_input, packed_weight, gemm_output, real_cal_num, out_channel, unit_size, out_channel, tmp_input_sum,
        bias_data, conv_param->conv_quant_arg_.left_shift_, conv_param->conv_quant_arg_.right_shift_,
        conv_param->conv_quant_arg_.quant_multiplier_, conv_param->conv_quant_arg_.output_quant_args_[0].zp_,
        conv_param->conv_quant_arg_.out_act_min_[0], conv_param->conv_quant_arg_.out_act_max_[0], per_channel);
#endif
    }
  }
}
