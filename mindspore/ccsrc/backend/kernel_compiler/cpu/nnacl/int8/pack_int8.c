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

#include "nnacl/int8/pack_int8.h"

#ifdef ENABLE_ARM32
void PackInputSum16x4PerChannelArm32(const int8_t *input_value, int32_t *input_sum, int32_t *filter_zp_ptr,
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

void PackInputSum16x4PerChannel(const int8_t *input_value, int32_t *input_sum, int32_t *filter_zp_ptr,
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
                        size_t output_channel, size_t plane_size, int32_t *filter_zp, size_t inputsum_stride) {
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
                       size_t plane_size, ConvParameter *conv_param) {
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

void Im2ColPackUnitInt8Opt(const int8_t *input_data, int8_t *packed_input, int8_t *matmul_input, int real_cal_num,
                           int block_index, int32_t *filter_zp, int32_t *input_sum, ConvParameter *conv_param,
                           bool per_channel, bool is_optimize) {
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

  for (int i = 0; i < real_cal_num; i++) {
    int block_start = block_index + i;
    int input_h = block_start / out_w * stride_h - pad_h;
    int input_w = block_start % out_w * stride_w - pad_w;
    int input_stride = input_h * in_w * in_channel + input_w * in_channel;
    int kh_s = MSMAX(0, UP_DIV(-input_h, dilation_h));
    int kh_e = MSMIN(kernel_h, UP_DIV(in_h - input_h, dilation_h));
    int kw_s = MSMAX(0, UP_DIV(-input_w, dilation_w));
    int kw_e = MSMIN(kernel_w, UP_DIV(in_w - input_w, dilation_w));
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

void PackInputToC8Int8(const int8_t *input_data, int16_t *packed_input, ConvParameter *conv_param) {
  int in_batch = conv_param->input_batch_;
  int in_channel = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int ic8_round = UP_ROUND(in_channel, C8NUM);
  int ic8 = in_channel / C8NUM * C8NUM;
  int in_plane = in_h * in_w;

  for (int b = 0; b < in_batch; b++) {
    int src_batch_offset = b * in_channel * in_plane;
    int dst_batch_offset = b * ic8_round * in_plane;
    for (int k = 0; k < in_plane; k++) {
      int src_plane_offset = src_batch_offset + k * in_channel;
      int dst_plane_offset = dst_batch_offset + k * C8NUM;
      for (int i = 0; i < ic8; i += 8) {
        int src_c_offset = src_plane_offset + i;
        int dst_c_offset = dst_plane_offset + i * in_plane;
#ifdef ENABLE_ARM
        vst1q_s16(packed_input + dst_c_offset, vmovl_s8(vld1_s8(input_data + src_c_offset)));
#else
        for (int j = 0; j < C8NUM; ++j) {
          (packed_input + dst_c_offset)[j] = (int16_t)(input_data + src_c_offset)[j];
        }
#endif
      }  // ic8_minus loop
      int res_c = in_channel - ic8;
      int tmp_ic_offset = ic8 * in_plane;
      for (int l = 0; l < res_c; ++l) {
        int src_c_offset = src_plane_offset + ic8 + l;
        int dst_c_offset = dst_plane_offset + tmp_ic_offset + l;
        (packed_input + dst_c_offset)[0] = (int16_t)(input_data + src_c_offset)[0];
      }  // res ic loop
      int res2 = ic8_round - in_channel;
      for (int l = 0; l < res2; ++l) {
        int dst_c_offset = dst_plane_offset + tmp_ic_offset + res_c + l;
        (packed_input + dst_c_offset)[0] = 0;
      }  // res ic loop
    }    // kh * kw loop
  }
}

void PackWeightToC8Int8(const int8_t *origin_weight_data, int16_t *packed_weight_data, ConvParameter *conv_param) {
  // origin weight format : ohwi
  int input_channel = conv_param->input_channel_;
  int ic8 = input_channel / C8NUM * C8NUM;
  int ic8_round = UP_ROUND(input_channel, C8NUM);
  int output_channel = conv_param->output_channel_;
  QuantArg *filter_zp = conv_param->conv_quant_arg_.filter_quant_args_;
  int kernel_plane = conv_param->kernel_h_ * conv_param->kernel_w_;

  for (int k = 0; k < kernel_plane; k++) {
    int src_kernel_offset = k * input_channel;
    int dst_kernel_offset = k * C8NUM;
    for (int o = 0; o < output_channel; o++) {
      int32_t zp;
      if (conv_param->conv_quant_arg_.filter_arg_num_ == 1) {
        zp = filter_zp[0].zp_;
      } else {
        zp = filter_zp[o].zp_;
      }
      int src_oc_offset = src_kernel_offset + o * kernel_plane * input_channel;
      int dst_oc_offset = dst_kernel_offset + o * ic8_round * kernel_plane;
      int i = 0;
      for (; i < ic8; i += C8NUM) {
        int src_ic_offset = src_oc_offset + i;
        int dst_ic_offset = dst_oc_offset + i * kernel_plane;
#ifdef ENABLE_ARM64
        int8x8_t src_s8 = vld1_s8(origin_weight_data + src_ic_offset);
        int16x8_t src_s16 = vmovl_s8(src_s8);
        int16x4_t src1_s16 = vget_low_s16(src_s16);
        int16x4_t src2_s16 = vget_high_s16(src_s16);
        int32x4_t src1_s32 = vmovl_s16(src1_s16);
        int32x4_t src2_s32 = vmovl_s16(src2_s16);
        int32x4_t zp_s32 = vdupq_n_s32(zp);
        int32x4_t dst1_s32 = vsubq_s32(src1_s32, zp_s32);
        int32x4_t dst2_s32 = vsubq_s32(src2_s32, zp_s32);
        int16x4_t dst1_s16 = vqmovn_s32(dst1_s32);
        int16x4_t dst2_s16 = vqmovn_s32(dst2_s32);
        vst1_s16(packed_weight_data + dst_ic_offset, dst1_s16);
        vst1_s16(packed_weight_data + dst_ic_offset + 4, dst2_s16);
#else
        for (int ci = 0; ci < C8NUM; ++ci) {
          (packed_weight_data + dst_ic_offset + ci)[0] = (int16_t)((origin_weight_data + src_ic_offset + ci)[0] - zp);
        }
#endif
      }
      dst_oc_offset += ic8 * kernel_plane;
      for (; i < input_channel; i++) {
        int c8_block_rem = i % C8NUM;
        int src_ic_offset = src_oc_offset + i;
        int dst_ic_offset = dst_oc_offset + c8_block_rem;
        (packed_weight_data + dst_ic_offset)[0] = (int16_t)((origin_weight_data + src_ic_offset)[0] - zp);
      }
    }
  }
}

void PackInputSum16x4Int8(const int8_t *input, int32_t *input_sum, int32_t *filter_zp, ConvParameter *conv_param) {
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

void PackInputSum16x4PerLayer(const int8_t *src, int32_t *dst, int32_t filter_zp, size_t row4, size_t col16) {
  /* normal matmul : 4x16 * 16x4 -> 4x4  */
#ifdef ENABLE_ARM
  PreSum4x16Int8Pert(src, dst, row4, col16, filter_zp);
#else
  for (int r = 0; r < row4; r++) {
    int32_t tmp_value = 0;
    for (int c = 0; c < col16; c++) {
      int r4div = r / C4NUM, r4mod = r % C4NUM, c16div = c / C16NUM, c16mod = c % C16NUM;
      int src_index = r4div * C4NUM * col16 + c16div * C16NUM * C4NUM + r4mod * C16NUM + c16mod;
      tmp_value += src[src_index];
    }
    dst[r] = tmp_value * filter_zp;
  }
#endif
  return;
}
void PackDepthwiseInt8Input(const int8_t *src, int16_t *dst, const ConvParameter *conv_param) {
  int input_zp = conv_param->conv_quant_arg_.input_quant_args_[0].zp_;
  int ic4 = UP_DIV(conv_param->input_channel_, C4NUM);
  int unit = conv_param->input_h_ * conv_param->input_w_;

  for (int b = 0; b < conv_param->input_batch_; b++) {
    const int8_t *src_b = src + b * unit * conv_param->input_channel_;
    int16_t *dst_b = dst + b * unit * ic4 * C4NUM;
    for (int k = 0; k < unit; k++) {
      const int8_t *src_k = src_b + k * conv_param->input_channel_;
      int16_t *dst_k = dst_b + k * ic4 * C4NUM;
      for (int c = 0; c < conv_param->input_channel_; c++) {
        dst_k[c] = (int16_t)(src_k[c] - input_zp);
      }
    }
  }
}

void PackDepthwiseInt8Weight(const int8_t *origin_weight, int16_t *packed_weight_, int plane, int channel,
                             ConvQuantArg *quant_qrg) {
  int weight_zp = quant_qrg->filter_quant_args_[0].zp_;
  for (int c = 0; c < channel; c++) {
    if (quant_qrg->per_channel_ & FILTER_PER_CHANNEL) {
      weight_zp = quant_qrg->filter_quant_args_[c].zp_;
    }
    int c8_block_num = c / C8NUM;
    int c8_block_rem = c % C8NUM;
    const int8_t *src_c = origin_weight + c * plane;
    int16_t *dst_c = packed_weight_ + c8_block_num * plane * C8NUM;
    for (int k = 0; k < plane; k++) {
      const int8_t *src_kernel = src_c + k;
      int16_t *dst_kernel = dst_c + C8NUM * k + c8_block_rem;
      *dst_kernel = (int16_t)(src_kernel[0] - weight_zp);
    }
  }
}

void PackDeconvDepthwiseInt8Weight(const int8_t *origin_weight, int16_t *packed_weight_, int plane, int channel,
                                   ConvQuantArg *quant_qrg) {
  int weight_zp = quant_qrg->filter_quant_args_[0].zp_;
  for (int c = 0; c < channel; c++) {
    if (quant_qrg->per_channel_ & FILTER_PER_CHANNEL) {
      weight_zp = quant_qrg->filter_quant_args_[c].zp_;
    }
    int c4_block_num = c / C4NUM;
    int c4_block_rem = c % C4NUM;
    const int8_t *src_c = origin_weight + c * plane;
    int16_t *dst_c = packed_weight_ + c4_block_num * plane * C4NUM;
    for (int k = 0; k < plane; k++) {
      const int8_t *src_kernel = src_c + k;
      int16_t *dst_kernel = dst_c + C4NUM * k + c4_block_rem;
      *dst_kernel = (int16_t)(src_kernel[0] - weight_zp);
    }
  }
}
void PackNHWCToNHWC4Int8(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  int c4_channel = c4 * C4NUM;
  int nhwc4_batch_unit_offset = c4 * C4NUM * plane;
  int ic_remainder_ = channel % C4NUM;
  if (ic_remainder_ != 0) {
    int nhwc4_batch_offset = 0;
    for (int b = 0; b < batch; b++) {
      int batch_offset = b * channel * plane;
      for (int i = 0; i < plane; i++) {
        int8_t *dst_per_plane = (int8_t *)dst + nhwc4_batch_offset + i * c4_channel;
        memcpy(dst_per_plane, (int8_t *)src + batch_offset + i * channel, channel);
        for (int j = channel; j < c4_channel; ++j) {
          dst_per_plane[j] = 0;
        }
      }
      nhwc4_batch_offset += nhwc4_batch_unit_offset;
    }
  } else {
    size_t ori_input_size = batch * plane * channel;
    memcpy((int8_t *)dst, (int8_t *)src, ori_input_size);
  }
}

void PackNHWC4ToNHWCInt8(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  int nhwc4_batch_unit_offset = c4 * C4NUM * plane;
  int ic_remainder_ = channel % C4NUM;
  if (ic_remainder_ != 0) {
    for (int b = 0; b < batch; b++) {
      int batch_offset = b * channel * plane;
      int nhwc4_batch_offset = b * nhwc4_batch_unit_offset;
      for (int i = 0; i < plane; i++) {
        memcpy((int8_t *)dst + batch_offset + i * channel, (int8_t *)src + nhwc4_batch_offset + i * c4 * C4NUM,
               channel);
      }
    }
  } else {
    size_t ori_input_size = batch * plane * channel;
    memcpy((int8_t *)dst, (int8_t *)src, ori_input_size);
  }
}

void PackNHWCToNHWC8Int8(const void *src, void *dst, int batch, int plane, int channel) {
  int c8 = UP_DIV(channel, C8NUM);
  int nhwc8_batch_unit_offset = c8 * C8NUM * plane;
  int ic_remainder_ = channel % C8NUM;
  if (ic_remainder_ != 0) {
    int nhwc8_batch_offset = 0;
    for (int b = 0; b < batch; b++) {
      int batch_offset = b * channel * plane;
      for (int i = 0; i < plane; i++) {
        memcpy((int8_t *)dst + nhwc8_batch_offset + i * c8 * C8NUM, (int8_t *)src + batch_offset + i * channel,
               channel);
      }
      nhwc8_batch_offset += nhwc8_batch_unit_offset;
    }
  } else {
    size_t ori_input_size = batch * plane * channel;
    memcpy((int8_t *)dst, (int8_t *)src, ori_input_size);
  }
}

void PackNHWC8ToNHWCInt8(const void *src, void *dst, int batch, int plane, int channel) {
  int c8 = UP_DIV(channel, C8NUM);
  int nhwc8_batch_unit_offset = c8 * C8NUM * plane;
  int ic_remainder_ = channel % C8NUM;
  if (ic_remainder_ != 0) {
    for (int b = 0; b < batch; b++) {
      int batch_offset = b * channel * plane;
      int nhwc8_batch_offset = b * nhwc8_batch_unit_offset;
      for (int i = 0; i < plane; i++) {
        memcpy((int8_t *)dst + batch_offset + i * channel, (int8_t *)src + nhwc8_batch_offset + i * c8 * C8NUM,
               channel);
      }
    }
  } else {
    size_t ori_input_size = batch * plane * channel;
    memcpy((int8_t *)dst, (int8_t *)src, ori_input_size);
  }
}

void PackNCHWToNC8HW8Int8(const void *src, void *dst, int batch, int plane, int channel) {
  int c8 = UP_DIV(channel, C8NUM);
  for (int b = 0; b < batch; b++) {
    int src_offset = b * plane * channel;
    int dst_offset = b * plane * c8 * C8NUM;
    for (int c = 0; c < channel; c++) {
      int c8_block_num = c / C8NUM;
      int c8_block_rem = c % C8NUM;
      int src_c_offset = src_offset + c * plane;
      int dst_c_offset = dst_offset + c8_block_num * plane * C8NUM;
      for (int k = 0; k < plane; k++) {
        int src_kernel_offset = src_c_offset + k;
        int dst_kernel_offset = dst_c_offset + C8NUM * k + c8_block_rem;
        ((int8_t *)dst + dst_kernel_offset)[0] = ((int8_t *)src + src_kernel_offset)[0];
      }
    }
  }
}

void PackNC4HW4ToNHWCInt8(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  for (int b = 0; b < batch; b++) {
    int src_offset = b * plane * c4 * C4NUM;
    int dst_offset = b * plane * channel;
    for (int k = 0; k < plane; k++) {
      int src_kernel_offset = src_offset + k * C4NUM;
      int dst_kernel_offset = dst_offset + k * channel;
      for (int c = 0; c < c4 - 1; c++) {
        int src_c_offset = src_kernel_offset + c * plane * C4NUM;
        int dst_c_offset = dst_kernel_offset + c * C4NUM;
        ((int8_t *)dst + dst_c_offset)[0] = ((int8_t *)src + src_c_offset)[0];
        ((int8_t *)dst + dst_c_offset)[1] = ((int8_t *)src + src_c_offset)[1];
        ((int8_t *)dst + dst_c_offset)[2] = ((int8_t *)src + src_c_offset)[2];
        ((int8_t *)dst + dst_c_offset)[3] = ((int8_t *)src + src_c_offset)[3];
      }
      // res part
      int res_c = channel - (c4 - 1) * C4NUM;
      for (int i = 0; i < res_c; i++) {
        int src_res_c_offset = src_kernel_offset + (c4 - 1) * C4NUM * plane + i;
        int dst_res_c_offset = dst_kernel_offset + (c4 - 1) * C4NUM + i;
        ((int8_t *)dst + dst_res_c_offset)[0] = ((int8_t *)src + src_res_c_offset)[0];
      }
    }
  }
}

void PackNHWCToC8HWN8Int8(const void *src, void *dst, int batch, int plane, int channel) {
  for (int n = 0; n < batch; n++) {
    for (int hw = 0; hw < plane; hw++) {
      for (int c = 0; c < channel; c++) {
        int c8div = c / C8NUM;
        int c8mod = c % C8NUM;
        int src_index = n * plane * channel + hw * channel + c;
        int dst_index = c8div * batch * plane * C8NUM + hw * batch * C8NUM + n * C8NUM + c8mod;
        ((int8_t *)dst)[dst_index] = ((int8_t *)src)[src_index];
      }
    }
  }
  return;
}

void PackNCHWToNHWCInt8(const void *src, void *dst, int batch, int plane, int channel) {
  for (int n = 0; n < batch; n++) {
    for (int c = 0; c < channel; c++) {
      for (int hw = 0; hw < plane; hw++) {
        int nhwc_index = n * channel * plane + hw * channel + c;
        int nchw_index = n * channel * plane + c * plane + hw;
        ((int8_t *)(dst))[nhwc_index] = ((const int8_t *)(src))[nchw_index];
      }
    }
  }
  return;
}

void PackNHWCToNCHWInt8(const void *src, void *dst, int batches, int plane, int channel) {
  int hw8 = plane / C8NUM * C8NUM;
  int c8 = channel / C8NUM * C8NUM;
  int batch = plane * channel;
  for (int n = 0; n < batches; n++) {
    const int8_t *src_batch = (const int8_t *)src + n * batch;
    int8_t *dst_batch = (int8_t *)dst + n * batch;
    int hw = 0;
    for (; hw < hw8; hw += C8NUM) {
      int c = 0;
      for (; c < c8; c += C8NUM) {
        const int8_t *src_ptr = src_batch + hw * channel + c;
        int8_t *dst_ptr = dst_batch + c * plane + hw;
#ifdef ENABLE_ARM64
        size_t srcStride = channel * sizeof(int8_t);
        size_t dstStride = plane * sizeof(int8_t);
        asm volatile(
          "mov x10, %[src_ptr]\n"
          "mov x11, %[dst_ptr]\n"

          "ld1 {v0.8b}, [x10], %[srcStride]\n"
          "ld1 {v1.8b}, [x10], %[srcStride]\n"
          "ld1 {v2.8b}, [x10], %[srcStride]\n"
          "ld1 {v3.8b}, [x10], %[srcStride]\n"

          "trn1 v4.8b, v0.8b, v1.8b\n"
          "trn2 v5.8b, v0.8b, v1.8b\n"
          "trn1 v6.8b, v2.8b, v3.8b\n"
          "trn2 v7.8b, v2.8b, v3.8b\n"

          "ld1 {v0.8b}, [x10], %[srcStride]\n"
          "ld1 {v1.8b}, [x10], %[srcStride]\n"
          "ld1 {v2.8b}, [x10], %[srcStride]\n"
          "ld1 {v3.8b}, [x10], %[srcStride]\n"

          "trn1 v8.4h, v4.4h, v6.4h\n"
          "trn2 v9.4h, v4.4h, v6.4h\n"
          "trn1 v10.4h, v5.4h, v7.4h\n"
          "trn2 v11.4h, v5.4h, v7.4h\n"

          "trn1 v4.8b, v0.8b, v1.8b\n"
          "trn2 v5.8b, v0.8b, v1.8b\n"
          "trn1 v6.8b, v2.8b, v3.8b\n"
          "trn2 v7.8b, v2.8b, v3.8b\n"

          "trn1 v12.4h, v4.4h, v6.4h\n"
          "trn2 v13.4h, v4.4h, v6.4h\n"
          "trn1 v14.4h, v5.4h, v7.4h\n"
          "trn2 v15.4h, v5.4h, v7.4h\n"

          "trn1 v0.2s, v8.2s, v12.2s\n"
          "trn2 v4.2s, v8.2s, v12.2s\n"
          "trn1 v1.2s, v10.2s, v14.2s\n"
          "trn2 v5.2s, v10.2s, v14.2s\n"
          "trn1 v2.2s, v9.2s, v13.2s\n"
          "trn2 v6.2s, v9.2s, v13.2s\n"
          "trn1 v3.2s, v11.2s, v15.2s\n"
          "trn2 v7.2s, v11.2s, v15.2s\n"

          "st1 {v0.8b}, [x11], %[dstStride]\n"
          "st1 {v1.8b}, [x11], %[dstStride]\n"
          "st1 {v2.8b}, [x11], %[dstStride]\n"
          "st1 {v3.8b}, [x11], %[dstStride]\n"
          "st1 {v4.8b}, [x11], %[dstStride]\n"
          "st1 {v5.8b}, [x11], %[dstStride]\n"
          "st1 {v6.8b}, [x11], %[dstStride]\n"
          "st1 {v7.8b}, [x11], %[dstStride]\n"
          :
          :
          [ dst_ptr ] "r"(dst_ptr), [ src_ptr ] "r"(src_ptr), [ srcStride ] "r"(srcStride), [ dstStride ] "r"(dstStride)
          : "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
            "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
            "v30", "v31");
#elif ENABLE_ARM32
        size_t srcStride = channel * sizeof(int8_t);
        size_t dstStride = plane * sizeof(int8_t);
        asm volatile(
          "mov r10, %[src_ptr]\n"
          "mov r12, %[dst_ptr]\n"

          "vld1.8 {d0}, [r10], %[srcStride]\n"
          "vld1.8 {d1}, [r10], %[srcStride]\n"
          "vld1.8 {d2}, [r10], %[srcStride]\n"
          "vld1.8 {d3}, [r10], %[srcStride]\n"
          "vld1.8 {d4}, [r10], %[srcStride]\n"
          "vld1.8 {d5}, [r10], %[srcStride]\n"
          "vld1.8 {d6}, [r10], %[srcStride]\n"
          "vld1.8 {d7}, [r10], %[srcStride]\n"

          "vtrn.8 d0, d1\n"
          "vtrn.8 d2, d3\n"
          "vtrn.8 d4, d5\n"
          "vtrn.8 d6, d7\n"

          "vtrn.16 d0, d2\n"
          "vtrn.16 d1, d3\n"
          "vtrn.16 d4, d6\n"
          "vtrn.16 d5, d7\n"

          "vtrn.32 d0, d4\n"
          "vtrn.32 d1, d5\n"
          "vtrn.32 d2, d6\n"
          "vtrn.32 d3, d7\n"

          "vst1.8 {d0}, [r12], %[dstStride]\n"
          "vst1.8 {d1}, [r12], %[dstStride]\n"
          "vst1.8 {d2}, [r12], %[dstStride]\n"
          "vst1.8 {d3}, [r12], %[dstStride]\n"
          "vst1.8 {d4}, [r12], %[dstStride]\n"
          "vst1.8 {d5}, [r12], %[dstStride]\n"
          "vst1.8 {d6}, [r12], %[dstStride]\n"
          "vst1.8 {d7}, [r12], %[dstStride]\n"
          :
          :
          [ dst_ptr ] "r"(dst_ptr), [ src_ptr ] "r"(src_ptr), [ srcStride ] "r"(srcStride), [ dstStride ] "r"(dstStride)
          : "r10", "r12", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14",
            "q15");
#else
        for (int tr = 0; tr < C8NUM; tr++) {
          for (int tc = 0; tc < C8NUM; tc++) {
            dst_ptr[tc * plane + tr] = src_ptr[tr * channel + tc];
          }
        }
#endif
      }
      for (; c < channel; c++) {
        const int8_t *src_ptr = src_batch + hw * channel + c;
        int8_t *dst_ptr = dst_batch + c * plane + hw;
        for (size_t i = 0; i < C8NUM; i++) {
          dst_ptr[i] = src_ptr[i * channel];
        }
      }
    }
    for (; hw < plane; hw++) {
      const int8_t *src_ptr = src_batch + hw * channel;
      int8_t *dst_ptr = dst_batch + hw;
      for (size_t i = 0; i < channel; i++) {
        dst_ptr[i * plane] = src_ptr[i];
      }
    }
  }
  return;
}
