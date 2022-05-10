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

void PackInputToC8Int8(const int8_t *input_data, int16_t *packed_input, const ConvParameter *conv_param) {
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

void PackWeightToC8Int8(const int8_t *origin_weight_data, int16_t *packed_weight_data,
                        const ConvParameter *conv_param) {
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

void PackInputSum16x4PerLayer(const int8_t *src, int32_t *dst, int32_t filter_zp, size_t row4, size_t col16) {
  /* normal matmul : 4x16 * 16x4 -> 4x4  */
#ifdef ENABLE_ARM
  PreSum4x16Int8Pert(src, dst, row4, col16, filter_zp);
#else
  for (size_t r = 0; r < row4; r++) {
    int32_t tmp_value = 0;
    for (size_t c = 0; c < col16; c++) {
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
                             const ConvQuantArg *quant_qrg) {
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
                                   const ConvQuantArg *quant_qrg) {
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
