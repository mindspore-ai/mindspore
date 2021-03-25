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

#include "nnacl/fp32/pack_fp32.h"

void PackWeightKHWToHWKFp32(const void *src, void *dst, int plane, int channel) {
  return PackNCHWToNHWCFp32(src, dst, 1, plane, channel, 0, 0);
}

void PackHWCToWHC(const float *src, float *dst, int height, int width, int channel) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      memcpy(dst + (j * height + i) * channel, src + (i * width + j) * channel, channel * sizeof(float));
    }
  }
}

void Im2ColPackUnitFp32(const float *input_data, const ConvParameter *conv_param, float *packed_input, int real_cal_num,
                        int block_index) {
  // input format : nhwc
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int kernel_plane = kernel_h * kernel_w;
  int dilation_h = conv_param->dilation_h_;
  int dilation_w = conv_param->dilation_w_;
  int in_channel = conv_param->input_channel_;
  int in_w = conv_param->input_w_;
  int out_w = conv_param->output_w_;

  for (int i = 0; i < real_cal_num; i++) {
    int block_start = block_index + i;
    int input_h = block_start / out_w * conv_param->stride_h_ - conv_param->pad_u_;
    int input_w = block_start % out_w * conv_param->stride_w_ - conv_param->pad_l_;
    int input_stride = (input_h * in_w + input_w) * in_channel;
    int kh_s = MSMAX(0, UP_DIV(-input_h, dilation_h));
    int kh_e = MSMIN(kernel_h, UP_DIV(conv_param->input_h_ - input_h, dilation_h));
    int kw_s = MSMAX(0, UP_DIV(-input_w, dilation_w));
    int kw_e = MSMIN(kernel_w, UP_DIV(in_w - input_w, dilation_w));
    if (dilation_w == 1 && dilation_h == 1) {
      for (int j = kh_s; j < kh_e; j++) {
        int input_y_stride = j * in_w * in_channel + input_stride;
        int input_x_stride = input_y_stride + kw_s * in_channel;
        int input_plane_offset = (j * kernel_w + kw_s) * in_channel + i * in_channel * kernel_plane;
        memcpy(packed_input + input_plane_offset, input_data + input_x_stride,
               (kw_e - kw_s) * in_channel * sizeof(float));
      }  // kernel_h loop
    } else {
      for (int j = kh_s; j < kh_e; j++) {
        int input_y_stride = j * dilation_h * in_w * in_channel + input_stride;
        for (int k = kw_s; k < kw_e; ++k) {
          int input_x_stride = input_y_stride + k * dilation_w * in_channel;
          int input_plane_offset = (j * kernel_w + k) * in_channel + i * in_channel * kernel_plane;
          memcpy(packed_input + input_plane_offset, input_data + input_x_stride, in_channel * sizeof(float));
        }
      }  // kernel_h loop
    }
  }  // tile num loop
}

void PackNHWCToNC4HW4Fp32(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  int c4_minus = c4 - 1;
  for (int b = 0; b < batch; b++) {
    int src_oc_offset = b * plane * channel;
    int dst_oc_offset = b * plane * c4 * C4NUM;
    for (int k = 0; k < plane; k++) {
      int src_kernel_offset = src_oc_offset + k * channel;
      int dst_kernel_offset = dst_oc_offset + k * C4NUM;
      for (int j = 0; j < c4_minus; ++j) {
        int src_ic_offset = src_kernel_offset + j * C4NUM;
        int dst_ic_offset = dst_kernel_offset + j * plane * C4NUM;
#ifdef ENABLE_ARM
        vst1q_f32((float *)dst + dst_ic_offset, vld1q_f32((float *)src + src_ic_offset));
#else
        for (int i = 0; i < C4NUM; ++i) {
          ((float *)dst + dst_ic_offset)[i] = ((float *)src + src_ic_offset)[i];
        }
#endif
      }
      int tmp_c = c4_minus * C4NUM;
      int tmp_c_offset = tmp_c * plane;
      int res_c = channel - tmp_c;
      for (int l = 0; l < res_c; ++l) {
        int src_ic_offset = src_kernel_offset + tmp_c + l;
        int dst_ic_offset = dst_kernel_offset + tmp_c_offset + l;
        ((float *)dst + dst_ic_offset)[0] = ((float *)src + src_ic_offset)[0];
      }
    }
  }
}

void PackNCHWToNC4HW4Fp32(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  for (int b = 0; b < batch; b++) {
    int src_offset = b * plane * channel;
    int dst_offset = b * plane * c4 * C4NUM;
    for (int c = 0; c < channel; c++) {
      int c4_block_num = c / C4NUM;
      int c4_block_rem = c % C4NUM;
      int src_c_offset = src_offset + c * plane;
      int dst_c_offset = dst_offset + c4_block_num * plane * C4NUM;
      for (int k = 0; k < plane; k++) {
        int src_kernel_offset = src_c_offset + k;
        int dst_kernel_offset = dst_c_offset + C4NUM * k + c4_block_rem;
        ((float *)dst + dst_kernel_offset)[0] = ((float *)src + src_kernel_offset)[0];
      }
    }
  }
}

void PackNHWCToNHWC4Fp32(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  int c4_channel = c4 * C4NUM;
  int nhwc4_batch_unit_offset = c4 * C4NUM * plane;
  int ic_remainder_ = channel % C4NUM;
  if (ic_remainder_ != 0) {
    int nhwc4_batch_offset = 0;
    for (int b = 0; b < batch; b++) {
      int batch_offset = b * channel * plane;
      for (int i = 0; i < plane; i++) {
        float *dst_per_plane = (float *)dst + nhwc4_batch_offset + i * c4_channel;
        memcpy(dst_per_plane, (float *)src + batch_offset + i * channel, channel * sizeof(float));
        for (int j = channel; j < c4_channel; ++j) {
          dst_per_plane[j] = 0;
        }
      }
      nhwc4_batch_offset += nhwc4_batch_unit_offset;
    }
  } else {
    size_t ori_input_size = batch * plane * channel * sizeof(float);
    memcpy((float *)dst, (float *)src, ori_input_size);
  }
}

void PackNHWCToNHWC8Fp32(const void *src, void *dst, int batch, int plane, int channel) {
  int c8 = UP_DIV(channel, C8NUM);
  int c8_channel = c8 * C8NUM;
  int nhwc8_batch_unit_offset = c8 * C8NUM * plane;
  int ic_remainder_ = channel % C8NUM;
  if (ic_remainder_ != 0) {
    int nhwc8_batch_offset = 0;
    for (int b = 0; b < batch; b++) {
      int batch_offset = b * channel * plane;
      for (int i = 0; i < plane; i++) {
        float *dst_per_plane = (float *)dst + nhwc8_batch_offset + i * c8_channel;
        memcpy(dst_per_plane, (float *)src + batch_offset + i * channel, channel * sizeof(float));
        for (int j = channel; j < c8_channel; ++j) {
          dst_per_plane[j] = 0;
        }
      }
      nhwc8_batch_offset += nhwc8_batch_unit_offset;
    }
  } else {
    size_t ori_input_size = batch * plane * channel * sizeof(float);
    memcpy((float *)dst, (float *)src, ori_input_size);
  }
}

void PackNHWC4ToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  int ic_remainder_ = channel % C4NUM;
  if (ic_remainder_ != 0) {
    int nhwc_batch_unit_offset = channel * plane;
    for (int b = 0; b < batch; b++) {
      int batch_offset = b * c4 * C4NUM * plane;
      for (int i = 0; i < plane; i++) {
        memcpy((float *)dst + b * nhwc_batch_unit_offset + i * channel, (float *)src + batch_offset + i * c4 * C4NUM,
               channel * sizeof(float));
      }
    }
  } else {
    size_t ori_input_size = batch * plane * channel * sizeof(float);
    memcpy((float *)dst, (float *)src, ori_input_size);
  }
}

void PackNC4HW4ToNHWC4Fp32(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  for (int b = 0; b < batch; b++) {
    int src_offset = b * plane * c4 * C4NUM;
    int dst_offset = b * plane * channel;
    for (int c = 0; c < channel; c++) {
      int c4_block_num = c / C4NUM;
      int c4_block_res = c % C4NUM;
      int src_c_offset = src_offset + c4_block_num * plane * C4NUM + c4_block_res;
      int dst_c_offset = dst_offset + c4_block_num * C4NUM + c4_block_res;
      for (int k = 0; k < plane; k++) {
        int src_kernel_offset = src_c_offset + k * C4NUM;
        int dst_kernel_offset = dst_c_offset + k * c4 * C4NUM;
        ((float *)dst + dst_kernel_offset)[0] = ((float *)src + src_kernel_offset)[0];
      }
    }
  }
}

void PackNC4HW4ToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel) {
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
#ifdef ENABLE_NEON
        vst1q_f32((float *)dst + dst_c_offset, vld1q_f32((float *)src + src_c_offset));
#else
        ((float *)dst + dst_c_offset)[0] = ((float *)src + src_c_offset)[0];
        ((float *)dst + dst_c_offset)[1] = ((float *)src + src_c_offset)[1];
        ((float *)dst + dst_c_offset)[2] = ((float *)src + src_c_offset)[2];
        ((float *)dst + dst_c_offset)[3] = ((float *)src + src_c_offset)[3];
#endif
      }
      // res part
      int res_c = channel - (c4 - 1) * C4NUM;
      for (int i = 0; i < res_c; i++) {
        int src_res_c_offset = src_kernel_offset + (c4 - 1) * C4NUM * plane + i;
        int dst_res_c_offset = dst_kernel_offset + (c4 - 1) * C4NUM + i;
        ((float *)dst + dst_res_c_offset)[0] = ((float *)src + src_res_c_offset)[0];
      }
    }
  }
}

void PackNHWCToC8HWN8Fp32(const void *src, void *dst, int batch, int plane, int channel) {
  for (int n = 0; n < batch; n++) {
    for (int hw = 0; hw < plane; hw++) {
      for (int c = 0; c < channel; c++) {
        int c8div = c / C8NUM;
        int c8mod = c % C8NUM;
        int src_index = n * plane * channel + hw * channel + c;
        int dst_index = c8div * batch * plane * C8NUM + hw * batch * C8NUM + n * C8NUM + c8mod;
        ((float *)dst)[dst_index] = ((float *)src)[src_index];
      }
    }
  }
  return;
}

void PackDepthwiseIndirectWeightC4Fp32(const void *src, void *dst, int height, int width, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  for (int c = 0; c < c4; c++) {
    int dst_off_c = c * C4NUM * height * width;
    for (int i = 0; i < C4NUM; i++) {
      int src_off_c = (c * C4NUM + i) * height * width;
      for (int kh = 0; kh < height; kh++) {
        int src_off_kh = src_off_c + kh * width;
        for (int kw = 0; kw < width; kw++) {
          int dst_off = dst_off_c + kw * height * C4NUM + kh * C4NUM + i;
          ((float *)dst)[dst_off] = ((float *)src)[src_off_kh + kw];
        }
      }
    }
  }
}

void PackDepthwiseIndirectWeightC8Fp32(const void *src, void *dst, int height, int width, int channel) {
  int c8 = UP_DIV(channel, C8NUM);
  for (int c = 0; c < c8; c++) {
    int dst_off_c = c * C8NUM * height * width;
    for (int i = 0; i < C8NUM; i++) {
      int src_off_c = (c * C8NUM + i) * height * width;
      for (int kh = 0; kh < height; kh++) {
        int src_off_kh = src_off_c + kh * width;
        for (int kw = 0; kw < width; kw++) {
          int dst_off = dst_off_c + kw * height * C8NUM + kh * C8NUM + i;
          ((float *)dst)[dst_off] = ((float *)src)[src_off_kh + kw];
        }
      }
    }
  }
}

void PackNHWCToNCHWFp32(const void *src, void *dst, int batches, int plane, int channel, int task_id,
                        int thread_count) {
#ifdef ENABLE_ARM64
  Transpose8X8Fp32Func Transpose8X8Fp32Func_ = Transpose8X8Fp32Arm64;
#elif defined(ENABLE_ARM32)
  Transpose8X8Fp32Func Transpose8X8Fp32Func_ = Transpose8X8Fp32Arm32;
#elif defined(ENABLE_AVX)
  Transpose8X8Fp32Func Transpose8X8Fp32Func_ = Transpose8X8Fp32Avx;
#elif defined(ENABLE_SSE) && !defined(ENABLE_AVX)
  Transpose8X8Fp32Func Transpose8X8Fp32Func_ = Transpose8X8Fp32Sse;
#endif
  int hw8 = plane / C8NUM;
  int task_start = 0;
  int task_end = plane;
  if (thread_count > 0) {
    int offset_hw = UP_DIV(hw8, thread_count) * C8NUM;
    task_start = offset_hw * task_id;
    int count = plane - task_start;
    if (count <= 0) {
      return;
    }
    task_end = (task_id + 1) == thread_count ? plane : MSMIN(plane, task_start + offset_hw);
    hw8 = task_start + ((task_end - task_start) >= offset_hw ? offset_hw : 0);
  } else {
    hw8 *= C8NUM;
  }
  int c8 = channel / C8NUM * C8NUM;
  int batch = plane * channel;
  for (int n = 0; n < batches; n++) {
    const float *src_batch = (const float *)src + n * batch;
    float *dst_batch = (float *)dst + n * batch;
    int hw = task_start;
    for (; hw < hw8; hw += C8NUM) {
      int c = 0;
      for (; c < c8; c += C8NUM) {
        const float *src_ptr = src_batch + hw * channel + c;
        float *dst_ptr = dst_batch + c * plane + hw;
#if defined(ENABLE_ARM64) || defined(ENABLE_AVX) || defined(ENABLE_SSE) || defined(ENABLE_ARM32)
        Transpose8X8Fp32Func_(src_ptr, dst_ptr, channel, plane);
#else
        for (int tr = 0; tr < C8NUM; tr++) {
          for (int tc = 0; tc < C8NUM; tc++) {
            dst_ptr[tc * plane + tr] = src_ptr[tr * channel + tc];
          }
        }
#endif
      }
      for (; c < channel; c++) {
        const float *src_ptr = src_batch + hw * channel + c;
        float *dst_ptr = dst_batch + c * plane + hw;
        for (size_t i = 0; i < C8NUM; i++) {
          dst_ptr[i] = src_ptr[i * channel];
        }
      }
    }
    for (; hw < task_end; hw++) {
      const float *src_ptr = src_batch + hw * channel;
      float *dst_ptr = dst_batch + hw;
      for (size_t i = 0; i < channel; i++) {
        dst_ptr[i * plane] = src_ptr[i];
      }
    }
  }
}

void PackNCHWToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel, int task_id, int thread_count) {
  return PackNHWCToNCHWFp32(src, dst, batch, channel, plane, task_id, thread_count);
}

#ifdef ENABLE_ARM64
inline void Transpose8X8Fp32Arm64(const float *src_ptr, float *dst_ptr, int src_stride, int dst_stride) {
  size_t srcStride = src_stride * sizeof(float);
  size_t dstStride = dst_stride * sizeof(float);
  asm volatile(
    "mov x10, %[src_ptr]\n"
    "mov x11, %[dst_ptr]\n"

    "ld1 {v0.4s, v1.4s}, [x10], %[srcStride]\n"
    "ld1 {v2.4s, v3.4s}, [x10], %[srcStride]\n"

    "zip1 v8.4s, v0.4s, v2.4s\n"
    "zip2 v9.4s, v0.4s, v2.4s\n"
    "zip1 v12.4s, v1.4s, v3.4s\n"
    "zip2 v13.4s, v1.4s, v3.4s\n"

    "ld1 {v4.4s, v5.4s}, [x10], %[srcStride]\n"
    "ld1 {v6.4s, v7.4s}, [x10], %[srcStride]\n"

    "zip1 v10.4s, v4.4s, v6.4s\n"
    "zip2 v11.4s, v4.4s, v6.4s\n"
    "zip1 v14.4s, v5.4s, v7.4s\n"
    "zip2 v15.4s, v5.4s, v7.4s\n"

    "ld1 {v0.4s, v1.4s}, [x10], %[srcStride]\n"
    "ld1 {v2.4s, v3.4s}, [x10], %[srcStride]\n"

    "trn1 v16.2d, v8.2d, v10.2d\n"
    "trn2 v18.2d, v8.2d, v10.2d\n"
    "trn1 v20.2d, v9.2d, v11.2d\n"
    "trn2 v22.2d, v9.2d, v11.2d\n"

    "ld1 {v4.4s, v5.4s}, [x10], %[srcStride]\n"
    "ld1 {v6.4s, v7.4s}, [x10], %[srcStride]\n"

    "trn1 v24.2d, v12.2d, v14.2d\n"
    "trn2 v26.2d, v12.2d, v14.2d\n"
    "trn1 v28.2d, v13.2d, v15.2d\n"
    "trn2 v30.2d, v13.2d, v15.2d\n"

    "zip1 v8.4s, v0.4s, v2.4s\n"
    "zip2 v9.4s, v0.4s, v2.4s\n"
    "zip1 v12.4s, v1.4s, v3.4s\n"
    "zip2 v13.4s, v1.4s, v3.4s\n"

    "zip1 v10.4s, v4.4s, v6.4s\n"
    "zip2 v11.4s, v4.4s, v6.4s\n"
    "zip1 v14.4s, v5.4s, v7.4s\n"
    "zip2 v15.4s, v5.4s, v7.4s\n"

    "trn1 v17.2d, v8.2d, v10.2d\n"
    "trn2 v19.2d, v8.2d, v10.2d\n"
    "trn1 v21.2d, v9.2d, v11.2d\n"
    "trn2 v23.2d, v9.2d, v11.2d\n"

    "trn1 v25.2d, v12.2d, v14.2d\n"
    "trn2 v27.2d, v12.2d, v14.2d\n"
    "trn1 v29.2d, v13.2d, v15.2d\n"
    "trn2 v31.2d, v13.2d, v15.2d\n"

    "st1 {v16.4s, v17.4s}, [x11], %[dstStride]\n"
    "st1 {v18.4s, v19.4s}, [x11], %[dstStride]\n"
    "st1 {v20.4s, v21.4s}, [x11], %[dstStride]\n"
    "st1 {v22.4s, v23.4s}, [x11], %[dstStride]\n"
    "st1 {v24.4s, v25.4s}, [x11], %[dstStride]\n"
    "st1 {v26.4s, v27.4s}, [x11], %[dstStride]\n"
    "st1 {v28.4s, v29.4s}, [x11], %[dstStride]\n"
    "st1 {v30.4s, v31.4s}, [x11], %[dstStride]\n"

    :
    : [ dst_ptr ] "r"(dst_ptr), [ src_ptr ] "r"(src_ptr), [ srcStride ] "r"(srcStride), [ dstStride ] "r"(dstStride)
    : "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
      "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
      "v31");
}
#endif

#ifdef ENABLE_ARM32
inline void Transpose8X8Fp32Arm32(const float *src_ptr, float *dst_ptr, int src_stride, int dst_stride) {
  size_t srcStride = src_stride * sizeof(float);
  size_t dstStride = dst_stride * sizeof(float);
  asm volatile(
    "mov r10, %[src_ptr]\n"
    "mov r12, %[dst_ptr]\n"

    "vld1.32 {q0, q1}, [r10], %[srcStride]\n"
    "vld1.32 {q2, q3}, [r10], %[srcStride]\n"

    "vtrn.32 d0, d4\n"
    "vtrn.32 d1, d5\n"
    "vtrn.32 d2, d6\n"
    "vtrn.32 d3, d7\n"

    "vld1.32 {q4, q5}, [r10], %[srcStride]\n"
    "vld1.32 {q6, q7}, [r10], %[srcStride]\n"

    "vtrn.32 d8, d12\n"
    "vtrn.32 d9, d13\n"
    "vtrn.32 d10, d14\n"
    "vtrn.32 d11, d15\n"

    "vld1.32 {q8, q9}, [r10], %[srcStride]\n"
    "vld1.32 {q10, q11}, [r10], %[srcStride]\n"

    "vswp d1, d8\n"
    "vswp d3, d10\n"
    "vswp d5, d12\n"
    "vswp d7, d14\n"

    "vtrn.32 d16, d20\n"
    "vtrn.32 d17, d21\n"
    "vtrn.32 d18, d22\n"
    "vtrn.32 d19, d23\n"

    "vld1.32 {q12, q13}, [r10], %[srcStride]\n"
    "vld1.32 {q14, q15}, [r10], %[srcStride]\n"

    "vtrn.32 d24, d28\n"
    "vtrn.32 d25, d29\n"
    "vtrn.32 d26, d30\n"
    "vtrn.32 d27, d31\n"

    "vswp d17, d24\n"
    "vswp d19, d26\n"
    "vswp d21, d28\n"
    "vswp d23, d30\n"

    "add r10, r12, #16\n"
    "vst1.32 {q0}, [r12], %[dstStride]\n"
    "vst1.32 {q8}, [r10], %[dstStride]\n"
    "vst1.32 {q2}, [r12], %[dstStride]\n"
    "vst1.32 {q10}, [r10], %[dstStride]\n"
    "vst1.32 {q4}, [r12], %[dstStride]\n"
    "vst1.32 {q12}, [r10], %[dstStride]\n"
    "vst1.32 {q6}, [r12], %[dstStride]\n"
    "vst1.32 {q14}, [r10], %[dstStride]\n"
    "vst1.32 {q1}, [r12], %[dstStride]\n"
    "vst1.32 {q9}, [r10], %[dstStride]\n"
    "vst1.32 {q3}, [r12], %[dstStride]\n"
    "vst1.32 {q11}, [r10], %[dstStride]\n"
    "vst1.32 {q5}, [r12], %[dstStride]\n"
    "vst1.32 {q13}, [r10], %[dstStride]\n"
    "vst1.32 {q7}, [r12], %[dstStride]\n"
    "vst1.32 {q15}, [r10], %[dstStride]\n"

    :
    : [ dst_ptr ] "r"(dst_ptr), [ src_ptr ] "r"(src_ptr), [ srcStride ] "r"(srcStride), [ dstStride ] "r"(dstStride)
    : "r10", "r12", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14",
      "q15");
}
#endif

#ifdef ENABLE_AVX
inline void Transpose8X8Fp32Avx(const float *src_ptr, float *dst_ptr, int src_stride, int dst_stride) {
  LOAD256X8_F32(src, src_ptr, src_stride)
  __m256 r1 = _mm256_unpacklo_ps(src1, src2);
  __m256 r2 = _mm256_unpackhi_ps(src1, src2);
  __m256 r3 = _mm256_unpacklo_ps(src3, src4);
  __m256 r4 = _mm256_unpackhi_ps(src3, src4);
  __m256 r5 = _mm256_unpacklo_ps(src5, src6);
  __m256 r6 = _mm256_unpackhi_ps(src5, src6);
  __m256 r7 = _mm256_unpacklo_ps(src7, src8);
  __m256 r8 = _mm256_unpackhi_ps(src7, src8);

  __m256 v;
  v = _mm256_shuffle_ps(r1, r3, 0x4E);
  src1 = _mm256_blend_ps(r1, v, 0xCC);
  src2 = _mm256_blend_ps(r3, v, 0x33);

  v = _mm256_shuffle_ps(r2, r4, 0x4E);
  src3 = _mm256_blend_ps(r2, v, 0xCC);
  src4 = _mm256_blend_ps(r4, v, 0x33);

  v = _mm256_shuffle_ps(r5, r7, 0x4E);
  src5 = _mm256_blend_ps(r5, v, 0xCC);
  src6 = _mm256_blend_ps(r7, v, 0x33);

  v = _mm256_shuffle_ps(r6, r8, 0x4E);
  src7 = _mm256_blend_ps(r6, v, 0xCC);
  src8 = _mm256_blend_ps(r8, v, 0x33);

  r1 = _mm256_permute2f128_ps(src1, src5, 0x20);
  r2 = _mm256_permute2f128_ps(src2, src6, 0x20);
  r3 = _mm256_permute2f128_ps(src3, src7, 0x20);
  r4 = _mm256_permute2f128_ps(src4, src8, 0x20);
  r5 = _mm256_permute2f128_ps(src1, src5, 0x31);
  r6 = _mm256_permute2f128_ps(src2, src6, 0x31);
  r7 = _mm256_permute2f128_ps(src3, src7, 0x31);
  r8 = _mm256_permute2f128_ps(src4, src8, 0x31);

  STORE256X8_F32(dst_ptr, dst_stride, r);
}
#endif

#if defined(ENABLE_SSE) && !defined(ENABLE_AVX)
inline void Transpose8X8Fp32Sse(const float *src_ptr, float *dst_ptr, int src_stride, int dst_stride) {
  __m128 v0_ma = _mm_loadu_ps(src_ptr);
  __m128 v1_ma = _mm_loadu_ps(src_ptr + src_stride);
  __m128 v2_ma = _mm_loadu_ps(src_ptr + 2 * src_stride);
  __m128 v3_ma = _mm_loadu_ps(src_ptr + 3 * src_stride);

  __m128 v4_ma = _mm_unpacklo_ps(v0_ma, v1_ma);
  __m128 v5_ma = _mm_unpackhi_ps(v0_ma, v1_ma);
  __m128 v6_ma = _mm_unpacklo_ps(v2_ma, v3_ma);
  __m128 v7_ma = _mm_unpackhi_ps(v2_ma, v3_ma);

  __m128 v8_ma = _mm_movelh_ps(v4_ma, v6_ma);
  __m128 v9_ma = _mm_movehl_ps(v6_ma, v4_ma);
  __m128 v10_ma = _mm_movelh_ps(v5_ma, v7_ma);
  __m128 v11_ma = _mm_movehl_ps(v7_ma, v5_ma);

  _mm_storeu_ps(dst_ptr, v8_ma);
  _mm_storeu_ps(dst_ptr + dst_stride, v9_ma);
  _mm_storeu_ps(dst_ptr + 2 * dst_stride, v10_ma);
  _mm_storeu_ps(dst_ptr + 3 * dst_stride, v11_ma);

  v0_ma = _mm_loadu_ps(src_ptr + C4NUM);
  v1_ma = _mm_loadu_ps(src_ptr + src_stride + C4NUM);
  v2_ma = _mm_loadu_ps(src_ptr + 2 * src_stride + C4NUM);
  v3_ma = _mm_loadu_ps(src_ptr + 3 * src_stride + C4NUM);

  v4_ma = _mm_unpacklo_ps(v0_ma, v1_ma);
  v5_ma = _mm_unpackhi_ps(v0_ma, v1_ma);
  v6_ma = _mm_unpacklo_ps(v2_ma, v3_ma);
  v7_ma = _mm_unpackhi_ps(v2_ma, v3_ma);

  v8_ma = _mm_movelh_ps(v4_ma, v6_ma);
  v9_ma = _mm_movehl_ps(v6_ma, v4_ma);
  v10_ma = _mm_movelh_ps(v5_ma, v7_ma);
  v11_ma = _mm_movehl_ps(v7_ma, v5_ma);

  _mm_storeu_ps(dst_ptr + C4NUM * dst_stride, v8_ma);
  _mm_storeu_ps(dst_ptr + (C4NUM + 1) * dst_stride, v9_ma);
  _mm_storeu_ps(dst_ptr + (C4NUM + 2) * dst_stride, v10_ma);
  _mm_storeu_ps(dst_ptr + (C4NUM + 3) * dst_stride, v11_ma);

  v0_ma = _mm_loadu_ps(src_ptr + C4NUM * src_stride);
  v1_ma = _mm_loadu_ps(src_ptr + (C4NUM + 1) * src_stride);
  v2_ma = _mm_loadu_ps(src_ptr + (C4NUM + 2) * src_stride);
  v3_ma = _mm_loadu_ps(src_ptr + (C4NUM + 3) * src_stride);

  v4_ma = _mm_unpacklo_ps(v0_ma, v1_ma);
  v5_ma = _mm_unpackhi_ps(v0_ma, v1_ma);
  v6_ma = _mm_unpacklo_ps(v2_ma, v3_ma);
  v7_ma = _mm_unpackhi_ps(v2_ma, v3_ma);

  v8_ma = _mm_movelh_ps(v4_ma, v6_ma);
  v9_ma = _mm_movehl_ps(v6_ma, v4_ma);
  v10_ma = _mm_movelh_ps(v5_ma, v7_ma);
  v11_ma = _mm_movehl_ps(v7_ma, v5_ma);

  _mm_storeu_ps(dst_ptr + C4NUM, v8_ma);
  _mm_storeu_ps(dst_ptr + dst_stride + C4NUM, v9_ma);
  _mm_storeu_ps(dst_ptr + 2 * dst_stride + C4NUM, v10_ma);
  _mm_storeu_ps(dst_ptr + 3 * dst_stride + C4NUM, v11_ma);

  v0_ma = _mm_loadu_ps(src_ptr + C4NUM * src_stride + C4NUM);
  v1_ma = _mm_loadu_ps(src_ptr + (C4NUM + 1) * src_stride + C4NUM);
  v2_ma = _mm_loadu_ps(src_ptr + (C4NUM + 2) * src_stride + C4NUM);
  v3_ma = _mm_loadu_ps(src_ptr + (C4NUM + 3) * src_stride + C4NUM);

  v4_ma = _mm_unpacklo_ps(v0_ma, v1_ma);
  v5_ma = _mm_unpackhi_ps(v0_ma, v1_ma);
  v6_ma = _mm_unpacklo_ps(v2_ma, v3_ma);
  v7_ma = _mm_unpackhi_ps(v2_ma, v3_ma);

  v8_ma = _mm_movelh_ps(v4_ma, v6_ma);
  v9_ma = _mm_movehl_ps(v6_ma, v4_ma);
  v10_ma = _mm_movelh_ps(v5_ma, v7_ma);
  v11_ma = _mm_movehl_ps(v7_ma, v5_ma);

  _mm_storeu_ps(dst_ptr + C4NUM * dst_stride + C4NUM, v8_ma);
  _mm_storeu_ps(dst_ptr + (C4NUM + 1) * dst_stride + C4NUM, v9_ma);
  _mm_storeu_ps(dst_ptr + (C4NUM + 2) * dst_stride + C4NUM, v10_ma);
  _mm_storeu_ps(dst_ptr + (C4NUM + 3) * dst_stride + C4NUM, v11_ma);
}
#endif

#if defined(ENABLE_ARM) || (defined(ENABLE_SSE) && !defined(ENABLE_AVX))
void PackWeightConvDw3x3Fp32(const void *src, void *dst, int channel) {
  // nchw to nc4hw4 with 1D F(2,3)
  for (int i = 0; i < channel; i++) {
    float *src_kernel = (float *)src + i * 9;
    float *dst_kernel = (float *)dst + (i / 4) * 48 + i % 4;
    for (int y = 0; y < 3; y++) {
      float g0 = src_kernel[3 * y];
      float g1 = src_kernel[3 * y + 1];
      float g2 = src_kernel[3 * y + 2];

      dst_kernel[16 * y] = g0;
      dst_kernel[16 * y + 4] = 0.5f * (g0 + g1 + g2);
      dst_kernel[16 * y + 8] = 0.5f * (g0 - g1 + g2);
      dst_kernel[16 * y + 12] = g2;
    }
  }
}
#endif
