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

#include "nnacl/fp16/pack_fp16.h"
#include <string.h>

#ifdef ENABLE_ARM
void PackWeightConvDw3x3Fp16(const void *src, void *dst, int channel) {
  // nchw to nc8hw8 with 1D F(2,3)
  for (int i = 0; i < channel; i++) {
    float16_t *src_kernel = (float16_t *)src + i * 9;
    float16_t *dst_kernel = (float16_t *)dst + (i / 8) * 96 + i % 8;
    for (int y = 0; y < 3; y++) {
      float16_t g0 = src_kernel[3 * y];
      float16_t g1 = src_kernel[3 * y + 1];
      float16_t g2 = src_kernel[3 * y + 2];

      dst_kernel[32 * y] = g0;
      dst_kernel[32 * y + 8] = (float16_t)0.5 * (g0 + g1 + g2);
      dst_kernel[32 * y + 16] = (float16_t)0.5 * (g0 - g1 + g2);
      dst_kernel[32 * y + 24] = g2;
    }
  }
}
#endif

void Im2ColPackUnitFp16(const float16_t *input_data, const ConvParameter *conv_param, float16_t *packed_input,
                        int real_cal_num, int block_index) {
  // input format : nhwc
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int kernel_plane = kernel_h * kernel_w;
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

  for (int i = 0; i < real_cal_num; i++) {
    int block_start = block_index + i;
    int input_h = block_start / out_w * stride_h - pad_h;
    int input_w = block_start % out_w * stride_w - pad_w;
    int input_stride = (input_h * in_w + input_w) * in_channel;
    int kh_s = MSMAX(0, UP_DIV(-input_h, dilation_h));
    int kh_e = MSMIN(kernel_h, UP_DIV(in_h - input_h, dilation_h));
    int kw_s = MSMAX(0, UP_DIV(-input_w, dilation_w));
    int kw_e = MSMIN(kernel_w, UP_DIV(in_w - input_w, dilation_w));
    if (dilation_h == 1 && dilation_w == 1) {
      for (int j = kh_s; j < kh_e; j++) {
        int input_y_stride = j * in_w * in_channel + input_stride;
        int input_x_stride = input_y_stride + kw_s * in_channel;
        int input_plane_offset = (j * kernel_w + kw_s) * in_channel + i * in_channel * kernel_plane;
        memcpy(packed_input + input_plane_offset, input_data + input_x_stride,
               (kw_e - kw_s) * in_channel * sizeof(float16_t));
      }  // kernel_h loop
    } else {
      for (int j = kh_s; j < kh_e; j++) {
        int input_y_stride = j * dilation_h * in_w * in_channel + input_stride;
        for (int n = kw_s; n < kw_e; n++) {
          int input_x_stride = input_y_stride + n * dilation_w * in_channel;
          int input_plane_offset = (j * kernel_w + n) * in_channel + i * in_channel * kernel_plane;
          memcpy(packed_input + input_plane_offset, input_data + input_x_stride, in_channel * sizeof(float16_t));
        }  // kernel_w loop
      }    // kernel_h loop
    }
  }  // tile num loop
}

void PackHWCToWHCFp16(const float16_t *src, float16_t *dst, int height, int width, int channel) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      memcpy(dst + (j * height + i) * channel, src + (i * width + j) * channel, channel * sizeof(float16_t));
    }
  }
}

void PackWeightToC8Fp16(const float16_t *origin_weight_data, float16_t *packed_weight_data,
                        const ConvParameter *conv_param) {
  // origin weight format : ohwi
  int input_channel = conv_param->input_channel_;
  int ic8 = UP_DIV(input_channel, C8NUM);
  int output_channel = conv_param->output_channel_;
  int kernel_plane = conv_param->kernel_h_ * conv_param->kernel_w_;

  for (int k = 0; k < kernel_plane; k++) {
    int src_kernel_offset = k * input_channel;
    int dst_kernel_offset = k * C8NUM;
    for (int o = 0; o < output_channel; o++) {
      int src_oc_offset = src_kernel_offset + o * kernel_plane * input_channel;
      int dst_oc_offset = dst_kernel_offset + o * ic8 * kernel_plane * C8NUM;
      for (int i = 0; i < input_channel; i++) {
        int c8_block_num = i / C8NUM;
        int c8_block_rem = i % C8NUM;
        int src_ic_offset = src_oc_offset + i;
        int dst_ic_offset = dst_oc_offset + c8_block_num * kernel_plane * C8NUM + c8_block_rem;
        (packed_weight_data + dst_ic_offset)[0] = (origin_weight_data + src_ic_offset)[0];
      }
    }
  }
}

void PackWeightToC4Fp16(const float16_t *origin_weight_data, float16_t *packed_weight_data,
                        const ConvParameter *conv_param) {
  // origin weight format : ohwi
  int input_channel = conv_param->input_channel_;
  int ic8 = UP_DIV(input_channel, C8NUM);
  int ic4 = ic8 * 2;
  int output_channel = conv_param->output_channel_;
  int kernel_plane = conv_param->kernel_h_ * conv_param->kernel_w_;

  for (int k = 0; k < kernel_plane; k++) {
    int src_kernel_offset = k * input_channel;
    int dst_kernel_offset = k * C4NUM;
    for (int o = 0; o < output_channel; o++) {
      int src_oc_offset = src_kernel_offset + o * kernel_plane * input_channel;
      int dst_oc_offset = dst_kernel_offset + o * ic4 * kernel_plane * C4NUM;
      for (int i = 0; i < input_channel; i++) {
        int c4_block_num = i / C4NUM;
        int c4_block_rem = i % C4NUM;
        int src_ic_offset = src_oc_offset + i;
        int dst_ic_offset = dst_oc_offset + c4_block_num * kernel_plane * C4NUM + c4_block_rem;
        (packed_weight_data + dst_ic_offset)[0] = (origin_weight_data + src_ic_offset)[0];
      }
    }
  }
}

void PackNHWCToNC4HW4Fp16(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  for (int b = 0; b < batch; b++) {
    int src_oc_offset = b * plane * channel;
    int dst_oc_offset = b * plane * c4 * C4NUM;
    for (int k = 0; k < plane; k++) {
      int src_kernel_offset = src_oc_offset + k * channel;
      int dst_kernel_offset = dst_oc_offset + k * C4NUM;
      for (int i = 0; i < channel; i++) {
        int c4_block_num = i / C4NUM;
        int c4_block_rem = i % C4NUM;
        int src_ic_offset = src_kernel_offset + i;
        int dst_ic_offset = dst_kernel_offset + c4_block_num * plane * C4NUM + c4_block_rem;
        ((float16_t *)dst + dst_ic_offset)[0] = ((float16_t *)src + src_ic_offset)[0];
      }
    }
  }
}

void PackNCHWToNC4HW4Fp16(const void *src, void *dst, int batch, int plane, int channel) {
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
        ((float16_t *)dst + dst_kernel_offset)[0] = ((float16_t *)src + src_kernel_offset)[0];
      }
    }
  }
}

void PackNHWCToNCHWFp16(const void *src, void *dst, int batches, int plane, int channel, int task_id,
                        int thread_count) {
#ifdef ENABLE_ARM64
  // Transpose16x8 in arm64
  const int hw_tile = C16NUM;
#else
  // Transpose8x8 in others
  const int hw_tile = C8NUM;
#endif
  int hw_align = plane / hw_tile;
  int task_start = 0;
  int task_end = plane;
  if (thread_count > 0) {
    int offset_hw = UP_DIV(hw_align, thread_count) * hw_tile;
    task_start = offset_hw * task_id;
    int count = plane - task_start;
    if (count <= 0) {
      return;
    }
    task_end = (task_id + 1) == thread_count ? plane : MSMIN(plane, task_start + offset_hw);
    hw_align = task_start + ((task_end - task_start) >= offset_hw ? offset_hw : 0);
  } else {
    hw_align *= hw_tile;
  }
  int c8 = channel / C8NUM * C8NUM;
  int batch = plane * channel;
  for (int n = 0; n < batches; n++) {
    const float16_t *src_batch = (const float16_t *)src + n * batch;
    float16_t *dst_batch = (float16_t *)dst + n * batch;
    int hw = task_start;
    for (; hw < hw_align; hw += hw_tile) {
      int c = 0;
      for (; c < c8; c += C8NUM) {
        const float16_t *src_ptr = src_batch + hw * channel + c;
        float16_t *dst_ptr = dst_batch + c * plane + hw;
#ifdef ENABLE_ARM64
        size_t src_stride = channel * sizeof(float16_t);
        size_t dst_stride = plane * sizeof(float16_t);
        Transpose16x8ARM64Fp16(src_ptr, dst_ptr, src_stride, dst_stride);
#elif defined(ENABLE_ARM82_A32)
        size_t src_stride = channel * sizeof(float16_t);
        size_t dst_stride = plane * sizeof(float16_t);
        Transpose8x8A32Fp16(src_ptr, dst_ptr, src_stride, dst_stride);
#else
        for (int tr = 0; tr < hw_tile; tr++) {
          for (int tc = 0; tc < C8NUM; tc++) {
            dst_ptr[tc * plane + tr] = src_ptr[tr * channel + tc];
          }
        }
#endif
      }
      for (; c < channel; c++) {
        const float16_t *src_ptr = src_batch + hw * channel + c;
        float16_t *dst_ptr = dst_batch + c * plane + hw;
        for (size_t i = 0; i < hw_tile; i++) {
          dst_ptr[i] = src_ptr[i * channel];
        }
      }
    }
    for (; hw < task_end; hw++) {
      const float16_t *src_ptr = src_batch + hw * channel;
      float16_t *dst_ptr = dst_batch + hw;
      for (size_t i = 0; i < channel; i++) {
        dst_ptr[i * plane] = src_ptr[i];
      }
    }
  }
}

void PackNCHWToNHWCFp16(const void *src, void *dst, int batch, int plane, int channel, int task_id, int thread_count) {
  return PackNHWCToNCHWFp16(src, dst, batch, channel, plane, task_id, thread_count);
}

void PackNHWCToNHWC4Fp16(const void *src, void *dst, int batch, int plane, int channel) {
  int ic4 = UP_DIV(channel, C4NUM);
  int c4_channel = ic4 * C4NUM;
  int nhwc4_batch_unit_offset = ic4 * C4NUM * plane;
  int ic_remainder_ = channel % C4NUM;
  if (ic_remainder_ != 0) {
    int nhwc4_batch_offset = 0;
    for (int b = 0; b < batch; b++) {
      int batch_offset = b * channel * plane;
      for (int i = 0; i < plane; i++) {
        float16_t *dst_per_plane = (float16_t *)dst + nhwc4_batch_offset + i * c4_channel;
        memcpy(dst_per_plane, (float16_t *)src + batch_offset + i * channel, channel * sizeof(float16_t));
        for (int j = channel; j < c4_channel; ++j) {
          dst_per_plane[j] = 0;
        }
      }
      nhwc4_batch_offset += nhwc4_batch_unit_offset;
    }
  } else {
    size_t ori_input_size = batch * plane * channel * sizeof(float16_t);
    memcpy(dst, src, ori_input_size);
  }
}

void PackNHWCToNHWC8Fp16(const void *src, void *dst, int batch, int plane, int channel) {
  int ic8 = UP_DIV(channel, C8NUM);
  int c8_channel = ic8 * C8NUM;
  int nhwc8_batch_unit_offset = ic8 * C8NUM * plane;
  int ic_remainder_ = channel % C8NUM;
  if (ic_remainder_ != 0) {
    int nhwc8_batch_offset = 0;
    for (int b = 0; b < batch; b++) {
      int batch_offset = b * channel * plane;
      for (int i = 0; i < plane; i++) {
        float16_t *dst_per_plane = (float16_t *)dst + nhwc8_batch_offset + i * c8_channel;
        memcpy(dst_per_plane, (float16_t *)src + batch_offset + i * channel, channel * sizeof(float16_t));
        for (int j = channel; j < c8_channel; ++j) {
          dst_per_plane[j] = 0;
        }
      }
      nhwc8_batch_offset += nhwc8_batch_unit_offset;
    }
  } else {
    size_t ori_input_size = batch * plane * channel * sizeof(float16_t);
    memcpy(dst, src, ori_input_size);
  }
}

void PackNHWC4ToNHWCFp16(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  int ic_remainder_ = channel % C4NUM;
  if (ic_remainder_ != 0) {
    int nhwc_batch_unit_offset = channel * plane;
    for (int b = 0; b < batch; b++) {
      int batch_offset = b * c4 * C4NUM * plane;
      for (int i = 0; i < plane; i++) {
        memcpy((float16_t *)dst + b * nhwc_batch_unit_offset + i * channel,
               (float16_t *)src + batch_offset + i * c4 * C4NUM, channel * sizeof(float16_t));
      }
    }
  } else {
    size_t ori_input_size = batch * plane * channel * sizeof(float16_t);
    memcpy((float16_t *)dst, (float16_t *)src, ori_input_size);
  }
}

void PackNCHWToNHWC4Fp16(const void *src, void *dst, int batch, int plane, int channel) {
  int nhwc4_batch_offset = 0;
  int ic4 = UP_DIV(channel, C4NUM);
  int nhwc4_batch_unit_offset = ic4 * C4NUM * plane;

  for (int b = 0; b < batch; b++) {
    int batch_offset = b * channel * plane;
    for (int c = 0; c < channel; c++) {
      int src_c_offset = batch_offset + c * plane;
      int dst_c_offset = nhwc4_batch_offset + c;
      for (int i = 0; i < plane; i++) {
        int src_plane_offset = src_c_offset + i;
        int dst_plane_offset = dst_c_offset + i * ic4 * C4NUM;
        ((float16_t *)dst)[dst_plane_offset] = ((float16_t *)src)[src_plane_offset];
      }
    }
    nhwc4_batch_offset += nhwc4_batch_unit_offset;
  }
}

void PackNC4HW4ToNHWC4Fp16(const void *src, void *dst, int batch, int plane, int channel) {
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
        ((float16_t *)dst + dst_kernel_offset)[0] = ((float16_t *)src + src_kernel_offset)[0];
      }
    }
  }
}

void PackNC4HW4ToNHWCFp16(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  for (int b = 0; b < batch; b++) {
    int src_offset = b * plane * c4 * C4NUM;
    int dst_offset = b * plane * channel;
    for (int c = 0; c < channel; c++) {
      int c4_block_num = c / C4NUM;
      int c4_block_res = c % C4NUM;
      int src_c_offset = src_offset + c4_block_num * plane * C4NUM + c4_block_res;
      int dst_c_offset = dst_offset + c;
      for (int k = 0; k < plane; k++) {
        int src_kernel_offset = src_c_offset + k * C4NUM;
        int dst_kernel_offset = dst_c_offset + k * channel;
        ((float16_t *)dst + dst_kernel_offset)[0] = ((float16_t *)src + src_kernel_offset)[0];
      }
    }
  }
}

void PackNC4HW4ToNCHWFp16(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  for (int b = 0; b < batch; b++) {
    int src_offset = b * plane * c4 * C4NUM;
    int dst_offset = b * plane * channel;
    for (int c = 0; c < channel; c++) {
      int c4_block_num = c / C4NUM;
      int c4_block_res = c % C4NUM;
      int src_c_offset = src_offset + c4_block_num * plane * C4NUM + c4_block_res;
      int dst_c_offset = dst_offset + c * plane;
      for (int k = 0; k < plane; k++) {
        int src_kernel_offset = src_c_offset + k * C4NUM;
        int dst_kernel_offset = dst_c_offset + k;
        ((float16_t *)dst + dst_kernel_offset)[0] = ((float16_t *)src + src_kernel_offset)[0];
      }
    }
  }
}

void PackNCHWFp32ToNC8HW8Fp16(const float *src, float16_t *dst, int batch, int plane, int channel) {
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
        (dst + dst_kernel_offset)[0] = (float16_t)(src + src_kernel_offset)[0];
      }
    }
  }
}

void PackNCHWFp16ToNC8HW8Fp16(const float16_t *src, float16_t *dst, int batch, int plane, int channel) {
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
        (dst + dst_kernel_offset)[0] = (float16_t)(src + src_kernel_offset)[0];
      }
    }
  }
}

#ifdef Debug
void PackNC8HW8ToNHWCFp16(const float16_t *src, float16_t *dst, int batch, int plane, int channel) {
  int block = UP_DIV(channel, C8NUM);
  int last_block_idx = block - 1;
  int last_src_col = channel - last_block_idx * C8NUM;
  for (size_t i = 0; i < block; i++) {
    size_t src_col = (i != last_block_idx) ? C8NUM : last_src_col;
    float16_t *dst_cur = dst + i * C8NUM;
    for (size_t j = 0; j < plane; j++) {
      memcpy(dst_cur, src, src_col * sizeof(float16_t));
      src += src_col;
      dst_cur += channel;
    }
  }
}
#endif

void PackNHWCFp32ToNHWC8Fp16(const float *src, float16_t *dst, int batch, int plane, int channel) {
  int c8_channel = UP_DIV(channel, C8NUM) * C8NUM;
  for (int b = 0; b < batch; b++) {
    float16_t *dst_batch = dst + b * plane * c8_channel;
    const float *src_batch = src + b * plane * channel;
    for (int i = 0; i < plane; i++) {
      float16_t *dst_plane = dst_batch + i * c8_channel;
      const float *src_plane = src_batch + i * channel;
      for (int c = 0; c < channel; c++) {
        dst_plane[c] = (float16_t)(src_plane[c]);
      }
    }
  }
}

void PackNHWCFp32ToC8HWN8Fp16(const float *src, float16_t *dst, int batch, int plane, int channel) {
  for (int n = 0; n < batch; n++) {
    for (int hw = 0; hw < plane; hw++) {
      for (int c = 0; c < channel; c++) {
        int c8div = c / C8NUM;
        int c8mod = c % C8NUM;
        int src_index = n * plane * channel + hw * channel + c;
        int dst_index = c8div * batch * plane * C8NUM + hw * batch * C8NUM + n * C8NUM + c8mod;
        dst[dst_index] = (float16_t)(src[src_index]);
      }
    }
  }
  return;
}

void PackNHWCFp16ToC8HWN8Fp16(const float16_t *src, float16_t *dst, int batch, int plane, int channel) {
  for (int n = 0; n < batch; n++) {
    for (int hw = 0; hw < plane; hw++) {
      for (int c = 0; c < channel; c++) {
        int c8div = c / C8NUM;
        int c8mod = c % C8NUM;
        int src_index = n * plane * channel + hw * channel + c;
        int dst_index = c8div * batch * plane * C8NUM + hw * batch * C8NUM + n * C8NUM + c8mod;
        dst[dst_index] = src[src_index];
      }
    }
  }
  return;
}

void PackNHWC8Fp16ToNHWCFp32(const float16_t *src, float *dst, int batch, int plane, int channel) {
  int c8_channel = UP_DIV(channel, C8NUM) * C8NUM;
  for (int b = 0; b < batch; b++) {
    const float16_t *src_batch = src + b * plane * c8_channel;
    float *dst_batch = dst + b * plane * channel;
    for (int i = 0; i < plane; i++) {
      const float16_t *src_plane = src_batch + i * c8_channel;
      float *dst_plane = dst_batch + i * channel;
      for (int c = 0; c < channel; c++) {
        dst_plane[c] = (float16_t)(src_plane[c]);
      }
    }
  }
}

void PackNHWC8ToNHWCFp16(const float16_t *src, float16_t *dst, int batch, int plane, int channel) {
  int c8_channel = UP_DIV(channel, C8NUM) * C8NUM;
  for (int b = 0; b < batch; b++) {
    const float16_t *src_batch = src + b * plane * c8_channel;
    float16_t *dst_batch = dst + b * plane * channel;
    for (int i = 0; i < plane; i++) {
      const float16_t *src_plane = src_batch + i * c8_channel;
      float16_t *dst_plane = dst_batch + i * channel;
      memcpy(dst_plane, src_plane, channel * sizeof(float16_t));
    }
  }
}

#ifdef ENABLE_ARM82_A32
inline void Transpose8x8A32Fp16(const float16_t *src, float16_t *dst, size_t src_stride, size_t dst_stride) {
  asm volatile(
    "mov r10, %[src]\n"
    "mov r12, %[dst]\n"
    "vld1.16 {q0}, [r10], %[src_stride]\n"
    "vld1.16 {q2}, [r10], %[src_stride]\n"
    "vld1.16 {q4}, [r10], %[src_stride]\n"
    "vld1.16 {q6}, [r10], %[src_stride]\n"

    "vtrn.16 d0, d4\n"
    "vtrn.16 d1, d5\n"
    "vtrn.16 d8, d12\n"
    "vtrn.16 d9, d13\n"

    "vld1.16 {q8}, [r10], %[src_stride]\n"
    "vld1.16 {q10}, [r10], %[src_stride]\n"
    "vld1.16 {q12}, [r10], %[src_stride]\n"
    "vld1.16 {q14}, [r10], %[src_stride]\n"

    "vtrn.32 d0, d8\n"
    "vtrn.32 d4, d12\n"
    "vtrn.32 d1, d9\n"
    "vtrn.32 d5, d13\n"

    "vtrn.16 d16, d20\n"
    "vtrn.16 d17, d21\n"
    "vtrn.16 d24, d28\n"
    "vtrn.16 d25, d29\n"

    "vtrn.32 d16, d24\n"
    "vtrn.32 d20, d28\n"
    "vtrn.32 d17, d25\n"
    "vtrn.32 d21, d29\n"

    "vswp d1, d16\n"
    "vswp d5, d20\n"
    "vswp d9, d24\n"
    "vswp d13, d28\n"

    "vst1.16 {q0}, [r12], %[dst_stride]\n"
    "vst1.16 {q2}, [r12], %[dst_stride]\n"
    "vst1.16 {q4}, [r12], %[dst_stride]\n"
    "vst1.16 {q6}, [r12], %[dst_stride]\n"

    "vst1.16 {q8}, [r12], %[dst_stride]\n"
    "vst1.16 {q10}, [r12], %[dst_stride]\n"
    "vst1.16 {q12}, [r12], %[dst_stride]\n"
    "vst1.16 {q14}, [r12], %[dst_stride]\n"

    :
    : [ dst ] "r"(dst), [ src ] "r"(src), [ src_stride ] "r"(src_stride), [ dst_stride ] "r"(dst_stride)
    : "r10", "r12", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14",
      "q15");
}

inline void Transpose12x8A32Fp16(const float16_t *src_c, float16_t *dst_c, size_t src_stride, size_t dst_stride) {
  asm volatile(
    "mov r10, %[src_c]\n"
    "mov r12, %[dst_c]\n"

    "vld1.16 {q0}, [r10], %[src_stride]\n"
    "vld1.16 {q2}, [r10], %[src_stride]\n"
    "vld1.16 {q4}, [r10], %[src_stride]\n"
    "vld1.16 {q6}, [r10], %[src_stride]\n"

    "vtrn.16 d0, d4\n"
    "vtrn.16 d1, d5\n"
    "vtrn.16 d8, d12\n"
    "vtrn.16 d9, d13\n"

    "vld1.16 {q8}, [r10], %[src_stride]\n"
    "vld1.16 {q10}, [r10], %[src_stride]\n"
    "vld1.16 {q12}, [r10], %[src_stride]\n"
    "vld1.16 {q14}, [r10], %[src_stride]\n"

    "vtrn.32 d0, d8\n"
    "vtrn.32 d4, d12\n"
    "vtrn.32 d1, d9\n"
    "vtrn.32 d5, d13\n"

    "vtrn.16 d16, d20\n"
    "vtrn.16 d17, d21\n"
    "vtrn.16 d24, d28\n"
    "vtrn.16 d25, d29\n"

    "vld1.16 {q1}, [r10], %[src_stride]\n"
    "vld1.16 {q3}, [r10], %[src_stride]\n"
    "vld1.16 {q5}, [r10], %[src_stride]\n"
    "vld1.16 {q7}, [r10], %[src_stride]\n"

    "vtrn.32 d16, d24\n"
    "vtrn.32 d20, d28\n"
    "vtrn.32 d17, d25\n"
    "vtrn.32 d21, d29\n"

    "vswp d1, d16\n"
    "vswp d5, d20\n"
    "vswp d9, d24\n"
    "vswp d13, d28\n"

    "vtrn.16 d2, d6\n"
    "vtrn.16 d3, d7\n"
    "vtrn.16 d10, d14\n"
    "vtrn.16 d11, d15\n"

    "vtrn.32 d2, d10\n"
    "vtrn.32 d6, d14\n"
    "vtrn.32 d3, d11\n"
    "vtrn.32 d7, d15\n"

    "vst1.16 {q0, d2}, [r12], %[dst_stride]\n"
    "vst1.16 {q2, d6}, [r12], %[dst_stride]\n"
    "vst1.16 {q4, d10}, [r12], %[dst_stride]\n"
    "vst1.16 {q6, d14}, [r12], %[dst_stride]\n"

    "vswp d3, d18\n"
    "vswp d7, d22\n"
    "vswp d11, d26\n"
    "vswp d15, d30\n"

    "vst1.16 {q8, d18}, [r12], %[dst_stride]\n"
    "vst1.16 {q10, d22}, [r12], %[dst_stride]\n"
    "vst1.16 {q12, d26}, [r12], %[dst_stride]\n"
    "vst1.16 {q14, d30}, [r12], %[dst_stride]\n"

    :
    : [ dst_c ] "r"(dst_c), [ src_c ] "r"(src_c), [ src_stride ] "r"(src_stride), [ dst_stride ] "r"(dst_stride)
    : "r10", "r12", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14",
      "q15");
}
#endif

#ifdef ENABLE_ARM64
inline void Transpose4x8ARM64Fp16(const float16_t *src_ptr, float16_t *dst_ptr, size_t src_stride, size_t dst_stride) {
  dst_stride += dst_stride;
  asm volatile(
    "mov x10, %[src_ptr]\n"
    "mov x11, %[dst_ptr]\n"

    "ld1 {v0.8h}, [x10], %[src_stride]\n"
    "ld1 {v1.8h}, [x10], %[src_stride]\n"
    "ld1 {v2.8h}, [x10], %[src_stride]\n"
    "ld1 {v3.8h}, [x10], %[src_stride]\n"
    "add x10, x11, %[dst_stride]\n"

    "zip1 v4.8h, v0.8h, v1.8h\n"
    "zip1 v5.8h, v2.8h, v3.8h\n"

    "trn1 v6.4s, v4.4s, v5.4s\n"
    "trn2 v7.4s, v4.4s, v5.4s\n"

    "trn1 v24.2d, v6.2d, v7.2d\n"
    "trn2 v25.2d, v6.2d, v7.2d\n"

    "zip2 v8.8h, v0.8h, v1.8h\n"
    "zip2 v9.8h, v2.8h, v3.8h\n"

    "trn1 v10.4s, v8.4s, v9.4s\n"
    "trn2 v11.4s, v8.4s, v9.4s\n"

    "trn1 v26.2d, v10.2d, v11.2d\n"
    "trn2 v27.2d, v10.2d, v11.2d\n"

    "st1 {v24.8h}, [x11], %[tow_dst_stride]\n"
    "st1 {v25.8h}, [x10], %[tow_dst_stride]\n"
    "st1 {v26.8h}, [x11], %[tow_dst_stride]\n"
    "st1 {v27.8h}, [x10], %[tow_dst_stride]\n"
    :
    : [ dst_ptr ] "r"(dst_ptr), [ src_ptr ] "r"(src_ptr), [ src_stride ] "r"(src_stride),
      [ dst_stride ] "r"(dst_stride), [ tow_dst_stride ] "r"(2 * dst_stride)
    : "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v24", "v25", "v26",
      "v27");
}

inline void Transpose8x8ARM64Fp16(const float16_t *src_ptr, float16_t *dst_ptr, size_t src_stride, size_t dst_stride) {
  asm volatile(
    "mov x10, %[src_ptr]\n"
    "mov x11, %[dst_ptr]\n"

    "ld1 {v0.8h}, [x10], %[src_stride]\n"
    "ld1 {v1.8h}, [x10], %[src_stride]\n"
    "ld1 {v2.8h}, [x10], %[src_stride]\n"
    "ld1 {v3.8h}, [x10], %[src_stride]\n"
    "ld1 {v4.8h}, [x10], %[src_stride]\n"
    "ld1 {v5.8h}, [x10], %[src_stride]\n"
    "ld1 {v6.8h}, [x10], %[src_stride]\n"
    "ld1 {v7.8h}, [x10], %[src_stride]\n"
    "add x10, x11, %[dst_stride]\n"

    "zip1 v16.8h, v0.8h, v1.8h\n"
    "zip1 v17.8h, v2.8h, v3.8h\n"
    "zip1 v18.8h, v4.8h, v5.8h\n"
    "zip1 v19.8h, v6.8h, v7.8h\n"

    "trn1 v20.4s, v16.4s, v17.4s\n"
    "trn2 v21.4s, v16.4s, v17.4s\n"
    "trn1 v22.4s, v18.4s, v19.4s\n"
    "trn2 v23.4s, v18.4s, v19.4s\n"

    "trn1 v24.2d, v20.2d, v22.2d\n"
    "trn2 v26.2d, v20.2d, v22.2d\n"
    "trn1 v25.2d, v21.2d, v23.2d\n"
    "trn2 v27.2d, v21.2d, v23.2d\n"

    "zip2 v8.8h, v0.8h, v1.8h\n"
    "zip2 v9.8h, v2.8h, v3.8h\n"
    "zip2 v10.8h, v4.8h, v5.8h\n"
    "zip2 v11.8h, v6.8h, v7.8h\n"

    "trn1 v12.4s, v8.4s, v9.4s\n"
    "trn2 v13.4s, v8.4s, v9.4s\n"
    "trn1 v14.4s, v10.4s, v11.4s\n"
    "trn2 v15.4s, v10.4s, v11.4s\n"

    "trn1 v28.2d, v12.2d, v14.2d\n"
    "trn2 v30.2d, v12.2d, v14.2d\n"
    "trn1 v29.2d, v13.2d, v15.2d\n"
    "trn2 v31.2d, v13.2d, v15.2d\n"

    "st1 {v24.8h}, [x11], %[tow_dst_stride]\n"
    "st1 {v25.8h}, [x10], %[tow_dst_stride]\n"
    "st1 {v26.8h}, [x11], %[tow_dst_stride]\n"
    "st1 {v27.8h}, [x10], %[tow_dst_stride]\n"
    "st1 {v28.8h}, [x11], %[tow_dst_stride]\n"
    "st1 {v29.8h}, [x10], %[tow_dst_stride]\n"
    "st1 {v30.8h}, [x11], %[tow_dst_stride]\n"
    "st1 {v31.8h}, [x10], %[tow_dst_stride]\n"
    :
    : [ dst_ptr ] "r"(dst_ptr), [ src_ptr ] "r"(src_ptr), [ src_stride ] "r"(src_stride),
      [ dst_stride ] "r"(dst_stride), [ tow_dst_stride ] "r"(2 * dst_stride)
    : "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
      "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
      "v31");
}

void Transpose12x8ARM64Fp16(const float16_t *src_ptr, float16_t *dst_ptr, size_t src_stride, size_t dst_stride) {
#ifdef ENABLE_DEBUG
  for (int tr = 0; tr < C12NUM; tr++) {
    for (int tc = 0; tc < C8NUM; tc++) {
      dst_ptr[tc * C12NUM + tr] = src_ptr[tr * col + tc];
    }
  }
#else
  asm volatile(
    "mov x10, %[src_ptr]\n"
    "mov x11, %[dst_ptr]\n"

    "ld1 {v0.8h}, [x10], %[src_stride]\n"
    "ld1 {v1.8h}, [x10], %[src_stride]\n"
    "ld1 {v2.8h}, [x10], %[src_stride]\n"
    "ld1 {v3.8h}, [x10], %[src_stride]\n"
    "ld1 {v4.8h}, [x10], %[src_stride]\n"
    "ld1 {v5.8h}, [x10], %[src_stride]\n"
    "ld1 {v6.8h}, [x10], %[src_stride]\n"
    "ld1 {v7.8h}, [x10], %[src_stride]\n"

    "zip1 v16.8h, v0.8h, v1.8h\n"
    "zip1 v17.8h, v2.8h, v3.8h\n"
    "zip1 v18.8h, v4.8h, v5.8h\n"
    "zip1 v19.8h, v6.8h, v7.8h\n"

    "ld1 {v8.8h}, [x10], %[src_stride]\n"
    "ld1 {v9.8h}, [x10], %[src_stride]\n"
    "ld1 {v10.8h}, [x10], %[src_stride]\n"
    "ld1 {v11.8h}, [x10], %[src_stride]\n"

    "trn1 v20.4s, v16.4s, v17.4s\n"
    "trn2 v21.4s, v16.4s, v17.4s\n"
    "trn1 v22.4s, v18.4s, v19.4s\n"
    "trn2 v23.4s, v18.4s, v19.4s\n"

    "trn1 v24.2d, v20.2d, v22.2d\n"
    "trn2 v25.2d, v20.2d, v22.2d\n"
    "trn1 v26.2d, v21.2d, v23.2d\n"
    "trn2 v27.2d, v21.2d, v23.2d\n"

    "zip1 v16.8h, v8.8h, v9.8h\n"
    "zip1 v17.8h, v10.8h, v11.8h\n"

    "trn1 v20.4s, v16.4s, v17.4s\n"
    "trn2 v21.4s, v16.4s, v17.4s\n"

    "trn1 v28.2d, v20.2d, v20.2d\n"
    "trn2 v29.2d, v20.2d, v20.2d\n"
    "trn1 v30.2d, v21.2d, v21.2d\n"
    "trn2 v31.2d, v21.2d, v21.2d\n"

    "add x10, x11, #16\n"
    "st1 {v24.8h}, [x11], %[dst_stride]\n"
    "st1 {v28.4h}, [x10], %[dst_stride]\n"
    "st1 {v26.8h}, [x11], %[dst_stride]\n"
    "st1 {v30.4h}, [x10], %[dst_stride]\n"
    "st1 {v25.8h}, [x11], %[dst_stride]\n"
    "st1 {v29.4h}, [x10], %[dst_stride]\n"
    "st1 {v27.8h}, [x11], %[dst_stride]\n"
    "st1 {v31.4h}, [x10], %[dst_stride]\n"

    "zip2 v16.8h, v0.8h, v1.8h\n"
    "zip2 v17.8h, v2.8h, v3.8h\n"
    "zip2 v18.8h, v4.8h, v5.8h\n"
    "zip2 v19.8h, v6.8h, v7.8h\n"

    "trn1 v20.4s, v16.4s, v17.4s\n"
    "trn2 v21.4s, v16.4s, v17.4s\n"
    "trn1 v22.4s, v18.4s, v19.4s\n"
    "trn2 v23.4s, v18.4s, v19.4s\n"

    "trn1 v24.2d, v20.2d, v22.2d\n"
    "trn2 v25.2d, v20.2d, v22.2d\n"
    "trn1 v26.2d, v21.2d, v23.2d\n"
    "trn2 v27.2d, v21.2d, v23.2d\n"

    "zip2 v16.8h, v8.8h, v9.8h\n"
    "zip2 v17.8h, v10.8h, v11.8h\n"

    "trn1 v20.4s, v16.4s, v17.4s\n"
    "trn2 v21.4s, v16.4s, v17.4s\n"

    "trn1 v28.2d, v20.2d, v20.2d\n"
    "trn2 v29.2d, v20.2d, v20.2d\n"
    "trn1 v30.2d, v21.2d, v21.2d\n"
    "trn2 v31.2d, v21.2d, v21.2d\n"

    "st1 {v24.8h}, [x11], %[dst_stride]\n"
    "st1 {v28.4h}, [x10], %[dst_stride]\n"
    "st1 {v26.8h}, [x11], %[dst_stride]\n"
    "st1 {v30.4h}, [x10], %[dst_stride]\n"
    "st1 {v25.8h}, [x11], %[dst_stride]\n"
    "st1 {v29.4h}, [x10], %[dst_stride]\n"
    "st1 {v27.8h}, [x11], %[dst_stride]\n"
    "st1 {v31.4h}, [x10], %[dst_stride]\n"
    :
    : [ dst_ptr ] "r"(dst_ptr), [ src_ptr ] "r"(src_ptr), [ src_stride ] "r"(src_stride), [ dst_stride ] "r"(dst_stride)
    : "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18",
      "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#endif
}

inline void Transpose16x8ARM64Fp16(const float16_t *src_ptr, float16_t *dst_ptr, size_t src_stride, size_t dst_stride) {
  asm volatile(
    "mov x10, %[src_ptr]\n"
    "mov x11, %[dst_ptr]\n"

    "ld1 {v0.8h}, [x10], %[src_stride]\n"
    "ld1 {v1.8h}, [x10], %[src_stride]\n"
    "ld1 {v2.8h}, [x10], %[src_stride]\n"
    "ld1 {v3.8h}, [x10], %[src_stride]\n"
    "ld1 {v4.8h}, [x10], %[src_stride]\n"
    "ld1 {v5.8h}, [x10], %[src_stride]\n"
    "ld1 {v6.8h}, [x10], %[src_stride]\n"
    "ld1 {v7.8h}, [x10], %[src_stride]\n"

    "zip1 v16.8h, v0.8h, v1.8h\n"
    "zip1 v17.8h, v2.8h, v3.8h\n"
    "zip1 v18.8h, v4.8h, v5.8h\n"
    "zip1 v19.8h, v6.8h, v7.8h\n"

    "ld1 {v8.8h}, [x10], %[src_stride]\n"
    "ld1 {v9.8h}, [x10], %[src_stride]\n"
    "ld1 {v10.8h}, [x10], %[src_stride]\n"
    "ld1 {v11.8h}, [x10], %[src_stride]\n"
    "ld1 {v12.8h}, [x10], %[src_stride]\n"
    "ld1 {v13.8h}, [x10], %[src_stride]\n"
    "ld1 {v14.8h}, [x10], %[src_stride]\n"
    "ld1 {v15.8h}, [x10], %[src_stride]\n"

    "trn1 v20.4s, v16.4s, v17.4s\n"
    "trn2 v21.4s, v16.4s, v17.4s\n"
    "trn1 v22.4s, v18.4s, v19.4s\n"
    "trn2 v23.4s, v18.4s, v19.4s\n"

    "trn1 v24.2d, v20.2d, v22.2d\n"
    "trn2 v25.2d, v20.2d, v22.2d\n"
    "trn1 v26.2d, v21.2d, v23.2d\n"
    "trn2 v27.2d, v21.2d, v23.2d\n"

    "zip1 v16.8h, v8.8h, v9.8h\n"
    "zip1 v17.8h, v10.8h, v11.8h\n"
    "zip1 v18.8h, v12.8h, v13.8h\n"
    "zip1 v19.8h, v14.8h, v15.8h\n"

    "trn1 v20.4s, v16.4s, v17.4s\n"
    "trn2 v21.4s, v16.4s, v17.4s\n"
    "trn1 v22.4s, v18.4s, v19.4s\n"
    "trn2 v23.4s, v18.4s, v19.4s\n"

    "trn1 v28.2d, v20.2d, v22.2d\n"
    "trn2 v29.2d, v20.2d, v22.2d\n"
    "trn1 v30.2d, v21.2d, v23.2d\n"
    "trn2 v31.2d, v21.2d, v23.2d\n"

    "add x10, x11, #16\n"
    "st1 {v24.8h}, [x11], %[dst_stride]\n"
    "st1 {v28.8h}, [x10], %[dst_stride]\n"
    "st1 {v26.8h}, [x11], %[dst_stride]\n"
    "st1 {v30.8h}, [x10], %[dst_stride]\n"
    "st1 {v25.8h}, [x11], %[dst_stride]\n"
    "st1 {v29.8h}, [x10], %[dst_stride]\n"
    "st1 {v27.8h}, [x11], %[dst_stride]\n"
    "st1 {v31.8h}, [x10], %[dst_stride]\n"

    "zip2 v16.8h, v0.8h, v1.8h\n"
    "zip2 v17.8h, v2.8h, v3.8h\n"
    "zip2 v18.8h, v4.8h, v5.8h\n"
    "zip2 v19.8h, v6.8h, v7.8h\n"

    "trn1 v20.4s, v16.4s, v17.4s\n"
    "trn2 v21.4s, v16.4s, v17.4s\n"
    "trn1 v22.4s, v18.4s, v19.4s\n"
    "trn2 v23.4s, v18.4s, v19.4s\n"

    "trn1 v24.2d, v20.2d, v22.2d\n"
    "trn2 v25.2d, v20.2d, v22.2d\n"
    "trn1 v26.2d, v21.2d, v23.2d\n"
    "trn2 v27.2d, v21.2d, v23.2d\n"

    "zip2 v16.8h, v8.8h, v9.8h\n"
    "zip2 v17.8h, v10.8h, v11.8h\n"
    "zip2 v18.8h, v12.8h, v13.8h\n"
    "zip2 v19.8h, v14.8h, v15.8h\n"

    "trn1 v20.4s, v16.4s, v17.4s\n"
    "trn2 v21.4s, v16.4s, v17.4s\n"
    "trn1 v22.4s, v18.4s, v19.4s\n"
    "trn2 v23.4s, v18.4s, v19.4s\n"

    "trn1 v28.2d, v20.2d, v22.2d\n"
    "trn2 v29.2d, v20.2d, v22.2d\n"
    "trn1 v30.2d, v21.2d, v23.2d\n"
    "trn2 v31.2d, v21.2d, v23.2d\n"

    "st1 {v24.8h}, [x11], %[dst_stride]\n"
    "st1 {v28.8h}, [x10], %[dst_stride]\n"
    "st1 {v26.8h}, [x11], %[dst_stride]\n"
    "st1 {v30.8h}, [x10], %[dst_stride]\n"
    "st1 {v25.8h}, [x11], %[dst_stride]\n"
    "st1 {v29.8h}, [x10], %[dst_stride]\n"
    "st1 {v27.8h}, [x11], %[dst_stride]\n"
    "st1 {v31.8h}, [x10], %[dst_stride]\n"
    :
    : [ dst_ptr ] "r"(dst_ptr), [ src_ptr ] "r"(src_ptr), [ src_stride ] "r"(src_stride), [ dst_stride ] "r"(dst_stride)
    : "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
      "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
      "v31");
}
#endif
