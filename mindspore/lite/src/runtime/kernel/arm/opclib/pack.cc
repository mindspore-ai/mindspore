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

#include "src/runtime/kernel/arm/opclib/pack.h"
#include <cstring>
#include <cstdlib>

void PackWeightFp32(float *weight_data, ConvParameter *conv_param, float *packed_weight) {
  // original weight format : ohwi
  // todo pack weight for arm32 platform
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int in_channel = conv_param->input_channel_;
  int out_channel = conv_param->output_channel_;
  int oc8 = UP_DIV(out_channel, C8NUM);
  int ic4 = UP_DIV(in_channel, C4NUM);
  int kernel_plane = kernel_h * kernel_w;
  int pack_weight_size = oc8 * C8NUM * ic4 * C4NUM * kernel_plane;

  int unit_size = C8NUM * C4NUM;
  int block_size = pack_weight_size / oc8;

  for (int m = 0; m < kernel_plane; m++) {
    int kernel_plane_stride = m * in_channel;
    int packed_kernel_plane_stride = m * unit_size * ic4;
    for (int i = 0; i < ic4; i++) {
      int channel_block_stride = kernel_plane_stride + i * C4NUM;
      int packed_channel_block_size = packed_kernel_plane_stride + i * unit_size;
      int ic_remainder = in_channel - i * C4NUM;
      int real_ic_num = ic_remainder < C4NUM ? ic_remainder : C4NUM;
      for (int h = 0; h < real_ic_num; h++) {
        int block_stride = channel_block_stride + h;
        int packed_block_stride = packed_channel_block_size + h * C8NUM;
        for (int j = 0; j < oc8; j++) {
          int kernel_block_stride = block_stride + j * C8NUM * kernel_plane * in_channel;
          int packed_kernel_block_size = packed_block_stride + j * block_size;
          int oc_remainder = out_channel - j * C8NUM;
          int real_oc_num = oc_remainder < C8NUM ? oc_remainder : C8NUM;
          for (int k = 0; k < real_oc_num; k++) {
            float *origin_data_ptr = weight_data + kernel_block_stride + k * kernel_plane * in_channel;
            float *packed_data_ptr = packed_weight + packed_kernel_block_size + k;
            *packed_data_ptr = *origin_data_ptr;
          }
        }  // kernel block loop
      }    // inchannel block loop
    }      // channel block loop
  }        // kernel plane loop
}

void PackWeightInt8(int8_t *weight_data, ConvParameter *conv_param, int8_t *packed_weight, int32_t *weight_sum) {
  // original weight format : ohwi
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int in_channel = conv_param->input_channel_;
  int out_channel = conv_param->output_channel_;
  int oc4 = UP_DIV(out_channel, C4NUM);
  int ic4 = UP_DIV(in_channel, C4NUM);
  int kernel_plane = kernel_h * kernel_w;
  int plane_c4 = UP_DIV(kernel_plane, C4NUM);
  int pack_weight_size = oc4 * C4NUM * ic4 * C4NUM * plane_c4 * C4NUM;
  int block_size = pack_weight_size / oc4;

  for (int m = 0; m < kernel_plane; m++) {
    int kernel_plane_stride = m * in_channel;
    int packed_kernel_plane_stride = m * C4NUM;
    for (int i = 0; i < ic4; i++) {
      int channel_block_stride = kernel_plane_stride + i * C4NUM;
      int packed_channel_block_size = packed_kernel_plane_stride + i * C4NUM * C4NUM * C4NUM;
      int ic_remainder = in_channel - i * C4NUM;
      int real_ic_num = ic_remainder < C4NUM ? ic_remainder : C4NUM;
      for (int h = 0; h < real_ic_num; h++) {
        int block_stride = channel_block_stride + h;
        int packed_block_stride = packed_channel_block_size + h;
        for (int j = 0; j < oc4; j++) {
          int kernel_block_stride = block_stride + j * C4NUM * kernel_plane * in_channel;
          int packed_kernel_block_size = packed_block_stride + j * block_size;
          int oc_remainder = out_channel - j * C4NUM;
          int real_oc_num = oc_remainder < C4NUM ? oc_remainder : C4NUM;
          for (int k = 0; k < real_oc_num; k++) {
            int8_t *origin_data_ptr = weight_data + kernel_block_stride + k * kernel_plane * in_channel;
            int8_t *packed_data_ptr = packed_weight + packed_kernel_block_size + k * C4NUM * C4NUM;
            *packed_data_ptr = origin_data_ptr[0];
            // value of weight must between [-127, 127]
            if (packed_data_ptr[0] == -128) {
              packed_data_ptr[0] = -127;
            }
            weight_sum[j * C4NUM + k] += (int32_t)packed_data_ptr[0];
          }
        }  // kernel block loop
      }    // inchannel block loop
    }      // channel block loop
  }        // kernel plane loop
}

void PackWeightInt8Opt(int8_t *weight_data, ConvParameter *conv_param, int8_t *packed_weight, int32_t *weight_sum) {
  // original weight format : ohwi
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int in_channel = conv_param->input_channel_;
  int out_channel = conv_param->output_channel_;
  int oc4 = UP_DIV(out_channel, C4NUM);
  int ic4 = UP_DIV(in_channel, C4NUM);
  int kernel_plane = kernel_h * kernel_w;
  int32_t filter_zp = conv_param->conv_quant_arg_.quant_args_[1][0].zp_;
  int pack_weight_size = oc4 * ic4 * C4NUM * C4NUM * kernel_plane;
  int unit_size = C4NUM * C4NUM;
  int block_size = pack_weight_size / oc4;

  for (int m = 0; m < kernel_plane; m++) {
    int kernel_plane_stride = m * in_channel;
    int packed_kernel_plane_stride = m * unit_size * ic4;
    for (int i = 0; i < ic4; i++) {
      int channel_block_stride = kernel_plane_stride + i * C4NUM;
      int packed_channel_block_size = packed_kernel_plane_stride + i * unit_size;
      int ic_remainder = in_channel - i * C4NUM;
      int real_ic_num = ic_remainder < C4NUM ? ic_remainder : C4NUM;
      for (int h = 0; h < real_ic_num; h++) {
        int block_stride = channel_block_stride + h;
        int packed_block_stride = packed_channel_block_size + h;
        for (int j = 0; j < oc4; j++) {
          int kernel_block_stride = block_stride + j * C4NUM * kernel_plane * in_channel;
          int packed_kernel_block_size = packed_block_stride + j * block_size;
          int oc_remainder = out_channel - j * C4NUM;
          int real_oc_num = oc_remainder < C4NUM ? oc_remainder : C4NUM;
          for (int k = 0; k < real_oc_num; k++) {
            int8_t *origin_data_ptr = weight_data + kernel_block_stride + k * kernel_plane * in_channel;
            int8_t *packed_data_ptr = packed_weight + packed_kernel_block_size + k * C4NUM;
            *packed_data_ptr = origin_data_ptr[0];
            if (packed_data_ptr[0] == -128) {
              packed_data_ptr[0] = -127;
            }
            weight_sum[j * C4NUM + k] += (int32_t)(packed_data_ptr[0] - filter_zp);
          }
        }  // kernel block loop
      }    // inchannel block loop
    }      // channel block loop
  }        // kernel plane loop
}

void Conv1x1InputPackFp32(const float *src, float *dst, ConvParameter *conv_param) {
  for (int c = 0; c < UP_DIV(conv_param->input_channel_, C4NUM); c++) {
    const float *src_c_ptr = src + c * conv_param->input_h_ * conv_param->input_w_ * C4NUM;
    float *dst_c_ptr = dst + c * conv_param->output_h_ * conv_param->output_w_ * C4NUM;
    for (int dst_h = 0; dst_h < conv_param->output_h_; dst_h++) {
      int src_h = dst_h * conv_param->stride_h_ - conv_param->pad_h_;
      if (src_h < 0 || src_h >= conv_param->input_h_) {
        continue;
      }
      const float *src_h_ptr = src_c_ptr + src_h * conv_param->input_w_ * C4NUM;
      float *dst_h_ptr = dst_c_ptr + dst_h * conv_param->output_w_ * C4NUM;
      for (int dst_w = 0; dst_w < conv_param->output_w_; dst_w++) {
        int src_w = dst_w * conv_param->stride_w_ - conv_param->pad_w_;
        if (src_w < 0 || src_w >= conv_param->input_w_) {
          continue;
        }
        memcpy(dst_h_ptr + dst_w * C4NUM, src_h_ptr + src_w * C4NUM, C4NUM * sizeof(float));
      }
    }
  }
  return;
}

void Pack1x1WeightFp32(const float *weight_data, float *packed_weight, ConvParameter *conv_param) {
  int c4 = UP_ROUND(conv_param->input_channel_, C4NUM);
  for (int ic = 0; ic < conv_param->input_channel_; ic++) {
    for (int oc = 0; oc < conv_param->output_channel_; oc++) {
      int oc4mod = oc % 4;
      int oc4div = oc / 4;
      int dst_index = oc4div * c4 * C4NUM + ic * C4NUM + oc4mod;
      int src_index = oc * conv_param->input_channel_ + ic;
      packed_weight[dst_index] = weight_data[src_index];
    }
  }
  return;
}

void Im2ColPackUnitFp32(const float *input_data, ConvParameter *conv_param, float *packed_input, int real_cal_num,
                        int block_index) {
  // input format : nhwc
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int stride_h = conv_param->stride_h_;
  int stride_w = conv_param->stride_w_;
  int pad_h = conv_param->pad_h_;
  int pad_w = conv_param->pad_w_;
  int dilation_h = conv_param->dilation_h_;
  int dilation_w = conv_param->dilation_w_;
  int in_channel = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int out_w = conv_param->output_w_;
  int ic4 = UP_DIV(in_channel, C4NUM);

  for (int i = 0; i < real_cal_num; i++) {
    int block_start = block_index + i;
    int input_h = block_start / out_w * stride_h - pad_h;
    int input_w = block_start % out_w * stride_w - pad_w;
    for (int j = 0; j < kernel_h; j++) {
      int input_y = input_h + j * dilation_h;
      if (input_y < 0 || input_y >= in_h) {
        continue;
      }
      int input_y_stride = input_y * in_w * ic4 * C4NUM;
      for (int n = 0; n < kernel_w; n++) {
        int input_x = input_w + n * dilation_w;
        if (input_x < 0 || input_x >= in_w) {
          continue;
        }
        int input_x_stride = input_y_stride + input_x * ic4 * C4NUM;
        int input_plane_offset = (j * kernel_w + n) * C8NUM * C4NUM * ic4 + i * C4NUM;
        for (int m = 0; m < ic4; m++) {
          int channel_block_stride = input_x_stride + m * C4NUM;
          int channel_block_offset = input_plane_offset + m * C8NUM * C4NUM;
#ifdef ENABLE_NEON
          vst1q_f32(packed_input + channel_block_offset, vld1q_f32(input_data + channel_block_stride));
#else
          (packed_input + channel_block_offset)[0] = (input_data + channel_block_stride)[0];
          (packed_input + channel_block_offset)[1] = (input_data + channel_block_stride)[1];
          (packed_input + channel_block_offset)[2] = (input_data + channel_block_stride)[2];
          (packed_input + channel_block_offset)[3] = (input_data + channel_block_stride)[3];
#endif
        }  // channel_block loop
      }    // kernel_w loop
    }      // kernel_h loop
  }        // tile num loop
}

void Im2ColPackUnitInt8(const int8_t *input_data, int8_t *packed_input, int real_cal_num, int block_index,
                        int32_t *input_sum, ConvParameter *conv_param) {
  // input format : nhwc
  int tile_num = conv_param->tile_num_;
  int32_t filter_zp = conv_param->conv_quant_arg_.quant_args_[1][0].zp_;
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int stride_h = conv_param->stride_h_;
  int stride_w = conv_param->stride_w_;
  int pad_h = conv_param->pad_h_;
  int pad_w = conv_param->pad_w_;
  int dilation_h = conv_param->dilation_h_;
  int dilation_w = conv_param->dilation_w_;
  int in_channel = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int ic4 = UP_DIV(in_channel, C4NUM);
  int out_w = conv_param->output_w_;

  for (int i = 0; i < real_cal_num; i++) {
    int block_start = block_index + i;
    int input_h = block_start / out_w * stride_h - pad_h;
    int input_w = block_start % out_w * stride_w - pad_w;
    int input_cal_num_offset = i * C4NUM * C4NUM;
    int32_t input_accumulator = 0;
    for (int j = 0; j < kernel_h; j++) {
      int input_y = input_h + j * dilation_h;
      if (input_y < 0 || input_y >= in_h) {
        continue;
      }
      int input_y_stride = input_y * in_w * ic4 * C4NUM;
      for (int n = 0; n < kernel_w; n++) {
        int input_x = input_w + n * dilation_w;
        if (input_x < 0 || input_x >= in_w) {
          continue;
        }
        int input_x_stride = input_y_stride + input_x * ic4 * C4NUM;
        int plane_c4_block = (j * kernel_w + n) / C4NUM;
        int plane_c4_res = (j * kernel_w + n) % C4NUM;
        int input_plane_offset =
          plane_c4_block * tile_num * C4NUM * C4NUM * ic4 + plane_c4_res * C4NUM + input_cal_num_offset;
        for (int m = 0; m < ic4; m++) {
          int channel_block_stride = input_x_stride + m * C4NUM;
          int channel_block_offset = input_plane_offset + m * tile_num * C4NUM * C4NUM;
          (packed_input + channel_block_offset)[0] = (input_data + channel_block_stride)[0];
          (packed_input + channel_block_offset)[1] = (input_data + channel_block_stride)[1];
          (packed_input + channel_block_offset)[2] = (input_data + channel_block_stride)[2];
          (packed_input + channel_block_offset)[3] = (input_data + channel_block_stride)[3];
          input_accumulator += (packed_input + channel_block_offset)[0];
          input_accumulator += (packed_input + channel_block_offset)[1];
          input_accumulator += (packed_input + channel_block_offset)[2];
          input_accumulator += (packed_input + channel_block_offset)[3];
        }  // channel_block loop
      }    // kernel_w loop
    }      // kernel_h loop
    input_sum[i] = input_accumulator * filter_zp;
  }  // tile num loop
}

void Im2ColPackUnitInt8Opt(const int8_t *input_data, int8_t *packed_input, int real_cal_num, int block_index,
                           int32_t *input_sum, ConvParameter *conv_param) {
  // input format : nhwc
  int tile_num = conv_param->tile_num_;
  int32_t filter_zp = conv_param->conv_quant_arg_.quant_args_[1][0].zp_;
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int stride_h = conv_param->stride_h_;
  int stride_w = conv_param->stride_w_;
  int pad_h = conv_param->pad_h_;
  int pad_w = conv_param->pad_w_;
  int dilation_h = conv_param->dilation_h_;
  int dilation_w = conv_param->dilation_w_;
  int in_channel = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int ic4 = UP_DIV(in_channel, C4NUM);
  int out_w = conv_param->output_w_;
  int block_size = kernel_h * kernel_w;

  for (int i = 0; i < real_cal_num; i++) {
    int block_start = block_index + i;
    int input_h = block_start / out_w * stride_h - pad_h;
    int input_w = block_start % out_w * stride_w - pad_w;
    for (int j = 0; j < kernel_h; j++) {
      int input_y = input_h + j * dilation_h;
      if (input_y < 0 || input_y >= in_h) {
        continue;
      }
      int input_y_stride = input_y * in_w * ic4 * C4NUM;
      for (int n = 0; n < kernel_w; n++) {
        int input_x = input_w + n * dilation_w;
        if (input_x < 0 || input_x >= in_w) {
          continue;
        }
        int input_x_stride = input_y_stride + input_x * ic4 * C4NUM;
        int input_plane_offset = (j * kernel_w + n) * tile_num * C4NUM * ic4 + i * C4NUM;
        for (int m = 0; m < ic4; m++) {
          int channel_block_stride = input_x_stride + m * C4NUM;
          int channel_block_offset = input_plane_offset + m * tile_num * C4NUM;
          (packed_input + channel_block_offset)[0] = (input_data + channel_block_stride)[0];
          (packed_input + channel_block_offset)[1] = (input_data + channel_block_stride)[1];
          (packed_input + channel_block_offset)[2] = (input_data + channel_block_stride)[2];
          (packed_input + channel_block_offset)[3] = (input_data + channel_block_stride)[3];
        }  // channel_block loop
      }    // kernel_w loop
    }      // kernel_h loop
    int32_t input_accumulator = 0;
    for (int j = 0; j < block_size; j++) {
      int block_offset = j * tile_num * ic4 * C4NUM + i * C4NUM;
      for (int c = 0; c < ic4; c++) {
        int ic4_offset = block_offset + c * tile_num * C4NUM;
        input_accumulator += (packed_input + ic4_offset)[0];
        input_accumulator += (packed_input + ic4_offset)[1];
        input_accumulator += (packed_input + ic4_offset)[2];
        input_accumulator += (packed_input + ic4_offset)[3];
      }
    }
    input_sum[i] = input_accumulator * filter_zp;
  }  // tile num loop
}

void PackInputToC8Int8(const int8_t *input_data, int16_t *packed_input, ConvParameter *conv_param) {
  int in_batch = conv_param->input_batch_;
  int in_channel = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int ic8 = UP_DIV(in_channel, C8NUM);

  for (int b = 0; b < in_batch; b++) {
    int src_batch_offset = b * in_channel * in_h * in_w;
    int dst_batch_offset = b * ic8 * C8NUM * in_h * in_w;
    for (int c = 0; c < in_channel; c++) {
      int ic8_block = c / C8NUM;
      int ic8_res = c % C8NUM;
      int src_c_offset = src_batch_offset + c;
      int dst_c_offset = dst_batch_offset + ic8_block * C8NUM * in_h * in_w + ic8_res;
      for (int k = 0; k < in_w * in_h; k++) {
        int src_plane_offset = src_c_offset + k * in_channel;
        int dst_plane_offset = dst_c_offset + k * C8NUM;
        (packed_input + dst_plane_offset)[0] = (int16_t)(input_data + src_plane_offset)[0];
      }
    }
  }
}

void PackWeightToC8Int8(const int8_t *origin_weight_data, int16_t *packed_weight_data, ConvParameter *conv_param) {
  // origin weight format : ohwi
  int input_channel = conv_param->input_channel_;
  int ic8 = UP_DIV(input_channel, C8NUM);
  int output_channel = conv_param->output_channel_;
  int filter_zp = conv_param->conv_quant_arg_.quant_args_[1][0].zp_;
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
        (packed_weight_data + dst_ic_offset)[0] = (int16_t)((origin_weight_data + src_ic_offset)[0] - filter_zp);
      }
    }
  }
}

void PackNHWCToNC4HW4Fp32(const void *src, void *dst, int batch, int plane, int channel) {
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
  int nhwc4_batch_unit_offset = c4 * C4NUM * plane;
  int ic_remainder_ = channel % C4NUM;
  if (ic_remainder_ != 0) {
    int nhwc4_batch_offset = 0;
    for (int b = 0; b < batch; b++) {
      int batch_offset = b * channel * plane;
      for (int i = 0; i < plane; i++) {
        memcpy((float *)dst + nhwc4_batch_offset + i * c4 * C4NUM, (float *)src + batch_offset + i * channel,
               channel * sizeof(float));
      }
      nhwc4_batch_offset += nhwc4_batch_unit_offset;
    }
  } else {
    size_t ori_input_size = batch * plane * channel * sizeof(float);
    memcpy(dst, src, ori_input_size);
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
    memcpy(dst, src, ori_input_size);
  }
}

void PackNCHWToNHWC4Fp32(const void *src, void *dst, int batch, int plane, int channel) {
  int nhwc4_batch_offset = 0;
  int c4 = UP_DIV(channel, C4NUM);
  int nhwc4_batch_unit_offset = c4 * C4NUM * plane;

  for (int b = 0; b < batch; b++) {
    int batch_offset = b * channel * plane;
    for (int c = 0; c < channel; c++) {
      int src_c_offset = batch_offset + c * plane;
      int dst_c_offset = nhwc4_batch_offset + c;
      for (int i = 0; i < plane; i++) {
        int src_plane_offset = src_c_offset + i;
        int dst_plane_offset = dst_c_offset + i * c4 * C4NUM;
        ((float *)dst)[dst_plane_offset] = ((float *)src)[src_plane_offset];
      }
    }
    nhwc4_batch_offset += nhwc4_batch_unit_offset;
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

void PackNC4HW4ToNCHWFp32(const void *src, void *dst, int batch, int plane, int channel) {
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
        ((float *)dst + dst_kernel_offset)[0] = ((float *)src + src_kernel_offset)[0];
      }
    }
  }
}

void PackNHWCToNHWC4Int8(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  int nhwc4_batch_unit_offset = c4 * C4NUM * plane;
  int ic_remainder_ = channel % C4NUM;
  if (ic_remainder_ != 0) {
    int nhwc4_batch_offset = 0;
    for (int b = 0; b < batch; b++) {
      int batch_offset = b * channel * plane;
      for (int i = 0; i < plane; i++) {
        memcpy((int8_t *)dst + nhwc4_batch_offset + i * c4 * C4NUM, (int8_t *)src + batch_offset + i * channel,
               channel);
      }
      nhwc4_batch_offset += nhwc4_batch_unit_offset;
    }
  } else {
    size_t ori_input_size = batch * plane * channel;
    memcpy(dst, src, ori_input_size);
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
        memcpy(reinterpret_cast<int8_t *>(dst) + batch_offset + i * channel,
               reinterpret_cast<const int8_t *>(src) + nhwc4_batch_offset + i * c4 * C4NUM, channel);
      }
    }
  } else {
    size_t ori_input_size = batch * plane * channel;
    memcpy(dst, src, ori_input_size);
  }
}

void PackNCHWToNHWC4Int8(const void *src, void *dst, int batch, int plane, int channel) {
  int nhwc4_batch_offset = 0;
  int c4 = UP_DIV(channel, C4NUM);
  int nhwc4_batch_unit_offset = c4 * C4NUM * plane;

  for (int b = 0; b < batch; b++) {
    int batch_offset = b * channel * plane;
    for (int c = 0; c < channel; c++) {
      int src_c_offset = batch_offset + c * plane;
      int dst_c_offset = nhwc4_batch_offset + c;
      for (int i = 0; i < plane; i++) {
        int src_plane_offset = src_c_offset + i;
        int dst_plane_offset = dst_c_offset + i * c4 * C4NUM;
        ((uint8_t *)dst)[dst_plane_offset] = ((uint8_t *)src)[src_plane_offset];
      }
    }
    nhwc4_batch_offset += nhwc4_batch_unit_offset;
  }
}

void PackNC4HW4ToNHWC4Int8(const void *src, void *dst, int batch, int plane, int channel) {
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
        ((uint8_t *)dst + dst_kernel_offset)[0] = ((uint8_t *)src + src_kernel_offset)[0];
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

void PackNC4HW4ToNCHWInt8(const void *src, void *dst, int batch, int plane, int channel) {
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
        ((uint8_t *)dst + dst_kernel_offset)[0] = ((uint8_t *)src + src_kernel_offset)[0];
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

void PackNHWCToNC8HW8Int8(const void *src, void *dst, int batch, int plane, int channel) {
  int c8 = UP_DIV(channel, C8NUM);
  for (int b = 0; b < batch; b++) {
    int src_oc_offset = b * plane * channel;
    int dst_oc_offset = b * plane * c8 * C8NUM;
    for (int k = 0; k < plane; k++) {
      int src_kernel_offset = src_oc_offset + k * channel;
      int dst_kernel_offset = dst_oc_offset + k * C8NUM;
      for (int i = 0; i < channel; i++) {
        int c8_block_num = i / C8NUM;
        int c8_block_rem = i % C8NUM;
        int src_ic_offset = src_kernel_offset + i;
        int dst_ic_offset = dst_kernel_offset + c8_block_num * plane * C8NUM + c8_block_rem;
        ((int8_t *)dst + dst_ic_offset)[0] = ((int8_t *)src + src_ic_offset)[0];
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

void PackNHWCToNCHWFp32(const void *src, void *dst, int batch, int plane, int channel) {
  for (int n = 0; n < batch; n++) {
    for (int c = 0; c < channel; c++) {
      for (int hw = 0; hw < plane; hw++) {
        int nhwc_index = n * channel * plane + hw * channel + c;
        int nchw_index = n * channel * plane + c * plane + hw;
        ((float *)dst)[nchw_index] = ((float *)src)[nhwc_index];
      }
    }
  }
  return;
}

void PackNCHWToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel) {
  for (int n = 0; n < batch; n++) {
    for (int c = 0; c < channel; c++) {
      for (int hw = 0; hw < plane; hw++) {
        int nhwc_index = n * channel * plane + hw * channel + c;
        int nchw_index = n * channel * plane + c * plane + hw;
        ((float *)dst)[nhwc_index] = ((float *)src)[nchw_index];
      }
    }
  }
  return;
}

void MatrixPackUnit(const float *src, float *dst, size_t row, size_t col, size_t src_stride, size_t dst_stride) {
  size_t copy_size = row * C4NUM * sizeof(float);
  for (int c = 0; c < col; c++) {
    memcpy(dst + c * dst_stride, src + c * src_stride, copy_size);
  }
}

void MatrixPack(const float *src, float *dst, int row, int ic4, int stride) {
  int row4mod = row % 4;
  int row4div = row / 4;

  for (int i = 0; i < row4div; i++) {
    MatrixPackUnit(src + i * 4 * 4, dst + i * 4 * ic4 * 4, 4, ic4, stride, 16);
  }

  if (row4mod > 0) {
    MatrixPackUnit(src + row4div * 4 * 4, dst + row4div * 4 * ic4 * 4, row4mod, ic4, stride, row4mod * 4);
  }
  return;
}

void PackDepthwiseInt8Input(const int8_t *src, int16_t *dst, const ConvParameter *conv_param) {
  auto input_zp = conv_param->conv_quant_arg_.quant_args_[0][0].zp_;
  int ic4 = UP_DIV(conv_param->input_channel_, C4NUM);
  int unit = conv_param->input_h_ * conv_param->input_w_;

  for (int b = 0; b < conv_param->input_batch_; b++) {
    auto src_b = src + b * unit * conv_param->input_channel_;
    auto dst_b = dst + b * unit * ic4 * C4NUM;
    for (int k = 0; k < unit; k++) {
      auto src_k = src_b + k * conv_param->input_channel_;
      auto dst_k = dst_b + k * ic4 * C4NUM;
      for (int c = 0; c < conv_param->input_channel_; c++) {
        dst_k[c] = (int16_t)(src_k[c] - input_zp);
      }
    }
  }
}

void PackDepthwiseInt8Weight(const int8_t *origin_weight, int16_t *packed_weight_, const ConvParameter *conv_param) {
  auto weight_zp = conv_param->conv_quant_arg_.quant_args_[1][0].zp_;
  int unit = conv_param->kernel_h_ * conv_param->kernel_w_;
  for (int c = 0; c < conv_param->output_channel_; c++) {
    int c4_block_num = c / C4NUM;
    int c4_block_rem = c % C4NUM;
    auto src_c = origin_weight + c * unit;
    auto dst_c = packed_weight_ + c4_block_num * unit * C4NUM;
    for (int k = 0; k < unit; k++) {
      auto src_kernel = src_c + k;
      auto dst_kernel = dst_c + C4NUM * k + c4_block_rem;
      *dst_kernel = (int16_t)(src_kernel[0] - weight_zp);
    }
  }
}
