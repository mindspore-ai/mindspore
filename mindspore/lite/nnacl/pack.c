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

#include <stdlib.h>
#include <string.h>
#include "nnacl/int8/conv_int8.h"
#include "nnacl/pack.h"

void PackWeightKHWToHWKFp32(const void *src, void *dst, int plane, int channel) {
  return PackNCHWToNHWCFp32(src, dst, 1, plane, channel);
}

void PackHWCToWHC(const float *src, float *dst, int height, int width, int channel) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      memcpy(dst + (j * height + i) * channel, src + (i * width + j) * channel, channel * sizeof(float));
    }
  }
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
  QuantArg *filter_args = conv_param->conv_quant_arg_.filter_quant_args_;

  for (int m = 0; m < kernel_plane; m++) {
    int kernel_plane_stride = m * in_channel;
    int plane_block = m / C4NUM;
    int plane_res = m % C4NUM;
    int packed_kernel_plane_stride = plane_block * C4NUM * C4NUM * ic4 * C4NUM + plane_res * C4NUM;
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
            int32_t f_zp;
            if (conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL) {
              f_zp = filter_args[j * C4NUM + k].zp_;
            } else {
              f_zp = filter_args[0].zp_;
            }
            weight_sum[j * C4NUM + k] += (int32_t)(packed_data_ptr[0] - f_zp);
          }
        }  // kernel block loop
      }    // inchannel block loop
    }      // channel block loop
  }        // kernel plane loop
}

void Conv1x1InputPack(const void *src_ptr, void *dst_ptr, ConvParameter *conv_param, int data_size) {
  /* support nhwc */
  char *src = (char *)src_ptr;
  char *dst = (char *)dst_ptr;
  for (int dst_h = 0; dst_h < conv_param->output_h_; dst_h++) {
    int src_h = dst_h * conv_param->stride_h_ - conv_param->pad_u_;
    if (src_h < 0 || src_h >= conv_param->input_h_) {
      continue;
    }
    const char *src_h_ptr = src + src_h * conv_param->input_w_ * conv_param->input_channel_ * data_size;
    char *dst_h_ptr = dst + dst_h * conv_param->output_w_ * conv_param->input_channel_ * data_size;
    for (int dst_w = 0; dst_w < conv_param->output_w_; dst_w++) {
      int src_w = dst_w * conv_param->stride_w_ - conv_param->pad_l_;
      if (src_w < 0 || src_w >= conv_param->input_w_) {
        continue;
      }
      memcpy(dst_h_ptr + dst_w * conv_param->input_channel_ * data_size,
             src_h_ptr + src_w * conv_param->input_channel_ * data_size, conv_param->input_channel_ * data_size);
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

#ifndef ENABLE_SSE
void PackNHWCToNCHWFp32(const void *src, void *dst, int batches, int plane, int channel) {
  int hw8 = plane / C8NUM * C8NUM;
  int c8 = channel / C8NUM * C8NUM;
  int batch = plane * channel;
  for (int n = 0; n < batches; n++) {
    const float *src_batch = (const float *)src + n * batch;
    float *dst_batch = (float *)dst + n * batch;
    int hw = 0;
    for (; hw < hw8; hw += C8NUM) {
      int c = 0;
      for (; c < c8; c += C8NUM) {
        const float *src_ptr = src_batch + hw * channel + c;
        float *dst_ptr = dst_batch + c * plane + hw;
#ifdef ENABLE_ARM64
        size_t srcStride = channel * sizeof(float);
        size_t dstStride = plane * sizeof(float);
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
          :
          [ dst_ptr ] "r"(dst_ptr), [ src_ptr ] "r"(src_ptr), [ srcStride ] "r"(srcStride), [ dstStride ] "r"(dstStride)
          : "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
            "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
            "v30", "v31");
#elif ENABLE_ARM32
        size_t srcStride = channel * sizeof(float);
        size_t dstStride = plane * sizeof(float);
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
        const float *src_ptr = src_batch + hw * channel + c;
        float *dst_ptr = dst_batch + c * plane + hw;
        for (size_t i = 0; i < C8NUM; i++) {
          dst_ptr[i] = src_ptr[i * channel];
        }
      }
    }
    for (; hw < plane; hw++) {
      const float *src_ptr = src_batch + hw * channel;
      float *dst_ptr = dst_batch + hw;
      for (size_t i = 0; i < channel; i++) {
        dst_ptr[i * plane] = src_ptr[i];
      }
    }
  }
  return;
}
#endif

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

void PackNCHWToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel) {
  return PackNHWCToNCHWFp32(src, dst, batch, channel, plane);
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
