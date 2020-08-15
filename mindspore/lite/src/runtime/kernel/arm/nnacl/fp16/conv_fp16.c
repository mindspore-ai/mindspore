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
#include "nnacl/fp16/conv_fp16.h"
#include <string.h>
#include "nnacl/fp16/pack_fp16.h"
#include "nnacl/fp16/winograd_transform_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifdef ENABLE_ARM64
void IndirectGemmFp16_16x8(float16_t *output, float16_t *input, float16_t *weight, float16_t *bias, size_t step,
                           size_t ic4, size_t oc8, size_t offset, size_t mode, size_t writeC4, size_t relu,
                           size_t relu6);
#endif

#ifdef __cplusplus
}
#endif
#ifndef ENABLE_NEON
void IndirectGemmFp16_16x8(float16_t *output, float16_t *input, float16_t *weight, float16_t *bias, size_t step,
                           size_t ic4, size_t out_channel, size_t offset, size_t mode, size_t writeC8, size_t relu,
                           size_t relu6) {
  if (!(mode && writeC8)) {
    IndirectGemmFp16_16x8_common(output, input, weight, bias, step, ic4, output, offset, relu, relu6);
  } else {
    IndirectGemmFp16_16x8_c8(output, input, weight, bias, step, ic4, output, offset, mode, writeC8, relu, relu6);
  }
}

void IndirectGemmFp16_16x8_common(float16_t *output, float16_t *input, float16_t *weight, float16_t *bias, size_t step,
                                  size_t ic4, size_t oc8, size_t offset, size_t relu, size_t relu6) {
  const int tile_n = 16;
  for (int i = 0; i < out_channel; i++) {
    int oc8_block = i / C8NUM;
    int oc8_res = i % C8NUM;
    int weight_oc_offset = oc8_block * step * ic4 * C4NUM * C8NUM + oc8_res;
    for (int k = 0; k < tile_n; k++) {
      int input_tile_offset = k * C4NUM;
      int out_tile_offset = i + k * out_channel;

      float16_t tmp_out = 0;
      for (int n = 0; n < step; n++) {
        int input_kw_offset = input_tile_offset + n * tile_n * ic4 * C4NUM;
        int weight_kw_offset = weight_oc_offset + n * ic4 * C4NUM * C8NUM;
        for (int j = 0; j < ic4; j++) {
          int input_ic4_offset = input_kw_offset + j * tile_n * C4NUM;
          int weight_ic4_offset = weight_kw_offset + j * C4NUM * C8NUM;
          for (int m = 0; m < C4NUM; m++) {
            int input_c4_offset = input_ic4_offset + m;
            int weight_c4_offset = weight_ic4_offset + m * C8NUM;
            tmp_out += (input + input_c4_offset)[0] * (weight + weight_c4_offset)[0];
          }
        }
      }

      float16_t *tmp = output + out_tile_offset;
      if (bias != NULL) {
        tmp[0] = tmp_out + bias[i];
      }
      if (relu) {
        tmp[0] = tmp[0] < 0 ? 0 : tmp[0];
      } else if (relu6) {
        mp[0] = tmp[0] < 0 ? 0 : tmp[0];
        tmp[0] = tmp[0] > 6 ? 6 : tmp[0];
      }
    }
  }
}

void IndirectGemmFp16_16x8_c8(float16_t *output, float16_t *input, float16_t *weight, float16_t *bias, size_t step,
                              size_t ic4, size_t output_channel, size_t offset, size_t mode, size_t writeC8,
                              size_t relu, size_t relu6) {
  const int tile_num = 16;
  if (mode && writeC8) {
    for (int i = 0; i < tile_num; i++) {
      int input_tile_offset = i * C4NUM;
      int output_tile_offset = i * output_channel * step;
      for (int j = 0; j < output_channel; j++) {
        int oc8_block = j / C8NUM;
        int oc8_res = j % C8NUM;
        int weight_oc_offset = oc8_block * step * ic4 * C4NUM * C8NUM + oc8_res;
        int out_oc_offset = output_tile_offset + oc8_block * step * C8NUM + oc8_res;

        for (int n = 0; n < step; n++) {
          int input_kw_offset = input_tile_offset + n * ic4 * C4NUM * tile_num;
          int weight_kw_offset = weight_oc_offset + n * ic4 * C4NUM * C8NUM;
          int output_kw_offset = out_oc_offset + n * C8NUM;
          float16_t acc = 0;

          for (int k = 0; k < ic4; k++) {
            int input_ic4_offset = input_kw_offset + k * tile_num * C4NUM;
            int weight_ic4_offset = weight_kw_offset + k * C4NUM * C8NUM;
            for (int m = 0; m < C4NUM; m++) {
              int input_ic_offset = input_ic4_offset + m;
              int weight_ic_offset = weight_ic4_offset + m * C8NUM;
              acc += (weight + weight_ic_offset)[0] * (input + input_ic_offset)[0];
            }
          }

          (output + output_kw_offset)[0] = acc;
        }
      }
    }
  } else {
  }
}
#endif

void SWBorderPixel(float16_t *dst, const float16_t *src, const float16_t *weight, const float16_t *bias, int height,
                   int width, int in_kh_step, int in_kw_step, int kernel_h, int kernel_w, int ic, bool is_relu,
                   bool is_relu6) {
  int ic8 = ic / C8NUM;
  int ic8_res = ic8 % C8NUM;
  int ic4 = ic8_res / C4NUM;
  for (int c = 0; c < C4NUM; c++) {
    dst[c] = 0;
  }
  const float16_t *weight_oc = weight;
  for (int oc = 0; oc < C4NUM; ++oc) {
    const float16_t *weight_kh = weight_oc;
    const float16_t *src_kh = src;
    for (int kh = 0; kh < height; kh++) {
      const float16_t *src_kw = src_kh;
      const float16_t *weight_kw = weight_kh;
      for (int kw = 0; kw < width; kw++) {
        const float16_t *src_ic8 = src_kw;
        const float16_t *weight_ic8 = weight_kw;

        for (int rc = 0; rc < ic8; ++rc) {
          for (int c = 0; c < C8NUM; c++) {
            dst[oc] += src_ic8[c] * weight_ic8[c];
          }
          src_ic8 += C8NUM;
          weight_ic8 += C8NUM;
        }  // ic8 loop

        const float16_t *src_ic4 = src_ic8;
        const float16_t *weight_ic4 = weight_ic8;
        for (int rc = 0; rc < ic4; ++rc) {
          for (int c = 0; c < C4NUM; c++) {
            dst[oc] += src_ic4[c] * weight_ic4[c];
          }
          src_ic4 += C4NUM;
          weight_ic4 += C4NUM;
        }  // ic4 loop

        src_kw += in_kw_step;
        weight_kw += ic4 * C4NUM;
      }  // kernel_w loop
      src_kh += in_kh_step;
      weight_kh += kernel_w * ic4 * C4NUM;
    }  // kernel_h loop
    dst[oc] += bias[oc];
    dst[oc] = (is_relu) ? (MSMAX(0, dst[oc])) : (dst[oc]);
    dst[oc] = (is_relu6) ? (MSMIN(6, MSMAX(0, dst[oc]))) : (dst[oc]);
    weight_oc += kernel_h * kernel_w * ic4 * C4NUM;
  }  // oc loop
}

void SWBorderFp16(float16_t *dst, const float16_t *src, const float16_t *weight, const float16_t *bias, int top,
                  int bottom, int left, int right, const ConvParameter *conv_param, const SlidingWindowParam *sliding) {
  float16_t *dst_h = dst + top * sliding->out_h_step_;
  for (int oh = top; oh < bottom; oh++) {
    int ih = oh * conv_param->stride_h_ - conv_param->pad_h_;
    int start_kh = MSMAX(0, UP_DIV(-ih, conv_param->dilation_h_));
    int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->input_h_ - ih, conv_param->dilation_h_));
    const float16_t *src_h = src + ih * sliding->in_h_step_;

    float16_t *dst_kernel = dst_h + left * sliding->block_channel_;
    for (int ow = left; ow < right; ow++) {
      int iw = ow * conv_param->stride_w_ - conv_param->pad_w_;
      int start_kw = MSMAX(0, UP_DIV(-iw, conv_param->dilation_w_));
      int end_kw = MSMIN(conv_param->kernel_w_, UP_DIV(conv_param->input_w_ - iw, conv_param->dilation_w_));
      const float16_t *src_w = src_h + iw * sliding->ic4_channel_;

      const float16_t *src_kernel = src_w + start_kh * sliding->in_kh_step_ + start_kw * sliding->in_kw_step_;
      const float16_t *weight_kernel = weight + (start_kh * conv_param->kernel_w_ + start_kw) * sliding->ic4_channel_;

      SWBorderPixel(dst_kernel, src_kernel, weight_kernel, bias, end_kh - start_kh, end_kw - start_kw,
                    sliding->in_kh_step_, sliding->in_kw_step_, conv_param->kernel_h_, conv_param->kernel_w_,
                    sliding->ic4_channel_, conv_param->is_relu_, conv_param->is_relu6_);

      dst_kernel += sliding->block_channel_;
    }  // width loop
    dst_h += sliding->out_h_step_;
  }  // height loop
}

void SWCenterFp16(float16_t *dst, const float16_t *src, const float16_t *weight, const float16_t *bias, int height,
                  int width, int kernel_h, int kernel_w, int out_h_step, int block_channel, int ic, int in_sh_step,
                  int in_sw_step, int in_kh_step, int in_kw_step, bool is_relu, bool is_relu6) {
  int ic8 = ic / C8NUM;
  int ic8_res = ic % C8NUM;
  int ic4 = ic8_res / C4NUM;
  float16_t *dst_h = dst;
  const float16_t *src_h = src;
  for (int oh = 0; oh < height; oh++) {
    float16_t *dst_w = dst_h;
    const float16_t *src_w = src_h;
    for (int ow = 0; ow < width; ow++) {
      const float16_t *weight_oc = weight;
      for (int c = 0; c < C4NUM; c++) {
        dst_w[c] = 0;
      }

      for (int oc = 0; oc < C4NUM; oc++) {
        const float16_t *weight_kh = weight_oc;
        const float16_t *src_kh = src_w;
        for (int kh = 0; kh < kernel_h; kh++) {
          const float16_t *src_kw = src_kh;
          const float16_t *weight_kw = weight_kh;
          for (int kw = 0; kw < kernel_w; kw++) {
            const float16_t *src_ic8 = src_kw;
            const float16_t *weight_ic8 = weight_kw;

            for (int rc = 0; rc < ic8; ++rc) {
              for (int c = 0; c < C8NUM; c++) {
                dst_w[oc] += src_ic8[c] * weight_ic8[c];
              }

              src_ic8 += C8NUM;
              weight_ic8 += C8NUM;
            }  // ic8 loop

            const float16_t *src_ic4 = src_ic8;
            const float16_t *weight_ic4 = weight_ic8;
            for (int rc = 0; rc < ic4; ++rc) {
              for (int c = 0; c < C4NUM; c++) {
                dst_w[oc] += src_ic4[c] * weight_ic4[c];
              }

              src_ic4 += C4NUM;
              weight_ic4 += C4NUM;
            }  // ic4 loop

            src_kw += in_kw_step;
            weight_kw += ic4 * C4NUM;
          }  // kernel_w loop
          src_kh += in_kh_step;
          weight_kh += kernel_w * ic4 * C4NUM;
        }  // kernel_h loop
        // add biad relu

        dst_w[oc] += bias[oc];
        dst_w[oc] = (is_relu) ? (MSMAX(0, dst_w[oc])) : (dst_w[oc]);
        dst_w[oc] = (is_relu6) ? (MSMIN(6, MSMAX(0, dst_w[oc]))) : (dst_w[oc]);
        weight_oc += kernel_h * kernel_w * ic4 * C4NUM;
      }  // oc block

      dst_w += block_channel;
      src_w += in_sw_step;
    }  // dst_width loop
    dst_h += out_h_step;
    src_h += in_sh_step;
  }  // dst_height loop
}

// fp16 conv sliding window
void ConvSWFp16(const float16_t *input_data, const float16_t *packed_weight, const float16_t *bias_data,
                float16_t *tmp_out_block, float16_t *output_data, int task_id, ConvParameter *conv_param,
                SlidingWindowParam *slidingWindow_param) {
  int oc4_res = conv_param->output_channel_ % C4NUM;
  const float16_t *src = input_data;
  float16_t *dst;
  if (oc4_res == 0) {
    dst = output_data;
  } else {
    dst = tmp_out_block;
  }

  for (int b = 0; b < conv_param->output_batch_; b++) {
    for (int oc = task_id; oc < slidingWindow_param->c_block_; oc += conv_param->thread_num_) {
      const float16_t *src_data = src;
      float16_t *dst_data = dst + oc * C4NUM;
      const float16_t *weight = packed_weight + oc * slidingWindow_param->kernel_step_;
      const float16_t *bias = bias_data + oc * C4NUM;
      SWBorderFp16(dst_data, src_data, weight, bias, 0, slidingWindow_param->top_, 0, conv_param->output_w_, conv_param,
                   slidingWindow_param);
      SWBorderFp16(dst_data, src_data, weight, bias, slidingWindow_param->bottom_, conv_param->output_h_, 0,
                   conv_param->output_w_, conv_param, slidingWindow_param);
      SWBorderFp16(dst_data, src_data, weight, bias, slidingWindow_param->top_, slidingWindow_param->bottom_, 0,
                   slidingWindow_param->left_, conv_param, slidingWindow_param);
      SWBorderFp16(dst_data, src_data, weight, bias, slidingWindow_param->top_, slidingWindow_param->bottom_,
                   slidingWindow_param->right_, conv_param->output_w_, conv_param, slidingWindow_param);

      if (slidingWindow_param->right_ > slidingWindow_param->left_ &&
          slidingWindow_param->bottom_ > slidingWindow_param->top_) {
        int in_h_start = slidingWindow_param->top_ * conv_param->stride_h_ - conv_param->pad_h_;
        int in_w_start = slidingWindow_param->left_ * conv_param->stride_w_ - conv_param->pad_w_;
        const float16_t *in_t =
          src_data + in_h_start * slidingWindow_param->in_h_step_ + in_w_start * slidingWindow_param->ic4_channel_;
        float16_t *out_t = dst_data + slidingWindow_param->top_ * slidingWindow_param->out_h_step_ +
                           slidingWindow_param->left_ * slidingWindow_param->block_channel_;
        SWCenterFp16(out_t, in_t, weight, bias, slidingWindow_param->bottom_ - slidingWindow_param->top_,
                     slidingWindow_param->right_ - slidingWindow_param->left_, conv_param->kernel_h_,
                     conv_param->kernel_w_, slidingWindow_param->out_h_step_, slidingWindow_param->block_channel_,
                     slidingWindow_param->ic4_channel_, slidingWindow_param->in_sh_step_,
                     slidingWindow_param->in_sw_step_, slidingWindow_param->in_kh_step_,
                     slidingWindow_param->in_kw_step_, conv_param->is_relu_, conv_param->is_relu6_);
      }
    }  // output C4 loop
    src += slidingWindow_param->in_step_;
    dst += slidingWindow_param->out_step_;
  }  // batch loop
}

// fp16 convolution common (im2col+gemm)
void ConvFp16(float16_t *input_data, float16_t *packed_input, float16_t *packed_weight, float16_t *bias_data,
              float16_t *tmp_out_block, float16_t *output_data, int task_id, ConvParameter *conv_param) {
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int in_batch = conv_param->input_batch_;
  int in_channel = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;
  int out_channel = conv_param->output_channel_;
  bool relu = conv_param->is_relu_;
  bool relu6 = conv_param->is_relu6_;
  // todo
  int thread_count = conv_param->thread_num_;
  int tile_n = 16;
  int output_count = out_h * out_w;
  int output_tile_count = UP_DIV(output_count, tile_n);

  int channel_block = UP_DIV(in_channel, C4NUM);
  int kernel_plane = kernel_h * kernel_w;
  int unit_size = kernel_plane * channel_block * C4NUM;
  int packed_input_size = output_tile_count * tile_n * unit_size;

  // we accumulate 4 channels per time for input blocks
  int ic4 = UP_DIV(in_channel, C4NUM);
  int conv_depth = kernel_h * kernel_w;
  // bytes from one output's i-th channel to the next output's i-th channel
  // we write 32 bytes per st1 instruction, after which the pointer in register will step 32B forward

  for (int b = 0; b < in_batch; b++) {
    int in_batch_offset = b * ic4 * C4NUM * in_h * in_w;
    int out_batch_offset = b * out_channel * out_h * out_w;
    int gemm_in_batch_offset = b * packed_input_size;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += thread_count) {
      int start_index = thread_id * tile_n;
      int real_cal_num = (output_count - start_index) < tile_n ? (output_count - start_index) : tile_n;
      float16_t *gemm_input = (float16_t *)(packed_input + thread_id * unit_size * tile_n + gemm_in_batch_offset);
      Im2ColPackUnitFp16(input_data + in_batch_offset, conv_param, gemm_input, real_cal_num, start_index);

      int out_offset = thread_id * tile_n * out_channel + out_batch_offset;
      if (real_cal_num == tile_n) {
        float16_t *gemm_output = output_data + out_offset;
        IndirectGemmFp16_16x8(gemm_output, gemm_input, packed_weight, bias_data, conv_depth, ic4, out_channel,
                              out_channel * sizeof(float16_t), 0, 0, relu, relu6);
      } else {
        // res part
        IndirectGemmFp16_16x8(tmp_out_block, gemm_input, packed_weight, bias_data, conv_depth, ic4, out_channel,
                              out_channel * sizeof(float16_t), 0, 0, relu, relu6);
        memcpy(output_data + out_offset, tmp_out_block, real_cal_num * out_channel * sizeof(float16_t));
      }
    }
  }
}

// fp16 conv3x3
void Conv3x3Fp16(float16_t *input_data, float16_t *transed_weight, const float16_t *bias_data, float16_t *output_data,
                 float16_t *tile_buffer, float16_t *block_unit_buffer, float16_t *tmp_dst_buffer, float16_t *tmp_out,
                 int task_id, ConvParameter *conv_param) {
  int thread_count = conv_param->thread_num_;
  int tile_num = 16;
  const int output_unit = 4;
  const int k_plane = 36;
  int ic4 = UP_DIV(conv_param->input_channel_, C4NUM);
  int oc8 = UP_DIV(conv_param->output_channel_, C8NUM);

  int out_w_block = UP_DIV(conv_param->output_w_, C4NUM);
  int out_h_block = UP_DIV(conv_param->output_h_, C4NUM);
  int output_count = out_w_block * out_h_block;
  int output_tile_count = UP_DIV(output_count, tile_num);
  int tile_buffer_offset = tile_num * k_plane * ic4 * C4NUM;
  int block_unit_buffer_offset = k_plane * C4NUM;
  int tmp_dst_buffer_offset = tile_num * k_plane * oc8 * C8NUM;

  int input_batch = conv_param->input_batch_;
  for (int batch = 0; batch < input_batch; batch++) {
    int in_batch_offset = batch * ic4 * C4NUM * conv_param->input_h_ * conv_param->input_w_;
    int tmp_out_batch_offset = batch * oc8 * C8NUM * out_w_block * out_h_block * output_unit * output_unit;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += thread_count) {
      int start_index = thread_id * tile_num;
      int real_cal_num = (output_count - start_index) < tile_num ? (output_count - start_index) : tile_num;

      Conv3x3Fp16InputTransform(input_data, tile_buffer + task_id * tile_buffer_offset,
                                block_unit_buffer + task_id * block_unit_buffer_offset, start_index, real_cal_num,
                                out_w_block, conv_param);

      IndirectGemmFp16_16x8(tmp_dst_buffer + task_id * tmp_dst_buffer_offset,
                            tile_buffer + task_id * tile_buffer_offset, transed_weight, NULL, 36, ic4, oc8 * C8NUM,
                            oc8 * C8NUM * 36 * sizeof(float16_t), 1, 1, 0, 0);

      Conv3x3Fp16OutputTransform(tmp_dst_buffer + task_id * tmp_dst_buffer_offset, tmp_out + tmp_out_batch_offset,
                                 bias_data, start_index, real_cal_num, out_w_block, conv_param);
    }
  }
}

// fp16 convolution winograd
void ConvWinogardFp16(float16_t *input_data, float16_t *trans_weight, const float16_t *bias_data,
                      TmpBufferAddressFp16 *buffer_list, int task_id, ConvParameter *conv_param,
                      InputTransformUnitFp16Func input_trans_func, OutputTransformUnitFp16Func output_trans_func) {
  int thread_num = conv_param->thread_num_;
  int input_unit = conv_param->input_unit_;
  int in_batch = conv_param->input_batch_;
  int in_channel = conv_param->input_channel_;
  int ic4 = UP_DIV(in_channel, C4NUM);
  int out_unit = conv_param->output_unit_;
  int out_w_block = UP_DIV(conv_param->output_w_, out_unit);
  int out_h_block = UP_DIV(conv_param->output_h_, out_unit);
  int tile_num = 16;
  int output_count = out_w_block * out_h_block;
  int output_tile_count = UP_DIV(output_count, tile_num);
  int out_channel = conv_param->output_channel_;
  int oc8 = UP_DIV(out_channel, C8NUM);
  int input_unit_square = input_unit * input_unit;
  size_t output_offset = oc8 * C8NUM * input_unit_square * sizeof(float16_t);

  float16_t *trans_input = buffer_list[0];
  float16_t *gemm_out = buffer_list[1];
  float16_t *tmp_out_data = buffer_list[2];
  float16_t *tmp_data = buffer_list[3];
  int trans_input_offset = tile_num * input_unit_square * ic4 * C4NUM;
  int gemm_out_offset = tile_num * input_unit_square * oc8 * C8NUM;
  int tmp_data_offset = input_unit_square * C4NUM;
  // step 1 : filter transform (pre-processed offline)
  // step 2 : input transform (online)
  for (int b = 0; b < in_batch; b++) {
    int in_batch_offset = b * ic4 * C4NUM * conv_param->input_h_ * conv_param->input_w_;
    int tmp_out_batch_offset = b * out_w_block * out_h_block * out_unit * out_unit * oc8 * C8NUM;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += thread_num) {
      int out_tile_index = thread_id * TILE_NUM;
      int cal_num = output_count - thread_id * tile_num;
      cal_num = cal_num > tile_num ? tile_num : cal_num;
      WinogradInputTransformFp16(input_data + in_batch_offset, trans_input + task_id * trans_input_offset,
                                 tmp_data + task_id * tmp_data_offset, cal_num, out_tile_index, out_w_block, conv_param,
                                 input_trans_func);
      // step 3 : gemm
      IndirectGemmFp16_16x8(gemm_out + task_id * gemm_out_offset, trans_input + task_id * trans_input_offset,
                            trans_weight, NULL, input_unit_square, ic4, oc8 * C8NUM, output_offset, 1, 1, 0, 0);

      // step 4 : output transform
      WinogradOutputTransformFp16(gemm_out + task_id * gemm_out_offset, tmp_out_data + tmp_out_batch_offset, bias_data,
                                  cal_num, out_tile_index, out_w_block, conv_param, output_trans_func);
    }
  }
}

void UnPackWinogradOutputFp16(const float16_t *src, float16_t *dst, int batch, int height, int width, int channel,
                              int output_unit) {
  int out_h_block_num = UP_DIV(height, output_unit);
  int out_w_block_num = UP_DIV(width, output_unit);
  int c8 = UP_DIV(channel, C8NUM);
  for (int b = 0; b < batch; b++) {
    int src_batch_offset = b * c8 * C8NUM * out_h_block_num * output_unit * out_w_block_num * output_unit;
    int dst_batch_offset = b * height * width * channel;
    for (int h = 0; h < height; h++) {
      int src_h_offset = src_batch_offset + C8NUM * (h * out_w_block_num * output_unit);
      int dst_h_offset = dst_batch_offset + h * width * channel;
      for (int w = 0; w < width; w++) {
        int src_w_offset = src_h_offset + w * C8NUM;
        int dst_w_offset = dst_h_offset + w * channel;
        for (int c = 0; c < c8 - 1; c++) {
          int src_c8_offset = src_w_offset + c * C8NUM * out_w_block_num * out_h_block_num * output_unit * output_unit;
          int dst_c8_offset = dst_w_offset + c * C8NUM;
#ifdef ENABLE_NEON
          vst1q_f16(dst + dst_c8_offset, vld1q_f16(src + src_c8_offset));
#else
          for (int i = 0; i < C8NUM; ++i) {
            dst[dst_c8_offset + i] = src[src_c8_offset + i];
          }
#endif
        }
        int c_res = channel - (c8 - 1) * C8NUM;
        int src_c_res_offset = (c8 - 1) * C8NUM * out_w_block_num * out_h_block_num * output_unit * output_unit;
        int dst_c_res_offset = (c8 - 1) * C8NUM;
        for (int c = 0; c < c_res; c++) {
          int src_c8_res_offset = src_w_offset + src_c_res_offset + c;
          int dst_c8_res_offset = dst_w_offset + dst_c_res_offset + c;
          dst[dst_c8_res_offset] = src[src_c8_res_offset];
        }
      }
    }
  }
}
