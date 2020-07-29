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
#include "src/runtime/kernel/arm/opclib/fp16/conv_fp16.h"
#include <string.h>
#include "src/runtime/kernel/arm/opclib/pack.h"
#include "src/runtime/kernel/arm/opclib/winograd_transform.h"

extern "C" {
#ifdef ENABLE_ARM64
#ifdef ENABLE_FP16
void IndirectGemmFp16_16x8(float16_t *output, float16_t *input, float16_t *weight, float16_t *bias, size_t step,
                           size_t ic4, size_t oc8, size_t offset, size_t mode, size_t writeC4, size_t relu,
                           size_t relu6);
#endif
#endif
}

#ifdef ENABLE_FP16
#ifndef ENABLE_NEON
void IndirectGemmFp16_16x8(float16_t *output, float16_t *input, float16_t *weight, float16_t *bias, size_t step,
                           size_t ic4, size_t out_channel, size_t offset, size_t mode, size_t writeC4, size_t relu,
                           size_t relu6) {
  int tile_n = 16;
  for (int i = 0; i < out_channel; i++) {
    int oc8_block = i / 8;
    int oc8_res = i % 8;
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

      (output + out_tile_offset)[0] = tmp_out;
    }
  }
}

void IndirectGemmFp16_16x8_tmp(float16_t *output, float16_t *input, float16_t *weight, const float16_t *bias,
                               size_t step, size_t ic4, size_t output_channel, size_t offset, size_t mode,
                               size_t writeC4, size_t relu, size_t relu6) {
  int tile_num = 16;
  if (mode) {
    for (int i = 0; i < tile_num; i++) {
      int input_tile_offset = i * C4NUM;
      int output_tile_offset = i * output_channel * 36;
      for (int j = 0; j < output_channel; j++) {
        int oc8_block = j / 8;
        int oc8_res = j % 8;
        int weight_oc_offset = oc8_block * 36 * ic4 * C4NUM * 8 + oc8_res;
        // todo nc4hw4 -> nhwc
        int out_oc_offset = output_tile_offset + oc8_block * 36 * C8NUM + oc8_res;

        for (int n = 0; n < step; n++) {
          int input_kw_offset = input_tile_offset + n * ic4 * C4NUM * tile_num;
          int weight_kw_offset = weight_oc_offset + n * ic4 * C4NUM * 8;
          int output_kw_offset = out_oc_offset + n * C8NUM;
          float16_t acc = 0;

          for (int k = 0; k < ic4; k++) {
            int input_ic4_offset = input_kw_offset + k * tile_num * C4NUM;
            int weight_ic4_offset = weight_kw_offset + k * C4NUM * 8;
            for (int m = 0; m < 4; m++) {
              int input_ic_offset = input_ic4_offset + m;
              int weight_ic_offset = weight_ic4_offset + m * 8;
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
  int oc8 = UP_DIV(in_channel, C8NUM);
  int conv_depth = kernel_h * kernel_w;
  // bytes from one output's i-th channel to the next output's i-th channel
  // we write 32 bytes per st1 instruction, after which the pointer in register will step 32B forward

  for (int b = 0; b < in_batch; b++) {
    int in_batch_offset = b * in_channel * in_h * in_w;
    int out_batch_offset = b * out_channel * out_h * out_w;
    int gemm_in_batch_offset = b * packed_input_size;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += thread_count) {
      int start_index = thread_id * tile_n;
      int real_cal_num = (output_count - start_index) < tile_n ? (output_count - start_index) : tile_n;
      float16_t *gemm_input =
        (float16_t *)(packed_input + thread_id * unit_size * tile_n + gemm_in_batch_offset);
      Im2ColPackUnitFp16(input_data + in_batch_offset, conv_param, gemm_input, real_cal_num, start_index);

      int out_offset = thread_id * tile_n * out_channel + out_batch_offset;
      if (real_cal_num == tile_n) {
        float16_t *gemm_output = output_data + out_offset;
        IndirectGemmFp16_16x8(gemm_output, gemm_input, packed_weight, bias_data, conv_depth, ic4, out_channel,
                              oc8 * C8NUM * sizeof(float16_t), 0, 0, 0, 0);
      } else {
        // res part
        IndirectGemmFp16_16x8(tmp_out_block, gemm_input, packed_weight, bias_data, conv_depth, ic4, out_channel,
                              oc8 * C8NUM * sizeof(float16_t), 0, 0, 0, 0);
        memcpy(output_data + out_offset, tmp_out_block, real_cal_num * out_channel * sizeof(float16_t));
      }
    }
  }
}

// fp16 conv3x3
void Conv3x3Fp16(float16_t *input_data, float16_t *transed_weight, const float16_t *bias_data, float16_t *output_data,
                 float16_t *tile_buffer, float16_t *block_unit_buffer, float16_t *tmp_dst_buffer, float16_t *tmp_out,
                 int task_id, ConvParameter *conv_param) {
  // todo
  int thread_count = conv_param->thread_num_;
  int tile_num = 16;
  int output_unit = 4;
  int ic4 = UP_DIV(conv_param->input_channel_, C4NUM);
  int oc8 = UP_DIV(conv_param->output_channel_, C8NUM);

  int output_batch = conv_param->output_batch_;
  int output_channel = conv_param->output_channel_;
  int output_w = conv_param->output_w_;
  int output_h = conv_param->output_h_;

  int out_w_block = UP_DIV(conv_param->output_w_, C4NUM);
  int out_h_block = UP_DIV(conv_param->output_h_, C4NUM);
  int output_count = out_w_block * out_h_block;
  int output_tile_count = UP_DIV(output_count, tile_num);

  int input_batch = conv_param->input_batch_;
  for (int batch = 0; batch < input_batch; batch++) {
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += thread_count) {
      int start_index = thread_id * tile_num;
      int real_cal_num = (output_count - start_index) < tile_num ? (output_count - start_index) : tile_num;

      Conv3x3Fp16InputTransform(input_data, tile_buffer, block_unit_buffer, start_index, real_cal_num, out_w_block,
                                conv_param);

      IndirectGemmFp16_16x8(tmp_dst_buffer, tile_buffer, transed_weight, NULL, 36, ic4, oc8 * C8NUM,
                            oc8 * C8NUM * 36 * sizeof(float16_t), 1, 1, 0, 0);

      Conv3x3Fp16OutputTransform(tmp_dst_buffer, tmp_out, bias_data, start_index, real_cal_num, out_w_block,
                                 conv_param);
    }
  }

  // get real output
  // todo
  for (int batch = 0; batch < output_batch; batch++) {
    int batch_size = batch * output_channel * output_h * output_w;
    for (int h = 0; h < output_h; h++) {
      for (int w = 0; w < output_w; w++) {
        for (int c = 0; c < output_channel; c++) {
          int oc8_block = c / C8NUM;
          int oc8_res = c % C8NUM;
          int src_offset = oc8_block * C8NUM * out_w_block * out_h_block * tile_num +
                           C8NUM * (h * out_w_block * output_unit + w) + oc8_res;
          int dst_offset = (h * output_w + w) * output_channel + c;
          (output_data + dst_offset)[0] = (tmp_out + src_offset)[0];
        }
      }
    }
  }
}
#endif

