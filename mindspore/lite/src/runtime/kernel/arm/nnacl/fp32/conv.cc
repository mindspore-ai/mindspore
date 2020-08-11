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

#include "nnacl/fp32/conv.h"
#include <string.h>
#include "nnacl/winograd_transform.h"

// fp32 conv common
void ConvFp32(float *input_data, float *packed_input, float *packed_weight, const float *bias_data,
              float *tmp_out_block, float *output_data, int task_id, ConvParameter *conv_param,
              GEMM_FUNC_FP32 gemm_func) {
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int in_batch = conv_param->input_batch_;
  int in_channel = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;
  int out_channel = conv_param->output_channel_;
  int thread_count = conv_param->thread_num_;
  int output_count = out_h * out_w;
  int output_tile_count = UP_DIV(output_count, TILE_NUM);
  int ic4 = UP_DIV(in_channel, C4NUM);
  int kernel_plane = kernel_h * kernel_w;
  int unit_size = kernel_plane * ic4 * C4NUM;
  int packed_input_size = output_tile_count * TILE_NUM * unit_size;

  // we accumulate 4 channels per time for input blocks
  int conv_depth = kernel_h * kernel_w;
  // bytes from one output's i-th channel to the next output's i-th channel
  // we write 32 bytes per st1 instruction, after which the pointer in register will step 32B forward
  size_t output_offset = out_channel * sizeof(float);

  for (int b = 0; b < in_batch; b++) {
    int in_batch_offset = b * in_channel * in_h * in_w;
    int out_batch_offset = b * out_channel * out_h * out_w;
    int gemm_in_batch_offset = b * packed_input_size;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += thread_count) {
      int start_index = thread_id * TILE_NUM;
      int real_cal_num = (output_count - start_index) < TILE_NUM ? (output_count - start_index) : TILE_NUM;
      float *gemm_input = packed_input + thread_id * unit_size * TILE_NUM + gemm_in_batch_offset;
      Im2ColPackUnitFp32(input_data + in_batch_offset, conv_param, gemm_input, real_cal_num, start_index);

      int out_offset = thread_id * TILE_NUM * out_channel + out_batch_offset;
      if (real_cal_num == TILE_NUM) {
        float *gemm_output = output_data + out_offset;
        gemm_func(gemm_output, gemm_input, packed_weight, bias_data, conv_depth, ic4, out_channel, output_offset, 0, 0,
                  conv_param->is_relu_, conv_param->is_relu6_);
      } else {
        // res part
        gemm_func(tmp_out_block, gemm_input, packed_weight, bias_data, conv_depth, ic4, out_channel, output_offset, 0,
                  0, conv_param->is_relu_, conv_param->is_relu6_);
        memcpy(output_data + out_offset, tmp_out_block, real_cal_num * out_channel * sizeof(float));
      }
    }
  }
}

// fp32 conv1x1 strassen matmul
int Conv1x1Fp32(const float *input_data, const float *weight_data, float *output_data, float *tmp_ptr,
                StrassenMatMulParameter matmul_param) {
  return StrassenMatmul(input_data, weight_data, output_data, &matmul_param, FP32_STRASSEN_MAX_RECURSION, 0, tmp_ptr);
}

// fp32 conv winograd
void ConvWinogardFp32(float *input_data, float *trans_weight, const float *bias_data, float *output_data,
                      TmpBufferAddress *buffer_list, int task_id, ConvParameter *conv_param,
                      InputTransformUnitFunc input_trans_func, OutputTransformUnitFunc output_trans_func,
                      GEMM_FUNC_FP32 gemm_func) {
  int thread_num = conv_param->thread_num_;
  int input_unit = conv_param->input_unit_;
  int in_batch = conv_param->input_batch_;
  int in_channel = conv_param->input_channel_;
  int ic4 = UP_DIV(in_channel, C4NUM);
  int out_unit = conv_param->output_unit_;
  int out_w_block = UP_DIV(conv_param->output_w_, out_unit);
  int out_h_block = UP_DIV(conv_param->output_h_, out_unit);
  int output_count = out_w_block * out_h_block;
  int output_tile_count = UP_DIV(output_count, TILE_NUM);
  int out_channel = conv_param->output_channel_;
  int out_batch = conv_param->output_batch_;
  int oc4 = UP_DIV(out_channel, C4NUM);
  int input_unit_square = input_unit * input_unit;
  size_t output_offset = oc4 * C4NUM * input_unit_square * sizeof(float);
  bool is_relu = conv_param->is_relu_;
  bool is_relu6 = conv_param->is_relu6_;

  float *trans_input = buffer_list[0];
  float *gemm_out = buffer_list[1];
  float *tmp_out_data = buffer_list[2];
  float *tmp_data = buffer_list[3];
  int trans_input_offset = TILE_NUM * input_unit_square * ic4 * C4NUM;
  int gemm_out_offset = TILE_NUM * input_unit_square * oc4 * C4NUM;
  int tmp_data_offset = input_unit_square * C4NUM;
  // step 1 : filter transform (pre-processed offline)
  // step 2 : input transform (online)
  for (int b = 0; b < in_batch; b++) {
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += thread_num) {
      int out_tile_index = thread_id * TILE_NUM;
      int cal_num = output_count - thread_id * TILE_NUM;
      cal_num = cal_num > TILE_NUM ? TILE_NUM : cal_num;
      WinogradInputTransform(input_data, trans_input + task_id * trans_input_offset,
                             tmp_data + task_id * tmp_data_offset, cal_num, out_tile_index, out_w_block, conv_param,
                             input_trans_func);
      // step 3 : gemm
      gemm_func(gemm_out + task_id * gemm_out_offset, trans_input + task_id * trans_input_offset, trans_weight, nullptr,
                input_unit_square, ic4, oc4 * C4NUM, output_offset, 1, 1, 0, 0);

      // step 4 : output transform
      WinogradOutputTransform(gemm_out + task_id * gemm_out_offset, tmp_out_data, bias_data, cal_num, out_tile_index,
                              out_w_block, conv_param, output_trans_func);
    }
  }
  // get real output
  UnPackWinogradOutput(tmp_out_data, output_data, out_batch, conv_param->output_h_, conv_param->output_w_, out_channel,
                       out_unit);
  int output_num = out_channel * conv_param->output_h_ * conv_param->output_w_ * conv_param->output_batch_;
  if (is_relu) {
    ReluFp32(output_data, output_data, output_num);
  } else if (is_relu6) {
    Relu6Fp32(output_data, output_data, output_num);
  } else {
    // do nothing
  }
}

void UnPackWinogradOutput(const float *src, float *dst, int batch, int height, int width, int channel,
                          int output_unit) {
  int out_h_block_num = UP_DIV(height, output_unit);
  int out_w_block_num = UP_DIV(width, output_unit);
  int c4 = UP_DIV(channel, C4NUM);
  for (int b = 0; b < batch; b++) {
    int src_batch_offset = b * c4 * C4NUM * out_h_block_num * output_unit * out_w_block_num * output_unit;
    int dst_batch_offset = b * height * width * channel;
    for (int h = 0; h < height; h++) {
      int src_h_offset = src_batch_offset + C4NUM * (h * out_w_block_num * output_unit);
      int dst_h_offset = dst_batch_offset + h * width * channel;
      for (int w = 0; w < width; w++) {
        int src_w_offset = src_h_offset + w * C4NUM;
        int dst_w_offset = dst_h_offset + w * channel;
        for (int c = 0; c < c4 - 1; c++) {
          int src_c4_offset = src_w_offset + c * C4NUM * out_w_block_num * out_h_block_num * output_unit * output_unit;
          int dst_c4_offset = dst_w_offset + c * C4NUM;
#ifdef ENABLE_NEON
          vst1q_f32(dst + dst_c4_offset, vld1q_f32(src + src_c4_offset));
#else
          dst[dst_c4_offset] = src[src_c4_offset];
          dst[dst_c4_offset + 1] = src[src_c4_offset + 1];
          dst[dst_c4_offset + 2] = src[src_c4_offset + 2];
          dst[dst_c4_offset + 3] = src[src_c4_offset + 3];
#endif
        }
        int c_res = channel - (c4 - 1) * C4NUM;
        int src_c_res_offset = (c4 - 1) * C4NUM * out_w_block_num * out_h_block_num * output_unit * output_unit;
        int dst_c_res_offset = (c4 - 1) * C4NUM;
        for (int c = 0; c < c_res; c++) {
          int src_c4_res_offset = src_w_offset + src_c_res_offset + c;
          int dst_c4_res_offset = dst_w_offset + dst_c_res_offset + c;
          dst[dst_c4_res_offset] = src[src_c4_res_offset];
        }
      }
    }
  }
}

// fp32 conv3x3
void Conv3x3Fp32(float *input_data, float *transed_weight, const float *bias_data, float *output_data,
                 TmpBufferAddress *buffer_list, int task_id, ConvParameter *conv_param, GEMM_FUNC_FP32 gemm_func) {
  int thread_count = conv_param->thread_num_;
  int ic4 = UP_DIV(conv_param->input_channel_, C4NUM);
  int output_channel = conv_param->output_channel_;
  int oc4 = UP_DIV(output_channel, C4NUM);
  int out_w_block = UP_DIV(conv_param->output_w_, OUPUT_UNIT);
  int out_h_block = UP_DIV(conv_param->output_h_, OUPUT_UNIT);
  int output_count = out_w_block * out_h_block;
  int output_tile_count = UP_DIV(output_count, TILE_NUM);
  int input_unit_square = 4 * 4;
  bool is_relu = conv_param->is_relu_;
  bool is_relu6 = conv_param->is_relu6_;
  float *tile_buffer = buffer_list[0];
  float *block_unit_buffer = buffer_list[1];
  float *tmp_dst_buffer = buffer_list[2];
  float *nc4hw4_out = buffer_list[3];
  int tile_buffer_offset = TILE_NUM * input_unit_square * ic4 * C4NUM;
  int block_unit_buffer_offset = input_unit_square * C4NUM;
  int tmp_dst_buffer_offset = TILE_NUM * input_unit_square * oc4 * C4NUM;

  int input_batch = conv_param->input_batch_;
  for (int batch = 0; batch < input_batch; batch++) {
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += thread_count) {
      int start_index = thread_id * TILE_NUM;
      int real_cal_num = (output_count - start_index) < TILE_NUM ? (output_count - start_index) : TILE_NUM;
      Conv3x3Fp32InputTransform(input_data, tile_buffer + task_id * tile_buffer_offset,
                                block_unit_buffer + task_id * block_unit_buffer_offset, start_index, real_cal_num,
                                out_w_block, conv_param);

      gemm_func(tmp_dst_buffer + task_id * tmp_dst_buffer_offset, tile_buffer + task_id * tile_buffer_offset,
                transed_weight, nullptr, input_unit_square, ic4, oc4 * C4NUM,
                oc4 * C4NUM * input_unit_square * sizeof(float), 1, 1, 0, 0);

      Conv3x3Fp32OutputTransform(tmp_dst_buffer + task_id * tmp_dst_buffer_offset, nc4hw4_out, bias_data, start_index,
                                 real_cal_num, out_w_block, conv_param);
    }
    PackNC4HW4ToNHWCFp32(nc4hw4_out, output_data, 1, conv_param->output_h_ * conv_param->output_w_, output_channel);
  }
  int output_num = output_channel * conv_param->output_h_ * conv_param->output_w_ * conv_param->output_batch_;
  if (is_relu) {
    ReluFp32(output_data, output_data, output_num);
  } else if (is_relu6) {
    Relu6Fp32(output_data, output_data, output_num);
  } else {
    // do nothing
  }
}
