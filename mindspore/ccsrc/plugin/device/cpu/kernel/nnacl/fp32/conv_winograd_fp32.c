/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/conv_winograd_fp32.h"
#include <string.h>
#include "nnacl/fp32/common_func_fp32.h"
#include "nnacl/fp32/winograd_transform.h"
#include "nnacl/fp32/matmul_fp32.h"

// fp32 conv winograd
void ConvWinogardFp32(const float *input_data, const float *trans_weight, const float *bias_data, float *output_data,
                      TmpBufferAddress *buffer_list, int task_id, const ConvParameter *conv_param,
                      TransFuncList trans_func) {
  if (conv_param->output_unit_ == 0) {
    return;
  }
  int in_channel = conv_param->input_channel_;
  int input_unit = conv_param->input_unit_;
  int out_w_block = UP_DIV(conv_param->output_w_, conv_param->output_unit_);
  int out_h_block = UP_DIV(conv_param->output_h_, conv_param->output_unit_);
  int output_count = out_w_block * out_h_block;
  const int tile_num = C12NUM;
  int output_tile_count = UP_DIV(output_count, tile_num);
#ifdef ENABLE_AVX
  const int col_tile = C16NUM;
  const int channel_pack_tile = C8NUM;
#else
  const int col_tile = C8NUM;
  const int channel_pack_tile = C4NUM;
#endif
  int oc_tile = UP_DIV(conv_param->output_channel_, col_tile);
  int oc8 = UP_DIV(conv_param->output_channel_, C8NUM);
  int input_unit_square = input_unit * input_unit;

  float *trans_input = buffer_list[0] + task_id * tile_num * input_unit_square * in_channel;
  float *gemm_out = buffer_list[1] + task_id * tile_num * input_unit_square * oc8 * C8NUM;
  float *tmp_data = buffer_list[2] + task_id * input_unit_square * channel_pack_tile;
  float *col_buffer = buffer_list[3] + task_id * tile_num * in_channel;
  // step 1 : filter transform (pre-processed offline)
  // step 2 : input transform (online)

  int block_per_thread = UP_DIV(output_tile_count, conv_param->thread_num_);
  int start_index = block_per_thread * task_id * tile_num;
  if (start_index >= output_count) {
    return;
  }
  int end_index = MSMIN(start_index + block_per_thread * tile_num, output_count);

  for (int b = 0; b < conv_param->input_batch_; b++) {
    int in_batch_offset = b * in_channel * conv_param->input_h_ * conv_param->input_w_;
    int out_batch_offset = b * conv_param->output_channel_ * conv_param->output_w_ * conv_param->output_h_;

    for (int out_tile_index = start_index; out_tile_index < end_index; out_tile_index += tile_num) {
      int cal_num = output_count - out_tile_index;
      cal_num = cal_num > tile_num ? tile_num : cal_num;
      if (cal_num <= 0) {
        return;
      }

#ifdef ENABLE_ARM64
      // Optimize input transform. Only valid for arm64, the tile num is 12, the channel_tile is 4.
      // For arm32, the tile_num is 4.
      // For x86_sse, the tile_num is 4, the channel_tile is 4.
      // For avx, the tile_num is 6, the channel_tile is 8.
      // N = input_unit, M = tile_num
      // The function(InputTransformNxNStep, InputTransform4x4PackM) needs to be rewritten.
      bool fused_pack =
        (cal_num == tile_num) && (trans_func.in_step_func_ != NULL) && (trans_func.in_pack_func_ != NULL);
      if (fused_pack) {
        float *opt_trans_input =
          buffer_list[4] + task_id * tile_num * input_unit_square * UP_ROUND(in_channel, channel_pack_tile);
        WinogradInputTransformOptStep(input_data + in_batch_offset, opt_trans_input, tmp_data, cal_num, out_tile_index,
                                      out_w_block, conv_param, trans_func.in_step_func_);

        for (int w_index = 0; w_index < input_unit; w_index++) {
          float *src_w = opt_trans_input + w_index * input_unit * tile_num * channel_pack_tile;
          for (int c = 0; c < UP_DIV(in_channel, channel_pack_tile); c++) {
            int real_c = in_channel - c * channel_pack_tile;
            real_c = real_c > channel_pack_tile ? channel_pack_tile : real_c;
            float *src_c = src_w + c * input_unit_square * tile_num * channel_pack_tile;
            float *dst_c = trans_input + c * tile_num * channel_pack_tile;
            trans_func.in_pack_func_(src_c, dst_c, channel_pack_tile, in_channel * tile_num, real_c);
          }

          for (int h_index = 0; h_index < input_unit; h_index++) {
            const float *gemm_input = trans_input + h_index * tile_num * in_channel;
            int point_index = h_index * input_unit + w_index;
            const float *gemm_weight = trans_weight + point_index * in_channel * oc_tile * col_tile;
            MatMulOpt(gemm_input, gemm_weight, gemm_out + point_index * C8NUM, NULL, 0, in_channel, cal_num,
                      oc8 * C8NUM, input_unit_square, OutType_TileC8);
          }
        }
      } else {
#endif
        WinogradInputTransform(input_data + in_batch_offset, trans_input, tmp_data, cal_num, out_tile_index,
                               out_w_block, conv_param, trans_func.in_func_);
        // step 3 : gemm
        float *src_ptr = trans_input;
        float *dst_ptr = gemm_out;
        float *tmp_col_ptr = col_buffer;
        for (int i = 0; i < input_unit_square; ++i) {
#ifdef ENABLE_AVX
          RowMajor2Col6Major(src_ptr + i * C12NUM * in_channel, tmp_col_ptr, C12NUM, in_channel);
#elif defined(ENABLE_ARM32) || defined(ENABLE_SSE)
        RowMajor2Col4Major(src_ptr + i * C12NUM * in_channel, tmp_col_ptr, C12NUM, in_channel);
#else
        RowMajor2Col12Major(src_ptr + i * C12NUM * in_channel, tmp_col_ptr, C12NUM, in_channel);
#endif
          MatMulOpt(tmp_col_ptr, trans_weight + i * in_channel * oc_tile * col_tile, dst_ptr + i * C8NUM, NULL, 0,
                    in_channel, cal_num, oc8 * C8NUM, input_unit_square, 2);
        }
#ifdef ENABLE_ARM64
      }
#endif

      // step 4 : output transform
      float *output_ptr = output_data + out_batch_offset;
      if (conv_param->out_format_ != Format_NC4HW4) {  // nc4hw4
        WinogradOutputNHWCTransform(gemm_out, output_ptr, bias_data, cal_num, out_tile_index, out_w_block, conv_param,
                                    trans_func.out_func_);
      } else {
#if defined(ENABLE_AVX) || defined(ENABLE_ARM64)
        WinogradOutputNC4HW4Transform(gemm_out, output_ptr, bias_data, cal_num, out_tile_index, out_w_block, conv_param,
                                      trans_func.out_func_);
#else
        WinogradOutputNHWCTransform(gemm_out, output_ptr, bias_data, cal_num, out_tile_index, out_w_block, conv_param,
                                    trans_func.out_func_);
#endif
      }
    }
  }
}

// fp32 conv winograd
void ConvWinogardFp32CutByBatch(const float *input_data, const float *trans_weight, const float *bias_data,
                                float *output_data, TmpBufferAddress *buffer_list, int task_id,
                                const ConvParameter *conv_param, TransFuncList trans_func) {
  if (conv_param->output_unit_ == 0) {
    return;
  }
  int in_channel = conv_param->input_channel_;
  int input_unit = conv_param->input_unit_;
  int out_w_block = UP_DIV(conv_param->output_w_, conv_param->output_unit_);
  int out_h_block = UP_DIV(conv_param->output_h_, conv_param->output_unit_);
  int output_count = out_w_block * out_h_block;
  const int tile_num = C12NUM;
#ifdef ENABLE_AVX
  const int col_tile = C16NUM;
  const int channel_pack_tile = C8NUM;
#else
  const int col_tile = C8NUM;
  const int channel_pack_tile = C4NUM;
#endif
  int oc_tile = UP_DIV(conv_param->output_channel_, col_tile);
  int oc8 = UP_DIV(conv_param->output_channel_, C8NUM);
  int input_unit_square = input_unit * input_unit;

  float *trans_input = buffer_list[0] + task_id * tile_num * input_unit_square * in_channel;
  float *gemm_out = buffer_list[1] + task_id * tile_num * input_unit_square * oc8 * C8NUM;
  float *tmp_data = buffer_list[2] + task_id * input_unit_square * channel_pack_tile;
  float *col_buffer = buffer_list[3] + task_id * tile_num * in_channel;
  // step 1 : filter transform (pre-processed offline)
  // step 2 : input transform (online)

  int block_batch_per_thread = UP_DIV(conv_param->input_batch_, conv_param->thread_num_);
  int start_batch = block_batch_per_thread * task_id;
  int end_batch = MSMIN(conv_param->input_batch_, (start_batch + block_batch_per_thread));

  for (int b = start_batch; b < end_batch; b++) {
    int in_batch_offset = b * in_channel * conv_param->input_h_ * conv_param->input_w_;
    int out_batch_offset = b * conv_param->output_channel_ * conv_param->output_w_ * conv_param->output_h_;

    for (int out_tile_index = 0; out_tile_index < output_count; out_tile_index += tile_num) {
      int cal_num = output_count - out_tile_index;
      cal_num = cal_num > tile_num ? tile_num : cal_num;
      if (cal_num <= 0) {
        return;
      }

#ifdef ENABLE_ARM64
      // Optimize input transform. Only valid for arm64, the tile num is 12, the channel_tile is 4.
      // For arm32, the tile_num is 4.
      // For x86_sse, the tile_num is 4, the channel_tile is 4.
      // For avx, the tile_num is 6, the channel_tile is 8.
      // N = input_unit, M = tile_num
      // The function(InputTransformNxNStep, InputTransform4x4PackM) needs to be rewritten.
      bool fused_pack =
        (cal_num == tile_num) && (trans_func.in_step_func_ != NULL) && (trans_func.in_pack_func_ != NULL);
      if (fused_pack) {
        float *opt_trans_input =
          buffer_list[4] + task_id * tile_num * input_unit_square * UP_ROUND(in_channel, channel_pack_tile);
        WinogradInputTransformOptStep(input_data + in_batch_offset, opt_trans_input, tmp_data, cal_num, out_tile_index,
                                      out_w_block, conv_param, trans_func.in_step_func_);

        for (int w_index = 0; w_index < input_unit; w_index++) {
          float *src_w = opt_trans_input + w_index * input_unit * tile_num * channel_pack_tile;
          for (int c = 0; c < UP_DIV(in_channel, channel_pack_tile); c++) {
            int real_c = in_channel - c * channel_pack_tile;
            real_c = real_c > channel_pack_tile ? channel_pack_tile : real_c;
            float *src_c = src_w + c * input_unit_square * tile_num * channel_pack_tile;
            float *dst_c = trans_input + c * tile_num * channel_pack_tile;
            trans_func.in_pack_func_(src_c, dst_c, channel_pack_tile, in_channel * tile_num, real_c);
          }

          for (int h_index = 0; h_index < input_unit; h_index++) {
            const float *gemm_input = trans_input + h_index * tile_num * in_channel;
            int point_index = h_index * input_unit + w_index;
            const float *gemm_weight = trans_weight + point_index * in_channel * oc_tile * col_tile;
            MatMulOpt(gemm_input, gemm_weight, gemm_out + point_index * C8NUM, NULL, 0, in_channel, cal_num,
                      oc8 * C8NUM, input_unit_square, OutType_TileC8);
          }
        }
      } else {
#endif
        WinogradInputTransform(input_data + in_batch_offset, trans_input, tmp_data, cal_num, out_tile_index,
                               out_w_block, conv_param, trans_func.in_func_);
        // step 3 : gemm
        float *src_ptr = trans_input;
        float *dst_ptr = gemm_out;
        float *tmp_col_ptr = col_buffer;
        for (int i = 0; i < input_unit_square; ++i) {
#ifdef ENABLE_AVX
          RowMajor2Col6Major(src_ptr + i * C12NUM * in_channel, tmp_col_ptr, C12NUM, in_channel);
#elif defined(ENABLE_ARM32) || defined(ENABLE_SSE)
        RowMajor2Col4Major(src_ptr + i * C12NUM * in_channel, tmp_col_ptr, C12NUM, in_channel);
#else
        RowMajor2Col12Major(src_ptr + i * C12NUM * in_channel, tmp_col_ptr, C12NUM, in_channel);
#endif
          MatMulOpt(tmp_col_ptr, trans_weight + i * in_channel * oc_tile * col_tile, dst_ptr + i * C8NUM, NULL, 0,
                    in_channel, cal_num, oc8 * C8NUM, input_unit_square, 2);
        }
#ifdef ENABLE_ARM64
      }
#endif

      // step 4 : output transform
      float *output_ptr = output_data + out_batch_offset;
      if (conv_param->out_format_ != Format_NC4HW4) {  // nc4hw4
        WinogradOutputNHWCTransform(gemm_out, output_ptr, bias_data, cal_num, out_tile_index, out_w_block, conv_param,
                                    trans_func.out_func_);
      } else {
#if defined(ENABLE_AVX) || defined(ENABLE_ARM64)
        WinogradOutputNC4HW4Transform(gemm_out, output_ptr, bias_data, cal_num, out_tile_index, out_w_block, conv_param,
                                      trans_func.out_func_);
#else
        WinogradOutputNHWCTransform(gemm_out, output_ptr, bias_data, cal_num, out_tile_index, out_w_block, conv_param,
                                    trans_func.out_func_);
#endif
      }
    }
  }
}
