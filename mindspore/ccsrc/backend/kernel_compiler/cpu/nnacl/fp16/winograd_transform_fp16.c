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

#include "nnacl/fp16/winograd_transform_fp16.h"

// fp16 common winograd
void WinogradInputTransformFp16(const float16_t *input_data, float16_t *trans_input, float16_t *tmp_data, int cal_num,
                                int out_tile_index, int out_w_block_num, const ConvParameter *conv_param,
                                InputTransFp16Func func) {
#ifdef ENABLE_ARM64
  const int tile_num = 16;
#else
  const int tile_num = 12;
#endif
  int input_unit = conv_param->input_unit_;
  int output_unit = conv_param->output_unit_;
  int in_channel = conv_param->input_channel_;
  int ic8 = UP_DIV(in_channel, C8NUM);
  int pad_h = conv_param->pad_u_;
  int pad_w = conv_param->pad_l_;
  int input_h = conv_param->input_h_;
  int input_w = conv_param->input_w_;
  if (out_w_block_num == 0) {
    return;
  }
  for (int c = 0; c < cal_num; c++) {  // actual tiled number
    int src_x_s = (out_tile_index % out_w_block_num) * output_unit - pad_w;
    int src_y_s = (out_tile_index / out_w_block_num) * output_unit - pad_h;
    int interval_x_s = src_x_s > 0 ? 0 : -src_x_s;
    int interval_y_s = src_y_s > 0 ? 0 : -src_y_s;
    int src_x_e = src_x_s + input_unit;
    int src_y_e = src_y_s + input_unit;
    int interval_x_e = src_x_e < input_w ? input_unit : (input_w - src_x_s);
    int interval_y_e = src_y_e < input_h ? input_unit : (input_h - src_y_s);

    int src_plane_offset = in_channel * (src_y_s * input_w + src_x_s);
    int dst_plane_offset = c * in_channel;
    for (int ic = 0; ic < ic8; ic++) {
      // clear tmp buffer
      memset(tmp_data, 0, input_unit * input_unit * C8NUM * sizeof(float16_t));

      int real_c = in_channel - ic * C8NUM;
      real_c = real_c > C8NUM ? C8NUM : real_c;
      int src_ic8_offset = src_plane_offset + ic * C8NUM;

      // get real input block with padding
      if (real_c == C8NUM) {
        for (int interval = interval_y_s; interval < interval_y_e; interval++) {
          int src_y_offset = src_ic8_offset + (interval * input_w + interval_x_s) * in_channel;
          int dst_y_offset = interval * input_unit * C8NUM + interval_x_s * C8NUM;
          for (int j = 0; j < (interval_x_e - interval_x_s); j++) {
            int src_x_offset = src_y_offset + j * in_channel;
            int dst_x_offset = dst_y_offset + j * C8NUM;
            const float16_t *src_addr = input_data + src_x_offset;
            float16_t *dst_addr = tmp_data + dst_x_offset;
#ifdef ENABLE_NEON
            vst1q_f16(dst_addr, vld1q_f16(src_addr));
#else
            for (int k = 0; k < C8NUM; k++) {
              dst_addr[k] = src_addr[k];
            }
#endif
          }
        }
      } else if (real_c < 8 && real_c >= 4) {
        for (int interval = interval_y_s; interval < interval_y_e; interval++) {
          int src_y_offset = src_ic8_offset + (interval * input_w + interval_x_s) * in_channel;
          int dst_y_offset = interval * input_unit * C8NUM + interval_x_s * C8NUM;
          for (int j = 0; j < (interval_x_e - interval_x_s); j++) {
            int src_x_offset = src_y_offset + j * in_channel;
            int dst_x_offset = dst_y_offset + j * C8NUM;
            const float16_t *src_addr = input_data + src_x_offset;
            float16_t *dst_addr = tmp_data + dst_x_offset;
            int rc = real_c - 4;
#ifdef ENABLE_NEON
            vst1_f16(dst_addr, vld1_f16(src_addr));
#else
            for (int k = 0; k < C4NUM; k++) {
              dst_addr[k] = src_addr[k];
            }
#endif
            src_addr += 4;
            dst_addr += 4;
            for (int i = 0; i < rc; ++i) {
              dst_addr[i] = src_addr[i];
            }
          }
        }
      } else {
        for (int interval = interval_y_s; interval < interval_y_e; interval++) {
          int src_y_offset = src_ic8_offset + (interval * input_w + interval_x_s) * in_channel;
          int dst_y_offset = interval * input_unit * C8NUM + interval_x_s * C8NUM;
          for (int j = 0; j < (interval_x_e - interval_x_s); j++) {
            int src_x_offset = src_y_offset + j * in_channel;
            int dst_x_offset = dst_y_offset + j * C8NUM;
            const float16_t *src_addr = input_data + src_x_offset;
            float16_t *dst_addr = tmp_data + dst_x_offset;
            for (int k = 0; k < real_c; k++) {
              dst_addr[k] = src_addr[k];
            }
          }
        }
      }

      // input transform
      int dst_ic8_offset = dst_plane_offset + ic * C8NUM;
      size_t dst_step = in_channel * tile_num;
      float16_t *trans_input_ptr = trans_input + dst_ic8_offset;
      func(tmp_data, trans_input_ptr, C8NUM, dst_step, real_c);
    }
    out_tile_index++;
  }  // cal_tile_num loop
}

void WinogradOutputNHWCTransformFp16(const float16_t *gemm_out, float16_t *tmp_out_data, const float16_t *bias_data,
                                     int cal_num, int out_tile_index, int output_unit_num,
                                     const ConvParameter *conv_param, OutputTransFp16Func func) {
  int output_unit = conv_param->output_unit_;
  int output_w = conv_param->output_w_;
  int output_h = conv_param->output_h_;
  int output_channel = conv_param->output_channel_;
  int oc8 = UP_DIV(output_channel, C8NUM);
  int input_unit = conv_param->input_unit_;
  NNACL_CHECK_ZERO_RETURN(output_unit_num);
  for (int i = 0; i < cal_num; i++) {
    int dst_x_s = out_tile_index % output_unit_num;
    int dst_y_s = out_tile_index / output_unit_num;
    int r_w = output_w - dst_x_s * output_unit;
    r_w = r_w > output_unit ? output_unit : r_w;
    int r_h = output_h - dst_y_s * output_unit;
    r_h = r_h > output_unit ? output_unit : r_h;
    int tmp_ix = dst_x_s * output_unit;
    dst_x_s = tmp_ix > output_w ? output_w : tmp_ix;
    int tmp_iy = dst_y_s * output_unit;
    dst_y_s = tmp_iy > output_h ? output_h : tmp_iy;

    int src_tile_offset = i * oc8 * C8NUM * input_unit * input_unit;
    int dst_tile_offset = output_channel * (dst_x_s + dst_y_s * output_w);

    for (int j = 0; j < oc8; j++) {
      int r_c = output_channel - j * C8NUM;
      r_c = r_c > C8NUM ? C8NUM : r_c;
      int src_oc8_offset = src_tile_offset + j * input_unit * input_unit * C8NUM;
      int dst_oc8_offset = dst_tile_offset + j * C8NUM;
      const float16_t *src_ptr = gemm_out + src_oc8_offset;
      const float16_t *bias_ptr = bias_data + j * C8NUM;
      float16_t *dst_ptr = tmp_out_data + dst_oc8_offset;
      func(src_ptr, dst_ptr, bias_ptr, C8NUM, output_w, output_channel, r_w, r_h, r_c);
    }
    out_tile_index++;
  }
}

void WinogradOutputNC8HW8TransformFp16(const float16_t *gemm_out, float16_t *tmp_out_data, const float16_t *bias_data,
                                       int cal_num, int out_tile_index, int output_unit_num,
                                       const ConvParameter *conv_param, OutputTransFp16Func func) {
  int output_unit = conv_param->output_unit_;
  int output_w = conv_param->output_w_;
  int output_h = conv_param->output_h_;
  int plane = output_w * output_h;
  int output_channel = conv_param->output_channel_;
  int oc8 = UP_DIV(output_channel, C8NUM);
  int input_unit = conv_param->input_unit_;
  NNACL_CHECK_ZERO_RETURN(output_unit_num);
  for (int i = 0; i < cal_num; i++) {
    int dst_x_s = out_tile_index % output_unit_num;
    int dst_y_s = out_tile_index / output_unit_num;
    int r_w = output_w - dst_x_s * output_unit;
    r_w = r_w > output_unit ? output_unit : r_w;
    int r_h = output_h - dst_y_s * output_unit;
    r_h = r_h > output_unit ? output_unit : r_h;
    int tmp_ix = dst_x_s * output_unit;
    dst_x_s = tmp_ix > output_w ? output_w : tmp_ix;
    int tmp_iy = dst_y_s * output_unit;
    dst_y_s = tmp_iy > output_h ? output_h : tmp_iy;

    int src_tile_offset = i * oc8 * C8NUM * input_unit * input_unit;
    int dst_tile_offset = dst_x_s + dst_y_s * output_w;

    for (int j = 0; j < oc8; j++) {
      int r_c = output_channel - j * C8NUM;
      r_c = r_c > C8NUM ? C8NUM : r_c;
      int src_oc8_offset = src_tile_offset + j * input_unit * input_unit * C8NUM;
      int dst_oc8_offset = (dst_tile_offset + plane * j) * C8NUM;
      const float16_t *src_ptr = gemm_out + src_oc8_offset;
      const float16_t *bias_ptr = bias_data + j * C8NUM;
      float16_t *dst_ptr = tmp_out_data + dst_oc8_offset;
      func(src_ptr, dst_ptr, bias_ptr, C8NUM, output_w, r_c, r_w, r_h, r_c);
    }
    out_tile_index++;
  }
}

int WinogradWeightTransformFp16(const float16_t *weight_data, float16_t *winograd_data, const float *matrix_g,
                                const float *matrix_gt, int oc_block, int input_unit, int kernel_unit,
                                int filter_channel, int filter_batch, bool pack) {
  // original weight format : ohwi
  int oc_block_num = UP_DIV(filter_batch, oc_block);
  int block_stride = filter_channel * oc_block;
  int block_num_stride = block_stride * oc_block_num;

  float16_t *matrix_gt_data_fp16 = (float16_t *)(malloc(input_unit * kernel_unit * sizeof(float16_t)));
  if (matrix_gt_data_fp16 == NULL) {
    return NNACL_ERRCODE_OP_FP16_WINOGRAD_GENERATOR;
  }
  Float32ToFloat16(matrix_gt, matrix_gt_data_fp16, input_unit * kernel_unit);

  // trans_filter = G*g*GT (g represents weight_data) = [(g * (G)T)T * (G)T]T
  // separate into two steps ===> tmp = (g * (G)T)T ===> out = [tmp * (G)T]T
  float16_t *tmp_data = (float16_t *)(malloc(filter_channel * input_unit * kernel_unit * sizeof(float16_t)));
  if (tmp_data == NULL) {
    free(matrix_gt_data_fp16);
    return NNACL_ERRCODE_OP_FP16_WINOGRAD_GENERATOR;
  }
  float16_t *trans_out_data = (float16_t *)(malloc(filter_channel * input_unit * input_unit * sizeof(float16_t)));
  if (trans_out_data == NULL) {
    free(tmp_data);
    free(matrix_gt_data_fp16);
    return NNACL_ERRCODE_OP_FP16_WINOGRAD_GENERATOR;
  }

#ifndef ENABLE_ARM64
  float16_t *tmp_data1 = (float16_t *)(malloc(filter_channel * input_unit * kernel_unit * sizeof(float16_t)));
  if (tmp_data1 == NULL) {
    free(tmp_data);
    free(matrix_gt_data_fp16);
    free(trans_out_data);
    return NNACL_ERRCODE_OP_FP16_WINOGRAD_GENERATOR;
  }
  float16_t *trans_out_data1 = (float16_t *)(malloc(filter_channel * input_unit * input_unit * sizeof(float16_t)));
  if (trans_out_data1 == NULL) {
    free(tmp_data);
    free(tmp_data1);
    free(matrix_gt_data_fp16);
    free(trans_out_data);
    return NNACL_ERRCODE_OP_FP16_WINOGRAD_GENERATOR;
  }
#endif

  int input_oz_offset = kernel_unit * kernel_unit * filter_channel;
  for (int i = 0; i < filter_batch; i++) {
    int out_c_block = i / oc_block;
    int out_c_res = i % oc_block;
    int output_oz_offset = out_c_block * block_stride + out_c_res;

#ifndef ENABLE_ARM64
    // tmp_data = g * GT
    MatrixMultiplyWinogradFp16(weight_data + i * input_oz_offset, matrix_gt_data_fp16, tmp_data, kernel_unit,
                               kernel_unit, input_unit, filter_channel);
    // tmp_data1 = (tmp_data)T
    PackHWCToWHCFp16(tmp_data, tmp_data1, kernel_unit, input_unit, filter_channel);
    // trans_out_data1 = tmp * GT
    MatrixMultiplyWinogradFp16(tmp_data1, matrix_gt_data_fp16, trans_out_data1, input_unit, kernel_unit, input_unit,
                               filter_channel);
    // trans_out_data = (trans_out_data1)T
    PackHWCToWHCFp16(trans_out_data1, trans_out_data, input_unit, input_unit, filter_channel);
#else
    // tmp = (g * GT)T
    MatrixMultiplyWinogradFp16(weight_data + i * input_oz_offset, matrix_gt_data_fp16, tmp_data, kernel_unit,
                               kernel_unit, input_unit, filter_channel);
    // trans = (tmp * GT)T
    MatrixMultiplyWinogradFp16(tmp_data, matrix_gt_data_fp16, trans_out_data, input_unit, kernel_unit, input_unit,
                               filter_channel);
#endif

    if (pack) {
      int in_offset = 0;
      for (int j = 0; j < input_unit; ++j) {
        for (int k = 0; k < input_unit; ++k) {
          for (int c = 0; c < filter_channel; ++c) {
            *(winograd_data + output_oz_offset + c * oc_block) = trans_out_data[in_offset + c];
          }
          in_offset += filter_channel;
          output_oz_offset += block_num_stride;
        }
      }
    } else {
      memcpy(winograd_data + i * filter_channel * input_unit * input_unit, trans_out_data,
             filter_channel * input_unit * input_unit * sizeof(float16_t));
    }
  }

#ifndef ENABLE_ARM64
  free(tmp_data1);
  free(trans_out_data1);
#endif
  free(tmp_data);
  free(trans_out_data);
  free(matrix_gt_data_fp16);
  return NNACL_OK;
}
