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

#include "nnacl/fp32/conv_common_fp32.h"
#include <string.h>
#ifdef ENABLE_AVX
#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif
#endif
#include "nnacl/fp32/matmul_fp32.h"
void Im2ColPackUnitFp32(const float *input_data, const ConvParameter *conv_param, float *packed_input, int real_cal_num,
                        int block_index) {
  // input format : nhwc
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int kernel_plane = kernel_h * kernel_w;
  int dilation_h = conv_param->dilation_h_;
  int dilation_w = conv_param->dilation_w_;
  int out_w = conv_param->output_w_;
  if (dilation_h == 0 || dilation_w == 0 || out_w == 0) {
    return;
  }
  int in_channel = conv_param->input_channel_;
  int in_w = conv_param->input_w_;
  for (int i = 0; i < real_cal_num; i++) {
    int block_start = block_index + i;
    int input_h = block_start / out_w * conv_param->stride_h_ - conv_param->pad_u_;
    int input_w = block_start % out_w * conv_param->stride_w_ - conv_param->pad_l_;
    if (conv_param->input_h_ - input_h < 0 || in_w - input_w < 0) {
      continue;
    }
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

// fp32 conv common
void ConvFp32(const float *input_data, float *packed_input, const float *packed_weight, const float *bias_data,
              float *col_major_input, float *output_data, int task_id, const ConvParameter *conv_param) {
  if (conv_param->thread_num_ == 0) {
    return;
  }
  Row2ColMajorFuncPtr Row2ColMajor = NULL;
  int output_hw = conv_param->output_h_ * conv_param->output_w_;
#ifdef ENABLE_AVX
  Row2ColMajor = RowMajor2Col6Major;
  const int cal_num = C6NUM;
#elif defined(ENABLE_SSE)
  Row2ColMajor = RowMajor2Col4Major;
  const int cal_num = C4NUM;
#elif defined(ENABLE_ARM64)
  MatmulFloatOptFuncPtr MatmulFloatOpt = NULL;
  int cal_num = 0;
  if (output_hw <= C4NUM) {
    Row2ColMajor = RowMajor2Col4Major;
    MatmulFloatOpt = MatmulFloatNeon64OptRow4;
    cal_num = C4NUM;
  } else if (output_hw <= C8NUM) {
    Row2ColMajor = RowMajor2Col8Major;
    MatmulFloatOpt = MatmulFloatNeon64OptRow8;
    cal_num = C8NUM;
  } else {
    Row2ColMajor = RowMajor2Col12Major;
    MatmulFloatOpt = MatmulFloatNeon64OptRow12;
    cal_num = C12NUM;
  }
#elif defined(ENABLE_ARM32)
  Row2ColMajor = RowMajor2Col12Major;
  const int cal_num = C12NUM;
#else
  Row2ColMajor = RowMajor2Col12Major;
  const int cal_num = C12NUM;
#endif

  int block_per_thread = UP_DIV(UP_DIV(output_hw, cal_num), conv_param->thread_num_);
  int start_block = block_per_thread * task_id;
  int start_hw = start_block * cal_num;
  int end_hw = MSMIN(output_hw, (start_block + block_per_thread) * cal_num);
  if (start_hw >= end_hw) {
    return;
  }
  int out_stride = conv_param->output_channel_ * cal_num;
  int deep = conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;
  packed_input += task_id * deep * cal_num;
  col_major_input += task_id * deep * cal_num;
  size_t input_size = deep * cal_num * sizeof(float);

  for (int b = 0; b < conv_param->input_batch_; b++) {
    int out_channel = conv_param->output_channel_;
    int in_offset = b * conv_param->input_channel_ * conv_param->input_h_ * conv_param->input_w_;
    int out_offset = b * out_channel * output_hw + start_hw * out_channel;
    for (int i = start_hw; i < end_hw; i += cal_num, out_offset += out_stride) {
      int real_cal_row = MSMIN(output_hw - i, cal_num);
      memset(packed_input, 0, input_size);
      Im2ColPackUnitFp32(input_data + in_offset, conv_param, packed_input, real_cal_row, i);
      Row2ColMajor(packed_input, col_major_input, cal_num, deep);
      float *gemm_output = output_data + out_offset;
// x86 func param types are different
#if ENABLE_AVX
      MatmulFloatAvxOpt(col_major_input, packed_weight, gemm_output, bias_data, (size_t)conv_param->act_type_, deep,
                        real_cal_row, out_channel, (size_t)out_channel, (size_t)OutType_Nhwc);
#elif ENABLE_SSE
      MatmulFloatSse64Opt(col_major_input, packed_weight, gemm_output, bias_data, (int)conv_param->act_type_, deep,
                          real_cal_row, out_channel, (size_t)out_channel, (int)OutType_Nhwc);
#elif ENABLE_ARM32
      MatmulFloatNeon32Opt12x4(col_major_input, packed_weight, gemm_output, bias_data, (int)conv_param->act_type_, deep,
                               real_cal_row, out_channel, out_channel, OutType_Nhwc);
#elif ENABLE_ARM64
      MatmulFloatOpt(col_major_input, packed_weight, gemm_output, bias_data, conv_param->act_type_, deep, real_cal_row,
                     out_channel, out_channel, OutType_Nhwc);
#else
      MatMul12x8(col_major_input, packed_weight, gemm_output, bias_data, (int)conv_param->act_type_, deep, real_cal_row,
                 out_channel, out_channel, OutType_Nhwc);
#endif
    }
  }
}

// fp32 conv common
void ConvFp32CutByBatch(const float *input_data, float *packed_input, const float *packed_weight,
                        const float *bias_data, float *col_major_input, float *output_data, int task_id,
                        const ConvParameter *conv_param) {
  if (conv_param->thread_num_ == 0) {
    return;
  }
  int output_hw = conv_param->output_h_ * conv_param->output_w_;
  Row2ColMajorFuncPtr Row2ColMajor = NULL;
#ifdef ENABLE_AVX
  const int cal_num = C6NUM;
  Row2ColMajor = RowMajor2Col6Major;
#elif defined(ENABLE_SSE)
  const int cal_num = C4NUM;
  Row2ColMajor = RowMajor2Col4Major;
#elif defined(ENABLE_ARM64)
  int cal_num = 0;
  MatmulFloatOptFuncPtr MatmulFloatOpt = NULL;
  if (output_hw <= C4NUM) {
    cal_num = C4NUM;
    Row2ColMajor = RowMajor2Col4Major;
    MatmulFloatOpt = MatmulFloatNeon64OptRow4;
  } else if (output_hw <= C8NUM) {
    cal_num = C8NUM;
    Row2ColMajor = RowMajor2Col8Major;
    MatmulFloatOpt = MatmulFloatNeon64OptRow8;
  } else {
    cal_num = C12NUM;
    Row2ColMajor = RowMajor2Col12Major;
    MatmulFloatOpt = MatmulFloatNeon64OptRow12;
  }
#elif defined(ENABLE_ARM32)
  const int cal_num = C12NUM;
  Row2ColMajor = RowMajor2Col12Major;
#else
  const int cal_num = C12NUM;
  Row2ColMajor = RowMajor2Col12Major;
#endif

  int block_batch_per_thread = UP_DIV(conv_param->input_batch_, conv_param->thread_num_);
  int start_batch = block_batch_per_thread * task_id;
  int end_batch = MSMIN(conv_param->input_batch_, (start_batch + block_batch_per_thread));

  int out_stride = conv_param->output_channel_ * cal_num;
  int deep = conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;
  packed_input += task_id * deep * cal_num;
  col_major_input += task_id * deep * cal_num;
  size_t input_size = deep * cal_num * sizeof(float);

  for (int b = start_batch; b < end_batch; b++) {
    int out_channel = conv_param->output_channel_;
    int in_offset = b * conv_param->input_channel_ * conv_param->input_h_ * conv_param->input_w_;
    int out_offset = b * out_channel * output_hw;
    for (int i = 0; i < output_hw; i += cal_num, out_offset += out_stride) {
      int real_cal_row = MSMIN(output_hw - i, cal_num);
      memset(packed_input, 0, input_size);
      Im2ColPackUnitFp32(input_data + in_offset, conv_param, packed_input, real_cal_row, i);
      Row2ColMajor(packed_input, col_major_input, cal_num, deep);
      float *gemm_output = output_data + out_offset;
// x86 func param types are different
#if ENABLE_AVX
      MatmulFloatAvxOpt(col_major_input, packed_weight, gemm_output, bias_data, (size_t)conv_param->act_type_, deep,
                        real_cal_row, out_channel, (size_t)out_channel, (size_t)OutType_Nhwc);
#elif ENABLE_SSE
      MatmulFloatSse64Opt(col_major_input, packed_weight, gemm_output, bias_data, (int)conv_param->act_type_, deep,
                          real_cal_row, out_channel, (size_t)out_channel, (int)OutType_Nhwc);
#elif ENABLE_ARM32
      MatmulFloatNeon32Opt12x4(col_major_input, packed_weight, gemm_output, bias_data, (int)conv_param->act_type_, deep,
                               real_cal_row, out_channel, out_channel, OutType_Nhwc);
#elif ENABLE_ARM64
      MatmulFloatOpt(col_major_input, packed_weight, gemm_output, bias_data, conv_param->act_type_, deep, real_cal_row,
                     out_channel, out_channel, OutType_Nhwc);
#else
      MatMul12x8(col_major_input, packed_weight, gemm_output, bias_data, (int)conv_param->act_type_, deep, real_cal_row,
                 out_channel, out_channel, OutType_Nhwc);
#endif
    }
  }
}

#if defined(ENABLE_AVX) || defined(ENABLE_ARM64)
void ConvFp32OutNC4HW4(const float *input_data, float *packed_input, const float *packed_weight, const float *bias_data,
                       float *col_major_input, float *output_data, int task_id, const ConvParameter *conv_param) {
  if (conv_param->thread_num_ == 0) {
    return;
  }
  int output_hw = conv_param->output_h_ * conv_param->output_w_;
  int out_channel = conv_param->output_channel_;
  int input_hw = conv_param->input_h_ * conv_param->input_w_;
  int in_channel = conv_param->input_channel_;
  Row2ColMajorFuncPtr Row2ColMajor = NULL;
  int cal_num = 0;
  int out_tile = 0;
#ifdef ENABLE_AVX
  cal_num = C6NUM;
  out_tile = C8NUM;
  Row2ColMajor = RowMajor2Col6Major;
  int align_channel = UP_DIV(out_channel, C16NUM) * C16NUM;
#else
  out_tile = C4NUM;
  MatmulFloatOptFuncPtr MatmulFloatOpt = NULL;
  if (output_hw <= C4NUM) {
    cal_num = C4NUM;
    Row2ColMajor = RowMajor2Col4Major;
    MatmulFloatOpt = MatmulFloatNeon64OptRow4;
  } else if (output_hw <= C8NUM) {
    cal_num = C8NUM;
    Row2ColMajor = RowMajor2Col8Major;
    MatmulFloatOpt = MatmulFloatNeon64OptRow8;
  } else {
    cal_num = C12NUM;
    Row2ColMajor = RowMajor2Col12Major;
    MatmulFloatOpt = MatmulFloatNeon64OptRow12;
  }
#endif
  int block_per_thread = UP_DIV(UP_DIV(output_hw, cal_num), conv_param->thread_num_);
  int start_block = block_per_thread * task_id;
  int start_hw = start_block * cal_num;
  int end_hw = MSMIN(output_hw, (start_block + block_per_thread) * cal_num);
  if (start_hw >= end_hw) {
    return;
  }
#ifdef ENABLE_AVX
  int act_type = 0;
  if (conv_param->act_type_ == ActType_Relu6) {
    act_type += 1;
  }
  if (conv_param->act_type_ == ActType_Relu || conv_param->act_type_ == ActType_Relu6) {
    act_type += 2;
  }
  int out_stride = out_tile * cal_num;
  int out_block_stride = output_hw * C8NUM;
#else
  int out_stride = MSMIN(out_channel, out_tile) * cal_num;
#endif
  int deep = conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;
  packed_input += task_id * deep * cal_num;
  col_major_input += task_id * deep * cal_num;
  size_t input_size = deep * cal_num * sizeof(float);

  for (int b = 0; b < conv_param->input_batch_; b++) {
    int in_offset = b * in_channel * input_hw;
#ifdef ENABLE_AVX
    int out_offset = b * align_channel * output_hw + start_hw * out_tile;
#else
    int out_offset = b * out_channel * output_hw + start_hw * MSMIN(out_channel, out_tile);
#endif
    for (int i = start_hw; i < end_hw; i += cal_num, out_offset += out_stride) {
      int real_cal_row = MSMIN(output_hw - i, cal_num);
      memset(packed_input, 0, input_size);
      Im2ColPackUnitFp32(input_data + in_offset, conv_param, packed_input, real_cal_row, i);
      Row2ColMajor(packed_input, col_major_input, cal_num, deep);
      float *gemm_output = output_data + out_offset;
#ifdef ENABLE_AVX
      for (int oc = 0; oc < out_channel; oc += C16NUM) {
        CommonConv6x16Kernel(gemm_output + oc * output_hw, col_major_input, packed_weight + oc * deep, bias_data + oc,
                             deep, out_block_stride, act_type, real_cal_row);
      }
#else
      MatmulFloatOpt(col_major_input, packed_weight, gemm_output, bias_data, conv_param->act_type_, deep, real_cal_row,
                     out_channel, output_hw, OutType_NC4HW4);
#endif
    }
  }
}
#endif

#ifdef ENABLE_AVX
void CommonConv6x16Kernel(float *dst, const float *src, const float *weight, const float *bias, const size_t depth,
                          const size_t out_step, const size_t act_flag, const size_t real_cal_row) {
#define Store1                   \
  _mm256_storeu_ps(dst, out[0]); \
  _mm256_storeu_ps(dst + out_step, out[1]);
#define Store2                                  \
  Store1 _mm256_storeu_ps(dst + C8NUM, out[2]); \
  _mm256_storeu_ps(dst + out_step + C8NUM, out[3]);
#define Store3                                   \
  Store2 _mm256_storeu_ps(dst + C16NUM, out[4]); \
  _mm256_storeu_ps(dst + out_step + C16NUM, out[5]);
#define Store4                                   \
  Store3 _mm256_storeu_ps(dst + C24NUM, out[6]); \
  _mm256_storeu_ps(dst + out_step + C24NUM, out[7]);
#define Store5                                   \
  Store4 _mm256_storeu_ps(dst + C32NUM, out[8]); \
  _mm256_storeu_ps(dst + out_step + C32NUM, out[9]);
#define Store6                                    \
  Store5 _mm256_storeu_ps(dst + C40NUM, out[10]); \
  _mm256_storeu_ps(dst + out_step + C40NUM, out[11]);

  __m256 out[12];
  if (bias != NULL) {
    out[0] = _mm256_loadu_ps(bias);
    out[1] = _mm256_loadu_ps(bias + C8NUM);
  } else {
    out[0] = _mm256_set1_ps(0.0f);
    out[1] = _mm256_set1_ps(0.0f);
  }
  out[2] = out[0];
  out[3] = out[1];
  out[4] = out[0];
  out[5] = out[1];
  out[6] = out[0];
  out[7] = out[1];
  out[8] = out[0];
  out[9] = out[1];
  out[10] = out[0];
  out[11] = out[1];
  for (int d = 0; d < depth; ++d) {
    __m256 w1 = _mm256_loadu_ps(weight);
    __m256 w2 = _mm256_loadu_ps(weight + C8NUM);
    __m256 s1 = _mm256_set1_ps(*src);
    __m256 s2 = _mm256_set1_ps(*(src + 1));
    out[0] = _mm256_fmadd_ps(s1, w1, out[0]);
    out[1] = _mm256_fmadd_ps(s1, w2, out[1]);
    out[2] = _mm256_fmadd_ps(s2, w1, out[2]);
    out[3] = _mm256_fmadd_ps(s2, w2, out[3]);
    s1 = _mm256_set1_ps(*(src + 2));
    s2 = _mm256_set1_ps(*(src + 3));
    out[4] = _mm256_fmadd_ps(s1, w1, out[4]);
    out[5] = _mm256_fmadd_ps(s1, w2, out[5]);
    out[6] = _mm256_fmadd_ps(s2, w1, out[6]);
    out[7] = _mm256_fmadd_ps(s2, w2, out[7]);
    s1 = _mm256_set1_ps(*(src + 4));
    s2 = _mm256_set1_ps(*(src + 5));
    out[8] = _mm256_fmadd_ps(s1, w1, out[8]);
    out[9] = _mm256_fmadd_ps(s1, w2, out[9]);
    out[10] = _mm256_fmadd_ps(s2, w1, out[10]);
    out[11] = _mm256_fmadd_ps(s2, w2, out[11]);
    weight += C16NUM;
    src += C6NUM;
  }
  __m256 six = _mm256_set1_ps(6.0f);
  __m256 zero = _mm256_set1_ps(0.0f);
  if (0x1 & act_flag) {  // relu6
    out[0] = _mm256_min_ps(out[0], six);
    out[1] = _mm256_min_ps(out[1], six);
    out[2] = _mm256_min_ps(out[2], six);
    out[3] = _mm256_min_ps(out[3], six);
    out[4] = _mm256_min_ps(out[4], six);
    out[5] = _mm256_min_ps(out[5], six);
    out[6] = _mm256_min_ps(out[6], six);
    out[7] = _mm256_min_ps(out[7], six);
    out[8] = _mm256_min_ps(out[8], six);
    out[9] = _mm256_min_ps(out[9], six);
    out[10] = _mm256_min_ps(out[10], six);
    out[11] = _mm256_min_ps(out[11], six);
  }
  if (0x2 & act_flag) {  // relu
    out[0] = _mm256_max_ps(out[0], zero);
    out[1] = _mm256_max_ps(out[1], zero);
    out[2] = _mm256_max_ps(out[2], zero);
    out[3] = _mm256_max_ps(out[3], zero);
    out[4] = _mm256_max_ps(out[4], zero);
    out[5] = _mm256_max_ps(out[5], zero);
    out[6] = _mm256_max_ps(out[6], zero);
    out[7] = _mm256_max_ps(out[7], zero);
    out[8] = _mm256_max_ps(out[8], zero);
    out[9] = _mm256_max_ps(out[9], zero);
    out[10] = _mm256_max_ps(out[10], zero);
    out[11] = _mm256_max_ps(out[11], zero);
  }
  if (real_cal_row == C6NUM) {
    Store6
  } else if (real_cal_row == C5NUM) {
    Store5
  } else if (real_cal_row == C4NUM) {
    Store4
  } else if (real_cal_row == C3NUM) {
    Store3
  } else if (real_cal_row == C2NUM) {
    Store2
  } else if (real_cal_row == C1NUM) {
    Store1
  }
}

#endif
