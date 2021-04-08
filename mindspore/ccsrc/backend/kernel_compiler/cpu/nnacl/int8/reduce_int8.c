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

#include <stdint.h>
#include "nnacl/int8/reduce_int8.h"
#include "nnacl/errorcode.h"
#include "nnacl/int8/fixed_point.h"
#include "nnacl/common_func.h"

int ReduceMeanN(int n, int h, int w, int c, int8_t *in_data, int8_t *out_data, QuantMulArg quant_arg) {
  return NNACL_OK;
}
int ReduceMeanH(int n, int h, int w, int c, int8_t *in_data, int8_t *out_data, QuantMulArg quant_arg) {
  return NNACL_OK;
}
int ReduceMeanW(int n, int h, int w, int c, int8_t *in_data, int8_t *out_data, QuantMulArg quant_arg) {
  return NNACL_OK;
}
int ReduceMeanC(int n, int h, int w, int c, int8_t *in_data, int8_t *out_data, QuantMulArg quant_arg) {
  return NNACL_OK;
}
int ReduceMeanNH(int n, int h, int w, int c, int8_t *in_data, int8_t *out_data, QuantMulArg quant_arg) {
  return NNACL_OK;
}
int ReduceMeanNW(int n, int h, int w, int c, int8_t *in_data, int8_t *out_data, QuantMulArg quant_arg) {
  return NNACL_OK;
}
int ReduceMeanNC(int n, int h, int w, int c, int8_t *in_data, int8_t *out_data, QuantMulArg quant_arg) {
  return NNACL_OK;
}
int ReduceMeanHW(int n, int plane, int count, int c, int8_t *in_data, int8_t *out_data, QuantMulArg quant_arg,
                 int32_t bias) {
  int stride = plane * UP_ROUND(c, C4NUM);
  for (int batch = 0; batch < n; ++batch) {
    int8_t *in_ptr = in_data + batch * stride;
    int8_t *out_ptr = out_data + batch * c;
    for (int i = 0; i < count; ++i) {
      int32_t sum_array = 0;
      int j = 0;
#ifdef ENABLE_ARM64
      for (; j < plane; j += 16) {
        int8x16_t in_data_vec = vld1q_s8(in_ptr);
        sum_array += vaddlvq_s8(in_data_vec);
        in_ptr += 16;
      }
      for (; j < plane; j += 8) {
        int8x8_t in_data_vec = vld1_s8(in_ptr);
        sum_array += vaddlv_s8(in_data_vec);
        in_ptr += 8;
      }
      for (; j < plane; j += 4) {
        int32x4_t in_data_vec;
        in_data_vec[0] = in_ptr[0];
        in_data_vec[1] = in_ptr[1];
        in_data_vec[2] = in_ptr[2];
        in_data_vec[3] = in_ptr[3];
        sum_array += vaddvq_s32(in_data_vec);
        in_ptr += 4;
      }
#elif ENABLE_ARM32
      int32x4_t accum = vmovq_n_s32(0);
      for (; j < plane; j += 16) {
        int32x4_t in_data_vec1;
        int32x4_t in_data_vec2;
        int32x4_t in_data_vec3;
        int32x4_t in_data_vec4;
        in_data_vec1[0] = in_ptr[0];
        in_data_vec1[1] = in_ptr[1];
        in_data_vec1[2] = in_ptr[2];
        in_data_vec1[3] = in_ptr[3];
        in_data_vec2[0] = in_ptr[4];
        in_data_vec2[1] = in_ptr[5];
        in_data_vec2[2] = in_ptr[6];
        in_data_vec2[3] = in_ptr[7];
        in_data_vec3[0] = in_ptr[8];
        in_data_vec3[1] = in_ptr[9];
        in_data_vec3[2] = in_ptr[10];
        in_data_vec3[3] = in_ptr[11];
        in_data_vec4[0] = in_ptr[12];
        in_data_vec4[1] = in_ptr[13];
        in_data_vec4[2] = in_ptr[14];
        in_data_vec4[3] = in_ptr[15];
        accum = vaddq_s32(accum, in_data_vec1);
        accum = vaddq_s32(accum, in_data_vec2);
        accum = vaddq_s32(accum, in_data_vec3);
        accum = vaddq_s32(accum, in_data_vec4);
        in_ptr += 16;
      }
      for (; j < plane; j += 8) {
        int32x4_t in_data_vec1;
        int32x4_t in_data_vec2;
        in_data_vec1[0] = in_ptr[0];
        in_data_vec1[1] = in_ptr[1];
        in_data_vec1[2] = in_ptr[2];
        in_data_vec1[3] = in_ptr[3];
        in_data_vec2[0] = in_ptr[4];
        in_data_vec2[1] = in_ptr[5];
        in_data_vec2[2] = in_ptr[6];
        in_data_vec2[3] = in_ptr[7];
        accum = vaddq_s32(accum, in_data_vec1);
        accum = vaddq_s32(accum, in_data_vec2);
        in_ptr += 8;
      }
      for (; j < plane; j += 4) {
        int32x4_t in_data_vec;
        in_data_vec[0] = in_ptr[0];
        in_data_vec[1] = in_ptr[1];
        in_data_vec[2] = in_ptr[2];
        in_data_vec[3] = in_ptr[3];
        accum = vaddq_s32(accum, in_data_vec);
        in_ptr += 4;
      }
      sum_array += accum[0];
      sum_array += accum[1];
      sum_array += accum[2];
      sum_array += accum[3];
#endif
      for (; j < plane; j++) {
        sum_array += in_ptr[0];
        in_ptr++;
      }
      int32_t mean =
        RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(sum_array * (1 << (unsigned int)quant_arg.left_shift_),
                                                              quant_arg.multiplier_),
                            quant_arg.right_shift_);
      mean += bias;
      *out_ptr++ = MSMAX(MSMIN(mean, INT8_MAX), INT8_MIN);
    }
  }
  return NNACL_OK;
}

int ReduceMeanHC(int n, int h, int w, int c, int8_t *in_data, int8_t *out_data, QuantMulArg quant_arg) {
  return NNACL_OK;
}

int ReduceMeanWC(int n, int h, int w, int c, int8_t *in_data, int8_t *out_data, QuantMulArg quant_arg) {
  return NNACL_OK;
}

int ReduceMeanNHW(int n, int h, int w, int c, int8_t *in_data, int8_t *out_data, QuantMulArg quant_arg) {
  return NNACL_OK;
}

int ReduceMeanNHC(int n, int h, int w, int c, int8_t *in_data, int8_t *out_data, QuantMulArg quant_arg) {
  return NNACL_OK;
}

int ReduceMeanNWC(int n, int h, int w, int c, int8_t *in_data, int8_t *out_data, QuantMulArg quant_arg) {
  return NNACL_OK;
}

int ReduceMeanHWC(int n, int h, int w, int c, int8_t *in_data, int8_t *out_data, QuantMulArg quant_arg) {
  return NNACL_OK;
}

int ReduceMeanNHWC(int n, int h, int w, int c, int8_t *in_data, int8_t *out_data, QuantMulArg quant_arg) {
  return NNACL_OK;
}

// Get x such that (x-zp_in) * scale_in = mean
// Assuming reduce n axes, this works for first n-1 reduce. One call for one reduce.
int ReduceMeanInt8(const int outer_size, const int inner_size, const int axis_size, const int32_t *src_data,
                   int32_t *dst_data, const ReduceQuantArg *quant, const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const int32_t *outer_src = src_data + j * axis_size * inner_size;
    int32_t *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int32_t *inner_src = outer_src + k;
      int32_t *inner_dst = outer_dst + k;
      int32_t sum = 0;
      for (i = 0; i < axis_size; i++) {
        int32_t tmp = inner_src[i * inner_size] - quant->in_zp_;
        if (isAddOverflow(sum, tmp)) {
          return NNACL_ERRCODE_ADD_OVERFLOW;
        }
        sum += tmp;
      }
      int32_t mean = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(sum * (1 << (unsigned int)quant->mean_left_shift_), quant->mean_multiplier_),
        quant->mean_right_shift_);
      if (isAddOverflow(mean, quant->in_zp_)) {
        return NNACL_ERRCODE_ADD_OVERFLOW;
      }
      *inner_dst = mean + quant->in_zp_;
    }
  }
  return NNACL_OK;
}

// suppose reduce n axes, this works for last reduce axis.
// get y such that (y-zp_out) * scale_out = mean(x-zp_in)*scale_in
int ReduceMeanLastAxis(const int outer_size, const int inner_size, const int axis_size, const int32_t *src_data,
                       int8_t *dst_data, const ReduceQuantArg *quant, const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const int32_t *outer_src = src_data + j * axis_size * inner_size;
    int8_t *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int32_t *inner_src = outer_src + k;
      int8_t *inner_dst = outer_dst + k;
      int32_t sum = 0;
      for (i = 0; i < axis_size; i++) {
        int32_t tmp = inner_src[i * inner_size] - quant->in_zp_;
        if (isAddOverflow(tmp, sum)) {
          return NNACL_ERRCODE_ADD_OVERFLOW;
        }
        sum += tmp;
      }
      int32_t mean = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(sum * (1 << (unsigned int)quant->mean_left_shift_), quant->mean_multiplier_),
        quant->mean_right_shift_);
      // trans to output scale
      int32_t mean_scaled =
        RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(mean * (1 << (unsigned int)quant->in_out_left_shift_),
                                                              quant->in_out_multiplier_),
                            quant->in_out_right_shift_);
      if (isAddOverflow(mean_scaled, quant->out_zp_)) {
        return NNACL_ERRCODE_ADD_OVERFLOW;
      }
      mean = mean_scaled + quant->out_zp_;

      *inner_dst = MSMAX(MSMIN(mean, INT8_MAX), INT8_MIN);
    }
  }
  return NNACL_OK;
}

// Get x such that (x-zp_in) * scale_in = sum(item-zp_in)*scale_in
// Assuming reduce n axes, this works for first n-1 reduce. One call for one reduce.
int ReduceSumInt8(const int outer_size, const int inner_size, const int axis_size, const int32_t *src_data,
                  int32_t *dst_data, const ReduceQuantArg *quant, const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const int32_t *outer_src = src_data + j * axis_size * inner_size;
    int32_t *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int32_t *inner_src = outer_src + k;
      int32_t *inner_dst = outer_dst + k;
      int32_t sum = 0;
      for (i = 0; i < axis_size; i++) {
        int32_t tmp = inner_src[i * inner_size] - quant->in_zp_;
        if (isAddOverflow(tmp, sum)) {
          return NNACL_ERRCODE_ADD_OVERFLOW;
        }
        sum += tmp;
      }

      if (isAddOverflow(quant->in_zp_, sum)) {
        return NNACL_ERRCODE_ADD_OVERFLOW;
      }
      *inner_dst = sum + quant->in_zp_;
    }
  }
  return NNACL_OK;
}

// suppose reduce n axes, this works for last reduce axis.
// get y such that (y-zp_out) * scale_out = sum(item-zp_in)*scale_in
int ReduceSumLastAxis(const int outer_size, const int inner_size, const int axis_size, const int32_t *src_data,
                      int8_t *dst_data, const ReduceQuantArg *quant, const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const int32_t *outer_src = src_data + j * axis_size * inner_size;
    int8_t *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int32_t *inner_src = outer_src + k;
      int8_t *inner_dst = outer_dst + k;
      int32_t sum = 0;
      for (i = 0; i < axis_size; i++) {
        int32_t tmp = inner_src[i * inner_size] - quant->in_zp_;
        if (isAddOverflow(tmp, sum)) {
          return NNACL_ERRCODE_ADD_OVERFLOW;
        }
        sum += tmp;
      }
      int32_t sum_scaled =
        RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(sum * (1 << (unsigned int)quant->in_out_left_shift_),
                                                              quant->in_out_multiplier_),
                            quant->in_out_right_shift_);
      if (isAddOverflow(sum_scaled, quant->out_zp_)) {
        return NNACL_ERRCODE_ADD_OVERFLOW;
      }
      sum = sum_scaled + quant->out_zp_;
      if (sum > INT8_MAX) {
        *inner_dst = INT8_MAX;
      } else if (sum < INT8_MIN) {
        *inner_dst = INT8_MIN;
      } else {
        *inner_dst = (int8_t)sum;
      }
    }
  }
  return NNACL_OK;
}

int ReduceMaxLastAxis(const int outer_size, const int inner_size, const int axis_size, const int32_t *src_data,
                      int8_t *dst_data, const ReduceQuantArg *quant, const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const int32_t *outer_src = src_data + j * axis_size * inner_size;
    int8_t *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int32_t *inner_src = outer_src + k;
      int8_t *inner_dst = outer_dst + k;
      int32_t tmp = INT8_MIN;
      for (i = 0; i < axis_size; i++) {
        tmp = tmp > inner_src[i * inner_size] ? tmp : inner_src[i * inner_size];
      }
      int32_t tmp_scaled = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul((tmp - quant->in_zp_) * (1 << (unsigned int)quant->in_out_left_shift_),
                                          quant->in_out_multiplier_),
        quant->in_out_right_shift_);
      if (isAddOverflow(tmp_scaled, quant->out_zp_)) {
        return NNACL_ERRCODE_ADD_OVERFLOW;
      }
      tmp = tmp_scaled + quant->out_zp_;
      if (tmp > INT8_MAX) {
        *inner_dst = INT8_MAX;
      } else if (tmp < INT8_MIN) {
        *inner_dst = INT8_MIN;
      } else {
        *inner_dst = (int8_t)tmp;
      }
    }
  }
  return NNACL_OK;
}

int ReduceMaxInt8(const int outer_size, const int inner_size, const int axis_size, const int32_t *src_data,
                  int32_t *dst_data, const ReduceQuantArg *quant, const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const int32_t *outer_src = src_data + j * axis_size * inner_size;
    int32_t *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int32_t *inner_src = outer_src + k;
      int32_t *inner_dst = outer_dst + k;
      int32_t tmp = INT8_MIN;
      for (i = 0; i < axis_size; i++) {
        tmp = tmp > inner_src[i * inner_size] ? tmp : inner_src[i * inner_size];
      }

      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}

int ReduceMinLastAxis(const int outer_size, const int inner_size, const int axis_size, const int32_t *src_data,
                      int8_t *dst_data, const ReduceQuantArg *quant, const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  const int base_offset = 20;
  for (j = tid; j < outer_size; j += thread_num) {
    const int32_t *outer_src = src_data + j * axis_size * inner_size;
    int8_t *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int32_t *inner_src = outer_src + k;
      int8_t *inner_dst = outer_dst + k;
      int32_t tmp = INT8_MAX;
      for (i = 0; i < axis_size; i++) {
        tmp = tmp < inner_src[i * inner_size] ? tmp : inner_src[i * inner_size];
      }
      int32_t tmp_scaled =
        RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(
                              (tmp - quant->in_zp_) * (1 << ((unsigned int)quant->in_out_left_shift_ + base_offset)),
                              quant->in_out_multiplier_),
                            quant->in_out_right_shift_ + base_offset);
      if (isAddOverflow(tmp_scaled, quant->out_zp_)) {
        return NNACL_ERRCODE_ADD_OVERFLOW;
      }
      tmp = tmp_scaled + quant->out_zp_;
      if (tmp > INT8_MAX) {
        *inner_dst = INT8_MAX;
      } else if (tmp < INT8_MIN) {
        *inner_dst = INT8_MIN;
      } else {
        *inner_dst = (int8_t)tmp;
      }
    }
  }
  return NNACL_OK;
}

int ReduceMinInt8(const int outer_size, const int inner_size, const int axis_size, const int32_t *src_data,
                  int32_t *dst_data, const ReduceQuantArg *quant, const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const int32_t *outer_src = src_data + j * axis_size * inner_size;
    int32_t *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int32_t *inner_src = outer_src + k;
      int32_t *inner_dst = outer_dst + k;
      int32_t tmp = INT8_MAX;
      for (i = 0; i < axis_size; i++) {
        tmp = tmp < inner_src[i * inner_size] ? tmp : inner_src[i * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}

int ReduceProdLastAxis(const int outer_size, const int inner_size, const int axis_size, const int32_t *src_data,
                       int8_t *dst_data, const ReduceQuantArg *quant, const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const int32_t *outer_src = src_data + j * axis_size * inner_size;
    int8_t *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int32_t *inner_src = outer_src + k;
      int8_t *inner_dst = outer_dst + k;
      int32_t prod = 1;
      for (i = 0; i < axis_size; i++) {
        int32_t tmp = inner_src[i * inner_size] - quant->in_zp_;
        if (isMulOverflow(prod, tmp)) {
          return NNACL_ERRCODE_MUL_OVERFLOW;
        }
        prod *= tmp;
      }
      prod = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(prod * (1 << (unsigned int)quant->prod_left_shift_), quant->prod_multiplier_),
        quant->prod_right_shift_);
      int32_t prod_scaled =
        RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(prod * (1 << (unsigned int)quant->in_out_left_shift_),
                                                              quant->in_out_multiplier_),
                            quant->in_out_right_shift_);
      if (isAddOverflow(prod_scaled, quant->out_zp_)) {
        return NNACL_ERRCODE_ADD_OVERFLOW;
      }
      prod = prod_scaled + quant->out_zp_;
      if (prod > INT8_MAX) {
        *inner_dst = INT8_MAX;
      } else if (prod < INT8_MIN) {
        *inner_dst = INT8_MIN;
      } else {
        *inner_dst = (int8_t)prod;
      }
    }
  }
  return NNACL_OK;
}

int ReduceProdInt8(const int outer_size, const int inner_size, const int axis_size, const int32_t *src_data,
                   int32_t *dst_data, const ReduceQuantArg *quant, const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const int32_t *outer_src = src_data + j * axis_size * inner_size;
    int32_t *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int32_t *inner_src = outer_src + k;
      int32_t *inner_dst = outer_dst + k;
      int32_t prod = 1;
      for (i = 0; i < axis_size; i++) {
        int32_t tmp = inner_src[i * inner_size] - quant->in_zp_;
        if (isMulOverflow(prod, tmp)) {
          return NNACL_ERRCODE_MUL_OVERFLOW;
        }
        prod *= tmp;
      }
      prod = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(prod * (1 << (unsigned int)quant->prod_left_shift_), quant->prod_multiplier_),
        quant->prod_right_shift_);
      if (isAddOverflow(prod, quant->in_zp_)) {
        return NNACL_ERRCODE_ADD_OVERFLOW;
      }
      *inner_dst = prod + quant->in_zp_;
    }
  }
  return NNACL_OK;
}

int ReduceSumSquareLastAxis(const int outer_size, const int inner_size, const int axis_size, const int32_t *src_data,
                            int8_t *dst_data, const ReduceQuantArg *quant, const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const int32_t *outer_src = src_data + j * axis_size * inner_size;
    int8_t *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int32_t *inner_src = outer_src + k;
      int8_t *inner_dst = outer_dst + k;
      int32_t sum = 0;
      for (i = 0; i < axis_size; i++) {
        int32_t tmp;
        if (isMulOverflow(inner_src[i * inner_size] - quant->in_zp_, inner_src[i * inner_size] - quant->in_zp_)) {
          return NNACL_ERRCODE_MUL_OVERFLOW;
        }
        tmp = (inner_src[i * inner_size] - quant->in_zp_) * (inner_src[i * inner_size] - quant->in_zp_);
        if (isAddOverflow(sum, tmp)) {
          return NNACL_ERRCODE_ADD_OVERFLOW;
        }
        sum += tmp;
      }
      int32_t sum_scaled =
        RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(sum * (1 << (unsigned int)quant->sum_square_left_shift_),
                                                              quant->sum_square_multiplier_),
                            quant->sum_square_right_shift_);
      if (isAddOverflow(sum_scaled, quant->out_zp_)) {
        return NNACL_ERRCODE_ADD_OVERFLOW;
      }
      sum = sum_scaled + quant->out_zp_;

      if (sum > INT8_MAX) {
        *inner_dst = INT8_MAX;
      } else if (sum < INT8_MIN) {
        *inner_dst = INT8_MIN;
      } else {
        *inner_dst = (int8_t)sum;
      }
    }
  }
  return NNACL_OK;
}

int ReduceSumSquareInt8(const int outer_size, const int inner_size, const int axis_size, const int32_t *src_data,
                        int32_t *dst_data, const ReduceQuantArg *quant, const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const int32_t *outer_src = src_data + j * axis_size * inner_size;
    int32_t *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int32_t *inner_src = outer_src + k;
      int32_t *inner_dst = outer_dst + k;
      int32_t sum = 0;
      for (i = 0; i < axis_size; i++) {
        int32_t tmp;
        if (isMulOverflow(inner_src[i * inner_size] - quant->in_zp_, inner_src[i * inner_size] - quant->in_zp_)) {
          return NNACL_ERRCODE_MUL_OVERFLOW;
        }
        tmp = (inner_src[i * inner_size] - quant->in_zp_) * (inner_src[i * inner_size] - quant->in_zp_);
        if (isAddOverflow(sum, tmp)) {
          return NNACL_ERRCODE_ADD_OVERFLOW;
        }
        sum += tmp;
      }
      sum =
        RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(sum * (1 << (unsigned int)quant->sum_square_left_shift_),
                                                              quant->sum_square_multiplier_),
                            quant->sum_square_right_shift_);
      if (isAddOverflow(sum, quant->in_zp_)) {
        return NNACL_ERRCODE_ADD_OVERFLOW;
      }
      *inner_dst = sum + quant->in_zp_;
    }
  }
  return NNACL_OK;
}
