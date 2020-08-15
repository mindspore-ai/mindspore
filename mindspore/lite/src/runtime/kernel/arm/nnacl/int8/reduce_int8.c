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
#include "nnacl/quantization/fixed_point.h"

inline bool isAddOverflow(int32_t x, int32_t y) {
  int32_t sum = x + y;
  return (x > 0 && y > 0 && sum < 0) || (x < 0 && y < 0 && sum > 0);
}

inline bool isMulOverflow(int32_t x, int32_t y) {
  int32_t p = x * y;
  return (x != 0) && (p / x != y);
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
      // (x - zp_in) * scale_in = mean[(item - zp_in) * scale_in]
      // x = mean(item-zp_in) + zp_in
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
        // y = mean(x-zp_in) * scale + zp_out
        int32_t tmp = inner_src[i * inner_size] - quant->in_zp_;
        if (isAddOverflow(tmp, sum)) {
          return NNACL_ERRCODE_ADD_OVERFLOW;
        }
        sum += tmp;
      }
      // sum / num
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

      if (mean > INT8_MAX) {
        *inner_dst = INT8_MAX;
      } else if (mean < INT8_MIN) {
        *inner_dst = INT8_MIN;
      } else {
        *inner_dst = (int8_t)mean;
      }
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
  int base_offset = 20;
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
                              (tmp - quant->in_zp_) * (1 << (unsigned int)quant->in_out_left_shift_ + base_offset),
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
        // quant_out = prod(quant_in-zp) * (scale_in^num/scale_out) + zp_out
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
      *inner_dst = prod + quant->in_zp_;  // todo overflow
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
      // quant_out = sum((quant_in - zp)^2) * scale_in^2 / scale_out + zp_out
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
