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

#include "nnacl/fp16/scale_fp16.h"

void Fp16ScaleInner(const float16_t *in_data, float16_t *out_data, const float16_t *scale, const float16_t *offset,
                    int outer_start, int outer_end, int axis_size, int inner_size) {
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size * inner_size;
    for (int i = 0; i < axis_size; i++) {
      int axis_offset = out_offset + i * inner_size;
      int in_index = 0;
#ifdef ENABLE_NEON
      for (; in_index < inner_size - 8; in_index += 8) {
        int in_offset = axis_offset + in_index;
        float16x8_t data = vld1q_f16(in_data + in_offset);
        float16x8_t scale_8 = vdupq_n_f16(scale[i]);
        float16x8_t offset_8 = vdupq_n_f16(offset[i]);
        float16x8_t result = vfmaq_f16(offset_8, data, scale_8);

        vst1q_f16(out_data + in_offset, result);
      }
#endif
      for (; in_index < inner_size; in_index++) {
        int in_offset = axis_offset + in_index;
        out_data[in_offset] = in_data[in_offset] * scale[i] + offset[i];
      }
    }
  }
}

void Fp16ScaleAxis(const float16_t *in_data, float16_t *out_data, const float16_t *scale, const float16_t *offset,
                   int outer_start, int outer_end, int axis_size) {
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size;
    int index = 0;
#ifdef ENABLE_NEON
    for (; index < axis_size - 8; index += 8) {
      int in_offset = out_offset + index;
      float16x8_t data = vld1q_f16(in_data + in_offset);
      float16x8_t scale_8 = vld1q_f16(scale + index);
      float16x8_t offset_8 = vld1q_f16(offset + index);
      float16x8_t result = vfmaq_f16(offset_8, data, scale_8);
      vst1q_f16(out_data + in_offset, result);
    }
#endif
    for (; index < axis_size; index++) {
      int in_offset = out_offset + index;
      out_data[in_offset] = in_data[in_offset] * scale[index] + offset[index];
    }
  }
}

void DoScaleFp16(const float16_t *in_data, float16_t *out_data, const float16_t *scale, const float16_t *offset,
                 int task_id, const ScaleParameter *scale_param) {
  NNACL_CHECK_ZERO_RETURN(scale_param->op_parameter_.thread_num_);
  int outer_step = UP_DIV(scale_param->outer_size_, scale_param->op_parameter_.thread_num_);
  int outer_start = task_id * outer_step;
  int outer_end = MSMIN(outer_start + outer_step, scale_param->outer_size_);

  if (scale_param->inner_size_ == 1) {
    Fp16ScaleAxis(in_data, out_data, scale, offset, outer_start, outer_end, scale_param->axis_size_);
  } else {
    Fp16ScaleInner(in_data, out_data, scale, offset, outer_start, outer_end, scale_param->axis_size_,
                   scale_param->inner_size_);
  }
}

void Fp16ScaleInnerRelu(const float16_t *in_data, float16_t *out_data, const float16_t *scale, const float16_t *offset,
                        int outer_start, int outer_end, int axis_size, int inner_size) {
#ifdef ENABLE_NEON
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size * inner_size;
    for (int i = 0; i < axis_size; i++) {
      int axis_offset = out_offset + i * inner_size;
      int in_index = 0;
#ifdef ENABLE_NEON
      for (; in_index < inner_size - 8; in_index += 8) {
        int in_offset = axis_offset + in_index;
        float16x8_t data = vld1q_f16(in_data + in_offset);
        float16x8_t scale_8 = vdupq_n_f16(scale[i]);
        float16x8_t offset_8 = vdupq_n_f16(offset[i]);
        float16x8_t tmp = vfmaq_f16(offset_8, data, scale_8);
        float16x8_t result = vmaxq_f16(tmp, zeros);
        vst1q_f16(out_data + in_offset, result);
      }
#endif
      for (; in_index < inner_size; in_index++) {
        int in_offset = axis_offset + in_index;
        float tmp = in_data[in_offset] * scale[i] + offset[i];
        out_data[in_offset] = tmp > 0.0f ? tmp : 0.0f;
      }
    }
  }
}

void Fp16ScaleAxisRelu(const float16_t *in_data, float16_t *out_data, const float16_t *scale, const float16_t *offset,
                       int outer_start, int outer_end, int axis_size) {
#ifdef ENABLE_NEON
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size;
    int index = 0;
#ifdef ENABLE_NEON
    for (; index < axis_size - 8; index += 8) {
      int in_offset = out_offset + index;
      float16x8_t data = vld1q_f16(in_data + in_offset);
      float16x8_t scale_8 = vld1q_f16(scale + index);
      float16x8_t offset_8 = vld1q_f16(offset + index);
      float16x8_t tmp = vfmaq_f16(offset_8, data, scale_8);
      float16x8_t result = vmaxq_f16(tmp, zeros);
      vst1q_f16(out_data + in_offset, result);
    }
#endif
    for (; index < axis_size; index++) {
      int in_offset = out_offset + index;
      float tmp = in_data[in_offset] * scale[index] + offset[index];
      out_data[in_offset] = tmp > 0.0f ? tmp : 0.0f;
    }
  }
}

void Fp16DoScaleRelu(const float16_t *in_data, float16_t *out_data, const float16_t *scale, const float16_t *offset,
                     int task_id, const ScaleParameter *scale_param) {
  NNACL_CHECK_ZERO_RETURN(scale_param->op_parameter_.thread_num_);
  int outer_step = UP_DIV(scale_param->outer_size_, scale_param->op_parameter_.thread_num_);
  int outer_start = task_id * outer_step;
  int outer_end = MSMIN(outer_start + outer_step, scale_param->outer_size_);

  if (scale_param->inner_size_ == 1) {
    Fp16ScaleAxisRelu(in_data, out_data, scale, offset, outer_start, outer_end, scale_param->axis_size_);
  } else {
    Fp16ScaleInnerRelu(in_data, out_data, scale, offset, outer_start, outer_end, scale_param->axis_size_,
                       scale_param->inner_size_);
  }
}

void Fp16ScaleInnerRelu6(const float16_t *in_data, float16_t *out_data, const float16_t *scale, const float16_t *offset,
                         int outer_start, int outer_end, int axis_size, int inner_size) {
#ifdef ENABLE_NEON
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
  float16x8_t bounds = {6, 6, 6, 6, 6, 6, 6, 6};
#endif
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size * inner_size;
    for (int i = 0; i < axis_size; i++) {
      int axis_offset = out_offset + i * inner_size;
      int in_index = 0;
#ifdef ENABLE_NEON
      for (; in_index < inner_size - 8; in_index += 8) {
        int in_offset = axis_offset + in_index;
        float16x8_t data = vld1q_f16(in_data + in_offset);
        float16x8_t scale_8 = vdupq_n_f16(scale[i]);
        float16x8_t offset_8 = vdupq_n_f16(offset[i]);
        float16x8_t tmp = vfmaq_f16(offset_8, data, scale_8);
        float16x8_t result = vminq_f16(vmaxq_f16(tmp, zeros), bounds);
        vst1q_f16(out_data + in_offset, result);
      }
#endif
      for (; in_index < inner_size; in_index++) {
        int in_offset = axis_offset + in_index;
        float tmp = in_data[in_offset] * scale[i] + offset[i];
        out_data[in_offset] = MSMIN(MSMAX(tmp, 0.0f), 6.0f);
      }
    }
  }
}

void Fp16ScaleAxisRelu6(const float16_t *in_data, float16_t *out_data, const float16_t *scale, const float16_t *offset,
                        int outer_start, int outer_end, int axis_size) {
#ifdef ENABLE_NEON
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
  float16x8_t bounds = {6, 6, 6, 6, 6, 6, 6, 6};
#endif
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size;
    int index = 0;
#ifdef ENABLE_NEON
    for (; index < axis_size - 8; index += 8) {
      int in_offset = out_offset + index;
      float16x8_t data = vld1q_f16(in_data + in_offset);
      float16x8_t scale_8 = vld1q_f16(scale + index);
      float16x8_t offset_8 = vld1q_f16(offset + index);
      float16x8_t tmp = vfmaq_f16(offset_8, data, scale_8);
      float16x8_t result = vminq_f16(vmaxq_f16(tmp, zeros), bounds);
      vst1q_f16(out_data + in_offset, result);
    }
#endif
    for (; index < axis_size; index++) {
      int in_offset = out_offset + index;
      float tmp = in_data[in_offset] * scale[index] + offset[index];
      out_data[in_offset] = MSMIN(MSMAX(tmp, 0.0f), 6.0f);
    }
  }
}

void DoScaleRelu6Fp16(const float16_t *in_data, float16_t *out_data, const float16_t *scale, const float16_t *offset,
                      int task_id, const ScaleParameter *scale_param) {
  NNACL_CHECK_ZERO_RETURN(scale_param->op_parameter_.thread_num_);
  int outer_step = UP_DIV(scale_param->outer_size_, scale_param->op_parameter_.thread_num_);
  int outer_start = task_id * outer_step;
  int outer_end = MSMIN(outer_start + outer_step, scale_param->outer_size_);

  if (scale_param->inner_size_ == 1) {
    Fp16ScaleAxisRelu6(in_data, out_data, scale, offset, outer_start, outer_end, scale_param->axis_size_);
  } else {
    Fp16ScaleInnerRelu6(in_data, out_data, scale, offset, outer_start, outer_end, scale_param->axis_size_,
                        scale_param->inner_size_);
  }
}
