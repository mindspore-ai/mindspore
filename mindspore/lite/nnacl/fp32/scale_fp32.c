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

#include "nnacl/fp32/scale_fp32.h"
#ifdef ENABLE_ARM
#include <arm_neon.h>
#endif
void ScaleInner(const float *in_data, float *out_data, const float *scale, const float *offset, int outer_start,
                int outer_end, int axis_size, int inner_size) {
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size * inner_size;
    for (int i = 0; i < axis_size; i++) {
      int axis_offset = out_offset + i * inner_size;
      int in_index = 0;
#ifdef ENABLE_ARM64
      for (; in_index < inner_size - 4; in_index += 4) {
        int in_offset = axis_offset + in_index;
        float32x4_t data = vld1q_f32(in_data + in_offset);
        float32x4_t scale_4 = vdupq_n_f32(scale[i]);
        float32x4_t offset_4 = vdupq_n_f32(offset[i]);
        float32x4_t result = vfmaq_f32(offset_4, data, scale_4);
        vst1q_f32(out_data + in_offset, result);
      }
#endif
      for (; in_index < inner_size; in_index++) {
        int in_offset = axis_offset + in_index;
        out_data[in_offset] = in_data[in_offset] * scale[i] + offset[i];
      }
    }
  }
}

void ScaleAxis(const float *in_data, float *out_data, const float *scale, const float *offset, int outer_start,
               int outer_end, int axis_size) {
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size;
    int index = 0;
#ifdef ENABLE_ARM64
    for (; index < axis_size - 4; index += 4) {
      int in_offset = out_offset + index;
      float32x4_t data = vld1q_f32(in_data + in_offset);
      float32x4_t scale_4 = vld1q_f32(scale + index);
      float32x4_t offset_4 = vld1q_f32(offset + index);
      float32x4_t result = vfmaq_f32(offset_4, data, scale_4);
      vst1q_f32(out_data + in_offset, result);
    }
#endif
    for (; index < axis_size; index++) {
      int in_offset = out_offset + index;
      out_data[in_offset] = in_data[in_offset] * scale[index] + offset[index];
    }
  }
}

void DoScale(const float *in_data, float *out_data, const float *scale, const float *offset, int task_id,
             const ScaleParameter *scale_param) {
  int outer_step = UP_DIV(scale_param->outer_size_, scale_param->op_parameter_.thread_num_);
  int outer_start = task_id * outer_step;
  int outer_end = MSMIN(outer_start + outer_step, scale_param->outer_size_);

  if (scale_param->inner_size_ == 1) {
    ScaleAxis(in_data, out_data, scale, offset, outer_start, outer_end, scale_param->axis_size_);
  } else {
    ScaleInner(in_data, out_data, scale, offset, outer_start, outer_end, scale_param->axis_size_,
               scale_param->inner_size_);
  }
}

void ScaleInnerRelu(const float *in_data, float *out_data, const float *scale, const float *offset, int outer_start,
                    int outer_end, int axis_size, int inner_size) {
#ifdef ENABLE_ARM64
  float32x4_t zeros = {0, 0, 0, 0};
#endif
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size * inner_size;
    for (int i = 0; i < axis_size; i++) {
      int axis_offset = out_offset + i * inner_size;
      int in_index = 0;
#ifdef ENABLE_ARM64
      for (; in_index < inner_size - 4; in_index += 4) {
        int in_offset = axis_offset + in_index;
        float32x4_t data = vld1q_f32(in_data + in_offset);
        float32x4_t scale_4 = vdupq_n_f32(scale[i]);
        float32x4_t offset_4 = vdupq_n_f32(offset[i]);
        float32x4_t tmp = vfmaq_f32(offset_4, data, scale_4);
        float32x4_t result = vmaxq_f32(tmp, zeros);
        vst1q_f32(out_data + in_offset, result);
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

void ScaleAxisRelu(const float *in_data, float *out_data, const float *scale, const float *offset, int outer_start,
                   int outer_end, int axis_size) {
#ifdef ENABLE_ARM64
  float32x4_t zeros = {0, 0, 0, 0};
#endif
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size;
    int index = 0;
#ifdef ENABLE_ARM64
    for (; index < axis_size - 4; index += 4) {
      int in_offset = out_offset + index;
      float32x4_t data = vld1q_f32(in_data + in_offset);
      float32x4_t scale_4 = vld1q_f32(scale + index);
      float32x4_t offset_4 = vld1q_f32(offset + index);
      float32x4_t tmp = vfmaq_f32(offset_4, data, scale_4);
      float32x4_t result = vmaxq_f32(tmp, zeros);
      vst1q_f32(out_data + in_offset, result);
    }
#endif
    for (; index < axis_size; index++) {
      int in_offset = out_offset + index;
      float tmp = in_data[in_offset] * scale[index] + offset[index];
      out_data[in_offset] = tmp > 0.0f ? tmp : 0.0f;
    }
  }
}

void DoScaleRelu(const float *in_data, float *out_data, const float *scale, const float *offset, int task_id,
                 const ScaleParameter *scale_param) {
  int outer_step = UP_DIV(scale_param->outer_size_, scale_param->op_parameter_.thread_num_);
  int outer_start = task_id * outer_step;
  int outer_end = MSMIN(outer_start + outer_step, scale_param->outer_size_);

  if (scale_param->inner_size_ == 1) {
    ScaleAxisRelu(in_data, out_data, scale, offset, outer_start, outer_end, scale_param->axis_size_);
  } else {
    ScaleInnerRelu(in_data, out_data, scale, offset, outer_start, outer_end, scale_param->axis_size_,
                   scale_param->inner_size_);
  }
}

void ScaleInnerRelu6(const float *in_data, float *out_data, const float *scale, const float *offset, int outer_start,
                     int outer_end, int axis_size, int inner_size) {
#ifdef ENABLE_ARM64
  float32x4_t zeros = {0, 0, 0, 0};
  float32x4_t bounds = {6, 6, 6, 6};
#endif
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size * inner_size;
    for (int i = 0; i < axis_size; i++) {
      int axis_offset = out_offset + i * inner_size;
      int in_index = 0;
#ifdef ENABLE_ARM64
      for (; in_index < inner_size - 4; in_index += 4) {
        int in_offset = axis_offset + in_index;
        float32x4_t data = vld1q_f32(in_data + in_offset);
        float32x4_t scale_4 = vdupq_n_f32(scale[i]);
        float32x4_t offset_4 = vdupq_n_f32(offset[i]);
        float32x4_t tmp = vfmaq_f32(offset_4, data, scale_4);
        float32x4_t result = vminq_f32(vmaxq_f32(tmp, zeros), bounds);
        vst1q_f32(out_data + in_offset, result);
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

void ScaleAxisRelu6(const float *in_data, float *out_data, const float *scale, const float *offset, int outer_start,
                    int outer_end, int axis_size) {
#ifdef ENABLE_ARM64
  float32x4_t zeros = {0, 0, 0, 0};
  float32x4_t bounds = {6, 6, 6, 6};
#endif
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size;
    int index = 0;
#ifdef ENABLE_ARM64
    for (; index < axis_size - 4; index += 4) {
      int in_offset = out_offset + index;
      float32x4_t data = vld1q_f32(in_data + in_offset);
      float32x4_t scale_4 = vld1q_f32(scale + index);
      float32x4_t offset_4 = vld1q_f32(offset + index);
      float32x4_t tmp = vfmaq_f32(offset_4, data, scale_4);
      float32x4_t result = vminq_f32(vmaxq_f32(tmp, zeros), bounds);
      vst1q_f32(out_data + in_offset, result);
    }
#endif
    for (; index < axis_size; index++) {
      int in_offset = out_offset + index;
      float tmp = in_data[in_offset] * scale[index] + offset[index];
      out_data[in_offset] = MSMIN(MSMAX(tmp, 0.0f), 6.0f);
    }
  }
}

void DoScaleRelu6(const float *in_data, float *out_data, const float *scale, const float *offset, int task_id,
                  const ScaleParameter *scale_param) {
  int outer_step = UP_DIV(scale_param->outer_size_, scale_param->op_parameter_.thread_num_);
  int outer_start = task_id * outer_step;
  int outer_end = MSMIN(outer_start + outer_step, scale_param->outer_size_);

  if (scale_param->inner_size_ == 1) {
    ScaleAxisRelu6(in_data, out_data, scale, offset, outer_start, outer_end, scale_param->axis_size_);
  } else {
    ScaleInnerRelu6(in_data, out_data, scale, offset, outer_start, outer_end, scale_param->axis_size_,
                    scale_param->inner_size_);
  }
}
