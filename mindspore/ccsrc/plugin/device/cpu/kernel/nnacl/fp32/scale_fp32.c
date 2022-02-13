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
#ifdef ENABLE_AVX
      MS_FLOAT32X8 scale_8 = MS_MOV256_F32(scale[i]);
      MS_FLOAT32X8 offset_8 = MS_MOV256_F32(offset[i]);
      for (; in_index <= inner_size - C8NUM; in_index += C8NUM) {
        int in_offset = axis_offset + in_index;
        MS_FLOAT32X8 data = MS_LD256_F32(in_data + in_offset);
        MS_FLOAT32X8 result = MS_MLA256_F32(offset_8, data, scale_8);
        MS_ST256_F32(out_data + in_offset, result);
      }
#endif
#if defined(ENABLE_ARM64) || defined(ENABLE_SSE)
      MS_FLOAT32X4 scale_4 = MS_MOVQ_F32(scale[i]);
      MS_FLOAT32X4 offset_4 = MS_MOVQ_F32(offset[i]);
      for (; in_index <= inner_size - C4NUM; in_index += C4NUM) {
        int in_offset = axis_offset + in_index;
        MS_FLOAT32X4 data = MS_LDQ_F32(in_data + in_offset);
        MS_FLOAT32X4 result = MS_MLAQ_F32(offset_4, data, scale_4);
        MS_STQ_F32(out_data + in_offset, result);
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
#if defined(ENABLE_AVX)
    for (; index <= axis_size - C8NUM; index += C8NUM) {
      int in_offset = out_offset + index;
      MS_FLOAT32X8 scale_8 = MS_LD256_F32(scale + index);
      MS_FLOAT32X8 offset_8 = MS_LD256_F32(offset + index);
      MS_FLOAT32X8 data = MS_LD256_F32(in_data + in_offset);
      MS_FLOAT32X8 result = MS_MLA256_F32(offset_8, data, scale_8);
      MS_ST256_F32(out_data + in_offset, result);
    }
#endif
#if defined(ENABLE_ARM64) || defined(ENABLE_SSE)
    for (; index <= axis_size - C4NUM; index += C4NUM) {
      MS_FLOAT32X4 scale_4 = MS_LDQ_F32(scale + index);
      MS_FLOAT32X4 offset_4 = MS_LDQ_F32(offset + index);
      int in_offset = out_offset + index;
      MS_FLOAT32X4 data = MS_LDQ_F32(in_data + in_offset);
      MS_FLOAT32X4 result = MS_MLAQ_F32(offset_4, data, scale_4);
      MS_STQ_F32(out_data + in_offset, result);
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
  if (scale_param->op_parameter_.thread_num_ == 0) {
    return;
  }
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
#ifdef ENABLE_AVX
  MS_FLOAT32X8 zeros_8 = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
#if defined(ENABLE_ARM64) || defined(ENABLE_SSE)
  MS_FLOAT32X4 zeros = {0, 0, 0, 0};
#endif
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size * inner_size;
    for (int i = 0; i < axis_size; i++) {
      int axis_offset = out_offset + i * inner_size;
      int in_index = 0;
#ifdef ENABLE_AVX
      MS_FLOAT32X8 scale_8 = MS_MOV256_F32(scale[i]);
      MS_FLOAT32X8 offset_8 = MS_MOV256_F32(offset[i]);
      for (; in_index <= inner_size - C8NUM; in_index += C8NUM) {
        int in_offset = axis_offset + in_index;
        MS_FLOAT32X8 data = MS_LD256_F32(in_data + in_offset);
        MS_FLOAT32X8 tmp = MS_MLA256_F32(offset_8, data, scale_8);
        MS_FLOAT32X8 result = MS_MAX256_F32(tmp, zeros_8);
        MS_ST256_F32(out_data + in_offset, result);
      }
#endif
#if defined(ENABLE_ARM64) || defined(ENABLE_SSE)
      MS_FLOAT32X4 scale_4 = MS_MOVQ_F32(scale[i]);
      MS_FLOAT32X4 offset_4 = MS_MOVQ_F32(offset[i]);
      for (; in_index <= inner_size - C4NUM; in_index += C4NUM) {
        int in_offset = axis_offset + in_index;
        MS_FLOAT32X4 data = MS_LDQ_F32(in_data + in_offset);
        MS_FLOAT32X4 tmp = MS_MLAQ_F32(offset_4, data, scale_4);
        MS_FLOAT32X4 result = MS_MAXQ_F32(tmp, zeros);
        MS_STQ_F32(out_data + in_offset, result);
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
#ifdef ENABLE_AVX
  MS_FLOAT32X8 zeros_8 = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
#if defined(ENABLE_ARM64) || defined(ENABLE_SSE)
  MS_FLOAT32X4 zeros = {0, 0, 0, 0};
#endif
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size;
    int index = 0;
#ifdef ENABLE_AVX
    for (; index <= axis_size - C8NUM; index += C8NUM) {
      int in_offset = out_offset + index;
      MS_FLOAT32X8 scale_8 = MS_LD256_F32(scale + index);
      MS_FLOAT32X8 offset_8 = MS_LD256_F32(offset + index);
      MS_FLOAT32X8 data = MS_LD256_F32(in_data + in_offset);
      MS_FLOAT32X8 tmp = MS_MLA256_F32(offset_8, data, scale_8);
      MS_FLOAT32X8 result = MS_MAX256_F32(tmp, zeros_8);
      MS_ST256_F32(out_data + in_offset, result);
    }
#endif
#if defined(ENABLE_ARM64) || defined(ENABLE_SSE)
    for (; index <= axis_size - C4NUM; index += C4NUM) {
      int in_offset = out_offset + index;
      MS_FLOAT32X4 data = MS_LDQ_F32(in_data + in_offset);
      MS_FLOAT32X4 scale_4 = MS_LDQ_F32(scale + index);
      MS_FLOAT32X4 offset_4 = MS_LDQ_F32(offset + index);
      MS_FLOAT32X4 tmp = MS_MLAQ_F32(offset_4, data, scale_4);
      MS_FLOAT32X4 result = MS_MAXQ_F32(tmp, zeros);
      MS_STQ_F32(out_data + in_offset, result);
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
  if (scale_param->op_parameter_.thread_num_ == 0) {
    return;
  }
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
#ifdef ENABLE_AVX
  MS_FLOAT32X8 zeros_8 = {0, 0, 0, 0, 0, 0, 0, 0};
  MS_FLOAT32X8 bounds_8 = {6, 6, 6, 6, 6, 6, 6, 6};
#endif
#if defined(ENABLE_ARM64) || defined(ENABLE_SSE)
  MS_FLOAT32X4 zeros = {0, 0, 0, 0};
  MS_FLOAT32X4 bounds = {6, 6, 6, 6};
#endif
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size * inner_size;
    for (int i = 0; i < axis_size; i++) {
      int axis_offset = out_offset + i * inner_size;
      int in_index = 0;
#if defined(ENABLE_AVX)
      MS_FLOAT32X8 scale_8 = MS_MOV256_F32(scale[i]);
      MS_FLOAT32X8 offset_8 = MS_MOV256_F32(offset[i]);
      for (; in_index <= inner_size - C8NUM; in_index += C8NUM) {
        int in_offset = axis_offset + in_index;
        MS_FLOAT32X8 data = MS_LD256_F32(in_data + in_offset);
        MS_FLOAT32X8 tmp = MS_MLA256_F32(offset_8, data, scale_8);
        MS_FLOAT32X8 result = MS_MIN256_F32(MS_MAX256_F32(tmp, zeros_8), bounds_8);
        MS_ST256_F32(out_data + in_offset, result);
      }
#endif
#if defined(ENABLE_ARM64) || defined(ENABLE_SSE)
      for (; in_index < inner_size - C4NUM; in_index += C4NUM) {
        int in_offset = axis_offset + in_index;
        MS_FLOAT32X4 data = MS_LDQ_F32(in_data + in_offset);
        MS_FLOAT32X4 scale_4 = MS_MOVQ_F32(scale[i]);
        MS_FLOAT32X4 offset_4 = MS_MOVQ_F32(offset[i]);
        MS_FLOAT32X4 tmp = MS_MLAQ_F32(offset_4, data, scale_4);
        MS_FLOAT32X4 result = MS_MINQ_F32(MS_MAXQ_F32(tmp, zeros), bounds);
        MS_STQ_F32(out_data + in_offset, result);
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
#ifdef ENABLE_AVX
  MS_FLOAT32X8 zeros_8 = {0, 0, 0, 0, 0, 0, 0, 0};
  MS_FLOAT32X8 bounds_8 = {6, 6, 6, 6, 6, 6, 6, 6};
#endif
#if defined(ENABLE_ARM64) || defined(ENABLE_SSE)
  MS_FLOAT32X4 zeros = {0, 0, 0, 0};
  MS_FLOAT32X4 bounds = {6, 6, 6, 6};
#endif
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size;
    int index = 0;
#ifdef ENABLE_AVX
    for (; index <= axis_size - C8NUM; index += C8NUM) {
      int in_offset = out_offset + index;
      MS_FLOAT32X8 data = MS_LD256_F32(in_data + in_offset);
      MS_FLOAT32X8 scale_8 = MS_LD256_F32(scale + index);
      MS_FLOAT32X8 offset_8 = MS_LD256_F32(offset + index);
      MS_FLOAT32X8 tmp = MS_MLA256_F32(offset_8, data, scale_8);
      MS_FLOAT32X8 result = MS_MIN256_F32(MS_MAX256_F32(tmp, zeros_8), bounds_8);
      MS_ST256_F32(out_data + in_offset, result);
    }
#endif
#if defined(ENABLE_ARM64) || defined(ENABLE_SSE)
    for (; index <= axis_size - C4NUM; index += C4NUM) {
      int in_offset = out_offset + index;
      MS_FLOAT32X4 data = MS_LDQ_F32(in_data + in_offset);
      MS_FLOAT32X4 scale_4 = MS_LDQ_F32(scale + index);
      MS_FLOAT32X4 offset_4 = MS_LDQ_F32(offset + index);
      MS_FLOAT32X4 tmp = MS_MLAQ_F32(offset_4, data, scale_4);
      MS_FLOAT32X4 result = MS_MINQ_F32(MS_MAXQ_F32(tmp, zeros), bounds);
      MS_STQ_F32(out_data + in_offset, result);
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
  if (scale_param->op_parameter_.thread_num_ == 0) {
    return;
  }
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
