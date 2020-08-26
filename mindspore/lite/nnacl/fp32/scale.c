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

#include "nnacl/fp32/scale.h"
#ifdef ENABLE_ARM
#include <arm_neon.h>
#endif
void ScaleInner(float *in_data, float *out_data, float *scale, float *offset, int outer_start, int outer_end,
                int axis_size, int inner_size) {
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
        float32x4_t reslut = vfmaq_f32(offset_4, data, scale_4);
        vst1q_f32(out_data + in_offset, reslut);
      }
#endif
      for (; in_index < inner_size; in_index++) {
        int in_offset = axis_offset + in_index;
        out_data[in_offset] = in_data[in_offset] * scale[i] + offset[i];
      }
    }
  }
}

void ScaleAxis(float *in_data, float *out_data, float *scale, float *offset, int outer_start, int outer_end,
               int axis_size) {
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size;
    int index = 0;
#ifdef ENABLE_ARM64
    for (; index < axis_size - 4; index += 4) {
      int in_offset = out_offset + index;
      float32x4_t data = vld1q_f32(in_data + in_offset);
      float32x4_t scale_4 = vld1q_f32(scale + index);
      float32x4_t offset_4 = vld1q_f32(offset + index);
      float32x4_t reslut = vfmaq_f32(offset_4, data, scale_4);
      vst1q_f32(out_data + in_offset, reslut);
    }
#endif
    for (; index < axis_size; index++) {
      int in_offset = out_offset + index;
      out_data[in_offset] = in_data[in_offset] * scale[index] + offset[index];
    }
  }
}

void DoScale(float *in_data, float *out_data, float *scale, float *offset, int task_id, ScaleParameter *scale_param) {
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
