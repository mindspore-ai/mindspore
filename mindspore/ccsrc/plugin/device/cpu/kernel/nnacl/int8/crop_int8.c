/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "nnacl/crop_parameter.h"
#include <string.h>
#include <math.h>
#include <float.h>
#include "nnacl/int8/crop_int8.h"

void Int8Crop1D(const int8_t *input, int8_t *output, int *output_shape, int64_t *in_offset, int task_id,
                int thread_count, const CropQuantArg *quant) {
  const int out_batch = output_shape[0];
  int64_t task_id_stride = thread_count > 1 ? UP_DIV(out_batch, thread_count) : out_batch;
  if (task_id_stride <= 0) {
    return;
  }

  float in_scale = quant->in_args_.scale_;
  int32_t in_zp = quant->in_args_.zp_;
  float out_scale = quant->out_args_.scale_;
  int32_t out_zp = quant->out_args_.zp_;
  float scale = in_scale / out_scale;
  float bias = -in_zp * scale;

  int n = task_id * task_id_stride;
  if (n >= out_batch) {
    return;
  }
  const int8_t *in_ptr = input + n + in_offset[0];
  int8_t *out_ptr = output + n;
  int64_t out_dist_stride = MSMIN(out_batch - task_id * task_id_stride, task_id_stride);
  if (fabs(in_scale - out_scale) <= FLT_EPSILON && in_zp == out_zp) {
    memcpy(out_ptr, in_ptr, sizeof(int8_t) * out_dist_stride);
  } else {
    for (int i = 0; i < out_dist_stride; i++) {
      int32_t output_tmp = round(in_ptr[i] * scale + bias) + out_zp;
      if (output_tmp > quant->output_activation_max_) {
        out_ptr[i] = quant->output_activation_max_;
      } else if (output_tmp < quant->output_activation_min_) {
        out_ptr[i] = quant->output_activation_min_;
      } else {
        out_ptr[i] = (int8_t)output_tmp;
      }
    }
  }
  return;
}

void Int8Crop2D(const int8_t *input, int8_t *output, int *input_shape, int *output_shape, int64_t *in_offset,
                int task_id, int thread_count, const CropQuantArg *quant) {
  const int in_height = input_shape[1];
  const int out_batch = output_shape[0];
  const int out_height = output_shape[1];
  int64_t task_id_stride = thread_count > 1 ? UP_DIV(out_height, thread_count) : out_height;
  if (task_id_stride <= 0) {
    return;
  }

  float in_scale = quant->in_args_.scale_;
  int32_t in_zp = quant->in_args_.zp_;
  float out_scale = quant->out_args_.scale_;
  int32_t out_zp = quant->out_args_.zp_;
  float scale = in_scale / out_scale;
  float bias = -in_zp * scale;

  for (int n = 0; n < out_batch; n++) {
    int h = task_id * task_id_stride;
    if (h >= out_height) {
      return;
    }
    const int8_t *in_ptr = input + (n + in_offset[0]) * in_height + h + in_offset[1];
    int8_t *out_ptr = output + n * out_height + h;
    int64_t out_dist_stride = MSMIN(out_height - task_id * task_id_stride, task_id_stride);
    if (fabs(in_scale - out_scale) <= FLT_EPSILON && in_zp == out_zp) {
      memcpy(out_ptr, in_ptr, sizeof(int8_t) * out_dist_stride);
    } else {
      for (int i = 0; i < out_dist_stride; i++) {
        int32_t output_tmp = round(in_ptr[i] * scale + bias) + out_zp;
        if (output_tmp > quant->output_activation_max_) {
          out_ptr[i] = quant->output_activation_max_;
        } else if (output_tmp < quant->output_activation_min_) {
          out_ptr[i] = quant->output_activation_min_;
        } else {
          out_ptr[i] = (int8_t)output_tmp;
        }
      }
    }
  }
  return;
}

void Int8Crop3D(const int8_t *input, int8_t *output, int *input_shape, int *output_shape, int64_t *in_offset,
                int task_id, int thread_count, const CropQuantArg *quant) {
  const int in_height = input_shape[1];
  const int in_width = input_shape[2];

  const int out_batch = output_shape[0];
  const int out_height = output_shape[1];
  const int out_width = output_shape[2];

  int64_t task_id_stride = thread_count > 1 ? UP_DIV(out_height, thread_count) : out_height;
  if (task_id_stride <= 0) {
    return;
  }

  const int in_stride_h = in_width;
  const int in_stride_n = in_stride_h * in_height;

  const int out_stride_h = out_width;
  const int out_stride_n = out_stride_h * out_height;

  float in_scale = quant->in_args_.scale_;
  int32_t in_zp = quant->in_args_.zp_;
  float out_scale = quant->out_args_.scale_;
  int32_t out_zp = quant->out_args_.zp_;
  float scale = in_scale / out_scale;
  float bias = -in_zp * scale;

  for (int n = 0; n < out_batch; n++) {
    for (int t = 0; t < task_id_stride; t++) {
      int h = t + task_id * task_id_stride;
      if (h >= out_height) {
        break;
      }
      const int8_t *in_ptr = input + (n + in_offset[0]) * in_stride_n + (h + in_offset[1]) * in_stride_h + in_offset[2];
      int8_t *out_ptr = output + n * out_stride_n + h * out_stride_h;
      if (fabs(in_scale - out_scale) <= FLT_EPSILON && in_zp == out_zp) {
        memcpy(out_ptr, in_ptr, sizeof(int8_t) * out_width);
      } else {
        for (int i = 0; i < out_width; i++) {
          int32_t output_tmp = round(in_ptr[i] * scale + bias) + out_zp;
          if (output_tmp > quant->output_activation_max_) {
            out_ptr[i] = quant->output_activation_max_;
          } else if (output_tmp < quant->output_activation_min_) {
            out_ptr[i] = quant->output_activation_min_;
          } else {
            out_ptr[i] = (int8_t)output_tmp;
          }
        }
      }
    }
  }
  return;
}

void Int8Crop4D(const int8_t *input, int8_t *output, int *input_shape, int *output_shape, int64_t *in_offset,
                int task_id, int thread_count, const CropQuantArg *quant) {
  const int in_height = input_shape[1];
  const int in_width = input_shape[2];
  const int in_channel = input_shape[3];

  const int out_batch = output_shape[0];
  const int out_height = output_shape[1];
  const int out_width = output_shape[2];
  const int out_channel = output_shape[3];

  int64_t task_id_stride = thread_count > 1 ? UP_DIV(out_height, thread_count) : out_height;
  if (task_id_stride <= 0) {
    return;
  }

  const int in_stride_w = in_channel;
  const int in_stride_h = in_channel * in_width;
  const int in_stride_n = in_stride_h * in_height;

  const int out_stride_w = out_channel;
  const int out_stride_h = out_channel * out_width;
  const int out_stride_n = out_stride_h * out_height;

  float in_scale = quant->in_args_.scale_;
  int32_t in_zp = quant->in_args_.zp_;
  float out_scale = quant->out_args_.scale_;
  int32_t out_zp = quant->out_args_.zp_;
  float scale = in_scale / out_scale;
  float bias = -in_zp * scale;

  for (int n = 0; n < out_batch; n++) {
    for (int t = 0; t < task_id_stride; t++) {
      int h = t + task_id * task_id_stride;
      if (h >= out_height) {
        break;
      }
      for (int w = 0; w < out_width; w++) {
        const int8_t *in_ptr = input + (n + in_offset[0]) * in_stride_n + (h + in_offset[1]) * in_stride_h +
                               (w + in_offset[2]) * in_stride_w + in_offset[3];
        int8_t *out_ptr = output + n * out_stride_n + h * out_stride_h + w * out_stride_w;
        if (fabs(in_scale - out_scale) <= FLT_EPSILON && in_zp == out_zp) {
          memcpy(out_ptr, in_ptr, sizeof(int8_t) * out_channel);
        } else {
          for (int i = 0; i < out_channel; i++) {
            int32_t output_tmp = round(in_ptr[i] * scale + bias) + out_zp;
            if (output_tmp > quant->output_activation_max_) {
              out_ptr[i] = quant->output_activation_max_;
            } else if (output_tmp < quant->output_activation_min_) {
              out_ptr[i] = quant->output_activation_min_;
            } else {
              out_ptr[i] = (int8_t)output_tmp;
            }
          }
        }
      }
    }
  }
  return;
}

void Int8Crop(const int8_t *input, int8_t *output, int *input_shape, int *output_shape, int64_t *in_offset,
              int input_dim, int task_id, int thread_count, const CropQuantArg *quant) {
  switch (input_dim) {
    case 1:
      Int8Crop1D(input, output, output_shape, in_offset, task_id, thread_count, quant);
      break;
    case 2:
      Int8Crop2D(input, output, input_shape, output_shape, in_offset, task_id, thread_count, quant);
      break;
    case 3:
      Int8Crop3D(input, output, input_shape, output_shape, in_offset, task_id, thread_count, quant);
      break;
    case 4:
      Int8Crop4D(input, output, input_shape, output_shape, in_offset, task_id, thread_count, quant);
      break;
  }
}
