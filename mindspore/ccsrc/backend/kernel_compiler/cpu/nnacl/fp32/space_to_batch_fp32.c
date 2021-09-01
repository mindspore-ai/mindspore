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
#include "nnacl/fp32/space_to_batch_fp32.h"
#include "nnacl/errorcode.h"

int DoSpaceToBatch(const float *input, float *output, const int *in_shape, const int *out_shape, const int *in_stride,
                   const int *out_stride, const int *blocks, const int *paddings, int thread, int task_id) {
  if (thread == 0) {
    return NNACL_ERR;
  }
  const int depth = in_shape[3];
  const int input_width = in_shape[2];
  const int input_height = in_shape[1];
  const int input_batch_size = in_shape[0];

  const int output_width = out_shape[2];
  const int output_height = out_shape[1];
  const int output_batch_size = out_shape[0];

  const int block_shape_height = blocks[0];
  const int block_shape_width = blocks[1];
  const int padding_top = paddings[0];
  const int padding_left = paddings[2];

  NNACL_CHECK_ZERO_RETURN_ERR(input_batch_size);
  NNACL_CHECK_ZERO_RETURN_ERR(block_shape_width);
  size_t copy_size = depth * sizeof(float);
  for (int out_b = task_id; out_b < output_batch_size; out_b += thread) {
    int input_batch = out_b % input_batch_size;
    int shift_w = (out_b / input_batch_size) % block_shape_width;
    int shift_h = (out_b / input_batch_size) / block_shape_width;
    for (int out_h = 0; out_h < output_height; out_h++) {
      for (int out_w = 0; out_w < output_width; out_w++) {
        float *out = output + out_b * out_stride[0] + out_h * out_stride[1] + out_w * out_stride[2];
        if (out_h * block_shape_height + shift_h < padding_top ||
            out_h * block_shape_height + shift_h >= padding_top + input_height ||
            out_w * block_shape_width + shift_w < padding_left ||
            out_w * block_shape_width + shift_w >= padding_left + input_width) {
          memset(out, 0, copy_size);
        } else {
          int in_h = (out_h * block_shape_height + shift_h) - padding_top;
          int in_w = (out_w * block_shape_width + shift_w) - padding_left;
          const float *in = input + input_batch * in_stride[0] + in_h * in_stride[1] + in_w * in_stride[2];
          memcpy(out, in, copy_size);
        }
      }
    }
  }
  return NNACL_OK;
}
