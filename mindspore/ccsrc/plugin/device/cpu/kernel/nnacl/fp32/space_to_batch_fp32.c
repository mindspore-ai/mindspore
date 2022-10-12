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

int DoSpaceToBatch(const void *input, void *output, SpaceToBatchParameter *param, int task_id) {
  if (param->op_parameter_.thread_num_ == 0) {
    return NNACL_ERR;
  }
  const int input_batch = param->input_shape_[0];
  const int input_height = param->input_shape_[1];
  const int input_width = param->input_shape_[2];

  const int output_batch = param->output_shape_[0];
  const int output_height = param->output_shape_[1];
  const int output_width = param->output_shape_[2];

  const int block_shape_height = param->block_sizes_[0];
  const int block_shape_width = param->block_sizes_[1];
  const int padding_top = param->paddings_[0];
  const int padding_left = param->paddings_[2];

  NNACL_CHECK_ZERO_RETURN_ERR(input_batch);
  NNACL_CHECK_ZERO_RETURN_ERR(block_shape_width);
  int copy_size = param->input_shape_[3] * param->data_type_len;
  for (int64_t out_b = task_id; out_b < output_batch; out_b += param->op_parameter_.thread_num_) {
    int in_b = out_b % input_batch;
    int shift_w = (out_b / input_batch) % block_shape_width;
    int shift_h = (out_b / input_batch) / block_shape_width;
    for (int out_h = 0; out_h < output_height; out_h++) {
      for (int out_w = 0; out_w < output_width; out_w++) {
        int64_t output_offset =
          out_b * param->out_stride_[0] + out_h * param->out_stride_[1] + out_w * param->out_stride_[2];
        if (out_h * block_shape_height + shift_h < padding_top ||
            out_h * block_shape_height + shift_h >= padding_top + input_height ||
            out_w * block_shape_width + shift_w < padding_left ||
            out_w * block_shape_width + shift_w >= padding_left + input_width) {
          memset((int8_t *)output + output_offset * param->data_type_len, 0, copy_size);
        } else {
          int in_h = (out_h * block_shape_height + shift_h) - padding_top;
          int in_w = (out_w * block_shape_width + shift_w) - padding_left;
          int input_offset = in_b * param->in_stride_[0] + in_h * param->in_stride_[1] + in_w * param->in_stride_[2];
          memcpy((int8_t *)output + output_offset * param->data_type_len,
                 (const int8_t *)input + input_offset * param->data_type_len, copy_size);
        }
      }
    }
  }
  return NNACL_OK;
}
