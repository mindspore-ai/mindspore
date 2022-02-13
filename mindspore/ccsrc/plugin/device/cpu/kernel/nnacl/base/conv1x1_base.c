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

#include "nnacl/base/conv1x1_base.h"

void Conv1x1InputPack(const void *src_ptr, void *dst_ptr, ConvParameter *conv_param, int data_size) {
  /* support nhwc */
  char *src = (char *)src_ptr;
  char *dst = (char *)dst_ptr;
  for (int dst_h = 0; dst_h < conv_param->output_h_; dst_h++) {
    int src_h = dst_h * conv_param->stride_h_ - conv_param->pad_u_;
    if (src_h < 0 || src_h >= conv_param->input_h_) {
      continue;
    }
    const char *src_h_ptr = src + src_h * conv_param->input_w_ * conv_param->input_channel_ * data_size;
    char *dst_h_ptr = dst + dst_h * conv_param->output_w_ * conv_param->input_channel_ * data_size;
    for (int dst_w = 0; dst_w < conv_param->output_w_; dst_w++) {
      int src_w = dst_w * conv_param->stride_w_ - conv_param->pad_l_;
      if (src_w < 0 || src_w >= conv_param->input_w_) {
        continue;
      }
      memcpy(dst_h_ptr + dst_w * conv_param->input_channel_ * data_size,
             src_h_ptr + src_w * conv_param->input_channel_ * data_size, conv_param->input_channel_ * data_size);
    }
  }
  return;
}
