/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "wrapper/base/affine_wrapper.h"

void FullSpliceRun(const float *in_data, float *out_data, const SpliceWrapperParam *param) {
  for (int r = 0; r < param->dst_row; ++r) {
    for (int off = 0; off < param->context_size; ++off) {
      int r_off = r - param->src_to_dst_row_offset + param->context[off];
      const float *tmp_src_data = in_data + r_off * param->src_col;
      float *tmp_dst_data = out_data + r * param->dst_col;
      memcpy(tmp_dst_data + off * param->src_col, tmp_src_data, param->src_col * sizeof(float));
    }
  }
}

void FullSpliceRunInt8(const int8_t *in_data, int8_t *out_data, const SpliceWrapperParam *param) {
  for (int r = 0; r < param->dst_row; ++r) {
    for (int off = 0; off < param->context_size; ++off) {
      int r_off = r - param->src_to_dst_row_offset + param->context[off];
      const int8_t *tmp_src_data = in_data + r_off * param->src_col;
      int8_t *tmp_dst_data = out_data + r * param->dst_col;
      memcpy(tmp_dst_data + off * param->src_col, tmp_src_data, param->src_col * sizeof(int8_t));
    }
  }
}

void IncrementSpliceRunInt8(const int8_t *in_data, int8_t *out_data, const SpliceWrapperParam *param) {
  int forward_offset = param->dst_row - 1 - param->src_to_dst_row_offset;
  // splice last context input to outputs
  for (int i = 0; i < param->context_size; ++i) {
    int forward_row = forward_offset + param->context[i];
    const int8_t *src_offset_ptr = in_data + forward_row * param->src_col;
    int8_t *splice_offset_ptr = out_data + i * param->src_col;
    memcpy(splice_offset_ptr, src_offset_ptr, param->src_col * sizeof(int8_t));
  }
}
