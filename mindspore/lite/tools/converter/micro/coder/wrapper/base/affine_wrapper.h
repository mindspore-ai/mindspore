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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_BASE_AFFINE_WRAPPER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_BASE_AFFINE_WRAPPER_H_
#include "nnacl/op_base.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct SpliceWrapperParam {
  int src_row;
  int src_col;
  int dst_row;
  int dst_col;
  int context_size;
  int context[MAX_SHAPE_SIZE];
  int src_to_dst_row_offset;
} SpliceWrapperParam;

void FullSpliceRunFp32(const float *in_data, float *out_data, const SpliceWrapperParam *param);
void FullSpliceRunInt8(const int8_t *in_data, int8_t *out_data, const SpliceWrapperParam *param);
void IncrementSpliceRunInt8(const int8_t *in_data, int8_t *out_data, const SpliceWrapperParam *param);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_BASE_AFFINE_WRAPPER_H_
