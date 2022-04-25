/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_NNACL_FP32_PACK_FP32_V2_H
#define MINDSPORE_NNACL_FP32_PACK_FP32_V2_H

#ifdef ENABLE_ARM64
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 *  Plan of packing supports granular multi-threads.
 */

void RowMajor2Col12MajorOpt(const float *src_ptr, float *dst_ptr, int64_t row, int64_t col, int64_t start, int64_t end);

void RowMajor2Row12MajorOpt(const float *src_ptr, float *dst_ptr, int64_t row, int64_t col, int64_t start, int64_t end);

#ifdef __cplusplus
}
#endif
#endif
#endif  // MINDSPORE_NNACL_FP32_PACK_FP32_V2_H
