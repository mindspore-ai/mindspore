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

#ifndef NNACL_GATHER_BASE_H_
#define NNACL_GATHER_BASE_H_

#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"

#ifdef __cplusplus
extern "C" {
#endif
int Gather(const void *input, int64_t outer_size, int64_t byte_inner_size, int64_t limit, const int *indices,
           int64_t index_num, void *output, int64_t byte_out_stride, int *error_index);
#ifdef __cplusplus
}
#endif

#endif  // NNACL_GATHER_BASE_H_
