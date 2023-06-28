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

#ifndef NNACL_INT8_PAD_INT8_H_
#define NNACL_INT8_PAD_INT8_H_

#include <string.h>
#include "nnacl/op_base.h"
#include "nnacl/pad_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
int PadConstant4D(const int8_t *in_data, int8_t *out_data, const int32_t *in_dims, const int32_t *out_dims,
                  const int32_t *paddings, const int tid, const int thread_num);
void MirrorPadInt8(const int8_t *in, int8_t *out, const int32_t *input_shape, int mirror_offset, const int *in_strides,
                   const int *out_strides, const int *paddings, int begin, int end);
#ifdef __cplusplus
}
#endif

#endif  // NNACL_INT8_PAD_INT8_H_
