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

#ifndef NNACL_GATHER_D_BASE_H_
#define NNACL_GATHER_D_BASE_H_

#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"

#ifdef __cplusplus
extern "C" {
#endif
#define GATHER_D(type0, type1, output, input, index, input_shape, input_shape_size, output_shape, output_shape_size, \
                 dim)                                                                                                \
  GatherD_Input_##type0##_Index_##type1(output, input, index, input_shape, input_shape_size, output_shape,           \
                                        output_shape_size, dim)

#define GATHER_D_IMPL_DECLARATION(type0, type1)                                                                \
  int GatherD_Input_##type0##_Index_##type1(                                                                   \
    type0 *output, const type0 *input, type1 *index, const size_t *input_shape, const size_t input_shape_size, \
    const size_t *output_shape, const size_t output_shape_size, const size_t dim)

GATHER_D_IMPL_DECLARATION(bool, int32_t);
GATHER_D_IMPL_DECLARATION(bool, int64_t);
GATHER_D_IMPL_DECLARATION(int16_t, int32_t);
GATHER_D_IMPL_DECLARATION(int16_t, int64_t);
GATHER_D_IMPL_DECLARATION(int32_t, int32_t);
GATHER_D_IMPL_DECLARATION(int32_t, int64_t);
GATHER_D_IMPL_DECLARATION(int64_t, int32_t);
GATHER_D_IMPL_DECLARATION(int64_t, int64_t);
GATHER_D_IMPL_DECLARATION(float, int32_t);
GATHER_D_IMPL_DECLARATION(float, int64_t);
#ifdef ENABLE_FP16
GATHER_D_IMPL_DECLARATION(float16_t, int32_t);
GATHER_D_IMPL_DECLARATION(float16_t, int64_t);
#endif

#ifdef __cplusplus
}
#endif

#endif  // NNACL_GATHER_D_BASE_H_
