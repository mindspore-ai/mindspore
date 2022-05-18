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

#ifndef MINDSPORE_NNACL_GATHER_D_BASE_H_
#define MINDSPORE_NNACL_GATHER_D_BASE_H_

#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"

#ifdef __cplusplus
extern "C" {
#endif
#define GATHER_D(type0, type1, output, input, index, input_shape, input_shape_size, output_shape, output_shape_size, \
                 dim)                                                                                                \
  GatherD_Input_##type0##_Index_##type1(output, input, index, input_shape, input_shape_size, output_shape,           \
                                        output_shape_size, dim)
int GatherD_Input_bool_Index_int32_t(bool *output, const bool *input, int32_t *index, const size_t *input_shape,
                                     const size_t input_shape_size, const size_t *output_shape,
                                     const size_t output_shape_size, const int dim);
int GatherD_Input_bool_Index_int64_t(bool *output, const bool *input, int64_t *index, const size_t *input_shape,
                                     const size_t input_shape_size, const size_t *output_shape,
                                     const size_t output_shape_size, const int dim);
int GatherD_Input_int16_t_Index_int32_t(int16_t *output, const int16_t *input, int32_t *index,
                                        const size_t *input_shape, const size_t input_shape_size,
                                        const size_t *output_shape, const size_t output_shape_size, const int dim);
int GatherD_Input_int16_t_Index_int64_t(int16_t *output, const int16_t *input, int64_t *index,
                                        const size_t *input_shape, const size_t input_shape_size,
                                        const size_t *output_shape, const size_t output_shape_size, const int dim);
int GatherD_Input_int32_t_Index_int32_t(int32_t *output, const int32_t *input, int *index, const size_t *input_shape,
                                        const size_t input_shape_size, const size_t *output_shape,
                                        const size_t output_shape_size, const int dim);
int GatherD_Input_int32_t_Index_int64_t(int32_t *output, const int32_t *input, int64_t *index,
                                        const size_t *input_shape, const size_t input_shape_size,
                                        const size_t *output_shape, const size_t output_shape_size, const int dim);
int GatherD_Input_int64_t_Index_int32_t(int64_t *output, const int64_t *input, int *index, const size_t *input_shape,
                                        const size_t input_shape_size, const size_t *output_shape,
                                        const size_t output_shape_size, const int dim);
int GatherD_Input_int64_t_Index_int64_t(int64_t *output, const int64_t *input, int64_t *index,
                                        const size_t *input_shape, const size_t input_shape_size,
                                        const size_t *output_shape, const size_t output_shape_size, const int dim);
int GatherD_Input_float_Index_int32_t(float *output, const float *input, int *index, const size_t *input_shape,
                                      const size_t input_shape_size, const size_t *output_shape,
                                      const size_t output_shape_size, const int dim);
int GatherD_Input_float_Index_int64_t(float *output, const float *input, int64_t *index, const size_t *input_shape,
                                      const size_t input_shape_size, const size_t *output_shape,
                                      const size_t output_shape_size, const int dim);
#ifdef ENABLE_FP16
int GatherD_Input_float16_t_Index_int32_t(float16_t *output, const float16_t *input, int *index,
                                          const size_t *input_shape, const size_t input_shape_size,
                                          const size_t *output_shape, const size_t output_shape_size, const int dim);
int GatherD_Input_float16_t_Index_int64_t(float16_t *output, const float16_t *input, int64_t *index,
                                          const size_t *input_shape, const size_t input_shape_size,
                                          const size_t *output_shape, const size_t output_shape_size, const int dim);
#endif
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_GATHER_D_BASE_H_
