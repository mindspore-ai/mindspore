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
#ifndef NNACL_FP16_RESIZE_FP16_H_
#define NNACL_FP16_RESIZE_FP16_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/resize_parameter.h"
#include "nnacl/op_base.h"
#include "nnacl/crop_parameter.h"
#include "nnacl/fp32/resize_fp32.h"

#ifdef __cplusplus
extern "C" {
#endif

int PrepareResizeBilinearFp16(const int *input_shape, const int *output_shape, CalculateOriginalCoordinate calculate,
                              int *y_bottoms, int *y_tops, int *x_lefts, int *x_rights, float16_t *y_bottom_weights,
                              float16_t *x_left_weights);

int PrepareResizeBicubicFp16(const int *input_shape, const int *output_shape, CalculateOriginalCoordinate calculate,
                             int *y_tops, int *x_lefts, float16_t *y_weights, float16_t *x_weights,
                             float16_t cubic_coeff);

int ResizeBilinearFp16(const float16_t *input_data, float16_t *output_data, const int *input_shape,
                       const int *output_shape, const int *y_bottoms, const int *y_tops, const int *x_lefts,
                       const int *x_rights, const float16_t *y_bottom_weights, const float16_t *x_left_weights,
                       float16_t *line0, float16_t *line1, const int h_begin, const int h_end);

int ResizeBicubicFp16(const float16_t *input_data, float16_t *output_data, const int *input_shape,
                      const int *output_shape, const int *y_tops, const int *x_lefts, const float16_t *y_weights,
                      const float16_t *x_weights, float16_t *line_buffer, const int h_begin, const int h_end);

int ResizeNearestNeighborFp16(const float16_t *input_data, float16_t *output_data, const int *input_shape,
                              const int *output_shape, CalculateOriginalCoordinate calculate,
                              int coordinate_transform_mode, int tid, int thread_num);

#ifdef __cplusplus
}
#endif

#endif  //  NNACL_FP16_RESIZE_FP16_H_
