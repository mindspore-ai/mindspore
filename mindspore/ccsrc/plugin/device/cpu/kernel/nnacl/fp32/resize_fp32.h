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
#ifndef MINDSPORE_NNACL_FP32_RESIZE_H_
#define MINDSPORE_NNACL_FP32_RESIZE_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include <memory.h>
#include "nnacl/resize_parameter.h"
#include "nnacl/op_base.h"
#include "nnacl/crop_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef float (*CalculateOriginalCoordinate)(int x_resized, int length_original, int length_resized);

int PrepareResizeBilinear(const int32_t *input_shape, const int32_t *output_shape,
                          CalculateOriginalCoordinate calculate, int32_t *y_bottoms, int32_t *y_tops, int32_t *x_lefts,
                          int32_t *x_rights, float *y_bottom_weights, float *x_left_weights);

int PrepareResizeBicubic(const int32_t *input_shape, const int32_t *output_shape, CalculateOriginalCoordinate calculate,
                         int32_t *y_tops, int32_t *x_lefts, float *y_weights, float *x_weights, float cubic_coeff);

int ResizeBilinear(const float *input_data, float *output_data, const int32_t *input_shape, const int32_t *output_shape,
                   const int32_t *y_bottoms, const int32_t *y_tops, const int32_t *x_lefts, const int32_t *x_rights,
                   const float *y_bottom_weights, const float *x_left_weights, float *line0, float *line1,
                   const int h_begin, const int h_end);

int ResizeBicubic(const float *input_data, float *output_data, const int32_t *input_shape, const int32_t *output_shape,
                  const int32_t *y_tops, const int32_t *x_lefts, const float *y_weights, const float *x_weights,
                  float *line_buffer, const int h_begin, const int h_end);

int PrepareCropAndResizeBilinear(const int32_t *input_shape, const float *boxes, const int32_t *box_idx,
                                 const int32_t *output_shape, int32_t *y_bottoms, int32_t *y_tops, int32_t *x_lefts,
                                 int32_t *x_rights, float *y_bottom_weights, float *x_left_weights);

int CropAndResizeBilinear(const float *input_data, float *output_data, const int32_t *box_idx, const float *boxes,
                          float extrapolation_value, const int32_t *input_shape, const int32_t *output_shape,
                          const int32_t *y_bottoms, const int32_t *y_tops, const int32_t *x_lefts,
                          const int32_t *x_rights, const float *y_bottom_weights, const float *x_left_weights,
                          float *line0, float *line1, const int h_begin, const int h_end);

int ResizeNearestNeighbor(const float *input_data, float *output_data, const int32_t *input_shape,
                          const int32_t *output_shape, CalculateOriginalCoordinate calculate,
                          int coordinate_transform_mode, int tid, int thread_num);

float CalculateAsymmetric(int x_resized, int length_original, int length_resized);

float CalculateAlignCorners(int x_resized, int length_original, int length_resized);

float CalculateHalfPixel(int x_resized, int length_original, int length_resized);

int CheckCropAndResizeBoxIdx(int32_t *box_idx, int32_t num_boxes, int32_t batch);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_RESIZE_H_
