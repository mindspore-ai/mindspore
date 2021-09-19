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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RESIZE_CUBIC_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RESIZE_CUBIC_OP_H_

#include <float.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <cmath>
#include <vector>
#include <utility>
#include <random>
#include "lite_cv/lite_mat.h"
#include "minddata/dataset/util/status.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace dataset {
/// \brief Calculate the coefficient for interpolation firstly
int calc_coeff(int input_size, int out_size, int input0, int input1, struct interpolation *interp,
               std::vector<int> &regions, std::vector<double> &coeffs_interp);

/// \brief Normalize the coefficient for interpolation
void normalize_coeff(int out_size, int kernel_size, const std::vector<double> &prekk, std::vector<int> &kk);

/// \brief Apply horizontal interpolation on input image
Status ImagingHorizontalInterp(LiteMat &output, LiteMat input, int offset, int kernel_size,
                               const std::vector<int> &regions, const std::vector<double> &prekk);

/// \brief Apply Vertical interpolation on input image
Status ImagingVerticalInterp(LiteMat &output, LiteMat input, int offset, int kernel_size,
                             const std::vector<int> &regions, const std::vector<double> &prekk);

/// \brief Mainly logic of Cubic interpolation
bool ImageInterpolation(LiteMat input, LiteMat &output, int x_size, int y_size, struct interpolation *interp,
                        const int rect[4]);

/// \brief Apply cubic interpolation on input image and obtain the output image
/// \param[in] input Input image
/// \param[out] dst Output image
/// \param[in] dst_w expected Output image width
/// \param[in] dst_h expected Output image height
bool ResizeCubic(const LiteMat &input, LiteMat &dst, int dst_w, int dst_h);
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RESIZE_CUBIC_OP_H_
