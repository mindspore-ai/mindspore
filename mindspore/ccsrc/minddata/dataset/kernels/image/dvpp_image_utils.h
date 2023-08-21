/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_IMAGE_UTILS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_IMAGE_UTILS_H_

#include <setjmp.h>

#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>
#if defined(_WIN32) || defined(_WIN64)
#undef HAVE_STDDEF_H
#undef HAVE_STDLIB_H
#elif __APPLE__
#include <sys/param.h>
#include <sys/mount.h>
#endif
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
/// \brief Returns Resized image.
/// \param input/output: Tensor of shape <N,H,W,C>, c == 1 or c == 3
/// \param output_height: height of output
/// \param output_width: width of output
/// \param fx: horizontal scale
/// \param fy: vertical scale
/// \param InterpolationMode: the interpolation mode
/// \param output: Resized image of shape <H,outputHeight,outputWidth,C> and same type as input
Status DvppResize(const std::shared_ptr<DeviceTensorAscend910B> &input, std::shared_ptr<DeviceTensorAscend910B> *output,
                  int32_t output_height, int32_t output_width, double fx = 0.0, double fy = 0.0,
                  InterpolationMode mode = InterpolationMode::kLinear);
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_IMAGE_UTILS_H_
