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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_DVPP_IMAGE_UTILS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_DVPP_IMAGE_UTILS_H_

#include <csetjmp>

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
#include "minddata/dataset/kernels/image/dvpp/utils/ErrorCode.h"

namespace mindspore {
namespace dataset {
const int kInvalidInterpolationMode = 100;

/// \brief Convert InterpolationMode to dvpp mode
inline int GetDVPPInterpolationMode(InterpolationMode mode) {
  switch (mode) {
    case InterpolationMode::kLinear:
      return 0;  // dvpp BILINEAR
    case InterpolationMode::kCubic:
      return 2;  // dvpp BICUBIC
    case InterpolationMode::kArea:
      return kInvalidInterpolationMode;
    case InterpolationMode::kNearestNeighbour:
      return 1;  // dvpp NEAREST
    default:
      return kInvalidInterpolationMode;
  }
}

/// \brief Returns Resized image.
/// \param input/output: Tensor of shape <N,H,W,C>, c == 1 or c == 3
/// \param output_height: height of output
/// \param output_width: width of output
/// \param fx: horizontal scale
/// \param fy: vertical scale
/// \param InterpolationMode: the interpolation mode
/// \param output: Resized image of shape <H,outputHeight,outputWidth,C> and same type as input
APP_ERROR DvppResize(const std::shared_ptr<DeviceTensorAscend910B> &input,
                     std::shared_ptr<DeviceTensorAscend910B> *output, int32_t output_height, int32_t output_width,
                     double fx = 0.0, double fy = 0.0, InterpolationMode mode = InterpolationMode::kLinear);

/// \brief Returns Decoded image
/// Supported images: JPEG JPG
/// \param input: input containing the not decoded image 1D bytes
/// \param output: Decoded image Tensor of shape <H,W,C> and type DE_UINT8. Pixel order is RGB
APP_ERROR DvppDecode(const std::shared_ptr<DeviceTensorAscend910B> &input,
                     std::shared_ptr<DeviceTensorAscend910B> *output);

/// \brief Returns Normalized image
/// \param input: Tensor of shape <H,W,C> in RGB order.
/// \param mean: Tensor of shape <3> and type DE_FLOAT32 which are mean of each channel in RGB order
/// \param std:  Tensor of shape <3> and type DE_FLOAT32 which are std of each channel in RGB order
/// \param is_hwc: Check if input is HWC/CHW format
/// \param output: Normalized image Tensor of same input shape and type DE_FLOAT32
APP_ERROR DvppNormalize(const std::shared_ptr<DeviceTensorAscend910B> &input,
                        std::shared_ptr<DeviceTensorAscend910B> *output, std::vector<float> mean,
                        std::vector<float> std, bool is_hwc);

/// \brief Returns image with adjusting brightness
/// \param input: Tensor of shape <H,W,C> format.
/// \param factor: brightness factor.
/// \param output: Augmented image Tensor of same input shape (type DE_FLOAT32 and DE_UINT8)
APP_ERROR DvppAdjustBrightness(const std::shared_ptr<DeviceTensorAscend910B> &input,
                               std::shared_ptr<DeviceTensorAscend910B> *output, float factor);

/// \brief Returns image with adjusting contrast
/// \param input: Tensor of shape <H,W,C> format.
/// \param factor: contrast factor.
/// \param output: Augmented image Tensor of same input shape (type DE_FLOAT32 and DE_UINT8)
APP_ERROR DvppAdjustContrast(const std::shared_ptr<DeviceTensorAscend910B> &input,
                             std::shared_ptr<DeviceTensorAscend910B> *output, float factor);

/// \brief Returns image with adjusting hue
/// \param input: Tensor of shape <H,W,C> format.
/// \param factor: hue factor.
/// \param output: Augmented image Tensor of same input shape (type DE_FLOAT32 and DE_UINT8)
APP_ERROR DvppAdjustHue(const std::shared_ptr<DeviceTensorAscend910B> &input,
                        std::shared_ptr<DeviceTensorAscend910B> *output, float factor);

/// \brief Returns image with adjusting saturation
/// \param input: Tensor of shape <H,W,C> format.
/// \param factor: saturation factor.
/// \param output: Augmented image Tensor of same input shape (type DE_FLOAT32 and DE_UINT8)
APP_ERROR DvppAdjustSaturation(const std::shared_ptr<DeviceTensorAscend910B> &input,
                               std::shared_ptr<DeviceTensorAscend910B> *output, float factor);

APP_ERROR GetSocName(std::string *soc_name);

APP_ERROR CreateAclTensor(const int64_t *view_dims, uint64_t view_dims_num, mindspore::TypeId data_type,
                          const int64_t *stride, int64_t offset, const int64_t *storage_dims, uint64_t storage_dims_num,
                          void *tensor_data, bool is_hwc, void **acl_tensor);
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_DVPP_IMAGE_UTILS_H_
