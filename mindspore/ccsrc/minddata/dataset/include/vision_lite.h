/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_VISION_LITE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_VISION_LITE_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "include/api/status.h"
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/include/transforms.h"

// FIXME - This internal IR header will be removed when external API classes are provided
#include "minddata/dataset/kernels/ir/vision/vision_ir.h"

namespace mindspore {
namespace dataset {

// Transform operations for performing computer vision.
namespace vision {

// Transform Op classes (in alphabetical order)
class CenterCropOperation;
class CropOperation;
class DecodeOperation;
class NormalizeOperation;
class ResizeOperation;
class RotateOperation;

/// \brief Function to create a CenterCrop TensorOperation.
/// \notes Crops the input image at the center to the given size.
/// \param[in] size A vector representing the output size of the cropped image.
///     If size is a single value, a square crop of size (size, size) is returned.
///     If size has 2 values, it should be (height, width).
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<CenterCropOperation> CenterCrop(std::vector<int32_t> size);

/// \brief Function to create a Crop TensorOp
/// \notes Crop an image based on location and crop size
/// \param[in] coordinates Starting location of crop. Must be a vector of two values, in the form of {x_coor, y_coor}
/// \param[in] size Size of the cropped area.
///     If size is a single value, a square crop of size (size, size) is returned.
///     If size has 2 values, it should be (height, width).
/// \return Shared pointer to the current TensorOp
std::shared_ptr<CropOperation> Crop(std::vector<int32_t> coordinates, std::vector<int32_t> size);

/// \brief Function to create a Decode TensorOperation.
/// \notes Decode the input image in RGB mode.
/// \param[in] rgb A boolean of whether to decode in RGB mode or not.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<DecodeOperation> Decode(bool rgb = true);

/// \brief Function to create a Normalize TensorOperation.
/// \notes Normalize the input image with respect to mean and standard deviation.
/// \param[in] mean A vector of mean values for each channel, w.r.t channel order.
///     The mean values must be in range [0.0, 255.0].
/// \param[in] std A vector of standard deviations for each channel, w.r.t. channel order.
///     The standard deviation values must be in range (0.0, 255.0]
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<NormalizeOperation> Normalize(std::vector<float> mean, std::vector<float> std);

/// \brief Function to create a Resize TensorOperation.
/// \notes Resize the input image to the given size.
/// \param[in] size A vector representing the output size of the resized image.
///     If size is a single value, the image will be resized to this value with
///     the same image aspect ratio. If size has 2 values, it should be (height, width).
/// \param[in] interpolation An enum for the mode of interpolation
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<ResizeOperation> Resize(std::vector<int32_t> size,
                                        InterpolationMode interpolation = InterpolationMode::kLinear);

/// \brief Applies an rotate transformation to an image.
/// \notes Rotate the input image using a specified angle id.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RotateOperation> Rotate();

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_VISION_LITE_H_
