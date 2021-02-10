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

namespace mindspore {
namespace dataset {

// Transform operations for performing computer vision.
namespace vision {

// Forward Declarations
class RotateOperation;

/// \brief CenterCrop TensorTransform.
/// \notes Crops the input image at the center to the given size.
class CenterCrop : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If size is a single value, a square crop of size (size, size) is returned.
  ///     If size has 2 values, it should be (height, width).
  explicit CenterCrop(std::vector<int32_t> size);

  /// \brief Destructor.
  ~CenterCrop() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<int32_t> size_;
};

/// \brief Crop TensorTransform.
/// \notes Crop an image based on location and crop size
class Crop : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] coordinates Starting location of crop. Must be a vector of two values, in the form of {x_coor, y_coor}
  /// \param[in] size Size of the cropped area.
  ///     If size is a single value, a square crop of size (size, size) is returned.
  ///     If size has 2 values, it should be (height, width).
  Crop(std::vector<int32_t> coordinates, std::vector<int32_t> size);

  /// \brief Destructor.
  ~Crop() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<int32_t> coordinates_;
  std::vector<int32_t> size_;
};

/// \brief Decode TensorTransform.
/// \notes Decode the input image in RGB mode.
class Decode : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] rgb A boolean of whether to decode in RGB mode or not.
  explicit Decode(bool rgb = true);

  /// \brief Destructor.
  ~Decode() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  bool rgb_;
};

/// \brief Normalize TensorTransform.
/// \notes Normalize the input image with respect to mean and standard deviation.
class Normalize : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] mean A vector of mean values for each channel, w.r.t channel order.
  ///     The mean values must be in range [0.0, 255.0].
  /// \param[in] std A vector of standard deviations for each channel, w.r.t. channel order.
  ///     The standard deviation values must be in range (0.0, 255.0]
  Normalize(std::vector<float> mean, std::vector<float> std);

  /// \brief Destructor.
  ~Normalize() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
};

/// \brief Resize TensorTransform.
/// \notes Resize the input image to the given size.
class Resize : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the resized image.
  ///     If size is a single value, the image will be resized to this value with
  ///     the same image aspect ratio. If size has 2 values, it should be (height, width).
  /// \param[in] interpolation An enum for the mode of interpolation
  explicit Resize(std::vector<int32_t> size, InterpolationMode interpolation = InterpolationMode::kLinear);

  /// \brief Destructor.
  ~Resize() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<int32_t> size_;
  InterpolationMode interpolation_;
};

/// \brief Rotate TensorTransform.
/// \notes Rotate the input image using a specified angle id.
class Rotate : public TensorTransform {
 public:
  /// \brief Constructor.
  Rotate();

  /// \brief Destructor.
  ~Rotate() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::shared_ptr<RotateOperation> op_;
};

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_VISION_LITE_H_
