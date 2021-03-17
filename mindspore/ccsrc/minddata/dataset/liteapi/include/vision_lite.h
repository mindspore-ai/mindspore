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
#include "include/constants.h"
#include "include/transforms.h"

namespace mindspore {
namespace dataset {

// Transform operations for performing computer vision.
namespace vision {

// Forward Declarations
class RotateOperation;

/// \brief Affine TensorTransform.
/// \notes Apply affine transform on input image.
class Affine final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] degrees The degrees to rotate the image by
  /// \param[in] translation The value representing vertical and horizontal translation (default = {0.0, 0.0})
  ///     The first value represent the x axis translation while the second represents y axis translation.
  /// \param[in] scale The scaling factor for the image (default = 0.0)
  /// \param[in] shear A float vector of size 2, representing the shear degrees (default = {0.0, 0.0})
  /// \param[in] interpolation An enum for the mode of interpolation
  /// \param[in] fill_value A vector representing the value to fill the area outside the transform
  ///    in the output image. If 1 value is provided, it is used for all RGB channels.
  ///    If 3 values are provided, it is used to fill R, G, B channels respectively.
  explicit Affine(float_t degrees, const std::vector<float> &translation = {0.0, 0.0}, float scale = 0.0,
                  const std::vector<float> &shear = {0.0, 0.0},
                  InterpolationMode interpolation = InterpolationMode::kNearestNeighbour,
                  const std::vector<uint8_t> &fill_value = {0, 0, 0});

  /// \brief Destructor.
  ~Affine() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief CenterCrop TensorTransform.
/// \notes Crops the input image at the center to the given size.
class CenterCrop final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If size is a single value, a square crop of size (size, size) is returned.
  ///     If size has 2 values, it should be (height, width).
  explicit CenterCrop(std::vector<int32_t> size);

  /// \brief Destructor.
  ~CenterCrop() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

  std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief ResizePreserveAR TensorTransform.
/// \notes Keep the original picture ratio and fill the rest.
class ResizePreserveAR final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] height The height of image output value after resizing.
  /// \param[in] width The width of image output value after resizing.
  /// \param[in] img_orientation Angle method of image rotation.
  ResizePreserveAR(int32_t height, int32_t width, int32_t img_orientation = 0);

  /// \brief Destructor.
  ~ResizePreserveAR() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief RGB2GRAY TensorTransform.
/// \notes Convert RGB image or color image to grayscale image
class RGB2GRAY : public TensorTransform {
 public:
  /// \brief Constructor.
  RGB2GRAY() = default;

  /// \brief Destructor.
  ~RGB2GRAY() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Crop TensorTransform.
/// \notes Crop an image based on location and crop size
class Crop final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] coordinates Starting location of crop. Must be a vector of two values, in the form of {x_coor, y_coor}
  /// \param[in] size Size of the cropped area.
  ///     If size is a single value, a square crop of size (size, size) is returned.
  ///     If size has 2 values, it should be (height, width).
  Crop(std::vector<int32_t> coordinates, std::vector<int32_t> size);

  /// \brief Destructor.
  ~Crop() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Decode TensorTransform.
/// \notes Decode the input image in RGB mode.
class Decode final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] rgb A boolean of whether to decode in RGB mode or not.
  explicit Decode(bool rgb = true);

  /// \brief Destructor.
  ~Decode() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

  std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Normalize TensorTransform.
/// \notes Normalize the input image with respect to mean and standard deviation.
class Normalize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] mean A vector of mean values for each channel, w.r.t channel order.
  ///     The mean values must be in range [0.0, 255.0].
  /// \param[in] std A vector of standard deviations for each channel, w.r.t. channel order.
  ///     The standard deviation values must be in range (0.0, 255.0]
  Normalize(std::vector<float> mean, std::vector<float> std);

  /// \brief Destructor.
  ~Normalize() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

  std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief RandomAffine TensorTransform.
/// \notes Applies a Random Affine transformation on input image in RGB or Greyscale mode.
class RandomAffine final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] degrees A float vector of size 2, representing the starting and ending degree
  /// \param[in] translate_range A float vector of size 2 or 4, representing percentages of translation on x and y axes.
  ///    if size is 2, (min_dx, max_dx, 0, 0)
  ///    if size is 4, (min_dx, max_dx, min_dy, max_dy)
  ///    all values are in range [-1, 1]
  /// \param[in] scale_range A float vector of size 2, representing the starting and ending scales in the range.
  /// \param[in] shear_ranges A float vector of size 2 or 4, representing the starting and ending shear degrees
  ///    vertically and horizontally.
  ///    if size is 2, (min_shear_x, max_shear_x, 0, 0)
  ///    if size is 4, (min_shear_x, max_shear_x, min_shear_y, max_shear_y)
  /// \param[in] interpolation An enum for the mode of interpolation
  /// \param[in] fill_value A vector representing the value to fill the area outside the transform
  ///    in the output image. If 1 value is provided, it is used for all RGB channels.
  ///    If 3 values are provided, it is used to fill R, G, B channels respectively.
  explicit RandomAffine(const std::vector<float_t> &degrees,
                        const std::vector<float_t> &translate_range = {0.0, 0.0, 0.0, 0.0},
                        const std::vector<float_t> &scale_range = {1.0, 1.0},
                        const std::vector<float_t> &shear_ranges = {0.0, 0.0, 0.0, 0.0},
                        InterpolationMode interpolation = InterpolationMode::kNearestNeighbour,
                        const std::vector<uint8_t> &fill_value = {0, 0, 0});

  /// \brief Destructor.
  ~RandomAffine() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Resize TensorTransform.
/// \notes Resize the input image to the given size.
class Resize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the resized image.
  ///     If size is a single value, the image will be resized to this value with
  ///     the same image aspect ratio. If size has 2 values, it should be (height, width).
  /// \param[in] interpolation An enum for the mode of interpolation
  explicit Resize(std::vector<int32_t> size, InterpolationMode interpolation = InterpolationMode::kLinear);

  /// \brief Destructor.
  ~Resize() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

  std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Rotate TensorTransform.
/// \notes Rotate the input image using a specified angle id.
class Rotate final : public TensorTransform {
 public:
  /// \brief Constructor.
  Rotate();

  /// \brief Destructor.
  ~Rotate() = default;

 protected:
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
