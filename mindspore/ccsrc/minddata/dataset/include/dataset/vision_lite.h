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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_LITE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_LITE_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "include/api/status.h"
#include "include/dataset/constants.h"
#include "include/dataset/transforms.h"

namespace mindspore {
namespace dataset {

// Transform operations for performing computer vision.
namespace vision {

// Forward Declarations
class RotateOperation;

/// \brief Apply affine transform on the input image.
class Affine final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] degrees The degrees to rotate the image.
  /// \param[in] translation The values representing vertical and horizontal translation (default = {0.0, 0.0}).
  ///     The first value represents the x axis translation while the second represents the y axis translation.
  /// \param[in] scale The scaling factor for the image (default = 0.0).
  /// \param[in] shear A float vector of size 2, representing the shear degrees (default = {0.0, 0.0}).
  /// \param[in] interpolation An enum for the mode of interpolation.
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation (Only supports this mode in Lite).
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  /// \param[in] fill_value A vector representing the value to fill the area outside the transformation
  ///    in the output image. If 1 value is provided, it is used for all RGB channels.
  ///    If 3 values are provided, it is used to fill R, G, B channels respectively.
  explicit Affine(float_t degrees, const std::vector<float> &translation = {0.0, 0.0}, float scale = 0.0,
                  const std::vector<float> &shear = {0.0, 0.0},
                  InterpolationMode interpolation = InterpolationMode::kNearestNeighbour,
                  const std::vector<uint8_t> &fill_value = {0, 0, 0});

  /// \brief Destructor.
  ~Affine() = default;

  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Crop the input image at the center to the given size.
class CenterCrop final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If the size is a single value, a squared crop of size (size, size) is returned.
  ///     If the size has 2 values, it should be (height, width).
  explicit CenterCrop(std::vector<int32_t> size);

  /// \brief Destructor.
  ~CenterCrop() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

  std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Crop an image based on location and crop size.
class Crop final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] coordinates Starting location of crop. Must be a vector of two values, in the form of {x_coor, y_coor}.
  /// \param[in] size Size of the cropped area.
  ///     If the size is a single value, a squared crop of size (size, size) is returned.
  ///     If the size has 2 values, it should be (height, width).
  Crop(std::vector<int32_t> coordinates, std::vector<int32_t> size);

  /// \brief Destructor.
  ~Crop() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Decode the input image in RGB mode.
class Decode final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] rgb A boolean indicating whether to decode the image in RGB mode or not.
  explicit Decode(bool rgb = true);

  /// \brief Destructor.
  ~Decode() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

  std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Blur the input image with the specified Gaussian kernel.
class GaussianBlur final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] kernel_size A vector of Gaussian kernel size for width and height. The value must be positive and odd.
  /// \param[in] sigma A vector of Gaussian kernel standard deviation sigma for width and height. The values must be
  ///     positive. Using default value 0 means to calculate the sigma according to the kernel size.
  GaussianBlur(const std::vector<int32_t> &kernel_size, const std::vector<float> &sigma = {0., 0.});

  /// \brief Destructor.
  ~GaussianBlur() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Normalize the input image with respect to mean and standard deviation.
class Normalize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] mean A vector of mean values for each channel, with respect to channel order.
  ///     The mean values must be in range [0.0, 255.0].
  /// \param[in] std A vector of standard deviations for each channel, with respect to channel order.
  ///     The standard deviation values must be in range (0.0, 255.0].
  Normalize(std::vector<float> mean, std::vector<float> std);

  /// \brief Destructor.
  ~Normalize() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

  std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply a Random Affine transformation on the input image in RGB or Greyscale mode.
class RandomAffine final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] degrees A float vector of size 2, representing the starting and ending degree.
  /// \param[in] translate_range A float vector of size 2 or 4, representing percentages of translation on x and y axes.
  ///    If the size is 2, (min_dx, max_dx, 0, 0).
  ///    If the size is 4, (min_dx, max_dx, min_dy, max_dy),
  ///    all values are in range [-1, 1].
  /// \param[in] scale_range A float vector of size 2, representing the starting and ending scales in the range.
  /// \param[in] shear_ranges A float vector of size 2 or 4, representing the starting and ending shear degrees
  ///    vertically and horizontally.
  ///    If the size is 2, (min_shear_x, max_shear_x, 0, 0),
  ///    if the size is 4, (min_shear_x, max_shear_x, min_shear_y, max_shear_y).
  /// \param[in] interpolation An enum for the mode of interpolation.
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation (Only supports this mode in Lite).
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  /// \param[in] fill_value A vector representing the value to fill the area outside the transform
  ///    in the output image. If 1 value is provided, it is used for all RGB channels.
  ///    If 3 values are provided, it is used to fill R, G and B channels respectively.
  explicit RandomAffine(const std::vector<float_t> &degrees,
                        const std::vector<float_t> &translate_range = {0.0, 0.0, 0.0, 0.0},
                        const std::vector<float_t> &scale_range = {1.0, 1.0},
                        const std::vector<float_t> &shear_ranges = {0.0, 0.0, 0.0, 0.0},
                        InterpolationMode interpolation = InterpolationMode::kNearestNeighbour,
                        const std::vector<uint8_t> &fill_value = {0, 0, 0});

  /// \brief Destructor.
  ~RandomAffine() = default;

  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Resize the input image to the given size.
class Resize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the resized image.
  ///     If the size is a single value, the image will be resized to this value with
  ///     the same image aspect ratio. If the size has 2 values, it should be (height, width).
  /// \param[in] interpolation An enum for the mode of interpolation.
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation (Only supports this mode in Lite).
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  ///   - InterpolationMode::kCubicPil, Interpolation method is bicubic interpolation like implemented in pillow.
  explicit Resize(std::vector<int32_t> size, InterpolationMode interpolation = InterpolationMode::kLinear);

  /// \brief Destructor.
  ~Resize() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

  std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Keep the original picture ratio and fills the rest.
class ResizePreserveAR final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] height The height of image output value after resizing.
  /// \param[in] width The width of image output value after resizing.
  /// \param[in] img_orientation optional rotation angle.
  ///     - img_orientation = 1, Rotate 0 degree.
  ///     - img_orientation = 2, Rotate 0 degree and apply horizontal flip.
  ///     - img_orientation = 3, Rotate 180 degree.
  ///     - img_orientation = 4, Rotate 180 degree and apply horizontal flip.
  ///     - img_orientation = 5, Rotate 90 degree and apply horizontal flip.
  ///     - img_orientation = 6, Rotate 90 degree.
  ///     - img_orientation = 7, Rotate 270 degree and apply horizontal flip.
  ///     - img_orientation = 8, Rotate 270 degree.
  ResizePreserveAR(int32_t height, int32_t width, int32_t img_orientation = 0);

  /// \brief Destructor.
  ~ResizePreserveAR() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief RGB2BGR TensorTransform.
/// \notes Convert the format of input image from RGB to BGR.
class RGB2BGR final : public TensorTransform {
 public:
  /// \brief Constructor.
  RGB2BGR() = default;

  /// \brief Destructor.
  ~RGB2BGR() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief RGB2GRAY TensorTransform.
/// \note Convert RGB image or color image to grayscale image.
/// \brief Convert a RGB image or color image to a grayscale one.
class RGB2GRAY final : public TensorTransform {
 public:
  /// \brief Constructor.
  RGB2GRAY() = default;

  /// \brief Destructor.
  ~RGB2GRAY() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Rotate the input image according to parameters.
class Rotate final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \note This api is only used in Lite, the interpolation mode is bilinear.
  /// \param[in] angle_id The fix rotation angle.
  ///     - FixRotationAngle::k0Degree = 1, Rotate 0 degree.
  ///     - FixRotationAngle::k0DegreeAndMirror = 2, Rotate 0 degree and apply horizontal flip.
  ///     - FixRotationAngle::k180Degree = 3, Rotate 180 degree.
  ///     - FixRotationAngle::k180DegreeAndMirror = 4, Rotate 180 degree and apply horizontal flip.
  ///     - FixRotationAngle::k90DegreeAndMirror = 5, Rotate 90 degree and apply horizontal flip.
  ///     - FixRotationAngle::k90Degree = 6, Rotate 90 degree.
  ///     - FixRotationAngle::k270DegreeAndMirror = 7, Rotate 270 degree and apply horizontal flip.
  ///     - FixRotationAngle::k270Degree = 8, Rotate 270 degree.
  explicit Rotate(FixRotationAngle angle_id = FixRotationAngle::k0Degree);

  /// \brief Constructor.
  /// \param[in] degrees A float value, representing the rotation degrees.
  /// \param[in] resample An enum for the mode of interpolation.
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  /// \param[in] expand A boolean representing whether the image is expanded after rotation.
  /// \param[in] center A float vector of size 2 or empty, representing the x and y center of rotation
  ///     or the center of the image.
  /// \param[in] fill_value A vector representing the value to fill the area outside the transform
  ///    in the output image. If 1 value is provided, it is used for all RGB channels.
  ///    If 3 values are provided, it is used to fill R, G, B channels respectively.
  Rotate(float degrees, InterpolationMode resample = InterpolationMode::kNearestNeighbour, bool expand = false,
         std::vector<float> center = {}, std::vector<uint8_t> fill_value = {0, 0, 0});

  /// \brief Destructor.
  ~Rotate() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::shared_ptr<RotateOperation> op_;
  struct Data;
  std::shared_ptr<Data> data_;
};

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_LITE_H_
