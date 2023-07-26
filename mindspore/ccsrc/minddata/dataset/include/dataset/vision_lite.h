/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
class DATASET_API Affine final : public TensorTransform {
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
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto affine_op = vision::Affine(30, {0.0, 0.0}, 0.8);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, affine_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit Affine(float_t degrees, const std::vector<float> &translation = {0.0, 0.0}, float scale = 0.0,
                  const std::vector<float> &shear = {0.0, 0.0},
                  InterpolationMode interpolation = InterpolationMode::kNearestNeighbour,
                  const std::vector<uint8_t> &fill_value = {0, 0, 0});

  /// \brief Destructor.
  ~Affine() override = default;

  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Crop the input image at the center to the given size.
class DATASET_API CenterCrop final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If the size is a single value, a squared crop of size (size, size) is returned.
  ///     If the size has 2 values, it should be (height, width).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto crop_op = vision::CenterCrop({32, 32});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, crop_op},  // operations
  ///                            {"image"});            // input columns
  /// \endcode
  explicit CenterCrop(const std::vector<int32_t> &size);

  /// \brief Destructor.
  ~CenterCrop() override = default;

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
class DATASET_API Crop final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] coordinates Starting location of crop. Must be a vector of two values, in the form of {x_coor, y_coor}.
  /// \param[in] size Size of the cropped area.
  ///     If the size is a single value, a squared crop of size (size, size) is returned.
  ///     If the size has 2 values, it should be (height, width).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto crop_op = vision::Crop({0, 0}, {32, 32});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, crop_op},  // operations
  ///                            {"image"});            // input columns
  /// \endcode
  Crop(const std::vector<int32_t> &coordinates, const std::vector<int32_t> &size);

  /// \brief Destructor.
  ~Crop() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Decode the input image in RGB mode.
class DATASET_API Decode final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] rgb A boolean indicating whether to decode the image in RGB mode or not.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op},  // operations
  ///                            {"image"});   // input columns
  /// \endcode
  explicit Decode(bool rgb = true);

  /// \brief Destructor.
  ~Decode() override = default;

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
class DATASET_API GaussianBlur final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] kernel_size A vector of Gaussian kernel size for width and height. The value must be positive and odd.
  /// \param[in] sigma A vector of Gaussian kernel standard deviation sigma for width and height. The values must be
  ///     positive. Using default value 0 means to calculate the sigma according to the kernel size.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto gaussian_op = vision::GaussianBlur({7, 7});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, gaussian_op},  // operations
  ///                            {"image"});                // input columns
  /// \endcode
  explicit GaussianBlur(const std::vector<int32_t> &kernel_size, const std::vector<float> &sigma = {0., 0.});

  /// \brief Destructor.
  ~GaussianBlur() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Transpose the input image; shape (H, W, C) to shape (C, H, W).
class DATASET_API HWC2CHW final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({std::make_shared<vision::Decode>(),
  ///                             std::make_shared<vision::HWC2CHW>()}, // operations
  ///                            {"image"});                            // input columns
  /// \endcode
  HWC2CHW();

  /// \brief Destructor.
  ~HWC2CHW() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Normalize the input image with respect to mean and standard deviation.
class DATASET_API Normalize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] mean A vector of mean values for each channel, with respect to channel order.
  ///     The mean values must be in range [0.0, 255.0].
  /// \param[in] std A vector of standard deviations for each channel, with respect to channel order.
  ///     The standard deviation values must be in range (0.0, 255.0].
  /// \param[in] is_hwc A boolean to indicate whether the input image is in HWC format (true) or CHW
  ///     format (false) (Lite only supports true) (default = true).

  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto normalize_op = vision::Normalize({128, 128, 128}, {1.0, 1.0, 1.0});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, normalize_op},  // operations
  ///                            {"image"});                 // input columns
  /// \endcode
  Normalize(const std::vector<float> &mean, const std::vector<float> &std, bool is_hwc = true);

  /// \brief Destructor.
  ~Normalize() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

  std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Pad the image according to padding parameters.
class DATASET_API Pad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] padding A vector representing the number of pixels to pad the image.
  ///    If the vector has one value, it pads all sides of the image with that value.
  ///    If the vector has two values, it pads left and right with the first and
  ///    top and bottom with the second value.
  ///    If the vector has four values, it pads left, top, right, and bottom with
  ///    those values respectively.
  /// \param[in] fill_value A vector representing the pixel intensity of the borders. Only valid if the
  ///    padding_mode is BorderType.kConstant. If 1 value is provided, it is used for all RGB channels.
  ///    If 3 values are provided, it is used to fill R, G, B channels respectively.
  /// \param[in] padding_mode The method of padding (default=BorderType.kConstant).
  ///    Can be any of
  ///    [BorderType.kConstant, BorderType.kEdge, BorderType.kReflect, BorderType.kSymmetric]
  ///    - BorderType.kConstant, means it fills the border with constant values
  ///    - BorderType.kEdge, means it pads with the last value on the edge
  ///    - BorderType.kReflect, means it reflects the values on the edge omitting the last value of edge
  ///    - BorderType.kSymmetric, means it reflects the values on the edge repeating the last value of edge
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto pad_op = vision::Pad({10, 10, 10, 10}, {255, 255, 255});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, pad_op},  // operations
  ///                            {"image"});           // input columns
  /// \endcode
  explicit Pad(const std::vector<int32_t> &padding, const std::vector<uint8_t> &fill_value = {0},
               BorderType padding_mode = BorderType::kConstant);

  /// \brief Destructor.
  ~Pad() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply a Random Affine transformation on the input image in RGB or Greyscale mode.
class DATASET_API RandomAffine final : public TensorTransform {
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
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomAffine({45, 90});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomAffine(const std::vector<float_t> &degrees,
                        const std::vector<float_t> &translate_range = {0.0, 0.0, 0.0, 0.0},
                        const std::vector<float_t> &scale_range = {1.0, 1.0},
                        const std::vector<float_t> &shear_ranges = {0.0, 0.0, 0.0, 0.0},
                        InterpolationMode interpolation = InterpolationMode::kNearestNeighbour,
                        const std::vector<uint8_t> &fill_value = {0, 0, 0});

  /// \brief Destructor.
  ~RandomAffine() override = default;

  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Rescale the pixel value of input image.
class DATASET_API Rescale final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] rescale Rescale factor.
  /// \param[in] shift Shift factor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto rescale_op = vision::Rescale(1.0, 0.0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, rescale_op},  // operations
  ///                            {"image"});               // input columns
  /// \endcode
  Rescale(float rescale, float shift);

  /// \brief Destructor.
  ~Rescale() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Resize the input image to the given size.
class DATASET_API Resize final : public TensorTransform {
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
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto resize_op = vision::Resize({224, 224}, InterpolationMode::kLinear);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, resize_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit Resize(const std::vector<int32_t> &size, InterpolationMode interpolation = InterpolationMode::kLinear);

  /// \brief Destructor.
  ~Resize() override = default;

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
class DATASET_API ResizePreserveAR final : public TensorTransform {
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
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto resize_op = vision::ResizePreserveAR(224, 224);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, resize_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  ResizePreserveAR(int32_t height, int32_t width, int32_t img_orientation = 0);

  /// \brief Destructor.
  ~ResizePreserveAR() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief RGB2BGR TensorTransform.
/// \note Convert the format of input image from RGB to BGR.
class DATASET_API RGB2BGR final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto rgb2bgr_op = vision::RGB2BGR();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, rgb2bgr_op},  // operations
  ///                            {"image"});               // input columns
  /// \endcode
  RGB2BGR() = default;

  /// \brief Destructor.
  ~RGB2BGR() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief RGB2GRAY TensorTransform.
/// \note Convert RGB image or color image to grayscale image.
/// \brief Convert a RGB image or color image to a grayscale one.
class DATASET_API RGB2GRAY final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto rgb2gray_op = vision::RGB2GRAY();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, rgb2gray_op},  // operations
  ///                            {"image"});                // input columns
  /// \endcode
  RGB2GRAY() = default;

  /// \brief Destructor.
  ~RGB2GRAY() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Rotate the input image according to parameters.
class DATASET_API Rotate final : public TensorTransform {
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
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto rotate_op = vision::Rotate(FixRotationAngle::k90Degree);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, rotate_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
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
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto rotate_op = vision::Rotate(90);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, rotate_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit Rotate(float degrees, InterpolationMode resample = InterpolationMode::kNearestNeighbour, bool expand = false,
                  const std::vector<float> &center = {}, const std::vector<uint8_t> &fill_value = {0, 0, 0});

  /// \brief Destructor.
  ~Rotate() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::shared_ptr<RotateOperation> op_;
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Swap the red and blue channels of the input image.
class DATASET_API SwapRedBlue final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto swap_red_blue_op = vision::SwapRedBlue();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, swap_red_blue_op},  // operations
  ///                            {"image"});                     // input columns
  /// \endcode
  SwapRedBlue();

  /// \brief Destructor.
  ~SwapRedBlue() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_LITE_H_
