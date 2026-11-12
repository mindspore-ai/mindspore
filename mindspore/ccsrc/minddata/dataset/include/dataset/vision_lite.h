/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include "utils/status.h"
#include "include/dataset/constants.h"
#include "include/dataset/transforms.h"
#include "minddata/dataset/util/visible.h"

namespace mindspore::dataset::vision {
// Forward Declarations
class RotateOperation;

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
}  // namespace mindspore::dataset::vision
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_LITE_H_
