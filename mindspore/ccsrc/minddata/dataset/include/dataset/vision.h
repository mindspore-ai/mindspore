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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "utils/dual_abi_helper.h"
#include "utils/status.h"
#include "include/dataset/constants.h"
#include "include/dataset/transforms.h"
#include "include/dataset/vision_lite.h"
#include "include/dataset/ms_tensor.h"
#include "minddata/dataset/util/visible.h"

namespace mindspore::dataset {
class TensorOperation;

// Transform operations for performing computer vision.
namespace vision {

/// \brief Randomly adjust the brightness, contrast, saturation, and hue of the input image.
class DATASET_API RandomColorAdjust final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] brightness Brightness adjustment factor. Must be a vector of one or two values
  ///     if it is a vector of two values it needs to be in the form of [min, max] (Default={1, 1}).
  /// \param[in] contrast Contrast adjustment factor. Must be a vector of one or two values
  ///     if it is a vector of two values, it needs to be in the form of [min, max] (Default={1, 1}).
  /// \param[in] saturation Saturation adjustment factor. Must be a vector of one or two values
  ///     if it is a vector of two values, it needs to be in the form of [min, max] (Default={1, 1}).
  /// \param[in] hue Hue adjustment factor. Must be a vector of one or two values
  ///     if it is a vector of two values, it must be in the form of [min, max] where -0.5 <= min <= max <= 0.5
  ///     (Default={0, 0}).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_color_adjust_op = vision::RandomColorAdjust({1.0, 5.0}, {10.0, 20.0}, {40.0, 40.0});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_color_adjust_op},  // operations
  ///                            {"image"});                           // input columns
  /// \endcode
  explicit RandomColorAdjust(const std::vector<float> &brightness = {1.0, 1.0},
                             const std::vector<float> &contrast = {1.0, 1.0},
                             const std::vector<float> &saturation = {1.0, 1.0},
                             const std::vector<float> &hue = {0.0, 0.0});

  /// \brief Destructor.
  ~RandomColorAdjust() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Crop the input image at a random location.
class DATASET_API RandomCrop final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If the size is a single value, a squared crop of size (size, size) is returned.
  ///     If the size has 2 values, it should be (height, width).
  /// \param[in] padding A vector representing the number of pixels to pad the image.
  ///    If the vector has one value, it pads all sides of the image with that value.
  ///    If the vector has two values, it pads left and right with the first and
  ///    top and bottom with the second value.
  ///    If the vector has four values, it pads left, top, right, and bottom with
  ///    those values respectively.
  /// \param[in] pad_if_needed A boolean indicating that whether to pad the image
  ///    if either side is smaller than the given output size.
  /// \param[in] fill_value A vector representing the pixel intensity of the borders if the padding_mode is
  ///     BorderType.kConstant. If 1 value is provided, it is used for all RGB channels.
  ///     If 3 values are provided, it is used to fill R, G, B channels respectively.
  /// \param[in] padding_mode The method of padding (default=BorderType::kConstant).It can be any of
  ///     [BorderType::kConstant, BorderType::kEdge, BorderType::kReflect, BorderType::kSymmetric].
  ///   - BorderType::kConstant, Fill the border with constant values.
  ///   - BorderType::kEdge, Fill the border with the last value on the edge.
  ///   - BorderType::kReflect, Reflect the values on the edge omitting the last value of edge.
  ///   - BorderType::kSymmetric, Reflect the values on the edge repeating the last value of edge.
  /// \note If the input image is more than one, then make sure that the image size is the same.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_crop_op = vision::RandomCrop({255, 255}, {10, 10, 10, 10});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_crop_op},  // operations
  ///                            {"image"});                   // input columns
  /// \endcode
  explicit RandomCrop(const std::vector<int32_t> &size, const std::vector<int32_t> &padding = {0, 0, 0, 0},
                      bool pad_if_needed = false, const std::vector<uint8_t> &fill_value = {0, 0, 0},
                      BorderType padding_mode = BorderType::kConstant);

  /// \brief Destructor.
  ~RandomCrop() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Equivalent to RandomResizedCrop TensorTransform, but crop the image before decoding.
class DATASET_API RandomCropDecodeResize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///               If the size is a single value, a squared crop of size (size, size) is returned.
  ///               If the size has 2 values, it should be (height, width).
  /// \param[in] scale Range [min, max) of respective size of the
  ///               original size to be cropped (default=(0.08, 1.0)).
  /// \param[in] ratio Range [min, max) of aspect ratio to be
  ///               cropped (default=(3. / 4., 4. / 3.)).
  /// \param[in] interpolation An enum for the mode of interpolation.
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  ///   - InterpolationMode::kCubicPil, Interpolation method is bicubic interpolation like implemented in pillow.
  /// \param[in] max_attempts The maximum number of attempts to propose a valid crop_area (default=10).
  ///               If exceeded, fall back to use center_crop instead.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto random_op = vision::RandomCropDecodeResize({255, 255}, {0.1, 0.5});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomCropDecodeResize(const std::vector<int32_t> &size, const std::vector<float> &scale = {0.08, 1.0},
                                  const std::vector<float> &ratio = {3. / 4., 4. / 3.},
                                  InterpolationMode interpolation = InterpolationMode::kLinear,
                                  int32_t max_attempts = 10);

  /// \brief Destructor.
  ~RandomCropDecodeResize() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly flip the input image horizontally with a given probability.
class DATASET_API RandomHorizontalFlip final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of flip.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomHorizontalFlip(0.8);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomHorizontalFlip(float prob = 0.5);

  /// \brief Destructor.
  ~RandomHorizontalFlip() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Crop the input image to a random size and aspect ratio.
class DATASET_API RandomResizedCrop final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If the size is a single value, a squared crop of size (size, size) is returned.
  ///     If the size has 2 values, it should be (height, width).
  /// \param[in] scale Range [min, max) of respective size of the original
  ///     size to be cropped (default=(0.08, 1.0)).
  /// \param[in] ratio Range [min, max) of aspect ratio to be cropped
  ///     (default=(3. / 4., 4. / 3.)).
  /// \param[in] interpolation Image interpolation mode (default=InterpolationMode::kLinear).
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  ///   - InterpolationMode::kCubicPil, Interpolation method is bicubic interpolation like implemented in pillow.
  /// \param[in] max_attempts The maximum number of attempts to propose a valid.
  ///     crop_area (default=10). If exceeded, fall back to use center_crop instead.
  /// \note If the input image is more than one, then make sure that the image size is the same.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomResizedCrop({32, 32}, {0.08, 1.0});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomResizedCrop(const std::vector<int32_t> &size, const std::vector<float> &scale = {0.08, 1.0},
                             const std::vector<float> &ratio = {3. / 4., 4. / 3.},
                             InterpolationMode interpolation = InterpolationMode::kLinear, int32_t max_attempts = 10);

  /// \brief Destructor.
  ~RandomResizedCrop() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly flip the input image vertically with a given probability.
class DATASET_API RandomVerticalFlip final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of flip.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomVerticalFlip();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomVerticalFlip(float prob = 0.5);

  /// \brief Destructor.
  ~RandomVerticalFlip() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Change the format of input tensor from 4-channel RGBA to 3-channel BGR.
class DATASET_API RGBA2BGR final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto rgb2bgr_op = vision::RGBA2BGR();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, rgb2bgr_op},  // operations
  ///                            {"image"});               // input columns
  /// \endcode
  RGBA2BGR();

  /// \brief Destructor.
  ~RGBA2BGR() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Change the input 4 channel RGBA tensor to 3 channel RGB.
class DATASET_API RGBA2RGB final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto rgba2rgb_op = vision::RGBA2RGB();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, rgba2rgb_op},  // operations
  ///                            {"image"});                // input columns
  /// \endcode
  RGBA2RGB();

  /// \brief Destructor.
  ~RGBA2RGB() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Divide the pixel values by 255 and convert from HWC format to CHW format with required datatype.
class DATASET_API ToTensor final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] output_type The type of the output tensor of type mindspore::DataType or String
  ///   (default=mindspore::DataType::kNumberTypeFloat32).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto to_tensor_op = vision::ToTensor();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({to_tensor_op},  // operations
  ///                            {"image"});  // input columns
  /// \endcode
  ToTensor();
  explicit ToTensor(std::string output_type);
  explicit ToTensor(mindspore::DataType output_type);

  /// \brief Destructor.
  ~ToTensor() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

}  // namespace vision
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_H_
