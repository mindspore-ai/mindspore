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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_VISION_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_VISION_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/api/status.h"
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/vision_lite.h"

namespace mindspore {
namespace dataset {

class TensorOperation;

// Transform operations for performing computer vision.
namespace vision {

/// \brief AutoContrast TensorTransform.
/// \notes Apply automatic contrast on input image.
class AutoContrast : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] cutoff Percent of pixels to cut off from the histogram, the valid range of cutoff value is 0 to 100.
  /// \param[in] ignore Pixel values to ignore.
  explicit AutoContrast(float cutoff = 0.0, std::vector<uint32_t> ignore = {});

  /// \brief Destructor.
  ~AutoContrast() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  float cutoff_;
  std::vector<uint32_t> ignore_;
};

/// \brief BoundingBoxAugment TensorTransform.
/// \notes  Apply a given image transform on a random selection of bounding box regions of a given image.
class BoundingBoxAugment : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] transform A TensorTransform transform.
  /// \param[in] ratio Ratio of bounding boxes to apply augmentation on. Range: [0, 1] (default=0.3).
  explicit BoundingBoxAugment(std::shared_ptr<TensorTransform> transform, float ratio = 0.3);

  /// \brief Destructor.
  ~BoundingBoxAugment() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::shared_ptr<TensorOperation> transform_;
  float ratio_;
};

/// \brief Constructor to apply CutMix on a batch of images
/// \notes Masks a random section of each image with the corresponding part of another randomly
///     selected image in that batch
class CutMixBatch : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] image_batch_format The format of the batch
  /// \param[in] alpha The hyperparameter of beta distribution (default = 1.0)
  /// \param[in] prob The probability by which CutMix is applied to each image (default = 1.0)
  explicit CutMixBatch(ImageBatchFormat image_batch_format, float alpha = 1.0, float prob = 1.0);

  /// \brief Destructor.
  ~CutMixBatch() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  float alpha_;
  float prob_;
  ImageBatchFormat image_batch_format_;
};

/// \brief CutOut TensorOp
/// \notes Randomly cut (mask) out a given number of square patches from the input image
class CutOut : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] length Integer representing the side length of each square patch
  /// \param[in] num_patches Integer representing the number of patches to be cut out of an image
  explicit CutOut(int32_t length, int32_t num_patches = 1);

  /// \brief Destructor.
  ~CutOut() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  int32_t length_;
  int32_t num_patches_;
};

/// \brief Equalize TensorTransform.
/// \notes Apply histogram equalization on input image.
class Equalize : public TensorTransform {
 public:
  /// \brief Constructor.
  Equalize();

  /// \brief Destructor.
  ~Equalize() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief HwcToChw TensorTransform.
/// \notes Transpose the input image; shape (H, W, C) to shape (C, H, W).
class HWC2CHW : public TensorTransform {
 public:
  /// \brief Constructor.
  HWC2CHW();

  /// \brief Destructor.
  ~HWC2CHW() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Invert TensorTransform.
/// \notes Apply invert on input image in RGB mode.
class Invert : public TensorTransform {
 public:
  /// \brief Constructor.
  Invert();

  /// \brief Destructor.
  ~Invert() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief MixUpBatch TensorTransform.
/// \notes Apply MixUp transformation on an input batch of images and labels. The labels must be in
///     one-hot format and Batch must be called before calling this function.
class MixUpBatch : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] alpha hyperparameter of beta distribution (default = 1.0)
  explicit MixUpBatch(float alpha = 1);

  /// \brief Destructor.
  ~MixUpBatch() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  float alpha_;
};

/// \brief NormalizePad TensorTransform.
/// \notes Normalize the input image with respect to mean and standard deviation and pad an extra
///     channel with value zero.
class NormalizePad : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] mean A vector of mean values for each channel, w.r.t channel order.
  ///     The mean values must be in range [0.0, 255.0].
  /// \param[in] std A vector of standard deviations for each channel, w.r.t. channel order.
  ///     The standard deviation values must be in range (0.0, 255.0]
  /// \param[in] dtype The output datatype of Tensor.
  ///     The standard deviation values must be "float32" or "float16"（default = "float32"）
  explicit NormalizePad(const std::vector<float> &mean, const std::vector<float> &std,
                        const std::string &dtype = "float32");

  /// \brief Destructor.
  ~NormalizePad() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
  std::string dtype_;
};

/// \brief Pad TensorOp
/// \notes Pads the image according to padding parameters
class Pad : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] padding A vector representing the number of pixels to pad the image
  ///    If vector has one value, it pads all sides of the image with that value.
  ///    If vector has two values, it pads left and top with the first and
  ///    right and bottom with the second value.
  ///    If vector has four values, it pads left, top, right, and bottom with
  ///    those values respectively.
  /// \param[in] fill_value A vector representing the pixel intensity of the borders if the padding_mode is
  ///    BorderType.kConstant. If 1 value is provided, it is used for all RGB channels. If 3 values are provided,
  ///    it is used to fill R, G, B channels respectively.
  /// \param[in] padding_mode The method of padding (default=BorderType.kConstant)
  ///    Can be any of
  ///    [BorderType.kConstant, BorderType.kEdge, BorderType.kReflect, BorderType.kSymmetric]
  ///    - BorderType.kConstant, means it fills the border with constant values
  ///    - BorderType.kEdge, means it pads with the last value on the edge
  ///    - BorderType.kReflect, means it reflects the values on the edge omitting the last value of edge
  ///    - BorderType.kSymmetric, means it reflects the values on the edge repeating the last value of edge
  explicit Pad(std::vector<int32_t> padding, std::vector<uint8_t> fill_value = {0},
               BorderType padding_mode = BorderType::kConstant);

  /// \brief Destructor.
  ~Pad() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<int32_t> padding_;
  std::vector<uint8_t> fill_value_;
  BorderType padding_mode_;
};

/// \brief RandomAffine TensorTransform.
/// \notes Applies a Random Affine transformation on input image in RGB or Greyscale mode.
class RandomAffine : public TensorTransform {
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
  std::vector<float_t> degrees_;          // min_degree, max_degree
  std::vector<float_t> translate_range_;  // maximum x translation percentage, maximum y translation percentage
  std::vector<float_t> scale_range_;      // min_scale, max_scale
  std::vector<float_t> shear_ranges_;     // min_x_shear, max_x_shear, min_y_shear, max_y_shear
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};

/// \brief Blends an image with its grayscale version with random weights
///        t and 1 - t generated from a given range. If the range is trivial
///        then the weights are determinate and t equals the bound of the interval
class RandomColor : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] t_lb Lower bound on the range of random weights
  /// \param[in] t_lb Upper bound on the range of random weights
  explicit RandomColor(float t_lb, float t_ub);

  /// \brief Destructor.
  ~RandomColor() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  float t_lb_;
  float t_ub_;
};

/// \brief RandomColorAdjust TensorTransform.
/// \brief Randomly adjust the brightness, contrast, saturation, and hue of the input image
class RandomColorAdjust : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] brightness Brightness adjustment factor. Must be a vector of one or two values
  ///     if it's a vector of two values it needs to be in the form of [min, max]. Default value is {1, 1}
  /// \param[in] contrast Contrast adjustment factor. Must be a vector of one or two values
  ///     if it's a vector of two values it needs to be in the form of [min, max]. Default value is {1, 1}
  /// \param[in] saturation Saturation adjustment factor. Must be a vector of one or two values
  ///     if it's a vector of two values it needs to be in the form of [min, max]. Default value is {1, 1}
  /// \param[in] hue Brightness adjustment factor. Must be a vector of one or two values
  ///     if it's a vector of two values it must be in the form of [min, max] where -0.5 <= min <= max <= 0.5
  ///     Default value is {0, 0}
  explicit RandomColorAdjust(std::vector<float> brightness = {1.0, 1.0}, std::vector<float> contrast = {1.0, 1.0},
                             std::vector<float> saturation = {1.0, 1.0}, std::vector<float> hue = {0.0, 0.0});

  /// \brief Destructor.
  ~RandomColorAdjust() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<float> brightness_;
  std::vector<float> contrast_;
  std::vector<float> saturation_;
  std::vector<float> hue_;
};

/// \brief RandomCrop TensorTransform.
/// \notes Crop the input image at a random location.
class RandomCrop : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If size is a single value, a square crop of size (size, size) is returned.
  ///     If size has 2 values, it should be (height, width).
  /// \param[in] padding A vector representing the number of pixels to pad the image
  ///    If vector has one value, it pads all sides of the image with that value.
  ///    If vector has two values, it pads left and top with the first and
  ///    right and bottom with the second value.
  ///    If vector has four values, it pads left, top, right, and bottom with
  ///    those values respectively.
  /// \param[in] pad_if_needed A boolean whether to pad the image if either side is smaller than
  ///     the given output size.
  /// \param[in] fill_value A vector representing the pixel intensity of the borders if the padding_mode is
  ///     BorderType.kConstant. If 1 value is provided, it is used for all RGB channels.
  ///     If 3 values are provided, it is used to fill R, G, B channels respectively.
  explicit RandomCrop(std::vector<int32_t> size, std::vector<int32_t> padding = {0, 0, 0, 0},
                      bool pad_if_needed = false, std::vector<uint8_t> fill_value = {0, 0, 0},
                      BorderType padding_mode = BorderType::kConstant);

  /// \brief Destructor.
  ~RandomCrop() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<int32_t> size_;
  std::vector<int32_t> padding_;
  bool pad_if_needed_;
  std::vector<uint8_t> fill_value_;
  BorderType padding_mode_;
};

/// \brief RandomCropDecodeResize TensorTransform.
/// \notes Equivalent to RandomResizedCrop, but crops before decodes.
class RandomCropDecodeResize : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///               If size is a single value, a square crop of size (size, size) is returned.
  ///               If size has 2 values, it should be (height, width).
  /// \param[in] scale Range [min, max) of respective size of the
  ///               original size to be cropped (default=(0.08, 1.0))
  /// \param[in] ratio Range [min, max) of aspect ratio to be
  ///               cropped (default=(3. / 4., 4. / 3.))
  /// \param[in] interpolation An enum for the mode of interpolation
  /// \param[in] The maximum number of attempts to propose a valid crop_area (default=10).
  ///               If exceeded, fall back to use center_crop instead.
  explicit RandomCropDecodeResize(std::vector<int32_t> size, std::vector<float> scale = {0.08, 1.0},
                                  std::vector<float> ratio = {3. / 4, 4. / 3},
                                  InterpolationMode interpolation = InterpolationMode::kLinear,
                                  int32_t max_attempts = 10);

  /// \brief Destructor.
  ~RandomCropDecodeResize() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<int32_t> size_;
  std::vector<float> scale_;
  std::vector<float> ratio_;
  InterpolationMode interpolation_;
  int32_t max_attempts_;
};

/// \brief RandomCropWithBBox TensorTransform.
/// \notes Crop the input image at a random location and adjust bounding boxes accordingly.
class RandomCropWithBBox : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If size is a single value, a square crop of size (size, size) is returned.
  ///     If size has 2 values, it should be (height, width).
  /// \param[in] padding A vector representing the number of pixels to pad the image
  ///    If vector has one value, it pads all sides of the image with that value.
  ///    If vector has two values, it pads left and top with the first and
  ///    right and bottom with the second value.
  ///    If vector has four values, it pads left, top, right, and bottom with
  ///    those values respectively.
  /// \param[in] pad_if_needed A boolean whether to pad the image if either side is smaller than
  ///     the given output size.
  /// \param[in] fill_value A vector representing the pixel intensity of the borders if the padding_mode is
  ///     BorderType.kConstant. If 1 value is provided, it is used for all RGB channels.
  ///     If 3 values are provided, it is used to fill R, G, B channels respectively.
  /// \param[in] padding_mode The method of padding (default=BorderType::kConstant).It can be any of
  ///     [BorderType::kConstant, BorderType::kEdge, BorderType::kReflect, BorderType::kSymmetric].
  explicit RandomCropWithBBox(std::vector<int32_t> size, std::vector<int32_t> padding = {0, 0, 0, 0},
                              bool pad_if_needed = false, std::vector<uint8_t> fill_value = {0, 0, 0},
                              BorderType padding_mode = BorderType::kConstant);

  /// \brief Destructor.
  ~RandomCropWithBBox() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<int32_t> size_;
  std::vector<int32_t> padding_;
  bool pad_if_needed_;
  std::vector<uint8_t> fill_value_;
  BorderType padding_mode_;
};

/// \brief RandomHorizontalFlip TensorTransform.
/// \notes Tensor operation to perform random horizontal flip.
class RandomHorizontalFlip : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of flip.
  explicit RandomHorizontalFlip(float prob = 0.5);

  /// \brief Destructor.
  ~RandomHorizontalFlip() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  float probability_;
};

/// \brief RandomHorizontalFlipWithBBox TensorTransform.
/// \notes Flip the input image horizontally, randomly with a given probability and adjust bounding boxes accordingly.
class RandomHorizontalFlipWithBBox : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of flip.
  explicit RandomHorizontalFlipWithBBox(float prob = 0.5);

  /// \brief Destructor.
  ~RandomHorizontalFlipWithBBox() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  float probability_;
};

/// \brief RandomPosterize TensorTransform.
/// \notes Tensor operation to perform random posterize.
class RandomPosterize : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] bit_range - uint8_t vector representing the minimum and maximum bit in range. (Default={4, 8})
  explicit RandomPosterize(const std::vector<uint8_t> &bit_range = {4, 8});

  /// \brief Destructor.
  ~RandomPosterize() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<uint8_t> bit_range_;
};

/// \brief RandomResize TensorTransform.
/// \notes Resize the input image using a randomly selected interpolation mode.
//      the same image aspect ratio. If size has 2 values, it should be (height, width).
class RandomResize : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the resized image.
  ///     If size is a single value, the smaller edge of the image will be resized to this value with
  explicit RandomResize(std::vector<int32_t> size);

  /// \brief Destructor.
  ~RandomResize() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<int32_t> size_;
};

/// \brief RandomResizeWithBBox TensorTransform.
/// \notes Resize the input image using a randomly selected interpolation mode and adjust
///     bounding boxes accordingly.
class RandomResizeWithBBox : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the resized image.
  ///     If size is a single value, the smaller edge of the image will be resized to this value with
  //      the same image aspect ratio. If size has 2 values, it should be (height, width).
  explicit RandomResizeWithBBox(std::vector<int32_t> size);

  /// \brief Destructor.
  ~RandomResizeWithBBox() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<int32_t> size_;
};

/// \brief RandomResizedCrop TensorTransform.
/// \notes Crop the input image to a random size and aspect ratio.
class RandomResizedCrop : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If size is a single value, a square crop of size (size, size) is returned.
  ///     If size has 2 values, it should be (height, width).
  /// \param[in] scale Range [min, max) of respective size of the original
  ///     size to be cropped (default=(0.08, 1.0))
  /// \param[in] ratio Range [min, max) of aspect ratio to be cropped
  ///     (default=(3. / 4., 4. / 3.)).
  /// \param[in] interpolation Image interpolation mode (default=InterpolationMode::kLinear)
  /// \param[in] max_attempts The maximum number of attempts to propose a valid
  ///     crop_area (default=10). If exceeded, fall back to use center_crop instead.
  explicit RandomResizedCrop(std::vector<int32_t> size, std::vector<float> scale = {0.08, 1.0},
                             std::vector<float> ratio = {3. / 4., 4. / 3.},
                             InterpolationMode interpolation = InterpolationMode::kLinear, int32_t max_attempts = 10);

  /// \brief Destructor.
  ~RandomResizedCrop() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<int32_t> size_;
  std::vector<float> scale_;
  std::vector<float> ratio_;
  InterpolationMode interpolation_;
  int32_t max_attempts_;
};

/// \brief RandomResizedCropWithBBox TensorTransform.
/// \notes Crop the input image to a random size and aspect ratio.
class RandomResizedCropWithBBox : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If size is a single value, a square crop of size (size, size) is returned.
  ///     If size has 2 values, it should be (height, width).
  /// \param[in] scale Range [min, max) of respective size of the original
  ///     size to be cropped (default=(0.08, 1.0))
  /// \param[in] ratio Range [min, max) of aspect ratio to be cropped
  ///     (default=(3. / 4., 4. / 3.)).
  /// \param[in] interpolation Image interpolation mode (default=InterpolationMode::kLinear)
  /// \param[in] max_attempts The maximum number of attempts to propose a valid
  ///     crop_area (default=10). If exceeded, fall back to use center_crop instead.
  RandomResizedCropWithBBox(std::vector<int32_t> size, std::vector<float> scale = {0.08, 1.0},
                            std::vector<float> ratio = {3. / 4., 4. / 3.},
                            InterpolationMode interpolation = InterpolationMode::kLinear, int32_t max_attempts = 10);

  /// \brief Destructor.
  ~RandomResizedCropWithBBox() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<int32_t> size_;
  std::vector<float> scale_;
  std::vector<float> ratio_;
  InterpolationMode interpolation_;
  int32_t max_attempts_;
};

/// \brief RandomRotation TensorOp
/// \notes Rotates the image according to parameters
class RandomRotation : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] degrees A float vector of size, representing the starting and ending degree
  /// \param[in] resample An enum for the mode of interpolation
  /// \param[in] expand A boolean representing whether the image is expanded after rotation
  /// \param[in] center A float vector of size 2, representing the x and y center of rotation.
  /// \param[in] fill_value A vector representing the value to fill the area outside the transform
  ///    in the output image. If 1 value is provided, it is used for all RGB channels.
  ///    If 3 values are provided, it is used to fill R, G, B channels respectively.
  RandomRotation(std::vector<float> degrees, InterpolationMode resample = InterpolationMode::kNearestNeighbour,
                 bool expand = false, std::vector<float> center = {-1, -1},
                 std::vector<uint8_t> fill_value = {0, 0, 0});

  /// \brief Destructor.
  ~RandomRotation() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<float> degrees_;
  InterpolationMode interpolation_mode_;
  std::vector<float> center_;
  bool expand_;
  std::vector<uint8_t> fill_value_;
};

/// \brief RandomSelectSubpolicy TensorTransform.
/// \notes Choose a random sub-policy from a list to be applied on the input image. A sub-policy is a list of tuples
///     (op, prob), where op is a TensorTransform operation and prob is the probability that this op will be applied.
///     Once a sub-policy is selected, each op within the subpolicy with be applied in sequence according to its
///     probability.
class RandomSelectSubpolicy : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] policy Vector of sub-policies to choose from.

  // FIXME - Provide TensorTransform support for policy
  explicit RandomSelectSubpolicy(std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> policy);
  // RandomSelectSubpolicy(std::vector<std::vector<std::pair<std::shared_ptr<TensorTransform>, double>>> policy);

  /// \brief Destructor.
  ~RandomSelectSubpolicy() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> policy_;
};

/// \brief RandomSharpness TensorTransform.
/// \notes Tensor operation to perform random sharpness.
class RandomSharpness : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] degrees A float vector of size 2, representing the starting and ending degree to uniformly
  ///     sample from, to select a degree to adjust sharpness.
  explicit RandomSharpness(std::vector<float> degrees = {0.1, 1.9});

  /// \brief Destructor.
  ~RandomSharpness() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<float> degrees_;
};

/// \brief RandomSolarize TensorTransform.
/// \notes Invert pixels randomly within specified range. If min=max, it is a single fixed magnitude operation
///     to inverts all pixel above that threshold
class RandomSolarize : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] threshold A vector with two elements specifying the pixel range to invert.
  explicit RandomSolarize(std::vector<uint8_t> threshold = {0, 255});

  /// \brief Destructor.
  ~RandomSolarize() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<uint8_t> threshold_;
};

/// \brief RandomVerticalFlip TensorTransform.
/// \notes Tensor operation to perform random vertical flip.
class RandomVerticalFlip : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of flip.
  explicit RandomVerticalFlip(float prob = 0.5);

  /// \brief Destructor.
  ~RandomVerticalFlip() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  float probability_;
};

/// \brief RandomVerticalFlipWithBBox TensorTransform.
/// \notes Flip the input image vertically, randomly with a given probability and adjust bounding boxes accordingly.
class RandomVerticalFlipWithBBox : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of flip.
  explicit RandomVerticalFlipWithBBox(float prob = 0.5);

  /// \brief Destructor.
  ~RandomVerticalFlipWithBBox() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  float probability_;
};

/// \brief RescaleOperation TensorTransform.
/// \notes Tensor operation to rescale the input image.
class Rescale : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] rescale Rescale factor.
  /// \param[in] shift Shift factor.
  Rescale(float rescale, float shift);

  /// \brief Destructor.
  ~Rescale() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  float rescale_;
  float shift_;
};

/// \brief ResizeWithBBox TensorTransform.
/// \notes Resize the input image to the given size and adjust bounding boxes accordingly.
class ResizeWithBBox : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size The output size of the resized image.
  ///     If size is an integer, smaller edge of the image will be resized to this value with the same image aspect
  ///     ratio. If size is a sequence of length 2, it should be (height, width).
  /// \param[in] interpolation An enum for the mode of interpolation (default=InterpolationMode::kLinear).
  explicit ResizeWithBBox(std::vector<int32_t> size, InterpolationMode interpolation = InterpolationMode::kLinear);

  /// \brief Destructor.
  ~ResizeWithBBox() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<int32_t> size_;
  InterpolationMode interpolation_;
};

/// \brief RgbaToBgr TensorTransform.
/// \notes Changes the input 4 channel RGBA tensor to 3 channel BGR.
class RGBA2BGR : public TensorTransform {
 public:
  /// \brief Constructor.
  RGBA2BGR();

  /// \brief Destructor.
  ~RGBA2BGR() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief RgbaToRgb TensorTransform.
/// \notes Changes the input 4 channel RGBA tensor to 3 channel RGB.
class RGBA2RGB : public TensorTransform {
 public:
  /// \brief Constructor.
  RGBA2RGB();

  /// \brief Destructor.
  ~RGBA2RGB() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief SoftDvppDecodeRandomCropResizeJpeg TensorTransform.
/// \notes Tensor operation to decode, random crop and resize JPEG image using the simulation algorithm of
///     Ascend series chip DVPP module. The usage scenario is consistent with SoftDvppDecodeResizeJpeg.
///     The input image size should be in range [32*32, 8192*8192].
///     The zoom-out and zoom-in multiples of the image length and width should in the range [1/32, 16].
///     Only images with an even resolution can be output. The output of odd resolution is not supported.
class SoftDvppDecodeRandomCropResizeJpeg : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the resized image.
  ///     If size is a single value, smaller edge of the image will be resized to this value with
  ///     the same image aspect ratio. If size has 2 values, it should be (height, width).
  SoftDvppDecodeRandomCropResizeJpeg(std::vector<int32_t> size, std::vector<float> scale = {0.08, 1.0},
                                     std::vector<float> ratio = {3. / 4., 4. / 3.}, int32_t max_attempts = 10);

  /// \brief Destructor.
  ~SoftDvppDecodeRandomCropResizeJpeg() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<int32_t> size_;
  std::vector<float> scale_;
  std::vector<float> ratio_;
  int32_t max_attempts_;
};

/// \brief SoftDvppDecodeResizeJpeg TensorTransform.
/// \notes Tensor operation to decode and resize JPEG image using the simulation algorithm of Ascend series
///     chip DVPP module. It is recommended to use this algorithm in the following scenarios:
///     When training, the DVPP of the Ascend chip is not used,
///     and the DVPP of the Ascend chip is used during inference,
///     and the accuracy of inference is lower than the accuracy of training;
///     and the input image size should be in range [32*32, 8192*8192].
///     The zoom-out and zoom-in multiples of the image length and width should in the range [1/32, 16].
///     Only images with an even resolution can be output. The output of odd resolution is not supported.
class SoftDvppDecodeResizeJpeg : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the resized image.
  ///     If size is a single value, smaller edge of the image will be resized to this value with
  ///     the same image aspect ratio. If size has 2 values, it should be (height, width).
  explicit SoftDvppDecodeResizeJpeg(std::vector<int32_t> size);

  /// \brief Destructor.
  ~SoftDvppDecodeResizeJpeg() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<int32_t> size_;
};

/// \brief SwapRedBlue TensorOp
/// \notes Swaps the red and blue channels in image
class SwapRedBlue : public TensorTransform {
 public:
  /// \brief Constructor.
  SwapRedBlue();

  /// \brief Destructor.
  ~SwapRedBlue() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief UniformAugment TensorTransform.
/// \notes Tensor operation to perform randomly selected augmentation.
class UniformAugment : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] transforms A vector of TensorTransform transforms.
  /// \param[in] num_ops An integer representing the number of OPs to be selected and applied.
  explicit UniformAugment(std::vector<std::shared_ptr<TensorTransform>> transforms, int32_t num_ops = 2);

  /// \brief Destructor.
  ~UniformAugment() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
  int32_t num_ops_;
};

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_VISION_H_
