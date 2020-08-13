/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_API_TRANSFORMS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_API_TRANSFORMS_H_

#include <vector>
#include <memory>
#include "minddata/dataset/core/constants.h"

namespace mindspore {
namespace dataset {

class TensorOp;

namespace api {
// Abstract class to represent a dataset in the data pipeline.
class TensorOperation : public std::enable_shared_from_this<TensorOperation> {
 public:
  /// \brief Constructor
  TensorOperation();

  /// \brief Destructor
  ~TensorOperation() = default;

  /// \brief Pure virtual function to convert a TensorOperation class into a runtime TensorOp object.
  /// \return shared pointer to the newly created TensorOp.
  virtual std::shared_ptr<TensorOp> Build() = 0;

  virtual bool ValidateParams() = 0;
};

// Transform operations for performing computer vision.
namespace vision {

// Transform Op classes (in alphabetical order)
class CenterCropOperation;
class CropOperation;
class CutOutOperation;
class DecodeOperation;
class HwcToChwOperation;
class MixUpBatchOperation;
class NormalizeOperation;
class OneHotOperation;
class PadOperation;
class RandomAffineOperation;
class RandomColorOperation;
class RandomColorAdjustOperation;
class RandomCropOperation;
class RandomHorizontalFlipOperation;
class RandomRotationOperation;
class RandomSharpnessOperation;
class RandomSolarizeOperation;
class RandomVerticalFlipOperation;
class ResizeOperation;
class RgbaToBgrOperation;
class RgbaToRgbOperation;
class SwapRedBlueOperation;
class UniformAugOperation;

/// \brief Function to create a CenterCrop TensorOperation.
/// \notes Crops the input image at the center to the given size.
/// \param[in] size - a vector representing the output size of the cropped image.
///      If size is a single value, a square crop of size (size, size) is returned.
///      If size has 2 values, it should be (height, width).
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<CenterCropOperation> CenterCrop(std::vector<int32_t> size);

/// \brief Function to create a Crop TensorOp
/// \notes Crop an image based on location and crop size
/// \param[in] coordinates Starting location of crop. Must be a vector of two values, in the form of {x_coor, y_coor}
/// \param[in] size Size of the cropped area. Must be a vector of two values, in the form of {height, width}
/// \return Shared pointer to the current TensorOp
std::shared_ptr<CropOperation> Crop(std::vector<int32_t> coordinates, std::vector<int32_t> size);

/// \brief Function to create a CutOut TensorOp
/// \notes Randomly cut (mask) out a given number of square patches from the input image
/// \param[in] length Integer representing the side length of each square patch
/// \param[in] num_patches Integer representing the number of patches to be cut out of an image
/// \return Shared pointer to the current TensorOp
std::shared_ptr<CutOutOperation> CutOut(int32_t length, int32_t num_patches = 1);

/// \brief Function to create a Decode TensorOperation.
/// \notes Decode the input image in RGB mode.
/// \param[in] rgb - a boolean of whether to decode in RGB mode or not.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<DecodeOperation> Decode(bool rgb = true);

/// \brief Function to create a HwcToChw TensorOperation.
/// \notes Transpose the input image; shape (H, W, C) to shape (C, H, W).
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<HwcToChwOperation> HWC2CHW();

/// \brief Function to create a MixUpBatch TensorOperation.
/// \notes Apply MixUp transformation on an input batch of images and labels. The labels must be in one-hot format and
///     Batch must be called before calling this function.
/// \param[in] alpha hyperparameter of beta distribution (default = 1.0)
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<MixUpBatchOperation> MixUpBatch(float alpha = 1);

/// \brief Function to create a Normalize TensorOperation.
/// \notes Normalize the input image with respect to mean and standard deviation.
/// \param[in] mean A vector of mean values for each channel, w.r.t channel order.
/// \param[in] std A vector of standard deviations for each channel, w.r.t. channel order.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<NormalizeOperation> Normalize(std::vector<float> mean, std::vector<float> std);

/// \brief Function to create a OneHot TensorOperation.
/// \notes Convert the labels into OneHot format.
/// \param[in] num_classes number of classes.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<OneHotOperation> OneHot(int32_t num_classes);

/// \brief Function to create a Pad TensorOp
/// \notes Pads the image according to padding parameters
/// \param[in] padding A vector representing the number of pixels to pad the image
///    If vector has one value, it pads all sides of the image with that value
///    If vector has two values, it pads left and right with the first and
///    top and bottom with the second value
///    If vector has four values, it pads left, top, right, and bottom with
///    those values respectively
/// \param[in] fill_value A vector representing the pixel intensity of the borders if the padding_mode is
///    BorderType.kConstant. If 3 values are provided,
///    it is used to fill R, G, B channels respectively
/// \param[in] padding_mode The method of padding (default=BorderType.kConstant)
///    Can be any of
///    [BorderType.kConstant, BorderType.kEdge, BorderType.kReflect, BorderType.kSymmetric]
///    - BorderType.kConstant, means it fills the border with constant values
///    - BorderType.kEdge, means it pads with the last value on the edge
///    - BorderType.kReflect, means it reflects the values on the edge omitting the last value of edge
///    - BorderType.kSymmetric, means it reflects the values on the edge repeating the last value of edge
/// \return Shared pointer to the current TensorOp
std::shared_ptr<PadOperation> Pad(std::vector<int32_t> padding, std::vector<uint8_t> fill_value = {0},
                                  BorderType padding_mode = BorderType::kConstant);

/// \brief Function to create a RandomAffine TensorOperation.
/// \notes Applies a Random Affine transformation on input image in RGB or Greyscale mode.
/// \param[in] degrees A float vector size 2, representing the starting and ending degree
/// \param[in] translate_range A float vector size 2, representing percentages of translation on x and y axes.
/// \param[in] scale_range A float vector size 2, representing the starting and ending scales in the range.
/// \param[in] shear_ranges A float vector size 4, representing the starting and ending shear degrees vertically and
///    horizontally.
/// \param[in] interpolation An enum for the mode of interpolation
/// \param[in] fill_value A uint8_t vector size 3, representing the pixel intensity of the borders, it is used to
///    fill R, G, B channels respectively.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomAffineOperation> RandomAffine(
  const std::vector<float_t> &degrees, const std::vector<float_t> &translate_range = {0.0, 0.0},
  const std::vector<float_t> &scale_range = {1.0, 1.0}, const std::vector<float_t> &shear_ranges = {0.0, 0.0, 0.0, 0.0},
  InterpolationMode interpolation = InterpolationMode::kNearestNeighbour,
  const std::vector<uint8_t> &fill_value = {0, 0, 0});

/// \brief Blends an image with its grayscale version with random weights
///        t and 1 - t generated from a given range. If the range is trivial
///        then the weights are determinate and t equals the bound of the interval
/// \param[in] t_lb lower bound on the range of random weights
/// \param[in] t_lb upper bound on the range of random weights
/// \return Shared pointer to the current TensorOp
std::shared_ptr<RandomColorOperation> RandomColor(float t_lb, float t_ub);

/// \brief Randomly adjust the brightness, contrast, saturation, and hue of the input image
/// \param[in] brightness Brightness adjustment factor. Must be a vector of one or two values
///     if it's a vector of two values it needs to be in the form of [min, max]. Default value is {1, 1}
/// \param[in] contrast Contrast adjustment factor. Must be a vector of one or two values
///     if it's a vector of two values it needs to be in the form of [min, max]. Default value is {1, 1}
/// \param[in] saturation Saturation adjustment factor. Must be a vector of one or two values
///     if it's a vector of two values it needs to be in the form of [min, max]. Default value is {1, 1}
/// \param[in] hue Brightness adjustment factor. Must be a vector of one or two values
///     if it's a vector of two values it must be in the form of [min, max] where -0.5 <= min <= max <= 0.5
///     Default value is {0, 0}
/// \return Shared pointer to the current TensorOp
std::shared_ptr<RandomColorAdjustOperation> RandomColorAdjust(std::vector<float> brightness = {1.0, 1.0},
                                                              std::vector<float> contrast = {1.0, 1.0},
                                                              std::vector<float> saturation = {1.0, 1.0},
                                                              std::vector<float> hue = {0.0, 0.0});

/// \brief Function to create a RandomCrop TensorOperation.
/// \notes Crop the input image at a random location.
/// \param[in] size - a vector representing the output size of the cropped image.
///               If size is a single value, a square crop of size (size, size) is returned.
///               If size has 2 values, it should be (height, width).
/// \param[in] padding - a vector with the value of pixels to pad the image. If 4 values are provided,
///                  it pads the left, top, right and bottom respectively.
/// \param[in] pad_if_needed - a boolean whether to pad the image if either side is smaller than
///                        the given output size.
/// \param[in] fill_value - a vector representing the pixel intensity of the borders, it is used to
///                     fill R, G, B channels respectively.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomCropOperation> RandomCrop(std::vector<int32_t> size, std::vector<int32_t> padding = {0, 0, 0, 0},
                                                bool pad_if_needed = false, std::vector<uint8_t> fill_value = {0, 0, 0},
                                                BorderType padding_mode = BorderType::kConstant);

/// \brief Function to create a RandomHorizontalFlip TensorOperation.
/// \notes Tensor operation to perform random horizontal flip.
/// \param[in] prob - float representing the probability of flip.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomHorizontalFlipOperation> RandomHorizontalFlip(float prob = 0.5);

/// \brief Function to create a RandomRotation TensorOp
/// \notes Rotates the image according to parameters
/// \param[in] degrees A float vector size 2, representing the starting and ending degree
/// \param[in] resample An enum for the mode of interpolation
/// \param[in] expand A boolean representing whether the image is expanded after rotation
/// \param[in] center A float vector size 2, representing the x and y center of rotation.
/// \param[in] fill_value A uint8_t vector size 3, representing the rgb value of the fill color
/// \return Shared pointer to the current TensorOp
std::shared_ptr<RandomRotationOperation> RandomRotation(
  std::vector<float> degrees, InterpolationMode resample = InterpolationMode::kNearestNeighbour, bool expand = false,
  std::vector<float> center = {-1, -1}, std::vector<uint8_t> fill_value = {0, 0, 0});

/// \brief Function to create a RandomSharpness TensorOperation.
/// \notes Tensor operation to perform random sharpness.
/// \param[in] start_degree - float representing the start of the range to uniformly sample the factor from it.
/// \param[in] end_degree - float representing the end of the range.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomSharpnessOperation> RandomSharpness(std::vector<float> degrees = {0.1, 1.9});

/// \brief Function to create a RandomSolarize TensorOperation.
/// \notes Invert pixels within specified range. If min=max, then it inverts all pixel above that threshold
/// \param[in] threshold_min - lower limit
/// \param[in] threshold_max - upper limit
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomSolarizeOperation> RandomSolarize(uint8_t threshold_min = 0, uint8_t threshold_max = 255);

/// \brief Function to create a RandomVerticalFlip TensorOperation.
/// \notes Tensor operation to perform random vertical flip.
/// \param[in] prob - float representing the probability of flip.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomVerticalFlipOperation> RandomVerticalFlip(float prob = 0.5);

/// \brief Function to create a RgbaToBgr TensorOperation.
/// \notes Changes the input 4 channel RGBA tensor to 3 channel BGR.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RgbaToBgrOperation> RGBA2BGR();

/// \brief Function to create a RgbaToRgb TensorOperation.
/// \notes Changes the input 4 channel RGBA tensor to 3 channel RGB.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RgbaToRgbOperation> RGBA2RGB();

/// \brief Function to create a Resize TensorOperation.
/// \notes Resize the input image to the given size.
/// \param[in] size - a vector representing the output size of the resized image.
///               If size is a single value, the image will be resized to this value with
///               the same image aspect ratio. If size has 2 values, it should be (height, width).
/// \param[in] interpolation An enum for the mode of interpolation
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<ResizeOperation> Resize(std::vector<int32_t> size,
                                        InterpolationMode interpolation = InterpolationMode::kLinear);

/// \brief Function to create a SwapRedBlue TensorOp
/// \notes Swaps the red and blue channels in image
/// \return Shared pointer to the current TensorOp
std::shared_ptr<SwapRedBlueOperation> SwapRedBlue();

/// \brief Function to create a UniformAugment TensorOperation.
/// \notes Tensor operation to perform randomly selected augmentation.
/// \param[in] transforms - a vector of TensorOperation transforms.
/// \param[in] num_ops - integer representing the number of OPs to be selected and applied.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<UniformAugOperation> UniformAugment(std::vector<std::shared_ptr<TensorOperation>> transforms,
                                                    int32_t num_ops = 2);

/* ####################################### Derived TensorOperation classes ################################# */

class CenterCropOperation : public TensorOperation {
 public:
  explicit CenterCropOperation(std::vector<int32_t> size);

  ~CenterCropOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  std::vector<int32_t> size_;
};

class CropOperation : public TensorOperation {
 public:
  CropOperation(std::vector<int32_t> coordinates, std::vector<int32_t> size);

  ~CropOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  std::vector<int32_t> coordinates_;
  std::vector<int32_t> size_;
};

class CutOutOperation : public TensorOperation {
 public:
  explicit CutOutOperation(int32_t length, int32_t num_patches = 1);

  ~CutOutOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  int32_t length_;
  int32_t num_patches_;
};

class DecodeOperation : public TensorOperation {
 public:
  explicit DecodeOperation(bool rgb = true);

  ~DecodeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  bool rgb_;
};

class HwcToChwOperation : public TensorOperation {
 public:
  ~HwcToChwOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;
};

class MixUpBatchOperation : public TensorOperation {
 public:
  explicit MixUpBatchOperation(float alpha = 1);

  ~MixUpBatchOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  float alpha_;
};

class NormalizeOperation : public TensorOperation {
 public:
  NormalizeOperation(std::vector<float> mean, std::vector<float> std);

  ~NormalizeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
};

class OneHotOperation : public TensorOperation {
 public:
  explicit OneHotOperation(int32_t num_classes_);

  ~OneHotOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  float num_classes_;
};

class PadOperation : public TensorOperation {
 public:
  PadOperation(std::vector<int32_t> padding, std::vector<uint8_t> fill_value = {0},
               BorderType padding_mode = BorderType::kConstant);

  ~PadOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  std::vector<int32_t> padding_;
  std::vector<uint8_t> fill_value_;
  BorderType padding_mode_;
};

class RandomAffineOperation : public TensorOperation {
 public:
  RandomAffineOperation(const std::vector<float_t> &degrees, const std::vector<float_t> &translate_range = {0.0, 0.0},
                        const std::vector<float_t> &scale_range = {1.0, 1.0},
                        const std::vector<float_t> &shear_ranges = {0.0, 0.0, 0.0, 0.0},
                        InterpolationMode interpolation = InterpolationMode::kNearestNeighbour,
                        const std::vector<uint8_t> &fill_value = {0, 0, 0});

  ~RandomAffineOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  std::vector<float_t> degrees_;          // min_degree, max_degree
  std::vector<float_t> translate_range_;  // maximum x translation percentage, maximum y translation percentage
  std::vector<float_t> scale_range_;      // min_scale, max_scale
  std::vector<float_t> shear_ranges_;     // min_x_shear, max_x_shear, min_y_shear, max_y_shear
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};

class RandomColorOperation : public TensorOperation {
 public:
  RandomColorOperation(float t_lb, float t_ub);

  ~RandomColorOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  float t_lb_;
  float t_ub_;
};

class RandomColorAdjustOperation : public TensorOperation {
 public:
  RandomColorAdjustOperation(std::vector<float> brightness = {1.0, 1.0}, std::vector<float> contrast = {1.0, 1.0},
                             std::vector<float> saturation = {1.0, 1.0}, std::vector<float> hue = {0.0, 0.0});

  ~RandomColorAdjustOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  std::vector<float> brightness_;
  std::vector<float> contrast_;
  std::vector<float> saturation_;
  std::vector<float> hue_;
};

class RandomCropOperation : public TensorOperation {
 public:
  RandomCropOperation(std::vector<int32_t> size, std::vector<int32_t> padding = {0, 0, 0, 0},
                      bool pad_if_needed = false, std::vector<uint8_t> fill_value = {0, 0, 0},
                      BorderType padding_mode = BorderType::kConstant);

  ~RandomCropOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  std::vector<int32_t> size_;
  std::vector<int32_t> padding_;
  bool pad_if_needed_;
  std::vector<uint8_t> fill_value_;
  BorderType padding_mode_;
};

class RandomHorizontalFlipOperation : public TensorOperation {
 public:
  explicit RandomHorizontalFlipOperation(float probability = 0.5);

  ~RandomHorizontalFlipOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  float probability_;
};

class RandomRotationOperation : public TensorOperation {
 public:
  RandomRotationOperation(std::vector<float> degrees, InterpolationMode interpolation_mode, bool expand,
                          std::vector<float> center, std::vector<uint8_t> fill_value);

  ~RandomRotationOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  std::vector<float> degrees_;
  InterpolationMode interpolation_mode_;
  std::vector<float> center_;
  bool expand_;
  std::vector<uint8_t> fill_value_;
};

class RandomSharpnessOperation : public TensorOperation {
 public:
  explicit RandomSharpnessOperation(std::vector<float> degrees = {0.1, 1.9});

  ~RandomSharpnessOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  std::vector<float> degrees_;
};

class RandomVerticalFlipOperation : public TensorOperation {
 public:
  explicit RandomVerticalFlipOperation(float probability = 0.5);

  ~RandomVerticalFlipOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  float probability_;
};

class ResizeOperation : public TensorOperation {
 public:
  explicit ResizeOperation(std::vector<int32_t> size,
                           InterpolationMode interpolation_mode = InterpolationMode::kLinear);

  ~ResizeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  std::vector<int32_t> size_;
  InterpolationMode interpolation_;
};

class RgbaToBgrOperation : public TensorOperation {
 public:
  RgbaToBgrOperation();

  ~RgbaToBgrOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;
};

class RgbaToRgbOperation : public TensorOperation {
 public:
  RgbaToRgbOperation();

  ~RgbaToRgbOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;
};

class UniformAugOperation : public TensorOperation {
 public:
  explicit UniformAugOperation(std::vector<std::shared_ptr<TensorOperation>> transforms, int32_t num_ops = 2);

  ~UniformAugOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
  int32_t num_ops_;
};

class SwapRedBlueOperation : public TensorOperation {
 public:
  SwapRedBlueOperation();

  ~SwapRedBlueOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;
};

class RandomSolarizeOperation : public TensorOperation {
 public:
  explicit RandomSolarizeOperation(uint8_t threshold_min, uint8_t threshold_max);

  ~RandomSolarizeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  uint8_t threshold_min_;
  uint8_t threshold_max_;
};
}  // namespace vision
}  // namespace api
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_API_TRANSFORMS_H_
