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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_VISION_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_VISION_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/vision_lite.h"
#include "minddata/dataset/include/status.h"
namespace mindspore {
namespace dataset {

// Transform operations for performing computer vision.
namespace vision {

// Char arrays storing name of corresponding classes (in alphabetical order)
constexpr char kAutoContrastOperation[] = "AutoContrast";
constexpr char kBoundingBoxAugmentOperation[] = "BoundingBoxAugment";
constexpr char kCutMixBatchOperation[] = "CutMixBatch";
constexpr char kCutOutOperation[] = "CutOut";
constexpr char kDvppDecodeResizeCropOperation[] = "DvppDecodeResizeCrop";
constexpr char kEqualizeOperation[] = "Equalize";
constexpr char kHwcToChwOperation[] = "HwcToChw";
constexpr char kInvertOperation[] = "Invert";
constexpr char kMixUpBatchOperation[] = "MixUpBatch";
constexpr char kNormalizePadOperation[] = "NormalizePad";
constexpr char kPadOperation[] = "Pad";
constexpr char kRandomAffineOperation[] = "RandomAffine";
constexpr char kRandomColorAdjustOperation[] = "RandomColorAdjust";
constexpr char kRandomColorOperation[] = "RandomColor";
constexpr char kRandomCropDecodeResizeOperation[] = "RandomCropDecodeResize";
constexpr char kRandomCropOperation[] = "RandomCrop";
constexpr char kRandomCropWithBBoxOperation[] = "RandomCropWithBBox";
constexpr char kRandomHorizontalFlipWithBBoxOperation[] = "RandomHorizontalFlipWithBBox";
constexpr char kRandomHorizontalFlipOperation[] = "RandomHorizontalFlip";
constexpr char kRandomPosterizeOperation[] = "RandomPosterize";
constexpr char kRandomResizedCropOperation[] = "RandomResizedCrop";
constexpr char kRandomResizedCropWithBBoxOperation[] = "RandomResizedCropWithBBox";
constexpr char kRandomResizeOperation[] = "RandomResize";
constexpr char kRandomResizeWithBBoxOperation[] = "RandomResizeWithBBox";
constexpr char kRandomRotationOperation[] = "RandomRotation";
constexpr char kRandomSolarizeOperation[] = "RandomSolarize";
constexpr char kRandomSharpnessOperation[] = "RandomSharpness";
constexpr char kRandomVerticalFlipOperation[] = "RandomVerticalFlip";
constexpr char kRandomVerticalFlipWithBBoxOperation[] = "RandomVerticalFlipWithBBox";
constexpr char kRescaleOperation[] = "Rescale";
constexpr char kResizeWithBBoxOperation[] = "ResizeWithBBox";
constexpr char kRgbaToBgrOperation[] = "RgbaToBgr";
constexpr char kRgbaToRgbOperation[] = "RgbaToRgb";
constexpr char kSoftDvppDecodeRandomCropResizeJpegOperation[] = "SoftDvppDecodeRandomCropResizeJpeg";
constexpr char kSoftDvppDecodeResizeJpegOperation[] = "SoftDvppDecodeResizeJpeg";
constexpr char kSwapRedBlueOperation[] = "SwapRedBlue";
constexpr char kUniformAugOperation[] = "UniformAug";

// Transform Op classes (in alphabetical order)
class AutoContrastOperation;
class BoundingBoxAugmentOperation;
class CutMixBatchOperation;
class CutOutOperation;
class DvppDecodeResizeCropOperation;
class EqualizeOperation;
class HwcToChwOperation;
class InvertOperation;
class MixUpBatchOperation;
class NormalizePadOperation;
class PadOperation;
class RandomAffineOperation;
class RandomColorOperation;
class RandomColorAdjustOperation;
class RandomCropOperation;
class RandomCropDecodeResizeOperation;
class RandomCropWithBBoxOperation;
class RandomHorizontalFlipOperation;
class RandomHorizontalFlipWithBBoxOperation;
class RandomPosterizeOperation;
class RandomResizeOperation;
class RandomResizeWithBBoxOperation;
class RandomResizedCropOperation;
class RandomResizedCropWithBBoxOperation;
class RandomRotationOperation;
class RandomSelectSubpolicyOperation;
class RandomSharpnessOperation;
class RandomSolarizeOperation;
class RandomVerticalFlipOperation;
class RandomVerticalFlipWithBBoxOperation;
class RescaleOperation;
class ResizeWithBBoxOperation;
class RgbaToBgrOperation;
class RgbaToRgbOperation;
class SoftDvppDecodeRandomCropResizeJpegOperation;
class SoftDvppDecodeResizeJpegOperation;
class SwapRedBlueOperation;
class UniformAugOperation;

/// \brief Function to create a AutoContrast TensorOperation.
/// \notes Apply automatic contrast on input image.
/// \param[in] cutoff Percent of pixels to cut off from the histogram, the valid range of cutoff value is 0 to 100.
/// \param[in] ignore Pixel values to ignore.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<AutoContrastOperation> AutoContrast(float cutoff = 0.0, std::vector<uint32_t> ignore = {});

/// \brief Function to create a BoundingBoxAugment TensorOperation.
/// \notes  Apply a given image transform on a random selection of bounding box regions of a given image.
/// \param[in] transform A TensorOperation transform.
/// \param[in] ratio Ratio of bounding boxes to apply augmentation on. Range: [0, 1] (default=0.3).
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<BoundingBoxAugmentOperation> BoundingBoxAugment(std::shared_ptr<TensorOperation> transform,
                                                                float ratio = 0.3);

/// \brief Function to apply CutMix on a batch of images
/// \notes Masks a random section of each image with the corresponding part of another randomly
///     selected image in that batch
/// \param[in] image_batch_format The format of the batch
/// \param[in] alpha The hyperparameter of beta distribution (default = 1.0)
/// \param[in] prob The probability by which CutMix is applied to each image (default = 1.0)
/// \return Shared pointer to the current TensorOp
std::shared_ptr<CutMixBatchOperation> CutMixBatch(ImageBatchFormat image_batch_format, float alpha = 1.0,
                                                  float prob = 1.0);

/// \brief Function to create a CutOut TensorOp
/// \notes Randomly cut (mask) out a given number of square patches from the input image
/// \param[in] length Integer representing the side length of each square patch
/// \param[in] num_patches Integer representing the number of patches to be cut out of an image
/// \return Shared pointer to the current TensorOp
std::shared_ptr<CutOutOperation> CutOut(int32_t length, int32_t num_patches = 1);

/// \brief Function to create a DvppDecodeResizeCropJpeg TensorOperation.
/// \notes Tensor operation to decode and resize JPEG image using the simulation algorithm of Ascend series
///     chip DVPP module. It is recommended to use this algorithm in the following scenarios:
///     When training, the DVPP of the Ascend chip is not used,
///     and the DVPP of the Ascend chip is used during inference,
///     and the accuracy of inference is lower than the accuracy of training;
///     and the input image size should be in range [16*16, 4096*4096].
///     Only images with an even resolution can be output. The output of odd resolution is not supported.
/// \param[in] crop vector representing the output size of the final crop image.
/// \param[in] size A vector representing the output size of the intermediate resized image.
///     If size is a single value, smaller edge of the image will be resized to this value with
///     the same image aspect ratio. If size has 2 values, it should be (height, width).
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<DvppDecodeResizeCropOperation> DvppDecodeResizeCropJpeg(std::vector<uint32_t> crop = {224, 224},
                                                                        std::vector<uint32_t> resize = {256, 256});

/// \brief Function to create a Equalize TensorOperation.
/// \notes Apply histogram equalization on input image.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<EqualizeOperation> Equalize();

/// \brief Function to create a HwcToChw TensorOperation.
/// \notes Transpose the input image; shape (H, W, C) to shape (C, H, W).
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<HwcToChwOperation> HWC2CHW();

/// \brief Function to create a Invert TensorOperation.
/// \notes Apply invert on input image in RGB mode.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<InvertOperation> Invert();

/// \brief Function to create a MixUpBatch TensorOperation.
/// \notes Apply MixUp transformation on an input batch of images and labels. The labels must be in
///     one-hot format and Batch must be called before calling this function.
/// \param[in] alpha hyperparameter of beta distribution (default = 1.0)
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<MixUpBatchOperation> MixUpBatch(float alpha = 1);

/// \brief Function to create a NormalizePad TensorOperation.
/// \notes Normalize the input image with respect to mean and standard deviation and pad an extra
///     channel with value zero.
/// \param[in] mean A vector of mean values for each channel, w.r.t channel order.
///     The mean values must be in range [0.0, 255.0].
/// \param[in] std A vector of standard deviations for each channel, w.r.t. channel order.
///     The standard deviation values must be in range (0.0, 255.0]
/// \param[in] dtype The output datatype of Tensor.
///     The standard deviation values must be "float32" or "float16"（default = "float32"）
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<NormalizePadOperation> NormalizePad(const std::vector<float> &mean, const std::vector<float> &std,
                                                    const std::string &dtype = "float32");

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
/// \param[in] fill_value A uint8_t vector of size 3, representing the pixel intensity of the borders, it is used to
///    fill R, G, B channels respectively.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomAffineOperation> RandomAffine(
  const std::vector<float_t> &degrees, const std::vector<float_t> &translate_range = {0.0, 0.0, 0.0, 0.0},
  const std::vector<float_t> &scale_range = {1.0, 1.0}, const std::vector<float_t> &shear_ranges = {0.0, 0.0, 0.0, 0.0},
  InterpolationMode interpolation = InterpolationMode::kNearestNeighbour,
  const std::vector<uint8_t> &fill_value = {0, 0, 0});

/// \brief Blends an image with its grayscale version with random weights
///        t and 1 - t generated from a given range. If the range is trivial
///        then the weights are determinate and t equals the bound of the interval
/// \param[in] t_lb Lower bound on the range of random weights
/// \param[in] t_lb Upper bound on the range of random weights
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
/// \param[in] size A vector representing the output size of the cropped image.
///     If size is a single value, a square crop of size (size, size) is returned.
///     If size has 2 values, it should be (height, width).
/// \param[in] padding A vector with the value of pixels to pad the image. If 4 values are provided,
///     it pads the left, top, right and bottom respectively.
/// \param[in] pad_if_needed A boolean whether to pad the image if either side is smaller than
///     the given output size.
/// \param[in] fill_value A vector representing the pixel intensity of the borders, it is used to
///     fill R, G, B channels respectively.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomCropOperation> RandomCrop(std::vector<int32_t> size, std::vector<int32_t> padding = {0, 0, 0, 0},
                                                bool pad_if_needed = false, std::vector<uint8_t> fill_value = {0, 0, 0},
                                                BorderType padding_mode = BorderType::kConstant);

/// \brief Function to create a RandomCropDecodeResize TensorOperation.
/// \notes Equivalent to RandomResizedCrop, but crops before decodes.
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
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomCropDecodeResizeOperation> RandomCropDecodeResize(
  std::vector<int32_t> size, std::vector<float> scale = {0.08, 1.0}, std::vector<float> ratio = {3. / 4, 4. / 3},
  InterpolationMode interpolation = InterpolationMode::kLinear, int32_t max_attempts = 10);

/// \brief Function to create a RandomCropWithBBox TensorOperation.
/// \Crop the input image at a random location and adjust bounding boxes accordingly.
/// \param[in] size A vector representing the output size of the cropped image.
///     If size is a single value, a square crop of size (size, size) is returned.
///     If size has 2 values, it should be (height, width).
/// \param[in] padding A vector with the value of pixels to pad the image. If 4 values are provided,
///     it pads the left, top, right and bottom respectively.
/// \param[in] pad_if_needed A boolean whether to pad the image if either side is smaller than
///     the given output size.
/// \param[in] fill_value A vector representing the pixel intensity of the borders, it is used to
///     fill R, G, B channels respectively.
/// \param[in] padding_mode The method of padding (default=BorderType::kConstant).It can be any of
///     [BorderType::kConstant, BorderType::kEdge, BorderType::kReflect, BorderType::kSymmetric].
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomCropWithBBoxOperation> RandomCropWithBBox(std::vector<int32_t> size,
                                                                std::vector<int32_t> padding = {0, 0, 0, 0},
                                                                bool pad_if_needed = false,
                                                                std::vector<uint8_t> fill_value = {0, 0, 0},
                                                                BorderType padding_mode = BorderType::kConstant);

/// \brief Function to create a RandomHorizontalFlip TensorOperation.
/// \notes Tensor operation to perform random horizontal flip.
/// \param[in] prob A float representing the probability of flip.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomHorizontalFlipOperation> RandomHorizontalFlip(float prob = 0.5);

/// \brief Function to create a RandomHorizontalFlipWithBBox TensorOperation.
/// \notes Flip the input image horizontally, randomly with a given probability and adjust bounding boxes accordingly.
/// \param[in] prob A float representing the probability of flip.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomHorizontalFlipWithBBoxOperation> RandomHorizontalFlipWithBBox(float prob = 0.5);

/// \brief Function to create a RandomPosterize TensorOperation.
/// \notes Tensor operation to perform random posterize.
/// \param[in] bit_range - uint8_t vector representing the minimum and maximum bit in range. (Default={4, 8})
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomPosterizeOperation> RandomPosterize(const std::vector<uint8_t> &bit_range = {4, 8});

/// \brief Function to create a RandomResize TensorOperation.
/// \notes Resize the input image using a randomly selected interpolation mode.
/// \param[in] size A vector representing the output size of the resized image.
///     If size is a single value, the smaller edge of the image will be resized to this value with
//      the same image aspect ratio. If size has 2 values, it should be (height, width).
std::shared_ptr<RandomResizeOperation> RandomResize(std::vector<int32_t> size);

/// \brief Function to create a RandomResizeWithBBox TensorOperation.
/// \notes Resize the input image using a randomly selected interpolation mode and adjust
///     bounding boxes accordingly.
/// \param[in] size A vector representing the output size of the resized image.
///     If size is a single value, the smaller edge of the image will be resized to this value with
//      the same image aspect ratio. If size has 2 values, it should be (height, width).
std::shared_ptr<RandomResizeWithBBoxOperation> RandomResizeWithBBox(std::vector<int32_t> size);

/// \brief Function to create a RandomResizedCrop TensorOperation.
/// \notes Crop the input image to a random size and aspect ratio.
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
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomResizedCropOperation> RandomResizedCrop(
  std::vector<int32_t> size, std::vector<float> scale = {0.08, 1.0}, std::vector<float> ratio = {3. / 4., 4. / 3.},
  InterpolationMode interpolation = InterpolationMode::kLinear, int32_t max_attempts = 10);

/// \brief Function to create a RandomResizedCropWithBBox TensorOperation.
/// \notes Crop the input image to a random size and aspect ratio.
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
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomResizedCropWithBBoxOperation> RandomResizedCropWithBBox(
  std::vector<int32_t> size, std::vector<float> scale = {0.08, 1.0}, std::vector<float> ratio = {3. / 4., 4. / 3.},
  InterpolationMode interpolation = InterpolationMode::kLinear, int32_t max_attempts = 10);

/// \brief Function to create a RandomRotation TensorOp
/// \notes Rotates the image according to parameters
/// \param[in] degrees A float vector of size, representing the starting and ending degree
/// \param[in] resample An enum for the mode of interpolation
/// \param[in] expand A boolean representing whether the image is expanded after rotation
/// \param[in] center A float vector of size 2, representing the x and y center of rotation.
/// \param[in] fill_value A uint8_t vector of size 3, representing the rgb value of the fill color
/// \return Shared pointer to the current TensorOp
std::shared_ptr<RandomRotationOperation> RandomRotation(
  std::vector<float> degrees, InterpolationMode resample = InterpolationMode::kNearestNeighbour, bool expand = false,
  std::vector<float> center = {-1, -1}, std::vector<uint8_t> fill_value = {0, 0, 0});

/// \brief Function to create a RandomSelectSubpolicy TensorOperation.
/// \notes Choose a random sub-policy from a list to be applied on the input image. A sub-policy is a list of tuples
///     (op, prob), where op is a TensorOp operation and prob is the probability that this op will be applied. Once
///     a sub-policy is selected, each op within the subpolicy with be applied in sequence according to its probability.
/// \param[in] policy Vector of sub-policies to choose from.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomSelectSubpolicyOperation> RandomSelectSubpolicy(
  std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> policy);

/// \brief Function to create a RandomSharpness TensorOperation.
/// \notes Tensor operation to perform random sharpness.
/// \param[in] degrees A float vector of size 2, representing the starting and ending degree to uniformly
///     sample from, to select a degree to adjust sharpness.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomSharpnessOperation> RandomSharpness(std::vector<float> degrees = {0.1, 1.9});

/// \brief Function to create a RandomSolarize TensorOperation.
/// \notes Invert pixels randomly within specified range. If min=max, it is a single fixed magnitude operation
///     to inverts all pixel above that threshold
/// \param[in] threshold A vector with two elements specifying the pixel range to invert.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomSolarizeOperation> RandomSolarize(std::vector<uint8_t> threshold = {0, 255});

/// \brief Function to create a RandomVerticalFlip TensorOperation.
/// \notes Tensor operation to perform random vertical flip.
/// \param[in] prob A float representing the probability of flip.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomVerticalFlipOperation> RandomVerticalFlip(float prob = 0.5);

/// \brief Function to create a RandomVerticalFlipWithBBox TensorOperation.
/// \notes Flip the input image vertically, randomly with a given probability and adjust bounding boxes accordingly.
/// \param[in] prob A float representing the probability of flip.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomVerticalFlipWithBBoxOperation> RandomVerticalFlipWithBBox(float prob = 0.5);

/// \brief Function to create a RescaleOperation TensorOperation.
/// \notes Tensor operation to rescale the input image.
/// \param[in] rescale Rescale factor.
/// \param[in] shift Shift factor.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RescaleOperation> Rescale(float rescale, float shift);

/// \brief Function to create a ResizeWithBBox TensorOperation.
/// \notes Resize the input image to the given size and adjust bounding boxes accordingly.
/// \param[in] size The output size of the resized image.
///     If size is an integer, smaller edge of the image will be resized to this value with the same image aspect ratio.
///     If size is a sequence of length 2, it should be (height, width).
/// \param[in] interpolation An enum for the mode of interpolation (default=InterpolationMode::kLinear).
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<ResizeWithBBoxOperation> ResizeWithBBox(std::vector<int32_t> size,
                                                        InterpolationMode interpolation = InterpolationMode::kLinear);

/// \brief Function to create a RgbaToBgr TensorOperation.
/// \notes Changes the input 4 channel RGBA tensor to 3 channel BGR.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RgbaToBgrOperation> RGBA2BGR();

/// \brief Function to create a RgbaToRgb TensorOperation.
/// \notes Changes the input 4 channel RGBA tensor to 3 channel RGB.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RgbaToRgbOperation> RGBA2RGB();

/// \brief Function to create a SoftDvppDecodeRandomCropResizeJpeg TensorOperation.
/// \notes Tensor operation to decode, random crop and resize JPEG image using the simulation algorithm of
///     Ascend series chip DVPP module. The usage scenario is consistent with SoftDvppDecodeResizeJpeg.
///     The input image size should be in range [32*32, 8192*8192].
///     The zoom-out and zoom-in multiples of the image length and width should in the range [1/32, 16].
///     Only images with an even resolution can be output. The output of odd resolution is not supported.
/// \param[in] size A vector representing the output size of the resized image.
///     If size is a single value, smaller edge of the image will be resized to this value with
///     the same image aspect ratio. If size has 2 values, it should be (height, width).
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<SoftDvppDecodeRandomCropResizeJpegOperation> SoftDvppDecodeRandomCropResizeJpeg(
  std::vector<int32_t> size, std::vector<float> scale = {0.08, 1.0}, std::vector<float> ratio = {3. / 4., 4. / 3.},
  int32_t max_attempts = 10);

/// \brief Function to create a SoftDvppDecodeResizeJpeg TensorOperation.
/// \notes Tensor operation to decode and resize JPEG image using the simulation algorithm of Ascend series
///     chip DVPP module. It is recommended to use this algorithm in the following scenarios:
///     When training, the DVPP of the Ascend chip is not used,
///     and the DVPP of the Ascend chip is used during inference,
///     and the accuracy of inference is lower than the accuracy of training;
///     and the input image size should be in range [32*32, 8192*8192].
///     The zoom-out and zoom-in multiples of the image length and width should in the range [1/32, 16].
///     Only images with an even resolution can be output. The output of odd resolution is not supported.
/// \param[in] size A vector representing the output size of the resized image.
///     If size is a single value, smaller edge of the image will be resized to this value with
///     the same image aspect ratio. If size has 2 values, it should be (height, width).
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<SoftDvppDecodeResizeJpegOperation> SoftDvppDecodeResizeJpeg(std::vector<int32_t> size);

/// \brief Function to create a SwapRedBlue TensorOp
/// \notes Swaps the red and blue channels in image
/// \return Shared pointer to the current TensorOp
std::shared_ptr<SwapRedBlueOperation> SwapRedBlue();

/// \brief Function to create a UniformAugment TensorOperation.
/// \notes Tensor operation to perform randomly selected augmentation.
/// \param[in] transforms A vector of TensorOperation transforms.
/// \param[in] num_ops An integer representing the number of OPs to be selected and applied.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<UniformAugOperation> UniformAugment(std::vector<std::shared_ptr<TensorOperation>> transforms,
                                                    int32_t num_ops = 2);

/* ####################################### Derived TensorOperation classes ################################# */

class AutoContrastOperation : public TensorOperation {
 public:
  explicit AutoContrastOperation(float cutoff = 0.0, std::vector<uint32_t> ignore = {});

  ~AutoContrastOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kAutoContrastOperation; }

 private:
  float cutoff_;
  std::vector<uint32_t> ignore_;
};

class BoundingBoxAugmentOperation : public TensorOperation {
 public:
  explicit BoundingBoxAugmentOperation(std::shared_ptr<TensorOperation> transform, float ratio = 0.3);

  ~BoundingBoxAugmentOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kBoundingBoxAugmentOperation; }

 private:
  std::shared_ptr<TensorOperation> transform_;
  float ratio_;
};

class CutMixBatchOperation : public TensorOperation {
 public:
  explicit CutMixBatchOperation(ImageBatchFormat image_batch_format, float alpha = 1.0, float prob = 1.0);

  ~CutMixBatchOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kCutMixBatchOperation; }

 private:
  float alpha_;
  float prob_;
  ImageBatchFormat image_batch_format_;
};

class CutOutOperation : public TensorOperation {
 public:
  explicit CutOutOperation(int32_t length, int32_t num_patches = 1);

  ~CutOutOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kCutOutOperation; }

 private:
  int32_t length_;
  int32_t num_patches_;
  ImageBatchFormat image_batch_format_;
};

class DvppDecodeResizeCropOperation : public TensorOperation {
 public:
  explicit DvppDecodeResizeCropOperation(const std::vector<uint32_t> &crop, const std::vector<uint32_t> &resize);

  ~DvppDecodeResizeCropOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDvppDecodeResizeCropOperation; }

 private:
  std::vector<uint32_t> crop_;
  std::vector<uint32_t> resize_;
};

class EqualizeOperation : public TensorOperation {
 public:
  ~EqualizeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kEqualizeOperation; }
};

class HwcToChwOperation : public TensorOperation {
 public:
  ~HwcToChwOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kHwcToChwOperation; }
};

class InvertOperation : public TensorOperation {
 public:
  ~InvertOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kInvertOperation; }
};

class MixUpBatchOperation : public TensorOperation {
 public:
  explicit MixUpBatchOperation(float alpha = 1);

  ~MixUpBatchOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kMixUpBatchOperation; }

 private:
  float alpha_;
};

class NormalizePadOperation : public TensorOperation {
 public:
  NormalizePadOperation(const std::vector<float> &mean, const std::vector<float> &std,
                        const std::string &dtype = "float32");

  ~NormalizePadOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kNormalizePadOperation; }

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
  std::string dtype_;
};

class PadOperation : public TensorOperation {
 public:
  PadOperation(std::vector<int32_t> padding, std::vector<uint8_t> fill_value = {0},
               BorderType padding_mode = BorderType::kConstant);

  ~PadOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kPadOperation; }

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

  Status ValidateParams() override;

  std::string Name() const override { return kRandomAffineOperation; }

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

  Status ValidateParams() override;

  std::string Name() const override { return kRandomColorOperation; }

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

  Status ValidateParams() override;

  std::string Name() const override { return kRandomColorAdjustOperation; }

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

  Status ValidateParams() override;

  std::string Name() const override { return kRandomCropOperation; }

 private:
  std::vector<int32_t> size_;
  std::vector<int32_t> padding_;
  bool pad_if_needed_;
  std::vector<uint8_t> fill_value_;
  BorderType padding_mode_;
};

class RandomCropDecodeResizeOperation : public TensorOperation {
 public:
  RandomCropDecodeResizeOperation(std::vector<int32_t> size, std::vector<float> scale, std::vector<float> ratio,
                                  InterpolationMode interpolation, int32_t max_attempts);

  ~RandomCropDecodeResizeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomCropDecodeResizeOperation; }

 private:
  std::vector<int32_t> size_;
  std::vector<float> scale_;
  std::vector<float> ratio_;
  InterpolationMode interpolation_;
  int32_t max_attempts_;
};

class RandomCropWithBBoxOperation : public TensorOperation {
 public:
  RandomCropWithBBoxOperation(std::vector<int32_t> size, std::vector<int32_t> padding = {0, 0, 0, 0},
                              bool pad_if_needed = false, std::vector<uint8_t> fill_value = {0, 0, 0},
                              BorderType padding_mode = BorderType::kConstant);

  ~RandomCropWithBBoxOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomCropWithBBoxOperation; }

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

  Status ValidateParams() override;

  std::string Name() const override { return kRandomHorizontalFlipOperation; }

 private:
  float probability_;
};

class RandomHorizontalFlipWithBBoxOperation : public TensorOperation {
 public:
  explicit RandomHorizontalFlipWithBBoxOperation(float probability = 0.5);

  ~RandomHorizontalFlipWithBBoxOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomHorizontalFlipWithBBoxOperation; }

 private:
  float probability_;
};

class RandomPosterizeOperation : public TensorOperation {
 public:
  explicit RandomPosterizeOperation(const std::vector<uint8_t> &bit_range = {4, 8});

  ~RandomPosterizeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomPosterizeOperation; }

 private:
  std::vector<uint8_t> bit_range_;
};

class RandomResizeOperation : public TensorOperation {
 public:
  explicit RandomResizeOperation(std::vector<int32_t> size);

  ~RandomResizeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomResizeOperation; }

 private:
  std::vector<int32_t> size_;
};

class RandomResizeWithBBoxOperation : public TensorOperation {
 public:
  explicit RandomResizeWithBBoxOperation(std::vector<int32_t> size);

  ~RandomResizeWithBBoxOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomResizeWithBBoxOperation; }

 private:
  std::vector<int32_t> size_;
};

class RandomResizedCropOperation : public TensorOperation {
 public:
  explicit RandomResizedCropOperation(std::vector<int32_t> size, std::vector<float> scale = {0.08, 1.0},
                                      std::vector<float> ratio = {3. / 4., 4. / 3.},
                                      InterpolationMode interpolation = InterpolationMode::kNearestNeighbour,
                                      int32_t max_attempts = 10);

  ~RandomResizedCropOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomResizedCropOperation; }

 private:
  std::vector<int32_t> size_;
  std::vector<float> scale_;
  std::vector<float> ratio_;
  InterpolationMode interpolation_;
  int32_t max_attempts_;
};

class RandomResizedCropWithBBoxOperation : public TensorOperation {
 public:
  explicit RandomResizedCropWithBBoxOperation(std::vector<int32_t> size, std::vector<float> scale = {0.08, 1.0},
                                              std::vector<float> ratio = {3. / 4., 4. / 3.},
                                              InterpolationMode interpolation = InterpolationMode::kNearestNeighbour,
                                              int32_t max_attempts = 10);

  ~RandomResizedCropWithBBoxOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomResizedCropWithBBoxOperation; }

 private:
  std::vector<int32_t> size_;
  std::vector<float> scale_;
  std::vector<float> ratio_;
  InterpolationMode interpolation_;
  int32_t max_attempts_;
};

class RandomRotationOperation : public TensorOperation {
 public:
  RandomRotationOperation(std::vector<float> degrees, InterpolationMode interpolation_mode, bool expand,
                          std::vector<float> center, std::vector<uint8_t> fill_value);

  ~RandomRotationOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomRotationOperation; }

 private:
  std::vector<float> degrees_;
  InterpolationMode interpolation_mode_;
  std::vector<float> center_;
  bool expand_;
  std::vector<uint8_t> fill_value_;
};

class RandomSelectSubpolicyOperation : public TensorOperation {
 public:
  explicit RandomSelectSubpolicyOperation(
    std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> policy);

  ~RandomSelectSubpolicyOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomSelectSubpolicyOperation; }

 private:
  std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> policy_;
};

class RandomSharpnessOperation : public TensorOperation {
 public:
  explicit RandomSharpnessOperation(std::vector<float> degrees = {0.1, 1.9});

  ~RandomSharpnessOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomSharpnessOperation; }

 private:
  std::vector<float> degrees_;
};

class RandomSolarizeOperation : public TensorOperation {
 public:
  explicit RandomSolarizeOperation(std::vector<uint8_t> threshold);

  ~RandomSolarizeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomSolarizeOperation; }

 private:
  std::vector<uint8_t> threshold_;
};

class RandomVerticalFlipOperation : public TensorOperation {
 public:
  explicit RandomVerticalFlipOperation(float probability = 0.5);

  ~RandomVerticalFlipOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomVerticalFlipOperation; }

 private:
  float probability_;
};

class RandomVerticalFlipWithBBoxOperation : public TensorOperation {
 public:
  explicit RandomVerticalFlipWithBBoxOperation(float probability = 0.5);

  ~RandomVerticalFlipWithBBoxOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomVerticalFlipWithBBoxOperation; }

 private:
  float probability_;
};

class RescaleOperation : public TensorOperation {
 public:
  explicit RescaleOperation(float rescale, float shift);

  ~RescaleOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRescaleOperation; }

 private:
  float rescale_;
  float shift_;
};

class ResizeWithBBoxOperation : public TensorOperation {
 public:
  explicit ResizeWithBBoxOperation(std::vector<int32_t> size,
                                   InterpolationMode interpolation_mode = InterpolationMode::kLinear);

  ~ResizeWithBBoxOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kResizeWithBBoxOperation; }

 private:
  std::vector<int32_t> size_;
  InterpolationMode interpolation_;
};

class RgbaToBgrOperation : public TensorOperation {
 public:
  RgbaToBgrOperation();

  ~RgbaToBgrOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRgbaToBgrOperation; }
};

class RgbaToRgbOperation : public TensorOperation {
 public:
  RgbaToRgbOperation();

  ~RgbaToRgbOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRgbaToRgbOperation; }
};

class SoftDvppDecodeRandomCropResizeJpegOperation : public TensorOperation {
 public:
  explicit SoftDvppDecodeRandomCropResizeJpegOperation(std::vector<int32_t> size, std::vector<float> scale,
                                                       std::vector<float> ratio, int32_t max_attempts);

  ~SoftDvppDecodeRandomCropResizeJpegOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kSoftDvppDecodeRandomCropResizeJpegOperation; }

 private:
  std::vector<int32_t> size_;
  std::vector<float> scale_;
  std::vector<float> ratio_;
  int32_t max_attempts_;
};

class SoftDvppDecodeResizeJpegOperation : public TensorOperation {
 public:
  explicit SoftDvppDecodeResizeJpegOperation(std::vector<int32_t> size);

  ~SoftDvppDecodeResizeJpegOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kSoftDvppDecodeResizeJpegOperation; }

 private:
  std::vector<int32_t> size_;
};

class SwapRedBlueOperation : public TensorOperation {
 public:
  SwapRedBlueOperation();

  ~SwapRedBlueOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kSwapRedBlueOperation; }
};

class UniformAugOperation : public TensorOperation {
 public:
  explicit UniformAugOperation(std::vector<std::shared_ptr<TensorOperation>> transforms, int32_t num_ops = 2);

  ~UniformAugOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kUniformAugOperation; }

 private:
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
  int32_t num_ops_;
};

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_VISION_H_
