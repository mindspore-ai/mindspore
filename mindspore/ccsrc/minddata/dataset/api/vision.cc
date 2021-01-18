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

#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/vision.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#endif
// Kernel image headers (in alphabetical order)
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/auto_contrast_op.h"
#include "minddata/dataset/kernels/image/bounding_box_augment_op.h"
#endif
#include "minddata/dataset/kernels/image/center_crop_op.h"
#include "minddata/dataset/kernels/image/crop_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/cutmix_batch_op.h"
#include "minddata/dataset/kernels/image/cut_out_op.h"
#endif
#include "minddata/dataset/kernels/image/decode_op.h"
#ifdef ENABLE_ACL
#include "minddata/dataset/kernels/image/dvpp/dvpp_decode_resize_crop_jpeg_op.h"
#endif
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/equalize_op.h"
#include "minddata/dataset/kernels/image/hwc_to_chw_op.h"
#include "minddata/dataset/kernels/image/invert_op.h"
#include "minddata/dataset/kernels/image/mixup_batch_op.h"
#endif
#include "minddata/dataset/kernels/image/normalize_op.h"
#include "minddata/dataset/kernels/image/normalize_pad_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/pad_op.h"
#include "minddata/dataset/kernels/image/random_affine_op.h"
#include "minddata/dataset/kernels/image/random_color_op.h"
#include "minddata/dataset/kernels/image/random_color_adjust_op.h"
#include "minddata/dataset/kernels/image/random_crop_and_resize_op.h"
#include "minddata/dataset/kernels/image/random_crop_op.h"
#include "minddata/dataset/kernels/image/random_crop_decode_resize_op.h"
#include "minddata/dataset/kernels/image/random_crop_with_bbox_op.h"
#include "minddata/dataset/kernels/image/random_crop_and_resize_with_bbox_op.h"
#include "minddata/dataset/kernels/image/random_horizontal_flip_op.h"
#include "minddata/dataset/kernels/image/random_horizontal_flip_with_bbox_op.h"
#include "minddata/dataset/kernels/image/random_posterize_op.h"
#include "minddata/dataset/kernels/image/random_resize_op.h"
#include "minddata/dataset/kernels/image/random_resize_with_bbox_op.h"
#include "minddata/dataset/kernels/image/random_rotation_op.h"
#include "minddata/dataset/kernels/image/random_select_subpolicy_op.h"
#include "minddata/dataset/kernels/image/random_sharpness_op.h"
#include "minddata/dataset/kernels/image/random_solarize_op.h"
#include "minddata/dataset/kernels/image/random_vertical_flip_op.h"
#include "minddata/dataset/kernels/image/random_vertical_flip_with_bbox_op.h"
#include "minddata/dataset/kernels/image/rescale_op.h"
#endif
#include "minddata/dataset/kernels/image/resize_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/resize_with_bbox_op.h"
#include "minddata/dataset/kernels/image/rgba_to_bgr_op.h"
#include "minddata/dataset/kernels/image/rgba_to_rgb_op.h"
#include "minddata/dataset/kernels/image/soft_dvpp/soft_dvpp_decode_random_crop_resize_jpeg_op.h"
#include "minddata/dataset/kernels/image/soft_dvpp/soft_dvpp_decode_resize_jpeg_op.h"
#include "minddata/dataset/kernels/image/swap_red_blue_op.h"
#include "minddata/dataset/kernels/image/uniform_aug_op.h"
#endif
#include "minddata/dataset/kernels/image/rotate_op.h"

namespace mindspore {
namespace dataset {

// Transform operations for computer vision.
namespace vision {
#ifndef ENABLE_ANDROID
// FUNCTIONS TO CREATE VISION TRANSFORM OPERATIONS
// (In alphabetical order)

// Function to create AutoContrastOperation.
std::shared_ptr<AutoContrastOperation> AutoContrast(float cutoff, std::vector<uint32_t> ignore) {
  auto op = std::make_shared<AutoContrastOperation>(cutoff, ignore);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create BoundingBoxAugmentOperation.
std::shared_ptr<BoundingBoxAugmentOperation> BoundingBoxAugment(std::shared_ptr<TensorOperation> transform,
                                                                float ratio) {
  auto op = std::make_shared<BoundingBoxAugmentOperation>(transform, ratio);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}
#endif

// Function to create CenterCropOperation.
std::shared_ptr<CenterCropOperation> CenterCrop(std::vector<int32_t> size) {
  auto op = std::make_shared<CenterCropOperation>(size);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create CropOperation.
std::shared_ptr<CropOperation> Crop(std::vector<int32_t> coordinates, std::vector<int32_t> size) {
  auto op = std::make_shared<CropOperation>(coordinates, size);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}
#ifndef ENABLE_ANDROID
// Function to create CutMixBatchOperation.
std::shared_ptr<CutMixBatchOperation> CutMixBatch(ImageBatchFormat image_batch_format, float alpha, float prob) {
  auto op = std::make_shared<CutMixBatchOperation>(image_batch_format, alpha, prob);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create CutOutOp.
std::shared_ptr<CutOutOperation> CutOut(int32_t length, int32_t num_patches) {
  auto op = std::make_shared<CutOutOperation>(length, num_patches);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create DecodeOperation.
std::shared_ptr<DecodeOperation> Decode(bool rgb) {
  auto op = std::make_shared<DecodeOperation>(rgb);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

#ifdef ENABLE_ACL
// Function to create DvppDecodeResizeCropOperation.
std::shared_ptr<DvppDecodeResizeCropOperation> DvppDecodeResizeCropJpeg(std::vector<uint32_t> crop,
                                                                        std::vector<uint32_t> resize) {
  auto op = std::make_shared<DvppDecodeResizeCropOperation>(crop, resize);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}
#endif

// Function to create EqualizeOperation.
std::shared_ptr<EqualizeOperation> Equalize() {
  auto op = std::make_shared<EqualizeOperation>();
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create HwcToChwOperation.
std::shared_ptr<HwcToChwOperation> HWC2CHW() {
  auto op = std::make_shared<HwcToChwOperation>();
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create InvertOperation.
std::shared_ptr<InvertOperation> Invert() {
  auto op = std::make_shared<InvertOperation>();
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create MixUpBatchOperation.
std::shared_ptr<MixUpBatchOperation> MixUpBatch(float alpha) {
  auto op = std::make_shared<MixUpBatchOperation>(alpha);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}
#endif

// Function to create NormalizeOperation.
std::shared_ptr<NormalizeOperation> Normalize(std::vector<float> mean, std::vector<float> std) {
  auto op = std::make_shared<NormalizeOperation>(mean, std);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

#ifndef ENABLE_ANDROID
// Function to create NormalizePadOperation.
std::shared_ptr<NormalizePadOperation> NormalizePad(const std::vector<float> &mean, const std::vector<float> &std,
                                                    const std::string &dtype) {
  auto op = std::make_shared<NormalizePadOperation>(mean, std, dtype);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create PadOperation.
std::shared_ptr<PadOperation> Pad(std::vector<int32_t> padding, std::vector<uint8_t> fill_value,
                                  BorderType padding_mode) {
  auto op = std::make_shared<PadOperation>(padding, fill_value, padding_mode);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomAffineOperation.
std::shared_ptr<RandomAffineOperation> RandomAffine(const std::vector<float_t> &degrees,
                                                    const std::vector<float_t> &translate_range,
                                                    const std::vector<float_t> &scale_range,
                                                    const std::vector<float_t> &shear_ranges,
                                                    InterpolationMode interpolation,
                                                    const std::vector<uint8_t> &fill_value) {
  auto op = std::make_shared<RandomAffineOperation>(degrees, translate_range, scale_range, shear_ranges, interpolation,
                                                    fill_value);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomColorOperation.
std::shared_ptr<RandomColorOperation> RandomColor(float t_lb, float t_ub) {
  auto op = std::make_shared<RandomColorOperation>(t_lb, t_ub);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

std::shared_ptr<TensorOp> RandomColorOperation::Build() {
  std::shared_ptr<RandomColorOp> tensor_op = std::make_shared<RandomColorOp>(t_lb_, t_ub_);
  return tensor_op;
}

// Function to create RandomColorAdjustOperation.
std::shared_ptr<RandomColorAdjustOperation> RandomColorAdjust(std::vector<float> brightness,
                                                              std::vector<float> contrast,
                                                              std::vector<float> saturation, std::vector<float> hue) {
  auto op = std::make_shared<RandomColorAdjustOperation>(brightness, contrast, saturation, hue);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomCropOperation.
std::shared_ptr<RandomCropOperation> RandomCrop(std::vector<int32_t> size, std::vector<int32_t> padding,
                                                bool pad_if_needed, std::vector<uint8_t> fill_value,
                                                BorderType padding_mode) {
  auto op = std::make_shared<RandomCropOperation>(size, padding, pad_if_needed, fill_value, padding_mode);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomCropDecodeResizeOperation.
std::shared_ptr<RandomCropDecodeResizeOperation> RandomCropDecodeResize(std::vector<int32_t> size,
                                                                        std::vector<float> scale,
                                                                        std::vector<float> ratio,
                                                                        InterpolationMode interpolation,
                                                                        int32_t max_attempts) {
  auto op = std::make_shared<RandomCropDecodeResizeOperation>(size, scale, ratio, interpolation, max_attempts);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomCropWithBBoxOperation.
std::shared_ptr<RandomCropWithBBoxOperation> RandomCropWithBBox(std::vector<int32_t> size, std::vector<int32_t> padding,
                                                                bool pad_if_needed, std::vector<uint8_t> fill_value,
                                                                BorderType padding_mode) {
  auto op = std::make_shared<RandomCropWithBBoxOperation>(size, padding, pad_if_needed, fill_value, padding_mode);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomHorizontalFlipOperation.
std::shared_ptr<RandomHorizontalFlipOperation> RandomHorizontalFlip(float prob) {
  auto op = std::make_shared<RandomHorizontalFlipOperation>(prob);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomHorizontalFlipOperation.
std::shared_ptr<RandomHorizontalFlipWithBBoxOperation> RandomHorizontalFlipWithBBox(float prob) {
  auto op = std::make_shared<RandomHorizontalFlipWithBBoxOperation>(prob);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomPosterizeOperation.
std::shared_ptr<RandomPosterizeOperation> RandomPosterize(const std::vector<uint8_t> &bit_range) {
  auto op = std::make_shared<RandomPosterizeOperation>(bit_range);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomResizeOperation.
std::shared_ptr<RandomResizeOperation> RandomResize(std::vector<int32_t> size) {
  auto op = std::make_shared<RandomResizeOperation>(size);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomResizeWithBBoxOperation.
std::shared_ptr<RandomResizeWithBBoxOperation> RandomResizeWithBBox(std::vector<int32_t> size) {
  auto op = std::make_shared<RandomResizeWithBBoxOperation>(size);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomResizedCropOperation.
std::shared_ptr<RandomResizedCropOperation> RandomResizedCrop(std::vector<int32_t> size, std::vector<float> scale,
                                                              std::vector<float> ratio, InterpolationMode interpolation,
                                                              int32_t max_attempts) {
  auto op = std::make_shared<RandomResizedCropOperation>(size, scale, ratio, interpolation, max_attempts);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomResizedCropOperation.
std::shared_ptr<RandomResizedCropWithBBoxOperation> RandomResizedCropWithBBox(std::vector<int32_t> size,
                                                                              std::vector<float> scale,
                                                                              std::vector<float> ratio,
                                                                              InterpolationMode interpolation,
                                                                              int32_t max_attempts) {
  auto op = std::make_shared<RandomResizedCropWithBBoxOperation>(size, scale, ratio, interpolation, max_attempts);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomRotationOperation.
std::shared_ptr<RandomRotationOperation> RandomRotation(std::vector<float> degrees, InterpolationMode resample,
                                                        bool expand, std::vector<float> center,
                                                        std::vector<uint8_t> fill_value) {
  auto op = std::make_shared<RandomRotationOperation>(degrees, resample, expand, center, fill_value);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomSharpnessOperation.
std::shared_ptr<RandomSharpnessOperation> RandomSharpness(std::vector<float> degrees) {
  auto op = std::make_shared<RandomSharpnessOperation>(degrees);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomSolarizeOperation.
std::shared_ptr<RandomSolarizeOperation> RandomSolarize(std::vector<uint8_t> threshold) {
  auto op = std::make_shared<RandomSolarizeOperation>(threshold);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomSelectSubpolicyOperation.
std::shared_ptr<RandomSelectSubpolicyOperation> RandomSelectSubpolicy(
  std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> policy) {
  auto op = std::make_shared<RandomSelectSubpolicyOperation>(policy);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomVerticalFlipOperation.
std::shared_ptr<RandomVerticalFlipOperation> RandomVerticalFlip(float prob) {
  auto op = std::make_shared<RandomVerticalFlipOperation>(prob);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomVerticalFlipWithBBoxOperation.
std::shared_ptr<RandomVerticalFlipWithBBoxOperation> RandomVerticalFlipWithBBox(float prob) {
  auto op = std::make_shared<RandomVerticalFlipWithBBoxOperation>(prob);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RescaleOperation.
std::shared_ptr<RescaleOperation> Rescale(float rescale, float shift) {
  auto op = std::make_shared<RescaleOperation>(rescale, shift);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

#endif
// Function to create ResizeOperation.
std::shared_ptr<ResizeOperation> Resize(std::vector<int32_t> size, InterpolationMode interpolation) {
  auto op = std::make_shared<ResizeOperation>(size, interpolation);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

#ifdef ENABLE_ANDROID
// Function to create RotateOperation.
std::shared_ptr<RotateOperation> Rotate() {
  auto op = std::make_shared<RotateOperation>();
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}
#endif

#ifndef ENABLE_ANDROID
// Function to create ResizeWithBBoxOperation.
std::shared_ptr<ResizeWithBBoxOperation> ResizeWithBBox(std::vector<int32_t> size, InterpolationMode interpolation) {
  auto op = std::make_shared<ResizeWithBBoxOperation>(size, interpolation);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RgbaToBgrOperation.
std::shared_ptr<RgbaToBgrOperation> RGBA2BGR() {
  auto op = std::make_shared<RgbaToBgrOperation>();
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RgbaToRgbOperation.
std::shared_ptr<RgbaToRgbOperation> RGBA2RGB() {
  auto op = std::make_shared<RgbaToRgbOperation>();
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create SoftDvppDecodeRandomCropResizeJpegOperation.
std::shared_ptr<SoftDvppDecodeRandomCropResizeJpegOperation> SoftDvppDecodeRandomCropResizeJpeg(
  std::vector<int32_t> size, std::vector<float> scale, std::vector<float> ratio, int32_t max_attempts) {
  auto op = std::make_shared<SoftDvppDecodeRandomCropResizeJpegOperation>(size, scale, ratio, max_attempts);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create SoftDvppDecodeResizeJpegOperation.
std::shared_ptr<SoftDvppDecodeResizeJpegOperation> SoftDvppDecodeResizeJpeg(std::vector<int32_t> size) {
  auto op = std::make_shared<SoftDvppDecodeResizeJpegOperation>(size);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create SwapRedBlueOperation.
std::shared_ptr<SwapRedBlueOperation> SwapRedBlue() {
  auto op = std::make_shared<SwapRedBlueOperation>();
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create UniformAugOperation.
std::shared_ptr<UniformAugOperation> UniformAugment(std::vector<std::shared_ptr<TensorOperation>> transforms,
                                                    int32_t num_ops) {
  auto op = std::make_shared<UniformAugOperation>(transforms, num_ops);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}
#endif

/* ####################################### Derived TensorOperation classes ################################# */

// (In alphabetical order)
#ifndef ENABLE_ANDROID

// AutoContrastOperation
AutoContrastOperation::AutoContrastOperation(float cutoff, std::vector<uint32_t> ignore)
    : cutoff_(cutoff), ignore_(ignore) {}

Status AutoContrastOperation::ValidateParams() {
  if (cutoff_ < 0 || cutoff_ > 100) {
    std::string err_msg = "AutoContrast: cutoff has to be between 0 and 100, got: " + std::to_string(cutoff_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  for (uint32_t single_ignore : ignore_) {
    if (single_ignore > 255) {
      std::string err_msg =
        "AutoContrast: invalid size, ignore has to be between 0 and 255, got: " + std::to_string(single_ignore);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> AutoContrastOperation::Build() {
  std::shared_ptr<AutoContrastOp> tensor_op = std::make_shared<AutoContrastOp>(cutoff_, ignore_);
  return tensor_op;
}

// BoundingBoxAugmentOperation
BoundingBoxAugmentOperation::BoundingBoxAugmentOperation(std::shared_ptr<TensorOperation> transform, float ratio)
    : transform_(transform), ratio_(ratio) {}

Status BoundingBoxAugmentOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorTransforms("BoundingBoxAugment", {transform_}));
  RETURN_IF_NOT_OK(ValidateProbability("BoundingBoxAugment", ratio_));
  return Status::OK();
}

std::shared_ptr<TensorOp> BoundingBoxAugmentOperation::Build() {
  std::shared_ptr<BoundingBoxAugmentOp> tensor_op = std::make_shared<BoundingBoxAugmentOp>(transform_->Build(), ratio_);
  return tensor_op;
}
#endif

// CenterCropOperation
CenterCropOperation::CenterCropOperation(std::vector<int32_t> size) : size_(size) {}

Status CenterCropOperation::ValidateParams() {
  if (size_.empty() || size_.size() > 2) {
    std::string err_msg = "CenterCrop: size vector has incorrect size.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // We have to limit crop size due to library restrictions, optimized to only iterate over size_ once
  for (int32_t i = 0; i < size_.size(); ++i) {
    if (size_[i] <= 0) {
      std::string err_msg = "CenterCrop: invalid size, size must be greater than 0, got: " + std::to_string(size_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    if (size_[i] == INT_MAX) {
      std::string err_msg = "CenterCrop: invalid size, size too large, got: " + std::to_string(size_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> CenterCropOperation::Build() {
  int32_t crop_height = size_[0];
  int32_t crop_width = size_[0];

  // User has specified crop_width.
  if (size_.size() == 2) {
    crop_width = size_[1];
  }

  std::shared_ptr<CenterCropOp> tensor_op = std::make_shared<CenterCropOp>(crop_height, crop_width);
  return tensor_op;
}

Status CenterCropOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["size"] = size_;
  return Status::OK();
}

// CropOperation.
CropOperation::CropOperation(std::vector<int32_t> coordinates, std::vector<int32_t> size)
    : coordinates_(coordinates), size_(size) {}

Status CropOperation::ValidateParams() {
  // Do some input validation.
  if (coordinates_.size() != 2) {
    std::string err_msg = "Crop: coordinates must be a vector of two values";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // we don't check the coordinates here because we don't have access to image dimensions
  if (size_.empty() || size_.size() > 2) {
    std::string err_msg = "Crop: size must be a vector of one or two values";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // We have to limit crop size due to library restrictions, optimized to only iterate over size_ once
  for (int32_t i = 0; i < size_.size(); ++i) {
    if (size_[i] <= 0) {
      std::string err_msg = "Crop: invalid size, size must be greater than 0, got: " + std::to_string(size_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    if (size_[i] == INT_MAX) {
      std::string err_msg = "Crop: invalid size, size too large, got: " + std::to_string(size_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  for (int32_t j = 0; j < coordinates_.size(); ++j) {
    if (coordinates_[j] < 0) {
      std::string err_msg =
        "Crop: invalid coordinates, coordinates must be greater than 0, got: " + std::to_string(coordinates_[j]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> CropOperation::Build() {
  int32_t x, y, height, width;

  x = coordinates_[0];
  y = coordinates_[1];

  height = size_[0];
  width = size_[0];
  // User has specified crop_width.
  if (size_.size() == 2) {
    width = size_[1];
  }

  std::shared_ptr<CropOp> tensor_op = std::make_shared<CropOp>(x, y, height, width);
  return tensor_op;
}

#ifndef ENABLE_ANDROID
// CutMixBatchOperation
CutMixBatchOperation::CutMixBatchOperation(ImageBatchFormat image_batch_format, float alpha, float prob)
    : image_batch_format_(image_batch_format), alpha_(alpha), prob_(prob) {}

Status CutMixBatchOperation::ValidateParams() {
  if (alpha_ <= 0) {
    std::string err_msg =
      "CutMixBatch: alpha must be a positive floating value however it is: " + std::to_string(alpha_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (prob_ < 0 || prob_ > 1) {
    std::string err_msg = "CutMixBatch: Probability has to be between 0 and 1.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> CutMixBatchOperation::Build() {
  std::shared_ptr<CutMixBatchOp> tensor_op = std::make_shared<CutMixBatchOp>(image_batch_format_, alpha_, prob_);
  return tensor_op;
}

// CutOutOperation
CutOutOperation::CutOutOperation(int32_t length, int32_t num_patches) : length_(length), num_patches_(num_patches) {}

Status CutOutOperation::ValidateParams() {
  if (length_ <= 0) {
    std::string err_msg = "CutOut: length must be positive, got: " + std::to_string(length_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (num_patches_ <= 0) {
    std::string err_msg = "CutOut: number of patches must be positive, got: " + std::to_string(num_patches_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> CutOutOperation::Build() {
  std::shared_ptr<CutOutOp> tensor_op = std::make_shared<CutOutOp>(length_, length_, num_patches_, false, 0, 0, 0);
  return tensor_op;
}

// DecodeOperation
DecodeOperation::DecodeOperation(bool rgb) : rgb_(rgb) {}

Status DecodeOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> DecodeOperation::Build() { return std::make_shared<DecodeOp>(rgb_); }

Status DecodeOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["rgb"] = rgb_;
  return Status::OK();
}

// EqualizeOperation
Status EqualizeOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> EqualizeOperation::Build() { return std::make_shared<EqualizeOp>(); }

// HwcToChwOperation
Status HwcToChwOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> HwcToChwOperation::Build() { return std::make_shared<HwcToChwOp>(); }

// InvertOperation
Status InvertOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> InvertOperation::Build() { return std::make_shared<InvertOp>(); }

// MixUpOperation
MixUpBatchOperation::MixUpBatchOperation(float alpha) : alpha_(alpha) {}

Status MixUpBatchOperation::ValidateParams() {
  if (alpha_ <= 0) {
    std::string err_msg =
      "MixUpBatch: alpha must be a positive floating value however it is: " + std::to_string(alpha_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> MixUpBatchOperation::Build() { return std::make_shared<MixUpBatchOp>(alpha_); }

#endif

#ifdef ENABLE_ACL
// DvppDecodeResizeCropOperation
DvppDecodeResizeCropOperation::DvppDecodeResizeCropOperation(const std::vector<uint32_t> &crop,
                                                             const std::vector<uint32_t> &resize)
    : crop_(crop), resize_(resize) {}

Status DvppDecodeResizeCropOperation::ValidateParams() {
  // size
  if (crop_.empty() || crop_.size() > 2) {
    std::string err_msg = "DvppDecodeResizeCropJpeg: crop resolution must be a vector of one or two elements, got: " +
                          std::to_string(crop_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (resize_.empty() || resize_.size() > 2) {
    std::string err_msg = "DvppDecodeResizeCropJpeg: resize resolution must be a vector of one or two elements, got: " +
                          std::to_string(resize_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (*min_element(crop_.begin(), crop_.end()) < 32 || *max_element(crop_.begin(), crop_.end()) > 2048) {
    std::string err_msg = "Dvpp module supports crop image with resolution in range [32, 2048], got Crop Parameters: ";
    if (crop_.size() == 2) {
      MS_LOG(ERROR) << err_msg << "[" << crop_[0] << ", " << crop_[1] << "]";
    } else {
      MS_LOG(ERROR) << err_msg << "[" << crop_[0] << ", " << crop_[0] << "]";
    }
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (*min_element(resize_.begin(), resize_.end()) < 32 || *max_element(resize_.begin(), resize_.end()) > 2048) {
    std::string err_msg =
      "Dvpp module supports resize image with resolution in range [32, 2048], got Crop Parameters: ";
    if (resize_.size() == 2) {
      MS_LOG(ERROR) << err_msg << "[" << resize_[0] << ", " << resize_[1] << "]";
    } else {
      MS_LOG(ERROR) << err_msg << "[" << resize_[0] << ", " << resize_[0] << "]";
    }
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (crop_.size() < resize_.size()) {
    if (crop_[0] > MIN(resize_[0], resize_[1])) {
      std::string err_msg =
        "Each value of crop parameter must be smaller than corresponding resize parameter, for example: x[0] <= "
        "y[0],  and x[1] <= y[1], please verify your input parameters.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  if (crop_.size() > resize_.size()) {
    if (MAX(crop_[0], crop_[1]) > resize_[0]) {
      std::string err_msg =
        "Each value of crop parameter must be smaller than corresponding resize parameter, for example: x[0] <= "
        "y[0],  and x[1] <= y[1], please verify your input parameters.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  if (crop_.size() == resize_.size()) {
    for (int32_t i = 0; i < crop_.size(); ++i) {
      if (crop_[i] > resize_[i]) {
        std::string err_msg =
          "Each value of crop parameter must be smaller than corresponding resize parameter, for example: x[0] <= "
          "y[0],  and x[1] <= y[1], please verify your input parameters.";
        MS_LOG(ERROR) << err_msg;
        RETURN_STATUS_SYNTAX_ERROR(err_msg);
      }
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> DvppDecodeResizeCropOperation::Build() {
  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  uint32_t cropHeight, cropWidth, resizeHeight, resizeWidth;
  if (crop_.size() == 1) {
    cropHeight = crop_[0];
    cropWidth = crop_[0];
  } else {
    cropHeight = crop_[0];
    cropWidth = crop_[1];
  }
  // User specified the width value.
  if (resize_.size() == 1) {
    resizeHeight = resize_[0];
    resizeWidth = resize_[0];
  } else {
    resizeHeight = resize_[0];
    resizeWidth = resize_[1];
  }
  std::shared_ptr<DvppDecodeResizeCropJpegOp> tensor_op =
    std::make_shared<DvppDecodeResizeCropJpegOp>(cropHeight, cropWidth, resizeHeight, resizeWidth);
  return tensor_op;
}
#endif

// NormalizeOperation
NormalizeOperation::NormalizeOperation(std::vector<float> mean, std::vector<float> std) : mean_(mean), std_(std) {}

Status NormalizeOperation::ValidateParams() {
  if (mean_.size() != 3) {
    std::string err_msg = "Normalize: mean vector has incorrect size: " + std::to_string(mean_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (std_.size() != 3) {
    std::string err_msg = "Normalize: std vector has incorrect size: " + std::to_string(std_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // check std/mean value
  for (int32_t i = 0; i < std_.size(); ++i) {
    if (std_[i] < 0.0f || std_[i] > 255.0f || CmpFloat(std_[i], 0.0f)) {
      std::string err_msg = "Normalize: std vector has incorrect value: " + std::to_string(std_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    if (mean_[i] < 0.0f || mean_[i] > 255.0f) {
      std::string err_msg = "Normalize: mean vector has incorrect value: " + std::to_string(mean_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> NormalizeOperation::Build() {
  return std::make_shared<NormalizeOp>(mean_[0], mean_[1], mean_[2], std_[0], std_[1], std_[2]);
}

Status NormalizeOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["mean"] = mean_;
  args["std"] = std_;
  *out_json = args;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
// NormalizePadOperation
NormalizePadOperation::NormalizePadOperation(const std::vector<float> &mean, const std::vector<float> &std,
                                             const std::string &dtype)
    : mean_(mean), std_(std), dtype_(dtype) {}

Status NormalizePadOperation::ValidateParams() {
  if (mean_.size() != 3) {
    std::string err_msg = "NormalizePad: mean vector has incorrect size: " + std::to_string(mean_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (std_.size() != 3) {
    std::string err_msg = "NormalizePad: std vector has incorrect size: " + std::to_string(std_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // check std/mean value
  for (int32_t i = 0; i < std_.size(); ++i) {
    if (std_[i] < 0.0f || std_[i] > 255.0f || CmpFloat(std_[i], 0.0f)) {
      std::string err_msg = "NormalizePad: std vector has incorrect value: " + std::to_string(std_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    if (mean_[i] < 0.0f || mean_[i] > 255.0f) {
      std::string err_msg = "NormalizePad: mean vector has incorrect value: " + std::to_string(mean_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  if (dtype_ != "float32" && dtype_ != "float16") {
    std::string err_msg = "NormalizePad: dtype must be float32 or float16, but got: " + dtype_;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> NormalizePadOperation::Build() {
  return std::make_shared<NormalizePadOp>(mean_[0], mean_[1], mean_[2], std_[0], std_[1], std_[2], dtype_);
}

// PadOperation
PadOperation::PadOperation(std::vector<int32_t> padding, std::vector<uint8_t> fill_value, BorderType padding_mode)
    : padding_(padding), fill_value_(fill_value), padding_mode_(padding_mode) {}

Status PadOperation::ValidateParams() {
  // padding
  RETURN_IF_NOT_OK(ValidateVectorPadding("Pad", padding_));
  // fill_value
  RETURN_IF_NOT_OK(ValidateVectorFillvalue("Pad", fill_value_));
  return Status::OK();
}

std::shared_ptr<TensorOp> PadOperation::Build() {
  int32_t pad_top, pad_bottom, pad_left, pad_right;
  switch (padding_.size()) {
    case 1:
      pad_left = padding_[0];
      pad_top = padding_[0];
      pad_right = padding_[0];
      pad_bottom = padding_[0];
      break;
    case 2:
      pad_left = padding_[0];
      pad_top = padding_[1];
      pad_right = padding_[0];
      pad_bottom = padding_[1];
      break;
    default:
      pad_left = padding_[0];
      pad_top = padding_[1];
      pad_right = padding_[2];
      pad_bottom = padding_[3];
  }
  uint8_t fill_r, fill_g, fill_b;

  fill_r = fill_value_[0];
  fill_g = fill_value_[0];
  fill_b = fill_value_[0];

  if (fill_value_.size() == 3) {
    fill_r = fill_value_[0];
    fill_g = fill_value_[1];
    fill_b = fill_value_[2];
  }

  std::shared_ptr<PadOp> tensor_op =
    std::make_shared<PadOp>(pad_top, pad_bottom, pad_left, pad_right, padding_mode_, fill_r, fill_g, fill_b);
  return tensor_op;
}

Status PadOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["padding"] = padding_;
  args["fill_value"] = fill_value_;
  args["padding_mode"] = padding_mode_;
  *out_json = args;
  return Status::OK();
}

// RandomAffineOperation
RandomAffineOperation::RandomAffineOperation(const std::vector<float_t> &degrees,
                                             const std::vector<float_t> &translate_range,
                                             const std::vector<float_t> &scale_range,
                                             const std::vector<float_t> &shear_ranges, InterpolationMode interpolation,
                                             const std::vector<uint8_t> &fill_value)
    : degrees_(degrees),
      translate_range_(translate_range),
      scale_range_(scale_range),
      shear_ranges_(shear_ranges),
      interpolation_(interpolation),
      fill_value_(fill_value) {
  random_op_ = true;
}

Status RandomAffineOperation::ValidateParams() {
  // Degrees
  if (degrees_.size() != 2) {
    std::string err_msg =
      "RandomAffine: degrees expecting size 2, got: degrees.size() = " + std::to_string(degrees_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (degrees_[0] > degrees_[1]) {
    std::string err_msg =
      "RandomAffine: minimum of degrees range is greater than maximum: min = " + std::to_string(degrees_[0]) +
      ", max = " + std::to_string(degrees_[1]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // Translate
  if (translate_range_.size() != 2 && translate_range_.size() != 4) {
    std::string err_msg = "RandomAffine: translate_range expecting size 2 or 4, got: translate_range.size() = " +
                          std::to_string(translate_range_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (translate_range_[0] > translate_range_[1]) {
    std::string err_msg = "RandomAffine: minimum of translate range on x is greater than maximum: min = " +
                          std::to_string(translate_range_[0]) + ", max = " + std::to_string(translate_range_[1]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (translate_range_[0] < -1 || translate_range_[0] > 1) {
    std::string err_msg = "RandomAffine: minimum of translate range on x is out of range of [-1, 1], value = " +
                          std::to_string(translate_range_[0]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (translate_range_[1] < -1 || translate_range_[1] > 1) {
    std::string err_msg = "RandomAffine: maximum of translate range on x is out of range of [-1, 1], value = " +
                          std::to_string(translate_range_[1]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (translate_range_.size() == 4) {
    if (translate_range_[2] > translate_range_[3]) {
      std::string err_msg = "RandomAffine: minimum of translate range on y is greater than maximum: min = " +
                            std::to_string(translate_range_[2]) + ", max = " + std::to_string(translate_range_[3]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    if (translate_range_[2] < -1 || translate_range_[2] > 1) {
      std::string err_msg = "RandomAffine: minimum of translate range on y is out of range of [-1, 1], value = " +
                            std::to_string(translate_range_[2]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    if (translate_range_[3] < -1 || translate_range_[3] > 1) {
      std::string err_msg = "RandomAffine: maximum of translate range on y is out of range of [-1, 1], value = " +
                            std::to_string(translate_range_[3]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  // Scale
  if (scale_range_.size() != 2) {
    std::string err_msg = "RandomAffine: scale_range vector has incorrect size: scale_range.size() = " +
                          std::to_string(scale_range_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (scale_range_[0] < 0) {
    std::string err_msg =
      "RandomAffine: min scale range must be greater than or equal to 0, got: " + std::to_string(scale_range_[0]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (scale_range_[1] <= 0) {
    std::string err_msg =
      "RandomAffine: max scale range must be greater than 0, got: " + std::to_string(scale_range_[1]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (scale_range_[0] > scale_range_[1]) {
    std::string err_msg =
      "RandomAffine: minimum of scale range is greater than maximum: min = " + std::to_string(scale_range_[0]) +
      ", max = " + std::to_string(scale_range_[1]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // Shear
  if (shear_ranges_.size() != 2 && shear_ranges_.size() != 4) {
    std::string err_msg = "RandomAffine: shear_ranges expecting size 2 or 4, got: shear_ranges.size() = " +
                          std::to_string(shear_ranges_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (shear_ranges_[0] > shear_ranges_[1]) {
    std::string err_msg = "RandomAffine: minimum of horizontal shear range is greater than maximum: min = " +
                          std::to_string(shear_ranges_[0]) + ", max = " + std::to_string(shear_ranges_[1]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (shear_ranges_.size() == 4 && shear_ranges_[2] > shear_ranges_[3]) {
    std::string err_msg = "RandomAffine: minimum of vertical shear range is greater than maximum: min = " +
                          std::to_string(shear_ranges_[2]) + ", max = " + std::to_string(scale_range_[3]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // Fill Value
  if (fill_value_.size() != 3) {
    std::string err_msg =
      "RandomAffine: fill_value vector has incorrect size: fill_value.size() = " + std::to_string(fill_value_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (int32_t i = 0; i < fill_value_.size(); ++i) {
    if (fill_value_[i] < 0 || fill_value_[i] > 255) {
      std::string err_msg =
        "RandomAffine: fill_value has to be between 0 and 255, got:" + std::to_string(fill_value_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomAffineOperation::Build() {
  if (shear_ranges_.size() == 2) {
    shear_ranges_.resize(4);
  }
  if (translate_range_.size() == 2) {
    translate_range_.resize(4);
  }
  auto tensor_op = std::make_shared<RandomAffineOp>(degrees_, translate_range_, scale_range_, shear_ranges_,
                                                    interpolation_, fill_value_);
  return tensor_op;
}

// RandomColorOperation.
RandomColorOperation::RandomColorOperation(float t_lb, float t_ub) : t_lb_(t_lb), t_ub_(t_ub) { random_op_ = true; }

Status RandomColorOperation::ValidateParams() {
  // Do some input validation.
  if (t_lb_ < 0 || t_ub_ < 0) {
    std::string err_msg =
      "RandomColor: lower bound or upper bound must be greater than or equal to 0, got t_lb: " + std::to_string(t_lb_) +
      ", t_ub: " + std::to_string(t_ub_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (t_lb_ > t_ub_) {
    std::string err_msg =
      "RandomColor: lower bound must be less or equal to upper bound, got t_lb: " + std::to_string(t_lb_) +
      ", t_ub: " + std::to_string(t_ub_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

// RandomColorAdjustOperation.
RandomColorAdjustOperation::RandomColorAdjustOperation(std::vector<float> brightness, std::vector<float> contrast,
                                                       std::vector<float> saturation, std::vector<float> hue)
    : brightness_(brightness), contrast_(contrast), saturation_(saturation), hue_(hue) {
  random_op_ = true;
}

Status RandomColorAdjustOperation::ValidateParams() {
  // brightness
  if (brightness_.empty() || brightness_.size() > 2) {
    std::string err_msg =
      "RandomColorAdjust: brightness must be a vector of one or two values, got: " + std::to_string(brightness_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (int32_t i = 0; i < brightness_.size(); ++i) {
    if (brightness_[i] < 0) {
      std::string err_msg =
        "RandomColorAdjust: brightness must be greater than or equal to 0, got: " + std::to_string(brightness_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  if (brightness_.size() == 2 && (brightness_[0] > brightness_[1])) {
    std::string err_msg = "RandomColorAdjust: brightness lower bound must be less or equal to upper bound, got lb: " +
                          std::to_string(brightness_[0]) + ", ub: " + std::to_string(brightness_[1]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // contrast
  if (contrast_.empty() || contrast_.size() > 2) {
    std::string err_msg =
      "RandomColorAdjust: contrast must be a vector of one or two values, got: " + std::to_string(contrast_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (int32_t i = 0; i < contrast_.size(); ++i) {
    if (contrast_[i] < 0) {
      std::string err_msg =
        "RandomColorAdjust: contrast must be greater than or equal to 0, got: " + std::to_string(contrast_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  if (contrast_.size() == 2 && (contrast_[0] > contrast_[1])) {
    std::string err_msg = "RandomColorAdjust: contrast lower bound must be less or equal to upper bound, got lb: " +
                          std::to_string(contrast_[0]) + ", ub: " + std::to_string(contrast_[1]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // saturation
  if (saturation_.empty() || saturation_.size() > 2) {
    std::string err_msg =
      "RandomColorAdjust: saturation must be a vector of one or two values, got: " + std::to_string(saturation_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (int32_t i = 0; i < saturation_.size(); ++i) {
    if (saturation_[i] < 0) {
      std::string err_msg =
        "RandomColorAdjust: saturation must be greater than or equal to 0, got: " + std::to_string(saturation_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  if (saturation_.size() == 2 && (saturation_[0] > saturation_[1])) {
    std::string err_msg = "RandomColorAdjust: saturation lower bound must be less or equal to upper bound, got lb: " +
                          std::to_string(saturation_[0]) + ", ub: " + std::to_string(saturation_[1]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // hue
  if (hue_.empty() || hue_.size() > 2) {
    std::string err_msg =
      "RandomColorAdjust: hue must be a vector of one or two values, got: " + std::to_string(hue_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (int32_t i = 0; i < hue_.size(); ++i) {
    if (hue_[i] < -0.5 || hue_[i] > 0.5) {
      std::string err_msg = "RandomColorAdjust: hue has to be between -0.5 and 0.5, got: " + std::to_string(hue_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  if (hue_.size() == 2 && (hue_[0] > hue_[1])) {
    std::string err_msg =
      "RandomColorAdjust: hue lower bound must be less or equal to upper bound, got lb: " + std::to_string(hue_[0]) +
      ", ub: " + std::to_string(hue_[1]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomColorAdjustOperation::Build() {
  float brightness_lb, brightness_ub, contrast_lb, contrast_ub, saturation_lb, saturation_ub, hue_lb, hue_ub;

  brightness_lb = brightness_[0];
  brightness_ub = brightness_[0];

  if (brightness_.size() == 2) brightness_ub = brightness_[1];

  contrast_lb = contrast_[0];
  contrast_ub = contrast_[0];

  if (contrast_.size() == 2) contrast_ub = contrast_[1];

  saturation_lb = saturation_[0];
  saturation_ub = saturation_[0];

  if (saturation_.size() == 2) saturation_ub = saturation_[1];

  hue_lb = hue_[0];
  hue_ub = hue_[0];

  if (hue_.size() == 2) hue_ub = hue_[1];

  std::shared_ptr<RandomColorAdjustOp> tensor_op = std::make_shared<RandomColorAdjustOp>(
    brightness_lb, brightness_ub, contrast_lb, contrast_ub, saturation_lb, saturation_ub, hue_lb, hue_ub);
  return tensor_op;
}

Status RandomColorAdjustOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["brightness"] = brightness_;
  args["contrast"] = contrast_;
  args["saturation"] = saturation_;
  args["hue"] = hue_;
  *out_json = args;
  return Status::OK();
}

// RandomCropOperation
RandomCropOperation::RandomCropOperation(std::vector<int32_t> size, std::vector<int32_t> padding, bool pad_if_needed,
                                         std::vector<uint8_t> fill_value, BorderType padding_mode)
    : TensorOperation(true),
      size_(size),
      padding_(padding),
      pad_if_needed_(pad_if_needed),
      fill_value_(fill_value),
      padding_mode_(padding_mode) {
  random_op_ = true;
}

Status RandomCropOperation::ValidateParams() {
  // size
  if (size_.empty() || size_.size() > 2) {
    std::string err_msg = "RandomCrop: size must be a vector of one or two values";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  RETURN_IF_NOT_OK(ValidateVectorPositive("RandomCrop", size_));
  // padding
  RETURN_IF_NOT_OK(ValidateVectorPadding("RandomCrop", padding_));
  // fill_value
  RETURN_IF_NOT_OK(ValidateVectorFillvalue("RandomCrop", fill_value_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomCropOperation::Build() {
  int32_t crop_height = size_[0];
  int32_t crop_width = size_[0];

  // User has specified the crop_width value.
  if (size_.size() == 2) {
    crop_width = size_[1];
  }

  int32_t pad_top, pad_bottom, pad_left, pad_right;
  switch (padding_.size()) {
    case 1:
      pad_left = padding_[0];
      pad_top = padding_[0];
      pad_right = padding_[0];
      pad_bottom = padding_[0];
      break;
    case 2:
      pad_left = padding_[0];
      pad_top = padding_[1];
      pad_right = padding_[0];
      pad_bottom = padding_[1];
      break;
    default:
      pad_left = padding_[0];
      pad_top = padding_[1];
      pad_right = padding_[2];
      pad_bottom = padding_[3];
  }

  uint8_t fill_r, fill_g, fill_b;
  fill_r = fill_value_[0];
  fill_g = fill_value_[0];
  fill_b = fill_value_[0];

  if (fill_value_.size() == 3) {
    fill_r = fill_value_[0];
    fill_g = fill_value_[1];
    fill_b = fill_value_[2];
  }

  auto tensor_op = std::make_shared<RandomCropOp>(crop_height, crop_width, pad_top, pad_bottom, pad_left, pad_right,
                                                  padding_mode_, pad_if_needed_, fill_r, fill_g, fill_b);
  return tensor_op;
}

Status RandomCropOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["size"] = size_;
  args["padding"] = padding_;
  args["pad_if_needed"] = pad_if_needed_;
  args["fill_value"] = fill_value_;
  args["padding_mode"] = padding_mode_;
  *out_json = args;
  return Status::OK();
}

// RandomCropDecodeResizeOperation
RandomCropDecodeResizeOperation::RandomCropDecodeResizeOperation(std::vector<int32_t> size, std::vector<float> scale,
                                                                 std::vector<float> ratio,
                                                                 InterpolationMode interpolation, int32_t max_attempts)
    : RandomResizedCropOperation(size, scale, ratio, interpolation, max_attempts) {}

std::shared_ptr<TensorOp> RandomCropDecodeResizeOperation::Build() {
  int32_t crop_height = size_[0];
  int32_t crop_width = size_[0];

  // User has specified the crop_width value.
  if (size_.size() == 2) {
    crop_width = size_[1];
  }

  float scale_lower_bound = scale_[0];
  float scale_upper_bound = scale_[1];

  float aspect_lower_bound = ratio_[0];
  float aspect_upper_bound = ratio_[1];

  auto tensor_op =
    std::make_shared<RandomCropDecodeResizeOp>(crop_height, crop_width, scale_lower_bound, scale_upper_bound,
                                               aspect_lower_bound, aspect_upper_bound, interpolation_, max_attempts_);
  return tensor_op;
}

RandomCropDecodeResizeOperation::RandomCropDecodeResizeOperation(const RandomResizedCropOperation &base)
    : RandomResizedCropOperation(base) {}

// RandomCropWithBBoxOperation
RandomCropWithBBoxOperation::RandomCropWithBBoxOperation(std::vector<int32_t> size, std::vector<int32_t> padding,
                                                         bool pad_if_needed, std::vector<uint8_t> fill_value,
                                                         BorderType padding_mode)
    : TensorOperation(true),
      size_(size),
      padding_(padding),
      pad_if_needed_(pad_if_needed),
      fill_value_(fill_value),
      padding_mode_(padding_mode) {}

Status RandomCropWithBBoxOperation::ValidateParams() {
  // size
  if (size_.empty() || size_.size() > 2) {
    std::string err_msg = "RandomCropWithBBox: size must be a vector of one or two values";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  RETURN_IF_NOT_OK(ValidateVectorPositive("RandomCropWithBBox", size_));
  // padding
  RETURN_IF_NOT_OK(ValidateVectorPadding("RandomCropWithBBox", padding_));
  // fill_value
  RETURN_IF_NOT_OK(ValidateVectorFillvalue("RandomCropWithBBox", fill_value_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomCropWithBBoxOperation::Build() {
  int32_t crop_height = size_[0];
  int32_t crop_width = size_[0];

  // User has specified the crop_width value.
  if (size_.size() == 2) {
    crop_width = size_[1];
  }

  int32_t pad_top, pad_bottom, pad_left, pad_right;
  switch (padding_.size()) {
    case 1:
      pad_left = padding_[0];
      pad_top = padding_[0];
      pad_right = padding_[0];
      pad_bottom = padding_[0];
      break;
    case 2:
      pad_left = padding_[0];
      pad_top = padding_[1];
      pad_right = padding_[0];
      pad_bottom = padding_[1];
      break;
    default:
      pad_left = padding_[0];
      pad_top = padding_[1];
      pad_right = padding_[2];
      pad_bottom = padding_[3];
  }

  uint8_t fill_r, fill_g, fill_b;
  fill_r = fill_value_[0];
  fill_g = fill_value_[0];
  fill_b = fill_value_[0];

  if (fill_value_.size() == 3) {
    fill_r = fill_value_[0];
    fill_g = fill_value_[1];
    fill_b = fill_value_[2];
  }

  auto tensor_op =
    std::make_shared<RandomCropWithBBoxOp>(crop_height, crop_width, pad_top, pad_bottom, pad_left, pad_right,
                                           padding_mode_, pad_if_needed_, fill_r, fill_g, fill_b);
  return tensor_op;
}

// RandomHorizontalFlipOperation
RandomHorizontalFlipOperation::RandomHorizontalFlipOperation(float probability)
    : TensorOperation(true), probability_(probability) {}

Status RandomHorizontalFlipOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateProbability("RandomHorizontalFlip", probability_));

  return Status::OK();
}

std::shared_ptr<TensorOp> RandomHorizontalFlipOperation::Build() {
  std::shared_ptr<RandomHorizontalFlipOp> tensor_op = std::make_shared<RandomHorizontalFlipOp>(probability_);
  return tensor_op;
}

// RandomHorizontalFlipWithBBoxOperation
RandomHorizontalFlipWithBBoxOperation::RandomHorizontalFlipWithBBoxOperation(float probability)
    : TensorOperation(true), probability_(probability) {}

Status RandomHorizontalFlipWithBBoxOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateProbability("RandomHorizontalFlipWithBBox", probability_));

  return Status::OK();
}

std::shared_ptr<TensorOp> RandomHorizontalFlipWithBBoxOperation::Build() {
  std::shared_ptr<RandomHorizontalFlipWithBBoxOp> tensor_op =
    std::make_shared<RandomHorizontalFlipWithBBoxOp>(probability_);
  return tensor_op;
}

// RandomPosterizeOperation
RandomPosterizeOperation::RandomPosterizeOperation(const std::vector<uint8_t> &bit_range)
    : TensorOperation(true), bit_range_(bit_range) {}

Status RandomPosterizeOperation::ValidateParams() {
  if (bit_range_.size() != 2) {
    std::string err_msg =
      "RandomPosterize: bit_range needs to be of size 2 but is of size: " + std::to_string(bit_range_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (bit_range_[0] < 1 || bit_range_[0] > 8) {
    std::string err_msg = "RandomPosterize: min_bit value is out of range [1-8]: " + std::to_string(bit_range_[0]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (bit_range_[1] < 1 || bit_range_[1] > 8) {
    std::string err_msg = "RandomPosterize: max_bit value is out of range [1-8]: " + std::to_string(bit_range_[1]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (bit_range_[1] < bit_range_[0]) {
    std::string err_msg = "RandomPosterize: max_bit value is less than min_bit: max =" + std::to_string(bit_range_[1]) +
                          ", min = " + std::to_string(bit_range_[0]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomPosterizeOperation::Build() {
  std::shared_ptr<RandomPosterizeOp> tensor_op = std::make_shared<RandomPosterizeOp>(bit_range_);
  return tensor_op;
}

// RandomResizeOperation
RandomResizeOperation::RandomResizeOperation(std::vector<int32_t> size) : TensorOperation(true), size_(size) {}

Status RandomResizeOperation::ValidateParams() {
  // size
  if (size_.size() != 2 && size_.size() != 1) {
    std::string err_msg =
      "RandomResize: size must be a vector of one or two values, got: " + std::to_string(size_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (size_[0] <= 0 || (size_.size() == 2 && size_[1] <= 0)) {
    std::string err_msg = "RandomResize: size must only contain positive integers.";
    MS_LOG(ERROR) << "RandomResize: size must only contain positive integers, got: " << size_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomResizeOperation::Build() {
  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  int32_t height = size_[0];
  int32_t width = 0;

  // User specified the width value.
  if (size_.size() == 2) {
    width = size_[1];
  }

  std::shared_ptr<RandomResizeOp> tensor_op = std::make_shared<RandomResizeOp>(height, width);
  return tensor_op;
}

// RandomResizeWithBBoxOperation
RandomResizeWithBBoxOperation::RandomResizeWithBBoxOperation(std::vector<int32_t> size)
    : TensorOperation(true), size_(size) {}

Status RandomResizeWithBBoxOperation::ValidateParams() {
  // size
  if (size_.size() != 2 && size_.size() != 1) {
    std::string err_msg =
      "RandomResizeWithBBox: size must be a vector of one or two values, got: " + std::to_string(size_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (size_[0] <= 0 || (size_.size() == 2 && size_[1] <= 0)) {
    std::string err_msg = "RandomResizeWithBBox: size must only contain positive integers.";
    MS_LOG(ERROR) << "RandomResizeWithBBox: size must only contain positive integers, got: " << size_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomResizeWithBBoxOperation::Build() {
  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  int32_t height = size_[0];
  int32_t width = 0;

  // User specified the width value.
  if (size_.size() == 2) {
    width = size_[1];
  }

  std::shared_ptr<RandomResizeWithBBoxOp> tensor_op = std::make_shared<RandomResizeWithBBoxOp>(height, width);
  return tensor_op;
}

// RandomResizedCropOperation
RandomResizedCropOperation::RandomResizedCropOperation(std::vector<int32_t> size, std::vector<float> scale,
                                                       std::vector<float> ratio, InterpolationMode interpolation,
                                                       int32_t max_attempts)
    : TensorOperation(true),
      size_(size),
      scale_(scale),
      ratio_(ratio),
      interpolation_(interpolation),
      max_attempts_(max_attempts) {}

Status RandomResizedCropOperation::ValidateParams() {
  // size
  if (size_.size() != 2 && size_.size() != 1) {
    std::string err_msg = Name() + ": size must be a vector of one or two values, got: " + std::to_string(size_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (size_[0] <= 0 || (size_.size() == 2 && size_[1] <= 0)) {
    std::string err_msg = Name() + ": size must only contain positive integers.";
    MS_LOG(ERROR) << Name() + ": size must only contain positive integers, got: " << size_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // scale
  if (scale_.size() != 2) {
    std::string err_msg = Name() + ": scale must be a vector of two values, got: " + std::to_string(scale_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (scale_[0] < 0) {
    std::string err_msg = Name() + ": min scale must be greater than or equal to 0.";
    MS_LOG(ERROR) << Name() + ": min scale must be greater than or equal to 0, got: " + std::to_string(scale_[0]);
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (scale_[1] <= 0) {
    std::string err_msg = Name() + ": max scale must be greater than 0.";
    MS_LOG(ERROR) << Name() + ": max scale must be greater than 0, got: " + std::to_string(scale_[1]);
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (scale_[1] < scale_[0]) {
    std::string err_msg = Name() + ": scale must have a size of two in the format of (min, max).";
    MS_LOG(ERROR) << Name() + ": scale must have a size of two in the format of (min, max), but got: " << scale_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // ratio
  if (ratio_.size() != 2) {
    std::string err_msg = Name() + ": ratio must be a vector of two values, got: " + std::to_string(ratio_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (ratio_[0] <= 0 || ratio_[1] <= 0) {
    std::string err_msg = Name() + ": ratio must be greater than 0.";
    MS_LOG(ERROR) << Name() + ": ratio must be greater than 0, got: " << ratio_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (ratio_[1] < ratio_[0]) {
    std::string err_msg = Name() + ": ratio must have a size of two in the format of (min, max).";
    MS_LOG(ERROR) << Name() + ": ratio must have a size of two in the format of (min, max), but got: " << ratio_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // max_attempts
  if (max_attempts_ < 1) {
    std::string err_msg =
      Name() + ": max_attempts must be greater than or equal to 1, got: " + std::to_string(max_attempts_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomResizedCropOperation::Build() {
  int32_t height = size_[0];
  int32_t width = size_[0];
  // User specified the width value.
  if (size_.size() == 2) {
    width = size_[1];
  }
  std::shared_ptr<RandomCropAndResizeOp> tensor_op = std::make_shared<RandomCropAndResizeOp>(
    height, width, scale_[0], scale_[1], ratio_[0], ratio_[1], interpolation_, max_attempts_);
  return tensor_op;
}

// RandomResizedCropWithBBoxOperation
RandomResizedCropWithBBoxOperation::RandomResizedCropWithBBoxOperation(std::vector<int32_t> size,
                                                                       std::vector<float> scale,
                                                                       std::vector<float> ratio,
                                                                       InterpolationMode interpolation,
                                                                       int32_t max_attempts)
    : size_(size), scale_(scale), ratio_(ratio), interpolation_(interpolation), max_attempts_(max_attempts) {}

Status RandomResizedCropWithBBoxOperation::ValidateParams() {
  // size
  if (size_.size() != 2 && size_.size() != 1) {
    std::string err_msg =
      "RandomResizedCropWithBBox: size must be a vector of one or two values, got: " + std::to_string(size_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (size_[0] <= 0 || (size_.size() == 2 && size_[1] <= 0)) {
    std::string err_msg = "RandomResizedCropWithBBox: size must only contain positive integers.";
    MS_LOG(ERROR) << "RandomResizedCropWithBBox: size must only contain positive integers, got: " << size_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // scale
  if (scale_.size() != 2) {
    std::string err_msg =
      "RandomResizedCropWithBBox: scale must be a vector of two values, got: " + std::to_string(scale_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (scale_[0] < 0) {
    std::string err_msg = "RandomResizedCropWithBBox: min scale must be greater than or equal to 0.";
    MS_LOG(ERROR) << "RandomResizedCropWithBBox: min scale must be greater than or equal to 0, got: " +
                       std::to_string(scale_[0]);
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (scale_[1] <= 0) {
    std::string err_msg = "RandomResizedCropWithBBox: max scale must be greater than 0.";
    MS_LOG(ERROR) << "RandomResizedCropWithBBox: max scale must be greater than 0, got: " + std::to_string(scale_[1]);
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (scale_[1] < scale_[0]) {
    std::string err_msg = "RandomResizedCropWithBBox: scale must have a size of two in the format of (min, max).";
    MS_LOG(ERROR) << "RandomResizedCropWithBBox: scale must have a size of two in the format of (min, max), but got: "
                  << scale_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // ratio
  if (ratio_.size() != 2) {
    std::string err_msg =
      "RandomResizedCropWithBBox: ratio must be a vector of two values, got: " + std::to_string(ratio_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (ratio_[0] <= 0 || ratio_[1] <= 0) {
    std::string err_msg = "RandomResizedCropWithBBox: ratio must be greater than 0.";
    MS_LOG(ERROR) << "RandomResizedCropWithBBox: ratio must be greater than 0, got: " << ratio_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (ratio_[1] < ratio_[0]) {
    std::string err_msg = "RandomResizedCropWithBBox: ratio must have a size of two in the format of (min, max).";
    MS_LOG(ERROR) << "RandomResizedCropWithBBox: ratio must have a size of two in the format of (min, max), but got: "
                  << ratio_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // max_attempts
  if (max_attempts_ < 1) {
    std::string err_msg = "RandomResizedCropWithBBox: max_attempts must be greater than or equal to 1, got: " +
                          std::to_string(max_attempts_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomResizedCropWithBBoxOperation::Build() {
  int32_t height = size_[0];
  int32_t width = size_[0];
  // User specified the width value.
  if (size_.size() == 2) {
    width = size_[1];
  }
  std::shared_ptr<RandomCropAndResizeWithBBoxOp> tensor_op = std::make_shared<RandomCropAndResizeWithBBoxOp>(
    height, width, scale_[0], scale_[1], ratio_[0], ratio_[1], interpolation_, max_attempts_);
  return tensor_op;
}

// Function to create RandomRotationOperation.
RandomRotationOperation::RandomRotationOperation(std::vector<float> degrees, InterpolationMode interpolation_mode,
                                                 bool expand, std::vector<float> center,
                                                 std::vector<uint8_t> fill_value)
    : TensorOperation(true),
      degrees_(degrees),
      interpolation_mode_(interpolation_mode),
      expand_(expand),
      center_(center),
      fill_value_(fill_value) {}

Status RandomRotationOperation::ValidateParams() {
  // degrees
  if (degrees_.size() != 2 && degrees_.size() != 1) {
    std::string err_msg =
      "RandomRotation: degrees must be a vector of one or two values, got: " + std::to_string(degrees_.size());
    MS_LOG(ERROR) << "RandomRotation: degrees must be a vector of one or two values, got: " << degrees_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if ((degrees_[1] < degrees_[0]) && (degrees_.size() == 2)) {
    std::string err_msg = "RandomRotation: degrees must be in the format of (min, max), got: (" +
                          std::to_string(degrees_[0]) + ", " + std::to_string(degrees_[1]) + ")";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  } else if ((degrees_[0] < 0) && degrees_.size() == 1) {
    std::string err_msg =
      "RandomRotation: if degrees only has one value, it must be greater than or equal to 0, got: " +
      std::to_string(degrees_[0]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // center
  if (center_.empty() || center_.size() != 2) {
    std::string err_msg =
      "RandomRotation: center must be a vector of two values, got: " + std::to_string(center_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // fill_value
  RETURN_IF_NOT_OK(ValidateVectorFillvalue("RandomRotation", fill_value_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomRotationOperation::Build() {
  float start_degree, end_degree;
  if (degrees_.size() == 1) {
    start_degree = -degrees_[0];
    end_degree = degrees_[0];
  } else if (degrees_.size() == 2) {
    start_degree = degrees_[0];
    end_degree = degrees_[1];
  }

  uint8_t fill_r, fill_g, fill_b;
  fill_r = fill_value_[0];
  fill_g = fill_value_[0];
  fill_b = fill_value_[0];

  if (fill_value_.size() == 3) {
    fill_r = fill_value_[0];
    fill_g = fill_value_[1];
    fill_b = fill_value_[2];
  }

  std::shared_ptr<RandomRotationOp> tensor_op = std::make_shared<RandomRotationOp>(
    start_degree, end_degree, center_[0], center_[1], interpolation_mode_, expand_, fill_r, fill_g, fill_b);
  return tensor_op;
}

Status RandomRotationOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["degrees"] = degrees_;
  args["interpolation_mode"] = interpolation_mode_;
  args["expand"] = expand_;
  args["center"] = center_;
  args["fill_value"] = fill_value_;
  *out_json = args;
  return Status::OK();
}

// RandomSelectSubpolicyOperation.
RandomSelectSubpolicyOperation::RandomSelectSubpolicyOperation(
  std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> policy)
    : TensorOperation(true), policy_(policy) {}

Status RandomSelectSubpolicyOperation::ValidateParams() {
  if (policy_.empty()) {
    std::string err_msg = "RandomSelectSubpolicy: policy must not be empty";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (int32_t i = 0; i < policy_.size(); i++) {
    if (policy_[i].empty()) {
      std::string err_msg = "RandomSelectSubpolicy: policy[" + std::to_string(i) + "] must not be empty";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    for (int32_t j = 0; j < policy_[i].size(); j++) {
      if (policy_[i][j].first == nullptr) {
        std::string transform_pos = "[" + std::to_string(i) + "]" + "[" + std::to_string(j) + "]";
        std::string err_msg = "RandomSelectSubpolicy: transform in policy" + transform_pos + " must not be null";
        MS_LOG(ERROR) << err_msg;
        RETURN_STATUS_SYNTAX_ERROR(err_msg);
      }
      if (policy_[i][j].second < 0.0 || policy_[i][j].second > 1.0) {
        std::string transform_pos = "[" + std::to_string(i) + "]" + "[" + std::to_string(j) + "]";
        std::string err_msg = "RandomSelectSubpolicy: probability of transform in policy" + transform_pos +
                              " must be between 0.0 and 1.0, got: " + std::to_string(policy_[i][j].second);
        MS_LOG(ERROR) << err_msg;
        RETURN_STATUS_SYNTAX_ERROR(err_msg);
      }
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomSelectSubpolicyOperation::Build() {
  std::vector<Subpolicy> policy_tensor_ops;
  for (int32_t i = 0; i < policy_.size(); i++) {
    Subpolicy sub_policy_tensor_ops;
    for (int32_t j = 0; j < policy_[i].size(); j++) {
      sub_policy_tensor_ops.push_back(std::make_pair(policy_[i][j].first->Build(), policy_[i][j].second));
    }
    policy_tensor_ops.push_back(sub_policy_tensor_ops);
  }
  std::shared_ptr<RandomSelectSubpolicyOp> tensor_op = std::make_shared<RandomSelectSubpolicyOp>(policy_tensor_ops);
  return tensor_op;
}

// Function to create RandomSharpness.
RandomSharpnessOperation::RandomSharpnessOperation(std::vector<float> degrees)
    : TensorOperation(true), degrees_(degrees) {}

Status RandomSharpnessOperation::ValidateParams() {
  if (degrees_.size() != 2 || degrees_[0] < 0 || degrees_[1] < 0) {
    std::string err_msg = "RandomSharpness: degrees must be a vector of two values and greater than or equal to 0.";
    MS_LOG(ERROR) << "RandomSharpness: degrees must be a vector of two values and greater than or equal to 0, got: "
                  << degrees_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (degrees_[1] < degrees_[0]) {
    std::string err_msg = "RandomSharpness: degrees must be in the format of (min, max).";
    MS_LOG(ERROR) << "RandomSharpness: degrees must be in the format of (min, max), got: " << degrees_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> RandomSharpnessOperation::Build() {
  std::shared_ptr<RandomSharpnessOp> tensor_op = std::make_shared<RandomSharpnessOp>(degrees_[0], degrees_[1]);
  return tensor_op;
}

// RandomSolarizeOperation.
RandomSolarizeOperation::RandomSolarizeOperation(std::vector<uint8_t> threshold)
    : TensorOperation(true), threshold_(threshold) {}

Status RandomSolarizeOperation::ValidateParams() {
  if (threshold_.size() != 2) {
    std::string err_msg =
      "RandomSolarize: threshold must be a vector of two values, got: " + std::to_string(threshold_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (int32_t i = 0; i < threshold_.size(); ++i) {
    if (threshold_[i] < 0 || threshold_[i] > 255) {
      std::string err_msg =
        "RandomSolarize: threshold has to be between 0 and 255, got:" + std::to_string(threshold_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  if (threshold_[0] > threshold_[1]) {
    std::string err_msg = "RandomSolarize: threshold must be passed in a (min, max) format";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomSolarizeOperation::Build() {
  std::shared_ptr<RandomSolarizeOp> tensor_op = std::make_shared<RandomSolarizeOp>(threshold_);
  return tensor_op;
}

// RandomVerticalFlipOperation
RandomVerticalFlipOperation::RandomVerticalFlipOperation(float probability)
    : TensorOperation(true), probability_(probability) {}

Status RandomVerticalFlipOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateProbability("RandomVerticalFlip", probability_));

  return Status::OK();
}

std::shared_ptr<TensorOp> RandomVerticalFlipOperation::Build() {
  std::shared_ptr<RandomVerticalFlipOp> tensor_op = std::make_shared<RandomVerticalFlipOp>(probability_);
  return tensor_op;
}

// RandomVerticalFlipWithBBoxOperation
RandomVerticalFlipWithBBoxOperation::RandomVerticalFlipWithBBoxOperation(float probability)
    : TensorOperation(true), probability_(probability) {}

Status RandomVerticalFlipWithBBoxOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateProbability("RandomVerticalFlipWithBBox", probability_));

  return Status::OK();
}

std::shared_ptr<TensorOp> RandomVerticalFlipWithBBoxOperation::Build() {
  std::shared_ptr<RandomVerticalFlipWithBBoxOp> tensor_op =
    std::make_shared<RandomVerticalFlipWithBBoxOp>(probability_);
  return tensor_op;
}

// RescaleOperation
RescaleOperation::RescaleOperation(float rescale, float shift) : rescale_(rescale), shift_(shift) {}

Status RescaleOperation::ValidateParams() {
  if (rescale_ < 0) {
    std::string err_msg = "Rescale: rescale must be greater than or equal to 0, got: " + std::to_string(rescale_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RescaleOperation::Build() {
  std::shared_ptr<RescaleOp> tensor_op = std::make_shared<RescaleOp>(rescale_, shift_);
  return tensor_op;
}

Status RescaleOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["rescale"] = rescale_;
  args["shift"] = shift_;
  *out_json = args;
  return Status::OK();
}

#endif
// ResizeOperation
ResizeOperation::ResizeOperation(std::vector<int32_t> size, InterpolationMode interpolation)
    : size_(size), interpolation_(interpolation) {}

Status ResizeOperation::ValidateParams() {
  // size
  if (size_.empty() || size_.size() > 2) {
    std::string err_msg = "Resize: size must be a vector of one or two values, got: " + std::to_string(size_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  RETURN_IF_NOT_OK(ValidateVectorPositive("Resize", size_));

  return Status::OK();
}

std::shared_ptr<TensorOp> ResizeOperation::Build() {
  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  int32_t height = size_[0];
  int32_t width = 0;

  // User specified the width value.
  if (size_.size() == 2) {
    width = size_[1];
  }

  return std::make_shared<ResizeOp>(height, width, interpolation_);
}

Status ResizeOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["size"] = size_;
  args["interpolation"] = interpolation_;
  *out_json = args;
  return Status::OK();
}

// RotateOperation
RotateOperation::RotateOperation() { rotate_op = std::make_shared<RotateOp>(0); }

Status RotateOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> RotateOperation::Build() { return rotate_op; }

void RotateOperation::setAngle(uint64_t angle_id) {
  std::dynamic_pointer_cast<RotateOp>(rotate_op)->setAngle(angle_id);
}

#ifndef ENABLE_ANDROID
// ResizeWithBBoxOperation
ResizeWithBBoxOperation::ResizeWithBBoxOperation(std::vector<int32_t> size, InterpolationMode interpolation)
    : size_(size), interpolation_(interpolation) {}

Status ResizeWithBBoxOperation::ValidateParams() {
  // size
  if (size_.empty() || size_.size() > 2) {
    std::string err_msg =
      "ResizeWithBBox: size must be a vector of one or two values, got: " + std::to_string(size_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  RETURN_IF_NOT_OK(ValidateVectorPositive("Resize", size_));

  return Status::OK();
}

std::shared_ptr<TensorOp> ResizeWithBBoxOperation::Build() {
  int32_t height = size_[0];
  int32_t width = 0;

  // User specified the width value.
  if (size_.size() == 2) {
    width = size_[1];
  }

  return std::make_shared<ResizeWithBBoxOp>(height, width, interpolation_);
}

// RgbaToBgrOperation.
RgbaToBgrOperation::RgbaToBgrOperation() {}

Status RgbaToBgrOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> RgbaToBgrOperation::Build() {
  std::shared_ptr<RgbaToBgrOp> tensor_op = std::make_shared<RgbaToBgrOp>();
  return tensor_op;
}

// RgbaToRgbOperation.
RgbaToRgbOperation::RgbaToRgbOperation() {}

Status RgbaToRgbOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> RgbaToRgbOperation::Build() {
  std::shared_ptr<RgbaToRgbOp> tensor_op = std::make_shared<RgbaToRgbOp>();
  return tensor_op;
}

// SoftDvppDecodeRandomCropResizeJpegOperation
SoftDvppDecodeRandomCropResizeJpegOperation::SoftDvppDecodeRandomCropResizeJpegOperation(std::vector<int32_t> size,
                                                                                         std::vector<float> scale,
                                                                                         std::vector<float> ratio,
                                                                                         int32_t max_attempts)
    : size_(size), scale_(scale), ratio_(ratio), max_attempts_(max_attempts) {}

Status SoftDvppDecodeRandomCropResizeJpegOperation::ValidateParams() {
  // size
  if (size_.size() != 2 && size_.size() != 1) {
    std::string err_msg = "SoftDvppDecodeRandomCropResizeJpeg: size must be a vector of one or two values, got: " +
                          std::to_string(size_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (size_[0] <= 0 || (size_.size() == 2 && size_[1] <= 0)) {
    std::string err_msg = "SoftDvppDecodeRandomCropResizeJpeg: size must only contain positive integers.";
    MS_LOG(ERROR) << "SoftDvppDecodeRandomCropResizeJpeg: size must only contain positive integers, got: " << size_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // scale
  if (scale_.size() != 2) {
    std::string err_msg =
      "SoftDvppDecodeRandomCropResizeJpeg: scale must be a vector of two values, got: " + std::to_string(scale_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (scale_[0] < 0) {
    std::string err_msg = "SoftDvppDecodeRandomCropResizeJpeg: min scale must be greater than or equal to 0.";
    MS_LOG(ERROR) << "SoftDvppDecodeRandomCropResizeJpeg: min scale must be greater than or equal to 0, got: " +
                       std::to_string(scale_[0]);
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (scale_[1] <= 0) {
    std::string err_msg = "SoftDvppDecodeRandomCropResizeJpeg: max scale must be greater than 0.";
    MS_LOG(ERROR) << "SoftDvppDecodeRandomCropResizeJpeg: max scale must be greater than 0, got: " +
                       std::to_string(scale_[1]);
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (scale_[1] < scale_[0]) {
    std::string err_msg = "SoftDvppDecodeRandomCropResizeJpeg: scale must be in the format of (min, max).";
    MS_LOG(ERROR) << "SoftDvppDecodeRandomCropResizeJpeg: scale must be in the format of (min, max), but got: "
                  << scale_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // ratio
  if (ratio_.size() != 2) {
    std::string err_msg =
      "SoftDvppDecodeRandomCropResizeJpeg: ratio must be a vector of two values, got: " + std::to_string(ratio_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (ratio_[0] <= 0 || ratio_[1] <= 0) {
    std::string err_msg = "SoftDvppDecodeRandomCropResizeJpeg: ratio must be greater than 0.";
    MS_LOG(ERROR) << "SoftDvppDecodeRandomCropResizeJpeg: ratio must be greater than 0, got: " << ratio_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (ratio_[1] < ratio_[0]) {
    std::string err_msg = "SoftDvppDecodeRandomCropResizeJpeg: ratio must be in the format of (min, max).";
    MS_LOG(ERROR) << "SoftDvppDecodeRandomCropResizeJpeg: ratio must be in the format of (min, max), but got: "
                  << ratio_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // max_attempts
  if (max_attempts_ < 1) {
    std::string err_msg = "SoftDvppDecodeRandomCropResizeJpeg: max_attempts must be greater than or equal to 1, got: " +
                          std::to_string(max_attempts_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> SoftDvppDecodeRandomCropResizeJpegOperation::Build() {
  int32_t height = size_[0];
  int32_t width = size_[0];
  // User specified the width value.
  if (size_.size() == 2) {
    width = size_[1];
  }

  auto tensor_op = std::make_shared<SoftDvppDecodeRandomCropResizeJpegOp>(height, width, scale_[0], scale_[1],
                                                                          ratio_[0], ratio_[1], max_attempts_);
  return tensor_op;
}

// SoftDvppDecodeResizeJpegOperation
SoftDvppDecodeResizeJpegOperation::SoftDvppDecodeResizeJpegOperation(std::vector<int32_t> size) : size_(size) {}

Status SoftDvppDecodeResizeJpegOperation::ValidateParams() {
  // size
  if (size_.empty() || size_.size() > 2) {
    std::string err_msg =
      "SoftDvppDecodeResizeJpeg: size must be a vector of one or two values, got: " + std::to_string(size_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  RETURN_IF_NOT_OK(ValidateVectorPositive("SoftDvppDecodeResizeJpeg", size_));

  return Status::OK();
}

std::shared_ptr<TensorOp> SoftDvppDecodeResizeJpegOperation::Build() {
  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  int32_t height = size_[0];
  int32_t width = 0;

  // User specified the width value.
  if (size_.size() == 2) {
    width = size_[1];
  }
  std::shared_ptr<SoftDvppDecodeResizeJpegOp> tensor_op = std::make_shared<SoftDvppDecodeResizeJpegOp>(height, width);
  return tensor_op;
}

// SwapRedBlueOperation.
SwapRedBlueOperation::SwapRedBlueOperation() {}

Status SwapRedBlueOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> SwapRedBlueOperation::Build() {
  std::shared_ptr<SwapRedBlueOp> tensor_op = std::make_shared<SwapRedBlueOp>();
  return tensor_op;
}

// UniformAugOperation
UniformAugOperation::UniformAugOperation(std::vector<std::shared_ptr<TensorOperation>> transforms, int32_t num_ops)
    : transforms_(transforms), num_ops_(num_ops) {}

Status UniformAugOperation::ValidateParams() {
  // transforms
  RETURN_IF_NOT_OK(ValidateVectorTransforms("UniformAug", transforms_));
  if (num_ops_ > transforms_.size()) {
    std::string err_msg = "UniformAug: num_ops is greater than transforms size, but got: " + std::to_string(num_ops_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // num_ops
  if (num_ops_ <= 0) {
    std::string err_msg = "UniformAug: num_ops must be greater than 0, but got: " + std::to_string(num_ops_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> UniformAugOperation::Build() {
  std::vector<std::shared_ptr<TensorOp>> tensor_ops;
  (void)std::transform(transforms_.begin(), transforms_.end(), std::back_inserter(tensor_ops),
                       [](std::shared_ptr<TensorOperation> op) -> std::shared_ptr<TensorOp> { return op->Build(); });
  std::shared_ptr<UniformAugOp> tensor_op = std::make_shared<UniformAugOp>(tensor_ops, num_ops_);
  return tensor_op;
}
#endif

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
