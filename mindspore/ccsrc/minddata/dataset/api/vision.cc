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

#include "minddata/dataset/include/vision.h"
#ifdef ENABLE_ACL
#include "minddata/dataset/include/vision_ascend.h"
#endif

#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/kernels/ir/vision/vision_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"

// Kernel image headers (in alphabetical order)
// FIXME - Delete these dvpp include header when dvpp TensorOperation moved to IR level
#ifdef ENABLE_ACL
#include "minddata/dataset/kernels/image/dvpp/dvpp_crop_jpeg_op.h"
#include "minddata/dataset/kernels/image/dvpp/dvpp_decode_resize_jpeg_op.h"
#include "minddata/dataset/kernels/image/dvpp/dvpp_decode_resize_crop_jpeg_op.h"
#include "minddata/dataset/kernels/image/dvpp/dvpp_decode_jpeg_op.h"
#include "minddata/dataset/kernels/image/dvpp/dvpp_decode_png_op.h"
#include "minddata/dataset/kernels/image/dvpp/dvpp_resize_jpeg_op.h"
#endif

namespace mindspore {
namespace dataset {

// Transform operations for computer vision.
namespace vision {
#ifndef ENABLE_ANDROID
// FUNCTIONS TO CREATE VISION TRANSFORM OPERATIONS
// (In alphabetical order)

// AutoContrast Transform Operation.
AutoContrast::AutoContrast(float cutoff, std::vector<uint32_t> ignore) : cutoff_(cutoff), ignore_(ignore) {}

std::shared_ptr<TensorOperation> AutoContrast::Parse() {
  return std::make_shared<AutoContrastOperation>(cutoff_, ignore_);
}

// BoundingBoxAugment Transform Operation.
BoundingBoxAugment::BoundingBoxAugment(std::shared_ptr<TensorTransform> transform, float ratio) {
  // Convert transform from TensorTransform to TensorOperation
  transform_ = transform->Parse();
  ratio_ = ratio;
}

std::shared_ptr<TensorOperation> BoundingBoxAugment::Parse() {
  return std::make_shared<BoundingBoxAugmentOperation>(transform_, ratio_);
}
#endif

// CenterCrop Transform Operation.
CenterCrop::CenterCrop(std::vector<int32_t> size) : size_(size) {}

std::shared_ptr<TensorOperation> CenterCrop::Parse() { return std::make_shared<CenterCropOperation>(size_); }

// Crop Transform Operation.
Crop::Crop(std::vector<int32_t> coordinates, std::vector<int32_t> size) : coordinates_(coordinates), size_(size) {}

std::shared_ptr<TensorOperation> Crop::Parse() { return std::make_shared<CropOperation>(coordinates_, size_); }

#ifndef ENABLE_ANDROID
// CutMixBatch Transform Operation.
CutMixBatch::CutMixBatch(ImageBatchFormat image_batch_format, float alpha, float prob)
    : image_batch_format_(image_batch_format), alpha_(alpha), prob_(prob) {}

std::shared_ptr<TensorOperation> CutMixBatch::Parse() {
  return std::make_shared<CutMixBatchOperation>(image_batch_format_, alpha_, prob_);
}

// CutOutOp.
CutOut::CutOut(int32_t length, int32_t num_patches) : length_(length), num_patches_(num_patches) {}

std::shared_ptr<TensorOperation> CutOut::Parse() { return std::make_shared<CutOutOperation>(length_, num_patches_); }

// Decode Transform Operation.
Decode::Decode(bool rgb) {}
std::shared_ptr<TensorOperation> Decode::Parse() { return std::make_shared<DecodeOperation>(rgb_); }

#endif
#ifdef ENABLE_ACL
// Function to create DvppResizeOperation.
std::shared_ptr<DvppCropJpegOperation> DvppCropJpeg(std::vector<uint32_t> crop) {
  auto op = std::make_shared<DvppCropJpegOperation>(crop);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create DvppDecodeResizeOperation.
std::shared_ptr<DvppDecodeResizeOperation> DvppDecodeResizeJpeg(std::vector<uint32_t> resize) {
  auto op = std::make_shared<DvppDecodeResizeOperation>(resize);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create DvppDecodeResizeCropOperation.
std::shared_ptr<DvppDecodeResizeCropOperation> DvppDecodeResizeCropJpeg(std::vector<uint32_t> crop,
                                                                        std::vector<uint32_t> resize) {
  auto op = std::make_shared<DvppDecodeResizeCropOperation>(crop, resize);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create DvppDecodeJpegOperation.
std::shared_ptr<DvppDecodeJpegOperation> DvppDecodeJpeg() {
  auto op = std::make_shared<DvppDecodeJpegOperation>();
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create DvppDecodePngOperation.
std::shared_ptr<DvppDecodePngOperation> DvppDecodePng() {
  auto op = std::make_shared<DvppDecodePngOperation>();
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create DvppResizeOperation.
std::shared_ptr<DvppResizeJpegOperation> DvppResizeJpeg(std::vector<uint32_t> resize) {
  auto op = std::make_shared<DvppResizeJpegOperation>(resize);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}
#endif

/*
// DvppResize Transform Operation.
DvppCropJpeg::DvppCropJpeg(std::vector<uint32_t> crop) : crop_(crop) {}

std::shared_ptr<TensorOperation> DvppCropJpeg::Parse() { return std::make_shared<DvppCropJpegOperation>(crop); }

// DvppDecodeResize Transform Operation.
DvppDecodeResize::DvppDecodeResizeJpeg(std::vector<uint32_t> resize) : resize_(resize) {}

std::shared_ptr<TensorOperation> DvppDecodeResizeJpeg::Parse() {
  return std::make_shared<DvppDecodeResizeOperation>(resize);
}

// DvppDecodeResizeCrop Transform Operation.
DvppDecodeResizeCrop::DvppDecodeResizeCropJpeg(std::vector<uint32_t> crop, std::vector<uint32_t> resize)
    : crop_(crop), resize_(resize) {}

std::shared_ptr<TensorOperation> DvppDecodeResizeCropJpeg::Parse() {
  return std::make_shared<DvppDecodeResizeCropOperation>(crop, resize);
}

// DvppCropOperation
DvppCropJpeg::DvppCropJpeg(const std::vector<uint32_t> &crop) : crop_(crop) {}

std::shared_ptr<TensorOperation> DvppCropJpeg::Parse() { return std::make_shared<DvppCropJpegOperation>(); }

// DvppDecodeResizeOperation
DvppDecodeResize::DvppDecodeResize(const std::vector<uint32_t> &resize) : resize_(resize) {}

std::shared_ptr<TensorOperation> DvppDecodeResize::Parse() { return std::make_shared<DvppDecodeResizeOperation>(); }

// DvppDecodeJpeg Transform Operation.
DvppDecodeJpeg::DvppDecodeJpeg() {}

std::shared_ptr<TensorOperation> DvppDecodeJpeg::Parse() { return std::make_shared<DvppDecodeJpegOperation>(); }

// DvppDecodePng Transform Operation.
DvppDecodePng::DvppDecodePng() {}

std::shared_ptr<TensorOperation> DvppDecodePng::Parse() { return std::make_shared<DvppDecodePngOperation>(); }

// DvppResizeOperation
DvppDecodeResize::DvppResizeJpeg(const std::vector<uint32_t> &resize) : resize_(resize) {}

std::shared_ptr<TensorOperation> DvppDecodeResize::Parse() { return std::make_shared<DvppDecodeResizeOperation>(); }

// DvppDecodeResizeCropOperation
DvppDecodeResizeCrop::DvppDecodeResizeCrop(const std::vector<uint32_t> &crop, const std::vector<uint32_t> &resize)
    : crop_(crop), resize_(resize) {}

std::shared_ptr<TensorOperation> DvppDecodeResizeCrop::Parse() {
  return std::make_shared<DvppDecodeResizeCropOperation>();
}

// DvppResize Transform Operation.
DvppResizeJpeg::DvppResizeJpeg(std::vector<uint32_t> resize) {}

std::shared_ptr<TensorOperation> DvppResizeJpeg::Parse() { return std::make_shared<DvppResizeJpegOperation>(resize); }
#endif

*/

#ifndef ENABLE_ANDROID
// Equalize Transform Operation.
Equalize::Equalize() {}

std::shared_ptr<TensorOperation> Equalize::Parse() { return std::make_shared<EqualizeOperation>(); }
// HwcToChw Transform Operation.
HWC2CHW::HWC2CHW() {}

std::shared_ptr<TensorOperation> HWC2CHW::Parse() { return std::make_shared<HwcToChwOperation>(); }

// Invert Transform Operation.
Invert::Invert() {}

std::shared_ptr<TensorOperation> Invert::Parse() { return std::make_shared<InvertOperation>(); }

// MixUpBatch Transform Operation.
MixUpBatch::MixUpBatch(float alpha) : alpha_(alpha) {}

std::shared_ptr<TensorOperation> MixUpBatch::Parse() { return std::make_shared<MixUpBatchOperation>(alpha_); }
#endif

// Normalize Transform Operation.
Normalize::Normalize(std::vector<float> mean, std::vector<float> std) : mean_(mean), std_(std) {}

std::shared_ptr<TensorOperation> Normalize::Parse() { return std::make_shared<NormalizeOperation>(mean_, std_); }

#ifndef ENABLE_ANDROID
// NormalizePad Transform Operation.
NormalizePad::NormalizePad(const std::vector<float> &mean, const std::vector<float> &std, const std::string &dtype)
    : mean_(mean), std_(std), dtype_(dtype) {}

std::shared_ptr<TensorOperation> NormalizePad::Parse() {
  return std::make_shared<NormalizePadOperation>(mean_, std_, dtype_);
}

// Pad Transform Operation.
Pad::Pad(std::vector<int32_t> padding, std::vector<uint8_t> fill_value, BorderType padding_mode)
    : padding_(padding), fill_value_(fill_value), padding_mode_(padding_mode) {}

std::shared_ptr<TensorOperation> Pad::Parse() {
  return std::make_shared<PadOperation>(padding_, fill_value_, padding_mode_);
}

// RandomAffine Transform Operation.
RandomAffine::RandomAffine(const std::vector<float_t> &degrees, const std::vector<float_t> &translate_range,
                           const std::vector<float_t> &scale_range, const std::vector<float_t> &shear_ranges,
                           InterpolationMode interpolation, const std::vector<uint8_t> &fill_value)
    : degrees_(degrees),
      translate_range_(translate_range),
      scale_range_(scale_range),
      shear_ranges_(shear_ranges),
      interpolation_(interpolation),
      fill_value_(fill_value) {}

std::shared_ptr<TensorOperation> RandomAffine::Parse() {
  return std::make_shared<RandomAffineOperation>(degrees_, translate_range_, scale_range_, shear_ranges_,
                                                 interpolation_, fill_value_);
}

// RandomColor Transform Operation.
RandomColor::RandomColor(float t_lb, float t_ub) : t_lb_(t_lb), t_ub_(t_ub) {}

std::shared_ptr<TensorOperation> RandomColor::Parse() { return std::make_shared<RandomColorOperation>(t_lb_, t_ub_); }

// RandomColorAdjust Transform Operation.
RandomColorAdjust::RandomColorAdjust(std::vector<float> brightness, std::vector<float> contrast,
                                     std::vector<float> saturation, std::vector<float> hue)
    : brightness_(brightness), contrast_(contrast), saturation_(saturation), hue_(hue) {}
std::shared_ptr<TensorOperation> RandomColorAdjust::Parse() {
  return std::make_shared<RandomColorAdjustOperation>(brightness_, contrast_, saturation_, hue_);
}

// RandomCrop Transform Operation.
RandomCrop::RandomCrop(std::vector<int32_t> size, std::vector<int32_t> padding, bool pad_if_needed,
                       std::vector<uint8_t> fill_value, BorderType padding_mode)
    : size_(size),
      padding_(padding),
      pad_if_needed_(pad_if_needed),
      fill_value_(fill_value),
      padding_mode_(padding_mode) {}

std::shared_ptr<TensorOperation> RandomCrop::Parse() {
  return std::make_shared<RandomCropOperation>(size_, padding_, pad_if_needed_, fill_value_, padding_mode_);
}

// RandomCropDecodeResize Transform Operation.
RandomCropDecodeResize::RandomCropDecodeResize(std::vector<int32_t> size, std::vector<float> scale,
                                               std::vector<float> ratio, InterpolationMode interpolation,
                                               int32_t max_attempts)
    : size_(size), scale_(scale), ratio_(ratio), interpolation_(interpolation), max_attempts_(max_attempts) {}

std::shared_ptr<TensorOperation> RandomCropDecodeResize::Parse() {
  return std::make_shared<RandomCropDecodeResizeOperation>(size_, scale_, ratio_, interpolation_, max_attempts_);
}

// RandomCropWithBBox Transform Operation.
RandomCropWithBBox::RandomCropWithBBox(std::vector<int32_t> size, std::vector<int32_t> padding, bool pad_if_needed,
                                       std::vector<uint8_t> fill_value, BorderType padding_mode)
    : size_(size),
      padding_(padding),
      pad_if_needed_(pad_if_needed),
      fill_value_(fill_value),
      padding_mode_(padding_mode) {}

std::shared_ptr<TensorOperation> RandomCropWithBBox::Parse() {
  return std::make_shared<RandomCropWithBBoxOperation>(size_, padding_, pad_if_needed_, fill_value_, padding_mode_);
}

// RandomHorizontalFlip.
RandomHorizontalFlip::RandomHorizontalFlip(float prob) : probability_(prob) {}

std::shared_ptr<TensorOperation> RandomHorizontalFlip::Parse() {
  return std::make_shared<RandomHorizontalFlipOperation>(probability_);
}

// RandomHorizontalFlipWithBBox
RandomHorizontalFlipWithBBox::RandomHorizontalFlipWithBBox(float prob) : probability_(prob) {}

std::shared_ptr<TensorOperation> RandomHorizontalFlipWithBBox::Parse() {
  return std::make_shared<RandomHorizontalFlipWithBBoxOperation>(probability_);
}

// RandomPosterize Transform Operation.
RandomPosterize::RandomPosterize(const std::vector<uint8_t> &bit_range) : bit_range_(bit_range) {}

std::shared_ptr<TensorOperation> RandomPosterize::Parse() {
  return std::make_shared<RandomPosterizeOperation>(bit_range_);
}

// RandomResize Transform Operation.
RandomResize::RandomResize(std::vector<int32_t> size) : size_(size) {}

std::shared_ptr<TensorOperation> RandomResize::Parse() { return std::make_shared<RandomResizeOperation>(size_); }

// RandomResizeWithBBox Transform Operation.
RandomResizeWithBBox::RandomResizeWithBBox(std::vector<int32_t> size) : size_(size) {}

std::shared_ptr<TensorOperation> RandomResizeWithBBox::Parse() {
  return std::make_shared<RandomResizeWithBBoxOperation>(size_);
}

// RandomResizedCrop Transform Operation.
RandomResizedCrop::RandomResizedCrop(std::vector<int32_t> size, std::vector<float> scale, std::vector<float> ratio,
                                     InterpolationMode interpolation, int32_t max_attempts)
    : size_(size), scale_(scale), ratio_(ratio), interpolation_(interpolation), max_attempts_(max_attempts) {}

std::shared_ptr<TensorOperation> RandomResizedCrop::Parse() {
  return std::make_shared<RandomResizedCropOperation>(size_, scale_, ratio_, interpolation_, max_attempts_);
}

// RandomResizedCrop Transform Operation.
RandomResizedCropWithBBox::RandomResizedCropWithBBox(std::vector<int32_t> size, std::vector<float> scale,
                                                     std::vector<float> ratio, InterpolationMode interpolation,
                                                     int32_t max_attempts)
    : size_(size), scale_(scale), ratio_(ratio), interpolation_(interpolation), max_attempts_(max_attempts) {}

std::shared_ptr<TensorOperation> RandomResizedCropWithBBox::Parse() {
  return std::make_shared<RandomResizedCropWithBBoxOperation>(size_, scale_, ratio_, interpolation_, max_attempts_);
}

// RandomRotation Transform Operation.
RandomRotation::RandomRotation(std::vector<float> degrees, InterpolationMode interpolation_mode, bool expand,
                               std::vector<float> center, std::vector<uint8_t> fill_value)
    : degrees_(degrees),
      interpolation_mode_(interpolation_mode),
      expand_(expand),
      center_(center),
      fill_value_(fill_value) {}

std::shared_ptr<TensorOperation> RandomRotation::Parse() {
  return std::make_shared<RandomRotationOperation>(degrees_, interpolation_mode_, expand_, center_, fill_value_);
}

// RandomSelectSubpolicy Transform Operation.
// FIXME - Provide TensorTransform support for policy
RandomSelectSubpolicy::RandomSelectSubpolicy(
  std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> policy)
    : policy_(policy) {}

std::shared_ptr<TensorOperation> RandomSelectSubpolicy::Parse() {
  return std::make_shared<RandomSelectSubpolicyOperation>(policy_);
}

// RandomSharpness Transform Operation.
RandomSharpness::RandomSharpness(std::vector<float> degrees) : degrees_(degrees) {}

std::shared_ptr<TensorOperation> RandomSharpness::Parse() {
  return std::make_shared<RandomSharpnessOperation>(degrees_);
}

// RandomSolarize Transform Operation.
RandomSolarize::RandomSolarize(std::vector<uint8_t> threshold) : threshold_(threshold) {}

std::shared_ptr<TensorOperation> RandomSolarize::Parse() {
  return std::make_shared<RandomSolarizeOperation>(threshold_);
}

// RandomVerticalFlip Transform Operation.
RandomVerticalFlip::RandomVerticalFlip(float prob) : probability_(prob) {}

std::shared_ptr<TensorOperation> RandomVerticalFlip::Parse() {
  return std::make_shared<RandomVerticalFlipOperation>(probability_);
}

// RandomVerticalFlipWithBBox Transform Operation.
RandomVerticalFlipWithBBox::RandomVerticalFlipWithBBox(float prob) : probability_(prob) {}

std::shared_ptr<TensorOperation> RandomVerticalFlipWithBBox::Parse() {
  return std::make_shared<RandomVerticalFlipWithBBoxOperation>(probability_);
}

// Rescale Transform Operation.
Rescale::Rescale(float rescale, float shift) : rescale_(rescale), shift_(shift) {}

std::shared_ptr<TensorOperation> Rescale::Parse() { return std::make_shared<RescaleOperation>(rescale_, shift_); }

#endif
// Resize Transform Operation.
Resize::Resize(std::vector<int32_t> size, InterpolationMode interpolation)
    : size_(size), interpolation_(interpolation) {}

std::shared_ptr<TensorOperation> Resize::Parse() { return std::make_shared<ResizeOperation>(size_, interpolation_); }

#ifdef ENABLE_ANDROID
// Rotate Transform Operation.
Rotate::Rotate() {}

std::shared_ptr<TensorOperation> Rotate::Parse() { return std::make_shared<RotateOperation>(); }
#endif

#ifndef ENABLE_ANDROID
// ResizeWithBBox Transform Operation.
ResizeWithBBox::ResizeWithBBox(std::vector<int32_t> size, InterpolationMode interpolation)
    : size_(size), interpolation_(interpolation) {}

std::shared_ptr<TensorOperation> ResizeWithBBox::Parse() {
  return std::make_shared<ResizeWithBBoxOperation>(size_, interpolation_);
}

// RgbaToBgr Transform Operation.
RGBA2BGR::RGBA2BGR() {}

std::shared_ptr<TensorOperation> RGBA2BGR::Parse() { return std::make_shared<RgbaToBgrOperation>(); }

// RgbaToRgb Transform Operation.
RGBA2RGB::RGBA2RGB() {}

std::shared_ptr<TensorOperation> RGBA2RGB::Parse() { return std::make_shared<RgbaToRgbOperation>(); }

// SoftDvppDecodeRandomCropResizeJpeg Transform Operation.
SoftDvppDecodeRandomCropResizeJpeg::SoftDvppDecodeRandomCropResizeJpeg(std::vector<int32_t> size,
                                                                       std::vector<float> scale,
                                                                       std::vector<float> ratio, int32_t max_attempts)
    : size_(size), scale_(scale), ratio_(ratio), max_attempts_(max_attempts) {}
std::shared_ptr<TensorOperation> SoftDvppDecodeRandomCropResizeJpeg::Parse() {
  return std::make_shared<SoftDvppDecodeRandomCropResizeJpegOperation>(size_, scale_, ratio_, max_attempts_);
}

// SoftDvppDecodeResizeJpeg Transform Operation.
SoftDvppDecodeResizeJpeg::SoftDvppDecodeResizeJpeg(std::vector<int32_t> size) : size_(size) {}

std::shared_ptr<TensorOperation> SoftDvppDecodeResizeJpeg::Parse() {
  return std::make_shared<SoftDvppDecodeResizeJpegOperation>(size_);
}

// SwapRedBlue Transform Operation.
SwapRedBlue::SwapRedBlue() {}

std::shared_ptr<TensorOperation> SwapRedBlue::Parse() { return std::make_shared<SwapRedBlueOperation>(); }

// UniformAug Transform Operation.
UniformAugment::UniformAugment(std::vector<std::shared_ptr<TensorTransform>> transforms, int32_t num_ops) {
  // Convert ops from TensorTransform to TensorOperation
  (void)std::transform(
    transforms.begin(), transforms.end(), std::back_inserter(transforms_),
    [](std::shared_ptr<TensorTransform> operation) -> std::shared_ptr<TensorOperation> { return operation->Parse(); });
  num_ops_ = num_ops;
}

std::shared_ptr<TensorOperation> UniformAugment::Parse() {
  return std::make_shared<UniformAugOperation>(transforms_, num_ops_);
}
#endif

// FIXME - Move these DVPP Derived TensorOperation classes to IR level
/* ####################################### Derived TensorOperation classes ################################# */

#ifdef ENABLE_ACL
// DvppCropOperation
DvppCropJpegOperation::DvppCropJpegOperation(const std::vector<uint32_t> &crop) : crop_(crop) {}

Status DvppCropJpegOperation::ValidateParams() {
  // size
  if (crop_.empty() || crop_.size() > 2) {
    std::string err_msg =
      "DvppCropJpeg: Crop resolution must be a vector of one or two elements, got: " + std::to_string(crop_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (*min_element(crop_.begin(), crop_.end()) < 32 || *max_element(crop_.begin(), crop_.end()) > 2048) {
    std::string err_msg = "Dvpp module supports crop image with resolution in range [32, 2048], got crop Parameters: ";
    if (crop_.size() == 2) {
      MS_LOG(ERROR) << err_msg << "[" << crop_[0] << ", " << crop_[1] << "]";
    } else {
      MS_LOG(ERROR) << err_msg << "[" << crop_[0] << ", " << crop_[0] << "]";
    }
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> DvppCropJpegOperation::Build() {
  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  uint32_t cropHeight, cropWidth;
  // User specified the width value.
  if (crop_.size() == 1) {
    cropHeight = crop_[0];
    cropWidth = crop_[0];
  } else {
    cropHeight = crop_[0];
    cropWidth = crop_[1];
  }
  std::shared_ptr<DvppCropJpegOp> tensor_op = std::make_shared<DvppCropJpegOp>(cropHeight, cropWidth);
  return tensor_op;
}

// DvppDecodeResizeOperation
DvppDecodeResizeOperation::DvppDecodeResizeOperation(const std::vector<uint32_t> &resize) : resize_(resize) {}

Status DvppDecodeResizeOperation::ValidateParams() {
  // size
  if (resize_.empty() || resize_.size() > 2) {
    std::string err_msg = "DvppDecodeResizeJpeg: resize resolution must be a vector of one or two elements, got: " +
                          std::to_string(resize_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (*min_element(resize_.begin(), resize_.end()) < 32 || *max_element(resize_.begin(), resize_.end()) > 2048) {
    std::string err_msg =
      "Dvpp module supports resize image with resolution in range [32, 2048], got resize Parameters: ";
    if (resize_.size() == 2) {
      MS_LOG(ERROR) << err_msg << "[" << resize_[0] << ", " << resize_[1] << "]";
    } else {
      MS_LOG(ERROR) << err_msg << "[" << resize_[0] << ", " << resize_[0] << "]";
    }
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> DvppDecodeResizeOperation::Build() {
  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  uint32_t resizeHeight, resizeWidth;
  // User specified the width value.
  if (resize_.size() == 1) {
    resizeHeight = resize_[0];
    resizeWidth = 0;
  } else {
    resizeHeight = resize_[0];
    resizeWidth = resize_[1];
  }
  std::shared_ptr<DvppDecodeResizeJpegOp> tensor_op =
    std::make_shared<DvppDecodeResizeJpegOp>(resizeHeight, resizeWidth);
  return tensor_op;
}

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
      MS_LOG(ERROR) << err_msg << "[" << resize_[0] << "]";
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
    resizeWidth = 0;
  } else {
    resizeHeight = resize_[0];
    resizeWidth = resize_[1];
  }
  std::shared_ptr<DvppDecodeResizeCropJpegOp> tensor_op =
    std::make_shared<DvppDecodeResizeCropJpegOp>(cropHeight, cropWidth, resizeHeight, resizeWidth);
  return tensor_op;
}

// DvppDecodeJPEG
Status DvppDecodeJpegOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> DvppDecodeJpegOperation::Build() { return std::make_shared<DvppDecodeJpegOp>(); }

// DvppDecodePNG
Status DvppDecodePngOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> DvppDecodePngOperation::Build() { return std::make_shared<DvppDecodePngOp>(); }

// DvppResizeOperation
DvppResizeJpegOperation::DvppResizeJpegOperation(const std::vector<uint32_t> &resize) : resize_(resize) {}

Status DvppResizeJpegOperation::ValidateParams() {
  // size
  if (resize_.empty() || resize_.size() > 2) {
    std::string err_msg = "DvppResizeJpeg: resize resolution must be a vector of one or two elements, got: " +
                          std::to_string(resize_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (*min_element(resize_.begin(), resize_.end()) < 32 || *max_element(resize_.begin(), resize_.end()) > 2048) {
    std::string err_msg =
      "Dvpp module supports resize image with resolution in range [32, 2048], got resize Parameters: ";
    if (resize_.size() == 2) {
      MS_LOG(ERROR) << err_msg << "[" << resize_[0] << ", " << resize_[1] << "]";
    } else {
      MS_LOG(ERROR) << err_msg << "[" << resize_[0] << ", " << resize_[0] << "]";
    }
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> DvppResizeJpegOperation::Build() {
  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  uint32_t resizeHeight, resizeWidth;
  // User specified the width value.
  if (resize_.size() == 1) {
    resizeHeight = resize_[0];
    resizeWidth = 0;
  } else {
    resizeHeight = resize_[0];
    resizeWidth = resize_[1];
  }
  std::shared_ptr<DvppResizeJpegOp> tensor_op = std::make_shared<DvppResizeJpegOp>(resizeHeight, resizeWidth);
  return tensor_op;
}
#endif

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
