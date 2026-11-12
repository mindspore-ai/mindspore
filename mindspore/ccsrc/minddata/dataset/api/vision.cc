/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/include/dataset/vision.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if defined(WITH_BACKEND) || defined(ENABLE_ACL) || defined(ENABLE_DVPP)
#include "minddata/dataset/include/dataset/vision_ascend.h"
#endif
#include "minddata/dataset/vision/transform/ascend_vision_ir.h"

#include "minddata/dataset/vision/transform/center_crop_ir.h"
#include "minddata/dataset/vision/transform/decode_ir.h"
#ifdef ENABLE_FFMPEG
#endif
#include "minddata/dataset/vision/transform/hwc_to_chw_ir.h"
#include "minddata/dataset/vision/transform/normalize_ir.h"
#include "minddata/dataset/vision/transform/random_color_adjust_ir.h"
#include "minddata/dataset/vision/transform/random_crop_decode_resize_ir.h"
#include "minddata/dataset/vision/transform/random_crop_ir.h"
#include "minddata/dataset/vision/transform/random_horizontal_flip_ir.h"
#include "minddata/dataset/vision/transform/random_resized_crop_ir.h"
#include "minddata/dataset/vision/transform/random_vertical_flip_ir.h"
#include "minddata/dataset/vision/transform/resize_ir.h"
#include "minddata/dataset/vision/transform/resize_preserve_ar_ir.h"
#include "minddata/dataset/vision/transform/rgb_to_bgr_ir.h"
#include "minddata/dataset/vision/transform/rgb_to_gray_ir.h"
#include "minddata/dataset/vision/transform/rgba_to_bgr_ir.h"
#include "minddata/dataset/vision/transform/rgba_to_rgb_ir.h"
#include "minddata/dataset/vision/transform/rotate_ir.h"
#include "minddata/dataset/vision/transform/to_tensor_ir.h"
#include "minddata/dataset/util/log_adapter.h"

#include "minddata/dataset/vision/transform/rescale_ir.h"
#include "minddata/dataset/vision/transform/swap_red_blue_ir.h"

#include "minddata/dataset/vision/kernels/image_utils.h"
#if defined(ENABLE_FFMPEG)
#include "minddata/dataset/vision/kernels/video_utils.h"
#endif

// Typecast between mindspore::DataType and dataset::DataType
#include "minddata/dataset/core/type_id.h"
#include "ir/dtype/type_id.h"

namespace mindspore::dataset::vision {
// CONSTRUCTORS FOR API CLASSES TO CREATE VISION TENSOR TRANSFORM OPERATIONS
// (In alphabetical order)

// Affine Transform Operation.

// AdjustBrightness Transform Operation.

// AdjustContrast Transform Operation.

// AdjustGamma Transform Operation.

// AdjustHue Transform Operation.

// AdjustSaturation Transform Operation.

// AdjustSharpness Transform Operation.

// AutoAugment Transform Operation.

// AutoContrast Transform Operation.

// BoundingBoxAugment Transform Operation.

// CenterCrop Transform Operation.
struct CenterCrop::Data {
  explicit Data(const std::vector<int32_t> &size) : size_(size) {}
  std::vector<int32_t> size_;
};

CenterCrop::CenterCrop(const std::vector<int32_t> &size) : data_(std::make_shared<Data>(size)) {}

std::shared_ptr<TensorOperation> CenterCrop::Parse() { return std::make_shared<CenterCropOperation>(data_->size_); }

std::shared_ptr<TensorOperation> CenterCrop::Parse(const MapTargetDevice &env) {
  if (env == MapTargetDevice::kAscend310) {
#if defined(WITH_BACKEND) || defined(ENABLE_ACL) || defined(ENABLE_DVPP)
    std::vector<uint32_t> usize_;
    usize_.reserve(data_->size_.size());
    std::transform(data_->size_.begin(), data_->size_.end(), std::back_inserter(usize_),
                   [](int32_t i) { return static_cast<uint32_t>(i); });
    return std::make_shared<DvppCropJpegOperation>(usize_);
#endif
  } else if (env == MapTargetDevice::kCpu) {
    return std::make_shared<CenterCropOperation>(data_->size_);
  }
  MS_LOG(ERROR) << "Unsupported MapTargetDevice, only supported kCpu and kAscend310.";
  return nullptr;
}

// ConvertColor Transform Operation.

// Crop Transform Operation.

// CutMixBatch Transform Operation.

// CutOutOp.

// Decode Transform Operation.
struct Decode::Data {
  explicit Data(bool rgb) : rgb_(rgb) {}
  bool rgb_;
};

Decode::Decode(bool rgb) : data_(std::make_shared<Data>(rgb)) {}

std::shared_ptr<TensorOperation> Decode::Parse() { return std::make_shared<DecodeOperation>(data_->rgb_); }

std::shared_ptr<TensorOperation> Decode::Parse(const MapTargetDevice &env) {
  if (env == MapTargetDevice::kAscend310) {
#if defined(WITH_BACKEND) || defined(ENABLE_ACL) || defined(ENABLE_DVPP)
    return std::make_shared<DvppDecodeJpegOperation>();
#endif
  } else if (env == MapTargetDevice::kCpu) {
    return std::make_shared<DecodeOperation>(data_->rgb_);
  }
  MS_LOG(ERROR) << "Unsupported MapTargetDevice, only supported kCpu and kAscend310.";
  return nullptr;
}

#if defined(ENABLE_FFMPEG)

#endif

#if defined(WITH_BACKEND) || defined(ENABLE_ACL) || defined(ENABLE_DVPP)
// DvppDecodeVideo Transform Operation.
struct DvppDecodeVideo::Data {
  Data(const std::vector<uint32_t> &size, VdecStreamFormat type, VdecOutputFormat out_format, const std::string &output)
      : size_(size), format_(out_format), en_type_(type), output_(output) {}

  std::vector<uint32_t> size_;
  VdecOutputFormat format_;
  VdecStreamFormat en_type_;
  std::string output_;
};

DvppDecodeVideo::DvppDecodeVideo(const std::vector<uint32_t> &size, VdecStreamFormat type, VdecOutputFormat out_format,
                                 const std::vector<char> &output)
    : data_(std::make_shared<Data>(size, type, out_format, CharToString(output))) {}

std::shared_ptr<TensorOperation> DvppDecodeVideo::Parse() {
  return std::make_shared<DvppDecodeVideoOperation>(data_->size_, data_->en_type_, data_->format_, data_->output_);
}

std::shared_ptr<TensorOperation> DvppDecodeVideo::Parse(const MapTargetDevice &env) {
  if (env == MapTargetDevice::kAscend310) {
    return std::make_shared<DvppDecodeVideoOperation>(data_->size_, data_->en_type_, data_->format_, data_->output_);
  }
  MS_LOG(ERROR) << "Unsupported MapTargetDevice, only kAscend310 is supported.";
  return nullptr;
}

// DvppDecodeResize Transform Operation.
struct DvppDecodeResizeJpeg::Data {
  explicit Data(const std::vector<uint32_t> &resize) : resize_(resize) {}
  std::vector<uint32_t> resize_;
};

DvppDecodeResizeJpeg::DvppDecodeResizeJpeg(const std::vector<uint32_t> &resize)
    : data_(std::make_shared<Data>(resize)) {}

std::shared_ptr<TensorOperation> DvppDecodeResizeJpeg::Parse() {
  return std::make_shared<DvppDecodeResizeOperation>(data_->resize_);
}

std::shared_ptr<TensorOperation> DvppDecodeResizeJpeg::Parse(const MapTargetDevice &env) {
  if (env == MapTargetDevice::kAscend310) {
    return std::make_shared<DvppDecodeResizeOperation>(data_->resize_);
  }
  MS_LOG(ERROR) << "Unsupported MapTargetDevice, only supported kAscend310.";
  return nullptr;
}

// DvppDecodeResizeCrop Transform Operation.
struct DvppDecodeResizeCropJpeg::Data {
  Data(const std::vector<uint32_t> &crop, const std::vector<uint32_t> &resize) : crop_(crop), resize_(resize) {}
  std::vector<uint32_t> crop_;
  std::vector<uint32_t> resize_;
};

DvppDecodeResizeCropJpeg::DvppDecodeResizeCropJpeg(const std::vector<uint32_t> &crop,
                                                   const std::vector<uint32_t> &resize)
    : data_(std::make_shared<Data>(crop, resize)) {}

std::shared_ptr<TensorOperation> DvppDecodeResizeCropJpeg::Parse() {
  return std::make_shared<DvppDecodeResizeCropOperation>(data_->crop_, data_->resize_);
}

std::shared_ptr<TensorOperation> DvppDecodeResizeCropJpeg::Parse(const MapTargetDevice &env) {
  if (env == MapTargetDevice::kAscend310) {
#if defined(WITH_BACKEND) || defined(ENABLE_ACL)
    return std::make_shared<DvppDecodeResizeCropOperation>(data_->crop_, data_->resize_);
#endif
  }
  MS_LOG(ERROR) << "Unsupported MapTargetDevice, only supported kAscend310.";
  return nullptr;
}

// DvppDecodePng Transform Operation.
DvppDecodePng::DvppDecodePng() {}

std::shared_ptr<TensorOperation> DvppDecodePng::Parse() { return std::make_shared<DvppDecodePngOperation>(); }

std::shared_ptr<TensorOperation> DvppDecodePng::Parse(const MapTargetDevice &env) {
  if (env == MapTargetDevice::kAscend310) {
    return std::make_shared<DvppDecodePngOperation>();
  }
  MS_LOG(ERROR) << "Unsupported MapTargetDevice, only supported kAscend310.";
  return nullptr;
}
#endif

// EncodeJpeg Function.

// EncodePng Function.

// Equalize Transform Operation.

// Erase Operation.

// GaussianBlur Transform Operation.

// GetImageNumChannels Function.

// GetImageSize Function.

// HorizontalFlip Transform Operation.

// HwcToChw Transform Operation.
HWC2CHW::HWC2CHW() = default;

std::shared_ptr<TensorOperation> HWC2CHW::Parse() { return std::make_shared<HwcToChwOperation>(); }

// Invert Transform Operation.

// MixUpBatch Transform Operation.

// Normalize Transform Operation.
struct Normalize::Data {
  Data(const std::vector<float> &mean, const std::vector<float> &std, bool is_hwc)
      : mean_(mean), std_(std), is_hwc_(is_hwc) {}
  std::vector<float> mean_;
  std::vector<float> std_;
  bool is_hwc_;
};

Normalize::Normalize(const std::vector<float> &mean, const std::vector<float> &std, bool is_hwc)
    : data_(std::make_shared<Data>(mean, std, is_hwc)) {}

std::shared_ptr<TensorOperation> Normalize::Parse() {
  return std::make_shared<NormalizeOperation>(data_->mean_, data_->std_, data_->is_hwc_);
}

std::shared_ptr<TensorOperation> Normalize::Parse(const MapTargetDevice &env) {
  if (data_->is_hwc_ == false) {
    MS_LOG(ERROR) << "Normalize op on Lite does not support 'is_hwc' = false.";
    return nullptr;
  }
  if (env == MapTargetDevice::kAscend310) {
#if defined(WITH_BACKEND) || defined(ENABLE_ACL) || defined(ENABLE_DVPP)
    return std::make_shared<DvppNormalizeOperation>(data_->mean_, data_->std_);
#endif
  } else if (env == MapTargetDevice::kCpu) {
    return std::make_shared<NormalizeOperation>(data_->mean_, data_->std_, data_->is_hwc_);
  }
  MS_LOG(ERROR) << "Unsupported MapTargetDevice, only supported kCpu and kAscend310.";
  return nullptr;
}

// NormalizePad Transform Operation.

// Pad Transform Operation.

// PadToSize Transform Operation.

// Perspective Transform Operation.

// Posterize Transform Operation

// RandAugment Transform Operation

// RandomAdjustSharpness Transform Operation.

// RandomAffine Transform Operation.

// RandomAutoContrast Transform Operation.

// RandomColor Transform Operation.

// RandomColorAdjust Transform Operation.
struct RandomColorAdjust::Data {
  Data(const std::vector<float> &brightness, const std::vector<float> &contrast, const std::vector<float> &saturation,
       const std::vector<float> &hue)
      : brightness_(brightness), contrast_(contrast), saturation_(saturation), hue_(hue) {}
  std::vector<float> brightness_;
  std::vector<float> contrast_;
  std::vector<float> saturation_;
  std::vector<float> hue_;
};

RandomColorAdjust::RandomColorAdjust(const std::vector<float> &brightness, const std::vector<float> &contrast,
                                     const std::vector<float> &saturation, const std::vector<float> &hue)
    : data_(std::make_shared<Data>(brightness, contrast, saturation, hue)) {}

std::shared_ptr<TensorOperation> RandomColorAdjust::Parse() {
  return std::make_shared<RandomColorAdjustOperation>(data_->brightness_, data_->contrast_, data_->saturation_,
                                                      data_->hue_);
}

// RandomCrop Transform Operation.
struct RandomCrop::Data {
  Data(const std::vector<int32_t> &size, const std::vector<int32_t> &padding, bool pad_if_needed,
       const std::vector<uint8_t> &fill_value, BorderType padding_mode)
      : size_(size),
        padding_(padding),
        pad_if_needed_(pad_if_needed),
        fill_value_(fill_value),
        padding_mode_(padding_mode) {}
  std::vector<int32_t> size_;
  std::vector<int32_t> padding_;
  bool pad_if_needed_;
  std::vector<uint8_t> fill_value_;
  BorderType padding_mode_;
};

RandomCrop::RandomCrop(const std::vector<int32_t> &size, const std::vector<int32_t> &padding, bool pad_if_needed,
                       const std::vector<uint8_t> &fill_value, BorderType padding_mode)
    : data_(std::make_shared<Data>(size, padding, pad_if_needed, fill_value, padding_mode)) {}

std::shared_ptr<TensorOperation> RandomCrop::Parse() {
  return std::make_shared<RandomCropOperation>(data_->size_, data_->padding_, data_->pad_if_needed_, data_->fill_value_,
                                               data_->padding_mode_);
}

// RandomCropDecodeResize Transform Operation.
struct RandomCropDecodeResize::Data {
  Data(const std::vector<int32_t> &size, const std::vector<float> &scale, const std::vector<float> &ratio,
       InterpolationMode interpolation, int32_t max_attempts)
      : size_(size), scale_(scale), ratio_(ratio), interpolation_(interpolation), max_attempts_(max_attempts) {}
  std::vector<int32_t> size_;
  std::vector<float> scale_;
  std::vector<float> ratio_;
  InterpolationMode interpolation_;
  int32_t max_attempts_;
};

RandomCropDecodeResize::RandomCropDecodeResize(const std::vector<int32_t> &size, const std::vector<float> &scale,
                                               const std::vector<float> &ratio, InterpolationMode interpolation,
                                               int32_t max_attempts)
    : data_(std::make_shared<Data>(size, scale, ratio, interpolation, max_attempts)) {}

std::shared_ptr<TensorOperation> RandomCropDecodeResize::Parse() {
  return std::make_shared<RandomCropDecodeResizeOperation>(data_->size_, data_->scale_, data_->ratio_,
                                                           data_->interpolation_, data_->max_attempts_);
}

// RandomCropWithBBox Transform Operation.

// RandomEqualize Transform Operation.

// RandomHorizontalFlip.
struct RandomHorizontalFlip::Data {
  explicit Data(float prob) : probability_(prob) {}
  float probability_;
};

RandomHorizontalFlip::RandomHorizontalFlip(float prob) : data_(std::make_shared<Data>(prob)) {}

std::shared_ptr<TensorOperation> RandomHorizontalFlip::Parse() {
  return std::make_shared<RandomHorizontalFlipOperation>(data_->probability_);
}

// RandomHorizontalFlipWithBBox

// RandomInvert Operation.

// RandomLighting Transform Operation.

// RandomPosterize Transform Operation.

// RandomResize Transform Operation.

// RandomResizeWithBBox Transform Operation.

// RandomResizedCrop Transform Operation.
struct RandomResizedCrop::Data {
  Data(const std::vector<int32_t> &size, const std::vector<float> &scale, const std::vector<float> &ratio,
       InterpolationMode interpolation, int32_t max_attempts)
      : size_(size), scale_(scale), ratio_(ratio), interpolation_(interpolation), max_attempts_(max_attempts) {}
  std::vector<int32_t> size_;
  std::vector<float> scale_;
  std::vector<float> ratio_;
  InterpolationMode interpolation_;
  int32_t max_attempts_;
};

RandomResizedCrop::RandomResizedCrop(const std::vector<int32_t> &size, const std::vector<float> &scale,
                                     const std::vector<float> &ratio, InterpolationMode interpolation,
                                     int32_t max_attempts)
    : data_(std::make_shared<Data>(size, scale, ratio, interpolation, max_attempts)) {}

std::shared_ptr<TensorOperation> RandomResizedCrop::Parse() {
  return std::make_shared<RandomResizedCropOperation>(data_->size_, data_->scale_, data_->ratio_, data_->interpolation_,
                                                      data_->max_attempts_);
}

// RandomResizedCrop Transform Operation.

// RandomRotation Transform Operation.

// RandomSelectSubpolicy Transform Operation.

// RandomSharpness Transform Operation.

// RandomSolarize Transform Operation.

// RandomVerticalFlip Transform Operation.
struct RandomVerticalFlip::Data {
  explicit Data(float prob) : probability_(prob) {}
  float probability_;
};

RandomVerticalFlip::RandomVerticalFlip(float prob) : data_(std::make_shared<Data>(prob)) {}

std::shared_ptr<TensorOperation> RandomVerticalFlip::Parse() {
  return std::make_shared<RandomVerticalFlipOperation>(data_->probability_);
}

// Rescale Transform Operation.
struct Rescale::Data {
  Data(float rescale, float shift) : rescale_(rescale), shift_(shift) {}
  float rescale_;
  float shift_;
};

Rescale::Rescale(float rescale, float shift) : data_(std::make_shared<Data>(rescale, shift)) {}

std::shared_ptr<TensorOperation> Rescale::Parse() {
  return std::make_shared<RescaleOperation>(data_->rescale_, data_->shift_);
}

// Resize Transform Operation.
struct Resize::Data {
  Data(const std::vector<int32_t> &size, InterpolationMode interpolation)
      : size_(size), interpolation_(interpolation) {}
  std::vector<int32_t> size_;
  InterpolationMode interpolation_;
};

Resize::Resize(const std::vector<int32_t> &size, InterpolationMode interpolation)
    : data_(std::make_shared<Data>(size, interpolation)) {}

std::shared_ptr<TensorOperation> Resize::Parse() {
  return std::make_shared<ResizeOperation>(data_->size_, data_->interpolation_);
}

std::shared_ptr<TensorOperation> Resize::Parse(const MapTargetDevice &env) {
  if (env == MapTargetDevice::kAscend310) {
#if defined(WITH_BACKEND) || defined(ENABLE_ACL) || defined(ENABLE_DVPP)
    std::vector<uint32_t> usize_;
    usize_.reserve(data_->size_.size());
    std::transform(data_->size_.begin(), data_->size_.end(), std::back_inserter(usize_),
                   [](int32_t i) { return static_cast<uint32_t>(i); });
    return std::make_shared<DvppResizeJpegOperation>(usize_);
#endif
  } else if (env == MapTargetDevice::kCpu) {
    return std::make_shared<ResizeOperation>(data_->size_, data_->interpolation_);
  }
  MS_LOG(ERROR) << "Unsupported MapTargetDevice, only supported kCpu and kAscend310.";
  return nullptr;
}

// ResizedCrop Transform Operation.

// ResizePreserveAR Transform Operation.
struct ResizePreserveAR::Data {
  Data(int32_t height, int32_t width, int32_t img_orientation)
      : height_(height), width_(width), img_orientation_(img_orientation) {}
  int32_t height_;
  int32_t width_;
  int32_t img_orientation_;
};

ResizePreserveAR::ResizePreserveAR(int32_t height, int32_t width, int32_t img_orientation)
    : data_(std::make_shared<Data>(height, width, img_orientation)) {}

std::shared_ptr<TensorOperation> ResizePreserveAR::Parse() {
  return std::make_shared<ResizePreserveAROperation>(data_->height_, data_->width_, data_->img_orientation_);
}

// ResizeWithBBox Transform Operation.

// RGB2BGR Transform Operation.
std::shared_ptr<TensorOperation> RGB2BGR::Parse() { return std::make_shared<RgbToBgrOperation>(); }

// RGB2GRAY Transform Operation.
std::shared_ptr<TensorOperation> RGB2GRAY::Parse() { return std::make_shared<RgbToGrayOperation>(); }

// Rotate Transform Operation.

// RgbaToBgr Transform Operation.
RGBA2BGR::RGBA2BGR() = default;

std::shared_ptr<TensorOperation> RGBA2BGR::Parse() { return std::make_shared<RgbaToBgrOperation>(); }

// RgbaToRgb Transform Operation.
RGBA2RGB::RGBA2RGB() = default;

std::shared_ptr<TensorOperation> RGBA2RGB::Parse() { return std::make_shared<RgbaToRgbOperation>(); }

// SlicePatches Transform Operation.

// Solarize Transform Operation.

// SwapRedBlue Transform Operation.
SwapRedBlue::SwapRedBlue() = default;
std::shared_ptr<TensorOperation> SwapRedBlue::Parse() { return std::make_shared<SwapRedBlueOperation>(); }

// ToTensor Transform Operation.
struct ToTensor::Data {
  explicit Data(const std::string &output_type) : output_type_(DataType(output_type)) {}
  explicit Data(const DataType::Type &output_type) : output_type_(output_type) {}
  explicit Data(const mindspore::DataType &output_type)
      : output_type_(dataset::MSTypeToDEType(static_cast<TypeId>(output_type))) {}
  DataType output_type_{};
};

ToTensor::ToTensor() : data_(std::make_shared<Data>(DataType::Type::DE_FLOAT32)) {}
ToTensor::ToTensor(std::string output_type) : data_(std::make_shared<Data>(output_type)) {}
ToTensor::ToTensor(mindspore::DataType output_type) : data_(std::make_shared<Data>(output_type)) {}

std::shared_ptr<TensorOperation> ToTensor::Parse() { return std::make_shared<ToTensorOperation>(data_->output_type_); }

// TrivialAugmentWide Transform Operation.

// UniformAug Transform Operation.

// VerticalFlip Transform Operation.

// WriteFile Function.

// WriteJpeg Function.

// WritePNG Function.

}  // namespace mindspore::dataset::vision
