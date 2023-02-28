/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#if defined(WITH_BACKEND) || defined(ENABLE_ACL) || defined(ENABLE_DVPP)
#include "minddata/dataset/include/dataset/vision_ascend.h"
#endif
#include "minddata/dataset/kernels/ir/vision/ascend_vision_ir.h"

#include "minddata/dataset/kernels/ir/vision/adjust_brightness_ir.h"
#include "minddata/dataset/kernels/ir/vision/adjust_contrast_ir.h"
#include "minddata/dataset/kernels/ir/vision/adjust_gamma_ir.h"
#include "minddata/dataset/kernels/ir/vision/adjust_hue_ir.h"
#include "minddata/dataset/kernels/ir/vision/adjust_saturation_ir.h"
#include "minddata/dataset/kernels/ir/vision/adjust_sharpness_ir.h"
#include "minddata/dataset/kernels/ir/vision/affine_ir.h"
#include "minddata/dataset/kernels/ir/vision/auto_augment_ir.h"
#include "minddata/dataset/kernels/ir/vision/auto_contrast_ir.h"
#include "minddata/dataset/kernels/ir/vision/bounding_box_augment_ir.h"
#include "minddata/dataset/kernels/ir/vision/center_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/convert_color_ir.h"
#include "minddata/dataset/kernels/ir/vision/crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/cutmix_batch_ir.h"
#include "minddata/dataset/kernels/ir/vision/cutout_ir.h"
#include "minddata/dataset/kernels/ir/vision/decode_ir.h"
#include "minddata/dataset/kernels/ir/vision/equalize_ir.h"
#include "minddata/dataset/kernels/ir/vision/erase_ir.h"
#include "minddata/dataset/kernels/ir/vision/gaussian_blur_ir.h"
#include "minddata/dataset/kernels/ir/vision/horizontal_flip_ir.h"
#include "minddata/dataset/kernels/ir/vision/hwc_to_chw_ir.h"
#include "minddata/dataset/kernels/ir/vision/invert_ir.h"
#include "minddata/dataset/kernels/ir/vision/mixup_batch_ir.h"
#include "minddata/dataset/kernels/ir/vision/normalize_ir.h"
#include "minddata/dataset/kernels/ir/vision/normalize_pad_ir.h"
#include "minddata/dataset/kernels/ir/vision/pad_to_size_ir.h"
#include "minddata/dataset/kernels/ir/vision/perspective_ir.h"
#include "minddata/dataset/kernels/ir/vision/posterize_ir.h"
#include "minddata/dataset/kernels/ir/vision/rand_augment_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_adjust_sharpness_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_affine_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_auto_contrast_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_color_adjust_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_color_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_crop_decode_resize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_crop_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_equalize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_horizontal_flip_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_horizontal_flip_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_invert_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_lighting_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_posterize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resize_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resized_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resized_crop_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_rotation_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_select_subpolicy_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_sharpness_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_solarize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_vertical_flip_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_vertical_flip_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/resize_ir.h"
#include "minddata/dataset/kernels/ir/vision/resize_preserve_ar_ir.h"
#include "minddata/dataset/kernels/ir/vision/resize_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/resized_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/rgb_to_bgr_ir.h"
#include "minddata/dataset/kernels/ir/vision/rgb_to_gray_ir.h"
#include "minddata/dataset/kernels/ir/vision/rgba_to_bgr_ir.h"
#include "minddata/dataset/kernels/ir/vision/rgba_to_rgb_ir.h"
#include "minddata/dataset/kernels/ir/vision/rotate_ir.h"
#include "minddata/dataset/kernels/ir/vision/slice_patches_ir.h"
#include "minddata/dataset/kernels/ir/vision/solarize_ir.h"
#include "minddata/dataset/kernels/ir/vision/to_tensor_ir.h"
#include "minddata/dataset/kernels/ir/vision/trivial_augment_wide_ir.h"
#include "minddata/dataset/kernels/ir/vision/uniform_aug_ir.h"
#include "minddata/dataset/kernels/ir/vision/vertical_flip_ir.h"
#include "minddata/dataset/util/log_adapter.h"

#if !defined(ENABLE_ANDROID) || defined(ENABLE_CLOUD_FUSION_INFERENCE)
#include "minddata/dataset/kernels/ir/vision/pad_ir.h"
#include "minddata/dataset/kernels/ir/vision/rescale_ir.h"
#include "minddata/dataset/kernels/ir/vision/swap_red_blue_ir.h"
#endif

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"

// Typecast between mindspore::DataType and dataset::DataType
#include "minddata/dataset/core/type_id.h"
#include "mindspore/core/ir/dtype/type_id.h"

namespace mindspore {
namespace dataset {
// Transform operations for computer vision.
namespace vision {
// CONSTRUCTORS FOR API CLASSES TO CREATE VISION TENSOR TRANSFORM OPERATIONS
// (In alphabetical order)

// Affine Transform Operation.
struct Affine::Data {
  Data(float_t degrees, const std::vector<float> &translation, float scale, const std::vector<float> &shear,
       InterpolationMode interpolation, const std::vector<uint8_t> &fill_value)
      : degrees_(degrees),
        translation_(translation),
        scale_(scale),
        shear_(shear),
        interpolation_(interpolation),
        fill_value_(fill_value) {}
  float degrees_;
  std::vector<float> translation_;
  float scale_;
  std::vector<float> shear_;
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};

Affine::Affine(float_t degrees, const std::vector<float> &translation, float scale, const std::vector<float> &shear,
               InterpolationMode interpolation, const std::vector<uint8_t> &fill_value)
    : data_(std::make_shared<Data>(degrees, translation, scale, shear, interpolation, fill_value)) {}

std::shared_ptr<TensorOperation> Affine::Parse() {
  return std::make_shared<AffineOperation>(data_->degrees_, data_->translation_, data_->scale_, data_->shear_,
                                           data_->interpolation_, data_->fill_value_);
}

#ifndef ENABLE_ANDROID
// AdjustBrightness Transform Operation.
struct AdjustBrightness::Data {
  explicit Data(float brightness_factor) : brightness_factor_(brightness_factor) {}
  float brightness_factor_;
};

AdjustBrightness::AdjustBrightness(float brightness_factor) : data_(std::make_shared<Data>(brightness_factor)) {}

std::shared_ptr<TensorOperation> AdjustBrightness::Parse() {
  return std::make_shared<AdjustBrightnessOperation>(data_->brightness_factor_);
}

// AdjustContrast Transform Operation.
struct AdjustContrast::Data {
  explicit Data(float contrast_factor) : contrast_factor_(contrast_factor) {}
  float contrast_factor_;
};

AdjustContrast::AdjustContrast(float contrast_factor) : data_(std::make_shared<Data>(contrast_factor)) {}

std::shared_ptr<TensorOperation> AdjustContrast::Parse() {
  return std::make_shared<AdjustContrastOperation>(data_->contrast_factor_);
}

// AdjustGamma Transform Operation.
struct AdjustGamma::Data {
  Data(float gamma, float gain) : gamma_(gamma), gain_(gain) {}
  float gamma_;
  float gain_;
};

AdjustGamma::AdjustGamma(float gamma, float gain) : data_(std::make_shared<Data>(gamma, gain)) {}

std::shared_ptr<TensorOperation> AdjustGamma::Parse() {
  return std::make_shared<AdjustGammaOperation>(data_->gamma_, data_->gain_);
}

// AdjustHue Transform Operation.
struct AdjustHue::Data {
  explicit Data(float hue_factor) : hue_factor_(hue_factor) {}
  float hue_factor_;
};

AdjustHue::AdjustHue(float hue_factor) : data_(std::make_shared<Data>(hue_factor)) {}

std::shared_ptr<TensorOperation> AdjustHue::Parse() { return std::make_shared<AdjustHueOperation>(data_->hue_factor_); }

// AdjustSaturation Transform Operation.
struct AdjustSaturation::Data {
  explicit Data(float saturation_factor) : saturation_factor_(saturation_factor) {}
  float saturation_factor_;
};

AdjustSaturation::AdjustSaturation(float saturation_factor) : data_(std::make_shared<Data>(saturation_factor)) {}

std::shared_ptr<TensorOperation> AdjustSaturation::Parse() {
  return std::make_shared<AdjustSaturationOperation>(data_->saturation_factor_);
}

// AdjustSharpness Transform Operation.
struct AdjustSharpness::Data {
  explicit Data(float sharpness_factor) : sharpness_factor_(sharpness_factor) {}
  float sharpness_factor_;
};

AdjustSharpness::AdjustSharpness(float sharpness_factor) : data_(std::make_shared<Data>(sharpness_factor)) {}

std::shared_ptr<TensorOperation> AdjustSharpness::Parse() {
  return std::make_shared<AdjustSharpnessOperation>(data_->sharpness_factor_);
}

// AutoAugment Transform Operation.
struct AutoAugment::Data {
  Data(AutoAugmentPolicy policy, InterpolationMode interpolation, const std::vector<uint8_t> &fill_value)
      : policy_(policy), interpolation_(interpolation), fill_value_(fill_value) {}
  AutoAugmentPolicy policy_;
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};

AutoAugment::AutoAugment(AutoAugmentPolicy policy, InterpolationMode interpolation,
                         const std::vector<uint8_t> &fill_value)
    : data_(std::make_shared<Data>(policy, interpolation, fill_value)) {}

std::shared_ptr<TensorOperation> AutoAugment::Parse() {
  return std::make_shared<AutoAugmentOperation>(data_->policy_, data_->interpolation_, data_->fill_value_);
}

// AutoContrast Transform Operation.
struct AutoContrast::Data {
  Data(float cutoff, const std::vector<uint32_t> &ignore) : cutoff_(cutoff), ignore_(ignore) {}
  float cutoff_;
  std::vector<uint32_t> ignore_;
};

AutoContrast::AutoContrast(float cutoff, const std::vector<uint32_t> &ignore)
    : data_(std::make_shared<Data>(cutoff, ignore)) {}

std::shared_ptr<TensorOperation> AutoContrast::Parse() {
  return std::make_shared<AutoContrastOperation>(data_->cutoff_, data_->ignore_);
}

// BoundingBoxAugment Transform Operation.
struct BoundingBoxAugment::Data {
  std::shared_ptr<TensorOperation> transform_;
  float ratio_;
};

BoundingBoxAugment::BoundingBoxAugment(TensorTransform *transform, float ratio) : data_(std::make_shared<Data>()) {
  data_->transform_ = transform ? transform->Parse() : nullptr;
  data_->ratio_ = ratio;
}

BoundingBoxAugment::BoundingBoxAugment(const std::shared_ptr<TensorTransform> &transform, float ratio)
    : data_(std::make_shared<Data>()) {
  data_->transform_ = transform ? transform->Parse() : nullptr;
  data_->ratio_ = ratio;
}

BoundingBoxAugment::BoundingBoxAugment(const std::reference_wrapper<TensorTransform> &transform, float ratio)
    : data_(std::make_shared<Data>()) {
  data_->transform_ = transform.get().Parse();
  data_->ratio_ = ratio;
}

std::shared_ptr<TensorOperation> BoundingBoxAugment::Parse() {
  return std::make_shared<BoundingBoxAugmentOperation>(data_->transform_, data_->ratio_);
}
#endif  // not ENABLE_ANDROID

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
                   [](int32_t i) { return (uint32_t)i; });
    return std::make_shared<DvppCropJpegOperation>(usize_);
#endif
  } else if (env == MapTargetDevice::kCpu) {
    return std::make_shared<CenterCropOperation>(data_->size_);
  }
  MS_LOG(ERROR) << "Unsupported MapTargetDevice, only supported kCpu and kAscend310.";
  return nullptr;
}

#ifndef ENABLE_ANDROID
// ConvertColor Transform Operation.
struct ConvertColor::Data {
  explicit Data(ConvertMode convert_mode) : convert_mode_(convert_mode) {}
  ConvertMode convert_mode_;
};

ConvertColor::ConvertColor(ConvertMode convert_mode) : data_(std::make_shared<Data>(convert_mode)) {}

std::shared_ptr<TensorOperation> ConvertColor::Parse() {
  return std::make_shared<ConvertColorOperation>(data_->convert_mode_);
}
#endif  // not ENABLE_ANDROID

// Crop Transform Operation.
struct Crop::Data {
  Data(const std::vector<int32_t> &coordinates, const std::vector<int32_t> &size)
      : coordinates_(coordinates), size_(size) {}
  std::vector<int32_t> coordinates_;
  std::vector<int32_t> size_;
};

Crop::Crop(const std::vector<int32_t> &coordinates, const std::vector<int32_t> &size)
    : data_(std::make_shared<Data>(coordinates, size)) {}

std::shared_ptr<TensorOperation> Crop::Parse() {
  return std::make_shared<CropOperation>(data_->coordinates_, data_->size_);
}

#ifndef ENABLE_ANDROID
// CutMixBatch Transform Operation.
struct CutMixBatch::Data {
  Data(ImageBatchFormat image_batch_format, float alpha, float prob)
      : image_batch_format_(image_batch_format), alpha_(alpha), prob_(prob) {}
  float alpha_;
  float prob_;
  ImageBatchFormat image_batch_format_;
};

CutMixBatch::CutMixBatch(ImageBatchFormat image_batch_format, float alpha, float prob)
    : data_(std::make_shared<Data>(image_batch_format, alpha, prob)) {}

std::shared_ptr<TensorOperation> CutMixBatch::Parse() {
  return std::make_shared<CutMixBatchOperation>(data_->image_batch_format_, data_->alpha_, data_->prob_);
}

// CutOutOp.
struct CutOut::Data {
  Data(int32_t length, int32_t num_patches, bool is_hwc)
      : length_(length), num_patches_(num_patches), is_hwc_(is_hwc) {}
  int32_t length_;
  int32_t num_patches_;
  bool is_hwc_;
};

CutOut::CutOut(int32_t length, int32_t num_patches, bool is_hwc)
    : data_(std::make_shared<Data>(length, num_patches, is_hwc)) {}

std::shared_ptr<TensorOperation> CutOut::Parse() {
  return std::make_shared<CutOutOperation>(data_->length_, data_->num_patches_, data_->is_hwc_);
}
#endif  // not ENABLE_ANDROID

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
#ifndef ENABLE_ANDROID

// EncodeJpeg Function.
Status EncodeJpeg(const mindspore::MSTensor &image, mindspore::MSTensor *output, int quality) {
  RETURN_UNEXPECTED_IF_NULL(output);
  std::shared_ptr<dataset::Tensor> input;
  RETURN_IF_NOT_OK(Tensor::CreateFromMSTensor(image, &input));
  std::shared_ptr<dataset::Tensor> de_tensor;
  RETURN_IF_NOT_OK(mindspore::dataset::EncodeJpeg(input, &de_tensor, quality));
  CHECK_FAIL_RETURN_UNEXPECTED(de_tensor->HasData(),
                               "EncodeJpeg: get an empty tensor with shape " + de_tensor->shape().ToString());
  *output = mindspore::MSTensor(std::make_shared<DETensor>(de_tensor));
  return Status::OK();
}

// EncodePng Function.
Status EncodePng(const mindspore::MSTensor &image, mindspore::MSTensor *output, int compression_level) {
  RETURN_UNEXPECTED_IF_NULL(output);
  std::shared_ptr<dataset::Tensor> input;
  RETURN_IF_NOT_OK(Tensor::CreateFromMSTensor(image, &input));
  TensorPtr de_tensor;
  RETURN_IF_NOT_OK(mindspore::dataset::EncodePng(input, &de_tensor, compression_level));
  CHECK_FAIL_RETURN_UNEXPECTED(de_tensor->HasData(),
                               "EncodePng: get an empty tensor with shape " + de_tensor->shape().ToString());
  *output = mindspore::MSTensor(std::make_shared<DETensor>(de_tensor));
  return Status::OK();
}

// Equalize Transform Operation.
Equalize::Equalize() = default;

std::shared_ptr<TensorOperation> Equalize::Parse() { return std::make_shared<EqualizeOperation>(); }

// Erase Operation.
struct Erase::Data {
  Data(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<uint8_t> &value, bool inplace)
      : top_(top), left_(left), height_(height), width_(width), value_(value), inplace_(inplace) {}
  int32_t top_;
  int32_t left_;
  int32_t height_;
  int32_t width_;
  std::vector<uint8_t> value_;
  bool inplace_;
};

Erase::Erase(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<uint8_t> &value, bool inplace)
    : data_(std::make_shared<Data>(top, left, height, width, value, inplace)) {}

std::shared_ptr<TensorOperation> Erase::Parse() {
  return std::make_shared<EraseOperation>(data_->top_, data_->left_, data_->height_, data_->width_, data_->value_,
                                          data_->inplace_);
}
#endif  // not ENABLE_ANDROID

// GaussianBlur Transform Operation.
struct GaussianBlur::Data {
  Data(const std::vector<int32_t> &kernel_size, const std::vector<float> &sigma)
      : kernel_size_(kernel_size), sigma_(sigma) {}
  std::vector<int32_t> kernel_size_;
  std::vector<float> sigma_;
};

GaussianBlur::GaussianBlur(const std::vector<int32_t> &kernel_size, const std::vector<float> &sigma)
    : data_(std::make_shared<Data>(kernel_size, sigma)) {}

std::shared_ptr<TensorOperation> GaussianBlur::Parse() {
  return std::make_shared<GaussianBlurOperation>(data_->kernel_size_, data_->sigma_);
}

#ifndef ENABLE_ANDROID
// GetImageNumChannels Function.
Status GetImageNumChannels(const mindspore::MSTensor &image, dsize_t *channels) {
  RETURN_UNEXPECTED_IF_NULL(channels);
  std::shared_ptr<dataset::Tensor> input;
  Status rc = Tensor::CreateFromMSTensor(image, &input);
  if (rc.IsError()) {
    RETURN_STATUS_UNEXPECTED("GetImageNumChannels: failed to create image tensor.");
  }
  return ImageNumChannels(input, channels);
}

// GetImageSize Function.
Status GetImageSize(const mindspore::MSTensor &image, std::vector<dsize_t> *size) {
  RETURN_UNEXPECTED_IF_NULL(size);
  std::shared_ptr<Tensor> input;
  Status rc = Tensor::CreateFromMSTensor(image, &input);
  if (rc.IsError()) {
    RETURN_STATUS_UNEXPECTED("GetImageSize: failed to create image tensor.");
  }
  return ImageSize(input, size);
}

// HorizontalFlip Transform Operation.
HorizontalFlip::HorizontalFlip() = default;

std::shared_ptr<TensorOperation> HorizontalFlip::Parse() { return std::make_shared<HorizontalFlipOperation>(); }
#endif  // not ENABLE_ANDROID

// HwcToChw Transform Operation.
HWC2CHW::HWC2CHW() = default;

std::shared_ptr<TensorOperation> HWC2CHW::Parse() { return std::make_shared<HwcToChwOperation>(); }

#ifndef ENABLE_ANDROID
// Invert Transform Operation.
Invert::Invert() = default;

std::shared_ptr<TensorOperation> Invert::Parse() { return std::make_shared<InvertOperation>(); }

// MixUpBatch Transform Operation.
struct MixUpBatch::Data {
  explicit Data(float alpha) : alpha_(alpha) {}
  float alpha_;
};

MixUpBatch::MixUpBatch(float alpha) : data_(std::make_shared<Data>(alpha)) {}

std::shared_ptr<TensorOperation> MixUpBatch::Parse() { return std::make_shared<MixUpBatchOperation>(data_->alpha_); }
#endif  // not ENABLE_ANDROID

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
#ifdef ENABLE_ANDROID
  if (data_->is_hwc_ == false) {
    MS_LOG(ERROR) << "Normalize op on Lite does not support 'is_hwc' = false.";
    return nullptr;
  }
#endif
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

#ifndef ENABLE_ANDROID
// NormalizePad Transform Operation.
struct NormalizePad::Data {
  Data(const std::vector<float> &mean, const std::vector<float> &std, const std::string &dtype, bool is_hwc)
      : mean_(mean), std_(std), dtype_(dtype), is_hwc_(is_hwc) {}
  std::vector<float> mean_;
  std::vector<float> std_;
  std::string dtype_;
  bool is_hwc_;
};

NormalizePad::NormalizePad(const std::vector<float> &mean, const std::vector<float> &std,
                           const std::vector<char> &dtype, bool is_hwc)
    : data_(std::make_shared<Data>(mean, std, CharToString(dtype), is_hwc)) {}

std::shared_ptr<TensorOperation> NormalizePad::Parse() {
  return std::make_shared<NormalizePadOperation>(data_->mean_, data_->std_, data_->dtype_, data_->is_hwc_);
}
#endif  // not ENABLE_ANDROID

// Pad Transform Operation.
struct Pad::Data {
  Data(const std::vector<int32_t> &padding, const std::vector<uint8_t> &fill_value, BorderType padding_mode)
      : padding_(padding), fill_value_(fill_value), padding_mode_(padding_mode) {}
  std::vector<int32_t> padding_;
  std::vector<uint8_t> fill_value_;
  BorderType padding_mode_;
};

Pad::Pad(const std::vector<int32_t> &padding, const std::vector<uint8_t> &fill_value, BorderType padding_mode)
    : data_(std::make_shared<Data>(padding, fill_value, padding_mode)) {}

std::shared_ptr<TensorOperation> Pad::Parse() {
#if !defined(ENABLE_ANDROID) || defined(ENABLE_CLOUD_FUSION_INFERENCE)
  return std::make_shared<PadOperation>(data_->padding_, data_->fill_value_, data_->padding_mode_);
#else
  MS_LOG(ERROR) << "Unsupported Pad.";
  return nullptr;
#endif
}

#ifndef ENABLE_ANDROID
// PadToSize Transform Operation.
struct PadToSize::Data {
  Data(const std::vector<int32_t> &size, const std::vector<int32_t> &offset, const std::vector<uint8_t> &fill_value,
       BorderType padding_mode)
      : size_(size), offset_(offset), fill_value_(fill_value), padding_mode_(padding_mode) {}
  std::vector<int32_t> size_;
  std::vector<int32_t> offset_;
  std::vector<uint8_t> fill_value_;
  BorderType padding_mode_;
};

PadToSize::PadToSize(const std::vector<int32_t> &size, const std::vector<int32_t> &offset,
                     const std::vector<uint8_t> &fill_value, BorderType padding_mode)
    : data_(std::make_shared<Data>(size, offset, fill_value, padding_mode)) {}

std::shared_ptr<TensorOperation> PadToSize::Parse() {
  return std::make_shared<PadToSizeOperation>(data_->size_, data_->offset_, data_->fill_value_, data_->padding_mode_);
}

// Perspective Transform Operation.
struct Perspective::Data {
  Data(const std::vector<std::vector<int32_t>> &start_points, const std::vector<std::vector<int32_t>> &end_points,
       InterpolationMode interpolation)
      : start_points_(start_points), end_points_(end_points), interpolation_(interpolation) {}
  std::vector<std::vector<int32_t>> start_points_;
  std::vector<std::vector<int32_t>> end_points_;
  InterpolationMode interpolation_;
};

Perspective::Perspective(const std::vector<std::vector<int32_t>> &start_points,
                         const std::vector<std::vector<int32_t>> &end_points, InterpolationMode interpolation)
    : data_(std::make_shared<Data>(start_points, end_points, interpolation)) {}

std::shared_ptr<TensorOperation> Perspective::Parse() {
  return std::make_shared<PerspectiveOperation>(data_->start_points_, data_->end_points_, data_->interpolation_);
}

// Posterize Transform Operation
struct Posterize::Data {
  explicit Data(uint8_t bits) : bits_(bits) {}
  uint8_t bits_;
};

Posterize::Posterize(uint8_t bits) : data_(std::make_shared<Data>(bits)) {}

std::shared_ptr<TensorOperation> Posterize::Parse() { return std::make_shared<PosterizeOperation>(data_->bits_); }

// RandAugment Transform Operation
struct RandAugment::Data {
  Data(int32_t num_ops, int32_t magnitude, int32_t num_magnitude_bins, InterpolationMode interpolation,
       const std::vector<uint8_t> &fill_value)
      : num_ops_(num_ops),
        magnitude_(magnitude),
        num_magnitude_bins_(num_magnitude_bins),
        interpolation_(interpolation),
        fill_value_(fill_value) {}
  int32_t num_ops_;
  int32_t magnitude_;
  int32_t num_magnitude_bins_;
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};

RandAugment::RandAugment(int32_t num_ops, int32_t magnitude, int32_t num_magnitude_bins,
                         InterpolationMode interpolation, const std::vector<uint8_t> &fill_value)
    : data_(std::make_shared<Data>(num_ops, magnitude, num_magnitude_bins, interpolation, fill_value)) {}

std::shared_ptr<TensorOperation> RandAugment::Parse() {
  return std::make_shared<RandAugmentOperation>(data_->num_ops_, data_->magnitude_, data_->num_magnitude_bins_,
                                                data_->interpolation_, data_->fill_value_);
}

// RandomAdjustSharpness Transform Operation.
struct RandomAdjustSharpness::Data {
  Data(float degree, float prob) : degree_(degree), probability_(prob) {}
  float degree_;
  float probability_;
};

RandomAdjustSharpness::RandomAdjustSharpness(float degree, float prob) : data_(std::make_shared<Data>(degree, prob)) {}

std::shared_ptr<TensorOperation> RandomAdjustSharpness::Parse() {
  return std::make_shared<RandomAdjustSharpnessOperation>(data_->degree_, data_->probability_);
}
#endif  // not ENABLE_ANDROID

// RandomAffine Transform Operation.
struct RandomAffine::Data {
  Data(const std::vector<float_t> &degrees, const std::vector<float_t> &translate_range,
       const std::vector<float_t> &scale_range, const std::vector<float_t> &shear_ranges,
       InterpolationMode interpolation, const std::vector<uint8_t> &fill_value)
      : degrees_(degrees),
        translate_range_(translate_range),
        scale_range_(scale_range),
        shear_ranges_(shear_ranges),
        interpolation_(interpolation),
        fill_value_(fill_value) {}
  std::vector<float_t> degrees_;          // min_degree, max_degree
  std::vector<float_t> translate_range_;  // maximum x translation percentage, maximum y translation percentage
  std::vector<float_t> scale_range_;      // min_scale, max_scale
  std::vector<float_t> shear_ranges_;     // min_x_shear, max_x_shear, min_y_shear, max_y_shear
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};

RandomAffine::RandomAffine(const std::vector<float_t> &degrees, const std::vector<float_t> &translate_range,
                           const std::vector<float_t> &scale_range, const std::vector<float_t> &shear_ranges,
                           InterpolationMode interpolation, const std::vector<uint8_t> &fill_value)
    : data_(std::make_shared<Data>(degrees, translate_range, scale_range, shear_ranges, interpolation, fill_value)) {}

std::shared_ptr<TensorOperation> RandomAffine::Parse() {
  return std::make_shared<RandomAffineOperation>(data_->degrees_, data_->translate_range_, data_->scale_range_,
                                                 data_->shear_ranges_, data_->interpolation_, data_->fill_value_);
}

#ifndef ENABLE_ANDROID
// RandomAutoContrast Transform Operation.
struct RandomAutoContrast::Data {
  Data(float cutoff, const std::vector<uint32_t> &ignore, float prob)
      : cutoff_(cutoff), ignore_(ignore), probability_(prob) {}
  float cutoff_;
  std::vector<uint32_t> ignore_;
  float probability_;
};

RandomAutoContrast::RandomAutoContrast(float cutoff, const std::vector<uint32_t> &ignore, float prob)
    : data_(std::make_shared<Data>(cutoff, ignore, prob)) {}

std::shared_ptr<TensorOperation> RandomAutoContrast::Parse() {
  return std::make_shared<RandomAutoContrastOperation>(data_->cutoff_, data_->ignore_, data_->probability_);
}

// RandomColor Transform Operation.
struct RandomColor::Data {
  Data(float t_lb, float t_ub) : t_lb_(t_lb), t_ub_(t_ub) {}
  float t_lb_;
  float t_ub_;
};

RandomColor::RandomColor(float t_lb, float t_ub) : data_(std::make_shared<Data>(t_lb, t_ub)) {}

std::shared_ptr<TensorOperation> RandomColor::Parse() {
  return std::make_shared<RandomColorOperation>(data_->t_lb_, data_->t_ub_);
}

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
struct RandomCropWithBBox::Data {
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

RandomCropWithBBox::RandomCropWithBBox(const std::vector<int32_t> &size, const std::vector<int32_t> &padding,
                                       bool pad_if_needed, const std::vector<uint8_t> &fill_value,
                                       BorderType padding_mode)
    : data_(std::make_shared<Data>(size, padding, pad_if_needed, fill_value, padding_mode)) {}

std::shared_ptr<TensorOperation> RandomCropWithBBox::Parse() {
  return std::make_shared<RandomCropWithBBoxOperation>(data_->size_, data_->padding_, data_->pad_if_needed_,
                                                       data_->fill_value_, data_->padding_mode_);
}

// RandomEqualize Transform Operation.
struct RandomEqualize::Data {
  explicit Data(float prob) : probability_(prob) {}
  float probability_;
};

RandomEqualize::RandomEqualize(float prob) : data_(std::make_shared<Data>(prob)) {}

std::shared_ptr<TensorOperation> RandomEqualize::Parse() {
  return std::make_shared<RandomEqualizeOperation>(data_->probability_);
}

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
struct RandomHorizontalFlipWithBBox::Data {
  explicit Data(float prob) : probability_(prob) {}
  float probability_;
};

RandomHorizontalFlipWithBBox::RandomHorizontalFlipWithBBox(float prob) : data_(std::make_shared<Data>(prob)) {}

std::shared_ptr<TensorOperation> RandomHorizontalFlipWithBBox::Parse() {
  return std::make_shared<RandomHorizontalFlipWithBBoxOperation>(data_->probability_);
}

// RandomInvert Operation.
struct RandomInvert::Data {
  explicit Data(float prob) : probability_(prob) {}
  float probability_;
};

RandomInvert::RandomInvert(float prob) : data_(std::make_shared<Data>(prob)) {}

std::shared_ptr<TensorOperation> RandomInvert::Parse() {
  return std::make_shared<RandomInvertOperation>(data_->probability_);
}

// RandomLighting Transform Operation.
struct RandomLighting::Data {
  explicit Data(float alpha) : alpha_(alpha) {}
  float alpha_;
};

RandomLighting::RandomLighting(float alpha) : data_(std::make_shared<Data>(alpha)) {}

std::shared_ptr<TensorOperation> RandomLighting::Parse() {
  return std::make_shared<RandomLightingOperation>(data_->alpha_);
}

// RandomPosterize Transform Operation.
struct RandomPosterize::Data {
  explicit Data(const std::vector<uint8_t> &bit_range) : bit_range_(bit_range) {}
  std::vector<uint8_t> bit_range_;
};

RandomPosterize::RandomPosterize(const std::vector<uint8_t> &bit_range) : data_(std::make_shared<Data>(bit_range)) {}

std::shared_ptr<TensorOperation> RandomPosterize::Parse() {
  return std::make_shared<RandomPosterizeOperation>(data_->bit_range_);
}

// RandomResize Transform Operation.
struct RandomResize::Data {
  explicit Data(const std::vector<int32_t> &size) : size_(size) {}
  std::vector<int32_t> size_;
};

RandomResize::RandomResize(const std::vector<int32_t> &size) : data_(std::make_shared<Data>(size)) {}

std::shared_ptr<TensorOperation> RandomResize::Parse() { return std::make_shared<RandomResizeOperation>(data_->size_); }

// RandomResizeWithBBox Transform Operation.
struct RandomResizeWithBBox::Data {
  explicit Data(const std::vector<int32_t> &size) : size_(size) {}
  std::vector<int32_t> size_;
};

RandomResizeWithBBox::RandomResizeWithBBox(const std::vector<int32_t> &size) : data_(std::make_shared<Data>(size)) {}

std::shared_ptr<TensorOperation> RandomResizeWithBBox::Parse() {
  return std::make_shared<RandomResizeWithBBoxOperation>(data_->size_);
}

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
struct RandomResizedCropWithBBox::Data {
  Data(const std::vector<int32_t> &size, const std::vector<float> &scale, const std::vector<float> &ratio,
       InterpolationMode interpolation, int32_t max_attempts)
      : size_(size), scale_(scale), ratio_(ratio), interpolation_(interpolation), max_attempts_(max_attempts) {}
  std::vector<int32_t> size_;
  std::vector<float> scale_;
  std::vector<float> ratio_;
  InterpolationMode interpolation_;
  int32_t max_attempts_;
};

RandomResizedCropWithBBox::RandomResizedCropWithBBox(const std::vector<int32_t> &size, const std::vector<float> &scale,
                                                     const std::vector<float> &ratio, InterpolationMode interpolation,
                                                     int32_t max_attempts)
    : data_(std::make_shared<Data>(size, scale, ratio, interpolation, max_attempts)) {}

std::shared_ptr<TensorOperation> RandomResizedCropWithBBox::Parse() {
  return std::make_shared<RandomResizedCropWithBBoxOperation>(data_->size_, data_->scale_, data_->ratio_,
                                                              data_->interpolation_, data_->max_attempts_);
}

// RandomRotation Transform Operation.
struct RandomRotation::Data {
  Data(const std::vector<float> &degrees, InterpolationMode resample, bool expand, const std::vector<float> &center,
       const std::vector<uint8_t> &fill_value)
      : degrees_(degrees), interpolation_mode_(resample), center_(center), expand_(expand), fill_value_(fill_value) {}
  std::vector<float> degrees_;
  InterpolationMode interpolation_mode_;
  std::vector<float> center_;
  bool expand_;
  std::vector<uint8_t> fill_value_;
};

RandomRotation::RandomRotation(const std::vector<float> &degrees, InterpolationMode resample, bool expand,
                               const std::vector<float> &center, const std::vector<uint8_t> &fill_value)
    : data_(std::make_shared<Data>(degrees, resample, expand, center, fill_value)) {}

std::shared_ptr<TensorOperation> RandomRotation::Parse() {
  return std::make_shared<RandomRotationOperation>(data_->degrees_, data_->interpolation_mode_, data_->expand_,
                                                   data_->center_, data_->fill_value_);
}

// RandomSelectSubpolicy Transform Operation.
struct RandomSelectSubpolicy::Data {
  std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> policy_;
};

RandomSelectSubpolicy::RandomSelectSubpolicy(
  const std::vector<std::vector<std::pair<TensorTransform *, double>>> &policy)
    : data_(std::make_shared<Data>()) {
  for (uint32_t i = 0; i < policy.size(); i++) {
    std::vector<std::pair<std::shared_ptr<TensorOperation>, double>> subpolicy;

    for (uint32_t j = 0; j < policy[i].size(); j++) {
      TensorTransform *op = policy[i][j].first;
      std::shared_ptr<TensorOperation> operation = (op ? op->Parse() : nullptr);
      double prob = policy[i][j].second;
      subpolicy.emplace_back(std::move(std::make_pair(operation, prob)));
    }
    data_->policy_.emplace_back(subpolicy);
  }
}

RandomSelectSubpolicy::RandomSelectSubpolicy(
  const std::vector<std::vector<std::pair<std::shared_ptr<TensorTransform>, double>>> &policy)
    : data_(std::make_shared<Data>()) {
  for (uint32_t i = 0; i < policy.size(); i++) {
    std::vector<std::pair<std::shared_ptr<TensorOperation>, double>> subpolicy;

    for (uint32_t j = 0; j < policy[i].size(); j++) {
      std::shared_ptr<TensorTransform> op = policy[i][j].first;
      std::shared_ptr<TensorOperation> operation = (op ? op->Parse() : nullptr);
      double prob = policy[i][j].second;
      subpolicy.emplace_back(std::move(std::make_pair(operation, prob)));
    }
    data_->policy_.emplace_back(subpolicy);
  }
}

RandomSelectSubpolicy::RandomSelectSubpolicy(
  const std::vector<std::vector<std::pair<std::reference_wrapper<TensorTransform>, double>>> &policy)
    : data_(std::make_shared<Data>()) {
  for (int32_t i = 0; i < policy.size(); i++) {
    std::vector<std::pair<std::shared_ptr<TensorOperation>, double>> subpolicy;

    for (int32_t j = 0; j < policy[i].size(); j++) {
      TensorTransform &op = policy[i][j].first;
      std::shared_ptr<TensorOperation> operation = op.Parse();
      double prob = policy[i][j].second;
      subpolicy.emplace_back(std::move(std::make_pair(operation, prob)));
    }
    data_->policy_.emplace_back(subpolicy);
  }
}

std::shared_ptr<TensorOperation> RandomSelectSubpolicy::Parse() {
  return std::make_shared<RandomSelectSubpolicyOperation>(data_->policy_);
}

// RandomSharpness Transform Operation.
struct RandomSharpness::Data {
  explicit Data(const std::vector<float> &degrees) : degrees_(degrees) {}
  std::vector<float> degrees_;
};

RandomSharpness::RandomSharpness(const std::vector<float> &degrees) : data_(std::make_shared<Data>(degrees)) {}

std::shared_ptr<TensorOperation> RandomSharpness::Parse() {
  return std::make_shared<RandomSharpnessOperation>(data_->degrees_);
}

// RandomSolarize Transform Operation.
struct RandomSolarize::Data {
  explicit Data(const std::vector<uint8_t> &threshold) : threshold_(threshold) {}
  std::vector<uint8_t> threshold_;
};

RandomSolarize::RandomSolarize(const std::vector<uint8_t> &threshold) : data_(std::make_shared<Data>(threshold)) {}

std::shared_ptr<TensorOperation> RandomSolarize::Parse() {
  return std::make_shared<RandomSolarizeOperation>(data_->threshold_);
}

// RandomVerticalFlip Transform Operation.
struct RandomVerticalFlip::Data {
  explicit Data(float prob) : probability_(prob) {}
  float probability_;
};

RandomVerticalFlip::RandomVerticalFlip(float prob) : data_(std::make_shared<Data>(prob)) {}

std::shared_ptr<TensorOperation> RandomVerticalFlip::Parse() {
  return std::make_shared<RandomVerticalFlipOperation>(data_->probability_);
}

// RandomVerticalFlipWithBBox Transform Operation.
struct RandomVerticalFlipWithBBox::Data {
  explicit Data(float prob) : probability_(prob) {}
  float probability_;
};

RandomVerticalFlipWithBBox::RandomVerticalFlipWithBBox(float prob) : data_(std::make_shared<Data>(prob)) {}

std::shared_ptr<TensorOperation> RandomVerticalFlipWithBBox::Parse() {
  return std::make_shared<RandomVerticalFlipWithBBoxOperation>(data_->probability_);
}

// ReadFile Function.
Status ReadFile(const std::string &filename, mindspore::MSTensor *output) {
  RETURN_UNEXPECTED_IF_NULL(output);

  std::shared_ptr<Tensor> de_tensor;
  RETURN_IF_NOT_OK(mindspore::dataset::ReadFile(filename, &de_tensor));
  CHECK_FAIL_RETURN_UNEXPECTED(de_tensor->HasData(),
                               "ReadFile: Get an empty tensor with shape " + de_tensor->shape().ToString());
  *output = mindspore::MSTensor(std::make_shared<DETensor>(de_tensor));

  return Status::OK();
}

// ReadImage Function.
Status ReadImage(const std::string &filename, mindspore::MSTensor *output, ImageReadMode mode) {
  RETURN_UNEXPECTED_IF_NULL(output);

  std::shared_ptr<Tensor> de_tensor;
  RETURN_IF_NOT_OK(mindspore::dataset::ReadImage(filename, &de_tensor, mode));
  CHECK_FAIL_RETURN_UNEXPECTED(de_tensor->HasData(),
                               "ReadImage: get an empty tensor with shape " + de_tensor->shape().ToString());
  *output = mindspore::MSTensor(std::make_shared<DETensor>(de_tensor));
  return Status::OK();
}
#endif  // not ENABLE_ANDROID

// Rescale Transform Operation.
struct Rescale::Data {
  Data(float rescale, float shift) : rescale_(rescale), shift_(shift) {}
  float rescale_;
  float shift_;
};

Rescale::Rescale(float rescale, float shift) : data_(std::make_shared<Data>(rescale, shift)) {}

std::shared_ptr<TensorOperation> Rescale::Parse() {
#if !defined(ENABLE_ANDROID) || defined(ENABLE_CLOUD_FUSION_INFERENCE)
  return std::make_shared<RescaleOperation>(data_->rescale_, data_->shift_);
#else
  MS_LOG(ERROR) << "Unsupported Rescale.";
  return nullptr;
#endif
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
                   [](int32_t i) { return (uint32_t)i; });
    return std::make_shared<DvppResizeJpegOperation>(usize_);
#endif
  } else if (env == MapTargetDevice::kCpu) {
    return std::make_shared<ResizeOperation>(data_->size_, data_->interpolation_);
  }
  MS_LOG(ERROR) << "Unsupported MapTargetDevice, only supported kCpu and kAscend310.";
  return nullptr;
}

#ifndef ENABLE_ANDROID
// ResizedCrop Transform Operation.
struct ResizedCrop::Data {
  Data(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<int32_t> &size,
       InterpolationMode interpolation)
      : top_(top), left_(left), height_(height), width_(width), size_(size), interpolation_(interpolation) {}
  int32_t top_;
  int32_t left_;
  int32_t height_;
  int32_t width_;
  std::vector<int32_t> size_;
  InterpolationMode interpolation_;
};

ResizedCrop::ResizedCrop(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<int32_t> &size,
                         InterpolationMode interpolation)
    : data_(std::make_shared<Data>(top, left, height, width, size, interpolation)) {}

std::shared_ptr<TensorOperation> ResizedCrop::Parse() {
  return std::make_shared<ResizedCropOperation>(data_->top_, data_->left_, data_->height_, data_->width_, data_->size_,
                                                data_->interpolation_);
}
#endif  // not ENABLE_ANDROID

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

#ifndef ENABLE_ANDROID
// ResizeWithBBox Transform Operation.
struct ResizeWithBBox::Data {
  Data(const std::vector<int32_t> &size, InterpolationMode interpolation)
      : size_(size), interpolation_(interpolation) {}
  std::vector<int32_t> size_;
  InterpolationMode interpolation_;
};

ResizeWithBBox::ResizeWithBBox(const std::vector<int32_t> &size, InterpolationMode interpolation)
    : data_(std::make_shared<Data>(size, interpolation)) {}

std::shared_ptr<TensorOperation> ResizeWithBBox::Parse() {
  return std::make_shared<ResizeWithBBoxOperation>(data_->size_, data_->interpolation_);
}
#endif  // not ENABLE_ANDROID

// RGB2BGR Transform Operation.
std::shared_ptr<TensorOperation> RGB2BGR::Parse() { return std::make_shared<RgbToBgrOperation>(); }

// RGB2GRAY Transform Operation.
std::shared_ptr<TensorOperation> RGB2GRAY::Parse() { return std::make_shared<RgbToGrayOperation>(); }

// Rotate Transform Operation.
struct Rotate::Data {
  Data(const float &degrees, InterpolationMode resample, bool expand, const std::vector<float> &center,
       const std::vector<uint8_t> &fill_value)
      : degrees_(degrees), interpolation_mode_(resample), center_(center), expand_(expand), fill_value_(fill_value) {}
  explicit Data(const FixRotationAngle &angle_id) : angle_id_(angle_id), lite_impl_(true) {}
  FixRotationAngle angle_id_{FixRotationAngle::k0Degree};
  bool lite_impl_{false};
  float degrees_{0};
  InterpolationMode interpolation_mode_{InterpolationMode::kNearestNeighbour};
  std::vector<float> center_{{}};
  bool expand_{false};
  std::vector<uint8_t> fill_value_{0, 0, 0};
};

Rotate::Rotate(FixRotationAngle angle_id) : data_(std::make_shared<Data>(angle_id)) {}

Rotate::Rotate(float degrees, InterpolationMode resample, bool expand, const std::vector<float> &center,
               const std::vector<uint8_t> &fill_value)
    : data_(std::make_shared<Data>(degrees, resample, expand, center, fill_value)) {}

std::shared_ptr<TensorOperation> Rotate::Parse() {
#ifndef ENABLE_ANDROID
  if (!data_->lite_impl_) {
    return std::make_shared<RotateOperation>(data_->degrees_, data_->interpolation_mode_, data_->expand_,
                                             data_->center_, data_->fill_value_);
  }
#else
  if (data_->lite_impl_) {
    return std::make_shared<RotateOperation>(data_->angle_id_);
  }
#endif  // not ENABLE_ANDROID
  std::string platform = data_->lite_impl_ ? "Cloud" : "Android";
  MS_LOG(ERROR) << "This Rotate API is not supported for " + platform + ", use another Rotate API.";
  return nullptr;
}

#ifndef ENABLE_ANDROID
// RgbaToBgr Transform Operation.
RGBA2BGR::RGBA2BGR() = default;

std::shared_ptr<TensorOperation> RGBA2BGR::Parse() { return std::make_shared<RgbaToBgrOperation>(); }

// RgbaToRgb Transform Operation.
RGBA2RGB::RGBA2RGB() = default;

std::shared_ptr<TensorOperation> RGBA2RGB::Parse() { return std::make_shared<RgbaToRgbOperation>(); }

// SlicePatches Transform Operation.
struct SlicePatches::Data {
  Data(int32_t num_height, int32_t num_width, SliceMode slice_mode, uint8_t fill_value)
      : num_height_(num_height), num_width_(num_width), slice_mode_(slice_mode), fill_value_(fill_value) {}
  int32_t num_height_;
  int32_t num_width_;
  SliceMode slice_mode_;
  uint8_t fill_value_;
};

SlicePatches::SlicePatches(int32_t num_height, int32_t num_width, SliceMode slice_mode, uint8_t fill_value)
    : data_(std::make_shared<Data>(num_height, num_width, slice_mode, fill_value)) {}

std::shared_ptr<TensorOperation> SlicePatches::Parse() {
  return std::make_shared<SlicePatchesOperation>(data_->num_height_, data_->num_width_, data_->slice_mode_,
                                                 data_->fill_value_);
}

// Solarize Transform Operation.
struct Solarize::Data {
  explicit Data(const std::vector<float> &threshold) : threshold_(threshold) {}
  std::vector<float> threshold_;
};

Solarize::Solarize(const std::vector<float> &threshold) : data_(std::make_shared<Data>(threshold)) {}

std::shared_ptr<TensorOperation> Solarize::Parse() { return std::make_shared<SolarizeOperation>(data_->threshold_); }
#endif  // not ENABLE_ANDROID

// SwapRedBlue Transform Operation.
SwapRedBlue::SwapRedBlue() = default;
std::shared_ptr<TensorOperation> SwapRedBlue::Parse() {
#if !defined(ENABLE_ANDROID) || defined(ENABLE_CLOUD_FUSION_INFERENCE)
  return std::make_shared<SwapRedBlueOperation>();
#else
  MS_LOG(ERROR) << "Unsupported SwapRedBlue.";
  return nullptr;
#endif
}

#ifndef ENABLE_ANDROID
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
struct TrivialAugmentWide::Data {
  Data(int32_t num_magnitude_bins, InterpolationMode interpolation, const std::vector<uint8_t> &fill_value)
      : num_magnitude_bins_(num_magnitude_bins), interpolation_(interpolation), fill_value_(fill_value) {}
  int32_t num_magnitude_bins_;
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};

TrivialAugmentWide::TrivialAugmentWide(int32_t num_magnitude_bins, InterpolationMode interpolation,
                                       const std::vector<uint8_t> &fill_value)
    : data_(std::make_shared<Data>(num_magnitude_bins, interpolation, fill_value)) {}

std::shared_ptr<TensorOperation> TrivialAugmentWide::Parse() {
  return std::make_shared<TrivialAugmentWideOperation>(data_->num_magnitude_bins_, data_->interpolation_,
                                                       data_->fill_value_);
}

// UniformAug Transform Operation.
struct UniformAugment::Data {
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
  int32_t num_ops_;
};

UniformAugment::UniformAugment(const std::vector<TensorTransform *> &transforms, int32_t num_ops)
    : data_(std::make_shared<Data>()) {
  (void)std::transform(
    transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
    [](TensorTransform *const op) -> std::shared_ptr<TensorOperation> { return op ? op->Parse() : nullptr; });
  data_->num_ops_ = num_ops;
}

UniformAugment::UniformAugment(const std::vector<std::shared_ptr<TensorTransform>> &transforms, int32_t num_ops)
    : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](const std::shared_ptr<TensorTransform> &op) -> std::shared_ptr<TensorOperation> {
                         return op ? op->Parse() : nullptr;
                       });
  data_->num_ops_ = num_ops;
}

UniformAugment::UniformAugment(const std::vector<std::reference_wrapper<TensorTransform>> &transforms, int32_t num_ops)
    : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](TensorTransform &op) -> std::shared_ptr<TensorOperation> { return op.Parse(); });
  data_->num_ops_ = num_ops;
}

std::shared_ptr<TensorOperation> UniformAugment::Parse() {
  return std::make_shared<UniformAugOperation>(data_->transforms_, data_->num_ops_);
}

// VerticalFlip Transform Operation.
VerticalFlip::VerticalFlip() = default;

std::shared_ptr<TensorOperation> VerticalFlip::Parse() { return std::make_shared<VerticalFlipOperation>(); }

// WriteFile Function.
Status WriteFile(const std::string &filename, const mindspore::MSTensor &data) {
  std::shared_ptr<dataset::Tensor> de_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateFromMSTensor(data, &de_tensor));
  RETURN_IF_NOT_OK(mindspore::dataset::WriteFile(filename, de_tensor));
  return Status::OK();
}

// WriteJpeg Function.
Status WriteJpeg(const std::string &filename, const mindspore::MSTensor &image, int quality) {
  std::shared_ptr<dataset::Tensor> image_de_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateFromMSTensor(image, &image_de_tensor));
  RETURN_IF_NOT_OK(mindspore::dataset::WriteJpeg(filename, image_de_tensor, quality));
  return Status::OK();
}

// WritePNG Function.
Status WritePng(const std::string &filename, const mindspore::MSTensor &image, int compression_level) {
  std::shared_ptr<dataset::Tensor> image_de_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateFromMSTensor(image, &image_de_tensor));
  RETURN_IF_NOT_OK(mindspore::dataset::WritePng(filename, image_de_tensor, compression_level));
  return Status::OK();
}
#endif  // not ENABLE_ANDROID
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
