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

#include "minddata/dataset/vision/transform/adjust_gamma_ir.h"
#include "minddata/dataset/vision/transform/auto_augment_ir.h"
#include "minddata/dataset/vision/transform/bounding_box_augment_ir.h"
#include "minddata/dataset/vision/transform/center_crop_ir.h"
#include "minddata/dataset/vision/transform/crop_ir.h"
#include "minddata/dataset/vision/transform/cutmix_batch_ir.h"
#include "minddata/dataset/vision/transform/decode_ir.h"
#ifdef ENABLE_FFMPEG
#include "minddata/dataset/vision/transform/decode_video_ir.h"
#endif
#include "minddata/dataset/vision/transform/erase_ir.h"
#include "minddata/dataset/vision/transform/hwc_to_chw_ir.h"
#include "minddata/dataset/vision/transform/mixup_batch_ir.h"
#include "minddata/dataset/vision/transform/normalize_ir.h"
#include "minddata/dataset/vision/transform/normalize_pad_ir.h"
#include "minddata/dataset/vision/transform/perspective_ir.h"
#include "minddata/dataset/vision/transform/posterize_ir.h"
#include "minddata/dataset/vision/transform/rand_augment_ir.h"
#include "minddata/dataset/vision/transform/random_color_adjust_ir.h"
#include "minddata/dataset/vision/transform/random_crop_decode_resize_ir.h"
#include "minddata/dataset/vision/transform/random_crop_ir.h"
#include "minddata/dataset/vision/transform/random_crop_with_bbox_ir.h"
#include "minddata/dataset/vision/transform/random_horizontal_flip_ir.h"
#include "minddata/dataset/vision/transform/random_horizontal_flip_with_bbox_ir.h"
#include "minddata/dataset/vision/transform/random_lighting_ir.h"
#include "minddata/dataset/vision/transform/random_posterize_ir.h"
#include "minddata/dataset/vision/transform/random_resize_with_bbox_ir.h"
#include "minddata/dataset/vision/transform/random_resized_crop_ir.h"
#include "minddata/dataset/vision/transform/random_resized_crop_with_bbox_ir.h"
#include "minddata/dataset/vision/transform/random_solarize_ir.h"
#include "minddata/dataset/vision/transform/random_vertical_flip_ir.h"
#include "minddata/dataset/vision/transform/random_vertical_flip_with_bbox_ir.h"
#include "minddata/dataset/vision/transform/resize_ir.h"
#include "minddata/dataset/vision/transform/resize_preserve_ar_ir.h"
#include "minddata/dataset/vision/transform/resize_with_bbox_ir.h"
#include "minddata/dataset/vision/transform/rgb_to_bgr_ir.h"
#include "minddata/dataset/vision/transform/rgb_to_gray_ir.h"
#include "minddata/dataset/vision/transform/rgba_to_bgr_ir.h"
#include "minddata/dataset/vision/transform/rgba_to_rgb_ir.h"
#include "minddata/dataset/vision/transform/rotate_ir.h"
#include "minddata/dataset/vision/transform/solarize_ir.h"
#include "minddata/dataset/vision/transform/to_tensor_ir.h"
#include "minddata/dataset/vision/transform/trivial_augment_wide_ir.h"
#include "minddata/dataset/util/log_adapter.h"

#include "minddata/dataset/vision/transform/pad_ir.h"
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

// AdjustSaturation Transform Operation.

// AdjustSharpness Transform Operation.

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
// DecodeVideo Transform Operation.
DecodeVideo::DecodeVideo() {}

std::shared_ptr<TensorOperation> DecodeVideo::Parse() { return std::make_shared<DecodeVideoOperation>(); }
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

// Erase Operation.
struct Erase::Data {
  Data(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<float> &value, bool inplace)
      : top_(top), left_(left), height_(height), width_(width), value_(value), inplace_(inplace) {}
  int32_t top_;
  int32_t left_;
  int32_t height_;
  int32_t width_;
  std::vector<float> value_;
  bool inplace_;
};

Erase::Erase(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<float> &value, bool inplace)
    : data_(std::make_shared<Data>(top, left, height, width, value, inplace)) {}

std::shared_ptr<TensorOperation> Erase::Parse() {
  return std::make_shared<EraseOperation>(data_->top_, data_->left_, data_->height_, data_->width_, data_->value_,
                                          data_->inplace_);
}

// GaussianBlur Transform Operation.

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

// HwcToChw Transform Operation.
HWC2CHW::HWC2CHW() = default;

std::shared_ptr<TensorOperation> HWC2CHW::Parse() { return std::make_shared<HwcToChwOperation>(); }

// Invert Transform Operation.

// MixUpBatch Transform Operation.
struct MixUpBatch::Data {
  explicit Data(float alpha) : alpha_(alpha) {}
  float alpha_;
};

MixUpBatch::MixUpBatch(float alpha) : data_(std::make_shared<Data>(alpha)) {}

std::shared_ptr<TensorOperation> MixUpBatch::Parse() { return std::make_shared<MixUpBatchOperation>(data_->alpha_); }

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
  return std::make_shared<PadOperation>(data_->padding_, data_->fill_value_, data_->padding_mode_);
}

// PadToSize Transform Operation.

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

// RandomSelectSubpolicy Transform Operation.

// RandomSharpness Transform Operation.

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

#ifdef ENABLE_FFMPEG
// ReadVideo Function.
Status ReadVideo(const std::string &filename, mindspore::MSTensor *video_output, mindspore::MSTensor *audio_output,
                 std::map<std::string, std::string> *metadata_output, float start_pts, float end_pts,
                 const std::string &pts_unit) {
  RETURN_UNEXPECTED_IF_NULL(video_output);
  RETURN_UNEXPECTED_IF_NULL(audio_output);
  RETURN_UNEXPECTED_IF_NULL(metadata_output);

  std::shared_ptr<Tensor> de_video_output;
  std::shared_ptr<Tensor> de_audio_output;
  RETURN_IF_NOT_OK(mindspore::dataset::ReadVideo(filename, &de_video_output, &de_audio_output, metadata_output,
                                                 start_pts, end_pts, pts_unit));
  *video_output = mindspore::MSTensor(std::make_shared<DETensor>(de_video_output));
  *audio_output = mindspore::MSTensor(std::make_shared<DETensor>(de_audio_output));
  return Status::OK();
}

// ReadVideoTimestamps.
Status ReadVideoTimestamps(const std::string &filename, std::tuple<std::vector<float>, float> *output,
                           const std::string &pts_unit) {
  RETURN_UNEXPECTED_IF_NULL(output);
  std::vector<int64_t> pts_int64_vector;
  float video_fps;
  float time_base;

  RETURN_IF_NOT_OK(
    mindspore::dataset::ReadVideoTimestamps(filename, &pts_int64_vector, &video_fps, &time_base, pts_unit));

  std::vector<float> pts_float_vector;

  if (pts_unit == "pts") {
    for (int64_t pts_int64 : pts_int64_vector) {
      pts_float_vector.push_back(static_cast<float>(pts_int64));
    }
  }
  if (pts_unit == "sec") {
    for (int64_t pts_int64 : pts_int64_vector) {
      pts_float_vector.push_back(static_cast<float>(pts_int64 * time_base));
    }
  }
  *output = std::make_tuple(pts_float_vector, video_fps);
  return Status::OK();
}
#endif

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
struct Solarize::Data {
  explicit Data(const std::vector<float> &threshold) : threshold_(threshold) {}
  std::vector<float> threshold_;
};

Solarize::Solarize(const std::vector<float> &threshold) : data_(std::make_shared<Data>(threshold)) {}

std::shared_ptr<TensorOperation> Solarize::Parse() { return std::make_shared<SolarizeOperation>(data_->threshold_); }

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

// VerticalFlip Transform Operation.

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
}  // namespace mindspore::dataset::vision
