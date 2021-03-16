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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_VISION_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_VISION_IR_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/api/status.h"
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/kernels/ir/tensor_operation.h"

namespace mindspore {
namespace dataset {

// Transform operations for performing computer vision.
namespace vision {

// Char arrays storing name of corresponding classes (in alphabetical order)
constexpr char kAffineOperation[] = "Affine";
constexpr char kAutoContrastOperation[] = "AutoContrast";
constexpr char kBoundingBoxAugmentOperation[] = "BoundingBoxAugment";
constexpr char kCenterCropOperation[] = "CenterCrop";
constexpr char kCropOperation[] = "Crop";
constexpr char kCutMixBatchOperation[] = "CutMixBatch";
constexpr char kCutOutOperation[] = "CutOut";
constexpr char kDecodeOperation[] = "Decode";
constexpr char kEqualizeOperation[] = "Equalize";
constexpr char kHwcToChwOperation[] = "HwcToChw";
constexpr char kInvertOperation[] = "Invert";
constexpr char kMixUpBatchOperation[] = "MixUpBatch";
constexpr char kNormalizeOperation[] = "Normalize";
constexpr char kNormalizePadOperation[] = "NormalizePad";
constexpr char kPadOperation[] = "Pad";
constexpr char kRandomAffineOperation[] = "RandomAffine";
constexpr char kRandomColorAdjustOperation[] = "RandomColorAdjust";
constexpr char kRandomColorOperation[] = "RandomColor";
constexpr char kRandomCropDecodeResizeOperation[] = "RandomCropDecodeResize";
constexpr char kRandomCropOperation[] = "RandomCrop";
constexpr char kRandomCropWithBBoxOperation[] = "RandomCropWithBBox";
constexpr char kRandomHorizontalFlipOperation[] = "RandomHorizontalFlip";
constexpr char kRandomHorizontalFlipWithBBoxOperation[] = "RandomHorizontalFlipWithBBox";
constexpr char kRandomPosterizeOperation[] = "RandomPosterize";
constexpr char kRandomResizedCropOperation[] = "RandomResizedCrop";
constexpr char kRandomResizedCropWithBBoxOperation[] = "RandomResizedCropWithBBox";
constexpr char kRandomResizeOperation[] = "RandomResize";
constexpr char kRandomResizeWithBBoxOperation[] = "RandomResizeWithBBox";
constexpr char kRandomRotationOperation[] = "RandomRotation";
constexpr char kRandomSelectSubpolicyOperation[] = "RandomSelectSubpolicy";
constexpr char kRandomSharpnessOperation[] = "RandomSharpness";
constexpr char kRandomSolarizeOperation[] = "RandomSolarize";
constexpr char kRandomVerticalFlipOperation[] = "RandomVerticalFlip";
constexpr char kRandomVerticalFlipWithBBoxOperation[] = "RandomVerticalFlipWithBBox";
constexpr char kRescaleOperation[] = "Rescale";
constexpr char kResizeOperation[] = "Resize";
constexpr char kResizePreserveAROperation[] = "ResizePreserveAR";
constexpr char kResizeWithBBoxOperation[] = "ResizeWithBBox";
constexpr char kRgbaToBgrOperation[] = "RgbaToBgr";
constexpr char kRgbaToRgbOperation[] = "RgbaToRgb";
constexpr char kRgbToGrayOperation[] = "RgbToGray";
constexpr char kRotateOperation[] = "Rotate";
constexpr char kSoftDvppDecodeRandomCropResizeJpegOperation[] = "SoftDvppDecodeRandomCropResizeJpeg";
constexpr char kSoftDvppDecodeResizeJpegOperation[] = "SoftDvppDecodeResizeJpeg";
constexpr char kSwapRedBlueOperation[] = "SwapRedBlue";
constexpr char kUniformAugOperation[] = "UniformAug";

/* ####################################### Derived TensorOperation classes ################################# */

class AffineOperation : public TensorOperation {
 public:
  explicit AffineOperation(float_t degrees, const std::vector<float> &translation, float scale,
                           const std::vector<float> &shear, InterpolationMode interpolation,
                           const std::vector<uint8_t> &fill_value);

  ~AffineOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kAffineOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  float degrees_;
  std::vector<float> translation_;
  float scale_;
  std::vector<float> shear_;
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};

class AutoContrastOperation : public TensorOperation {
 public:
  explicit AutoContrastOperation(float cutoff, std::vector<uint32_t> ignore);

  ~AutoContrastOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kAutoContrastOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  float cutoff_;
  std::vector<uint32_t> ignore_;
};

class BoundingBoxAugmentOperation : public TensorOperation {
 public:
  explicit BoundingBoxAugmentOperation(std::shared_ptr<TensorOperation> transform, float ratio);

  ~BoundingBoxAugmentOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kBoundingBoxAugmentOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::shared_ptr<TensorOperation> transform_;
  float ratio_;
};

class CenterCropOperation : public TensorOperation {
 public:
  explicit CenterCropOperation(std::vector<int32_t> size);

  ~CenterCropOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kCenterCropOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::vector<int32_t> size_;
};

class RgbToGrayOperation : public TensorOperation {
 public:
  RgbToGrayOperation() = default;

  ~RgbToGrayOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRgbToGrayOperation; }
};

class CropOperation : public TensorOperation {
 public:
  CropOperation(std::vector<int32_t> coordinates, std::vector<int32_t> size);

  ~CropOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kCropOperation; }

 private:
  std::vector<int32_t> coordinates_;
  std::vector<int32_t> size_;
};

class CutMixBatchOperation : public TensorOperation {
 public:
  explicit CutMixBatchOperation(ImageBatchFormat image_batch_format, float alpha, float prob);

  ~CutMixBatchOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kCutMixBatchOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  float alpha_;
  float prob_;
  ImageBatchFormat image_batch_format_;
};

class CutOutOperation : public TensorOperation {
 public:
  explicit CutOutOperation(int32_t length, int32_t num_patches);

  ~CutOutOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kCutOutOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t length_;
  int32_t num_patches_;
};

class DecodeOperation : public TensorOperation {
 public:
  explicit DecodeOperation(bool rgb);

  ~DecodeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDecodeOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  bool rgb_;
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
  explicit MixUpBatchOperation(float alpha);

  ~MixUpBatchOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kMixUpBatchOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  float alpha_;
};

class NormalizeOperation : public TensorOperation {
 public:
  NormalizeOperation(std::vector<float> mean, std::vector<float> std);

  ~NormalizeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kNormalizeOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
};

class NormalizePadOperation : public TensorOperation {
 public:
  NormalizePadOperation(const std::vector<float> &mean, const std::vector<float> &std, const std::string &dtype);

  ~NormalizePadOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kNormalizePadOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
  std::string dtype_;
};

class PadOperation : public TensorOperation {
 public:
  PadOperation(std::vector<int32_t> padding, std::vector<uint8_t> fill_value, BorderType padding_mode);

  ~PadOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kPadOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::vector<int32_t> padding_;
  std::vector<uint8_t> fill_value_;
  BorderType padding_mode_;
};

class RandomAffineOperation : public TensorOperation {
 public:
  RandomAffineOperation(const std::vector<float_t> &degrees, const std::vector<float_t> &translate_range,
                        const std::vector<float_t> &scale_range, const std::vector<float_t> &shear_ranges,
                        InterpolationMode interpolation, const std::vector<uint8_t> &fill_value);

  ~RandomAffineOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomAffineOperation; }

  Status to_json(nlohmann::json *out_json) override;

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

  Status to_json(nlohmann::json *out_json) override;

 private:
  float t_lb_;
  float t_ub_;
};

class RandomColorAdjustOperation : public TensorOperation {
 public:
  RandomColorAdjustOperation(std::vector<float> brightness, std::vector<float> contrast, std::vector<float> saturation,
                             std::vector<float> hue);

  ~RandomColorAdjustOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomColorAdjustOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::vector<float> brightness_;
  std::vector<float> contrast_;
  std::vector<float> saturation_;
  std::vector<float> hue_;
};

class RandomCropOperation : public TensorOperation {
 public:
  RandomCropOperation(std::vector<int32_t> size, std::vector<int32_t> padding, bool pad_if_needed,
                      std::vector<uint8_t> fill_value, BorderType padding_mode);

  ~RandomCropOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomCropOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::vector<int32_t> size_;
  std::vector<int32_t> padding_;
  bool pad_if_needed_;
  std::vector<uint8_t> fill_value_;
  BorderType padding_mode_;
};

class RandomResizedCropOperation : public TensorOperation {
 public:
  RandomResizedCropOperation(std::vector<int32_t> size, std::vector<float> scale, std::vector<float> ratio,
                             InterpolationMode interpolation, int32_t max_attempts);

  /// \brief default copy constructor
  explicit RandomResizedCropOperation(const RandomResizedCropOperation &) = default;

  ~RandomResizedCropOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomResizedCropOperation; }

  Status to_json(nlohmann::json *out_json) override;

 protected:
  std::vector<int32_t> size_;
  std::vector<float> scale_;
  std::vector<float> ratio_;
  InterpolationMode interpolation_;
  int32_t max_attempts_;
};

class RandomCropDecodeResizeOperation : public RandomResizedCropOperation {
 public:
  RandomCropDecodeResizeOperation(std::vector<int32_t> size, std::vector<float> scale, std::vector<float> ratio,
                                  InterpolationMode interpolation, int32_t max_attempts);

  explicit RandomCropDecodeResizeOperation(const RandomResizedCropOperation &base);

  ~RandomCropDecodeResizeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  std::string Name() const override { return kRandomCropDecodeResizeOperation; }

  Status to_json(nlohmann::json *out_json) override;
};

class RandomCropWithBBoxOperation : public TensorOperation {
 public:
  RandomCropWithBBoxOperation(std::vector<int32_t> size, std::vector<int32_t> padding, bool pad_if_needed,
                              std::vector<uint8_t> fill_value, BorderType padding_mode);

  ~RandomCropWithBBoxOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomCropWithBBoxOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::vector<int32_t> size_;
  std::vector<int32_t> padding_;
  bool pad_if_needed_;
  std::vector<uint8_t> fill_value_;
  BorderType padding_mode_;
};

class RandomHorizontalFlipOperation : public TensorOperation {
 public:
  explicit RandomHorizontalFlipOperation(float probability);

  ~RandomHorizontalFlipOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomHorizontalFlipOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  float probability_;
};

class RandomHorizontalFlipWithBBoxOperation : public TensorOperation {
 public:
  explicit RandomHorizontalFlipWithBBoxOperation(float probability);

  ~RandomHorizontalFlipWithBBoxOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomHorizontalFlipWithBBoxOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  float probability_;
};

class RandomPosterizeOperation : public TensorOperation {
 public:
  explicit RandomPosterizeOperation(const std::vector<uint8_t> &bit_range);

  ~RandomPosterizeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomPosterizeOperation; }

  Status to_json(nlohmann::json *out_json) override;

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

  Status to_json(nlohmann::json *out_json) override;

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

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::vector<int32_t> size_;
};

class RandomResizedCropWithBBoxOperation : public TensorOperation {
 public:
  explicit RandomResizedCropWithBBoxOperation(std::vector<int32_t> size, std::vector<float> scale,
                                              std::vector<float> ratio, InterpolationMode interpolation,
                                              int32_t max_attempts);

  ~RandomResizedCropWithBBoxOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomResizedCropWithBBoxOperation; }

  Status to_json(nlohmann::json *out_json) override;

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

  Status to_json(nlohmann::json *out_json) override;

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

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> policy_;
};

class RandomSharpnessOperation : public TensorOperation {
 public:
  explicit RandomSharpnessOperation(std::vector<float> degrees);

  ~RandomSharpnessOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomSharpnessOperation; }

  Status to_json(nlohmann::json *out_json) override;

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

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::vector<uint8_t> threshold_;
};

class RandomVerticalFlipOperation : public TensorOperation {
 public:
  explicit RandomVerticalFlipOperation(float probability);

  ~RandomVerticalFlipOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomVerticalFlipOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  float probability_;
};

class RandomVerticalFlipWithBBoxOperation : public TensorOperation {
 public:
  explicit RandomVerticalFlipWithBBoxOperation(float probability);

  ~RandomVerticalFlipWithBBoxOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomVerticalFlipWithBBoxOperation; }

  Status to_json(nlohmann::json *out_json) override;

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

  Status to_json(nlohmann::json *out_json) override;

 private:
  float rescale_;
  float shift_;
};

class ResizeOperation : public TensorOperation {
 public:
  explicit ResizeOperation(std::vector<int32_t> size, InterpolationMode interpolation_mode);

  ~ResizeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kResizeOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::vector<int32_t> size_;
  InterpolationMode interpolation_;
};

class ResizePreserveAROperation : public TensorOperation {
 public:
  ResizePreserveAROperation(int32_t height, int32_t width, int32_t img_orientation);

  ~ResizePreserveAROperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kResizePreserveAROperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t height_;
  int32_t width_;
  int32_t img_orientation_;
};

class ResizeWithBBoxOperation : public TensorOperation {
 public:
  explicit ResizeWithBBoxOperation(std::vector<int32_t> size, InterpolationMode interpolation_mode);

  ~ResizeWithBBoxOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kResizeWithBBoxOperation; }

  Status to_json(nlohmann::json *out_json) override;

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

class RotateOperation : public TensorOperation {
 public:
  RotateOperation();

  ~RotateOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRotateOperation; }

  void setAngle(uint64_t angle_id);

 private:
  std::shared_ptr<TensorOp> rotate_op;
};

class SoftDvppDecodeRandomCropResizeJpegOperation : public TensorOperation {
 public:
  explicit SoftDvppDecodeRandomCropResizeJpegOperation(std::vector<int32_t> size, std::vector<float> scale,
                                                       std::vector<float> ratio, int32_t max_attempts);

  ~SoftDvppDecodeRandomCropResizeJpegOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kSoftDvppDecodeRandomCropResizeJpegOperation; }

  Status to_json(nlohmann::json *out_json) override;

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

  Status to_json(nlohmann::json *out_json) override;

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
  explicit UniformAugOperation(std::vector<std::shared_ptr<TensorOperation>> transforms, int32_t num_ops);

  ~UniformAugOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kUniformAugOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
  int32_t num_ops_;
};

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_VISION_IR_H_
