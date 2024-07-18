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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_TENSOR_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_TENSOR_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

#include "minddata/dataset/core/device_resource.h"
#include "minddata/dataset/core/device_tensor.h"
#if !defined(BUILD_LITE) && defined(ENABLE_D)
#include "minddata/dataset/core/device_tensor_ascend910b.h"
#endif
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/engine/perf/info_collector.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

#define IO_CHECK(input, output)                             \
  do {                                                      \
    if (input == nullptr || output == nullptr) {            \
      RETURN_STATUS_UNEXPECTED("input or output is null."); \
    }                                                       \
  } while (false)

#define IO_CHECK_VECTOR(input, output)              \
  do {                                              \
    if (output == nullptr) {                        \
      RETURN_STATUS_UNEXPECTED("output is null.");  \
    }                                               \
    for (auto &_i : input) {                        \
      if (_i == nullptr) {                          \
        RETURN_STATUS_UNEXPECTED("input is null."); \
      }                                             \
    }                                               \
  } while (false)

namespace mindspore {
namespace dataset {
// base class
constexpr char kTensorOp[] = "TensorOp";

// image
constexpr char kAdjustBrightnessOp[] = "AdjustBrightnessOp";
constexpr char kAdjustContrastOp[] = "AdjustContrastOp";
constexpr char kAdjustGammaOp[] = "AdjustGammaOp";
constexpr char kAdjustHueOp[] = "AdjustHueOp";
constexpr char kAdjustSaturationOp[] = "AdjustSaturationOp";
constexpr char kAffineOp[] = "AffineOp";
constexpr char kAutoAugmentOp[] = "AutoAugmentOp";
constexpr char kAutoContrastOp[] = "AutoContrastOp";
constexpr char kBoundingBoxAugmentOp[] = "BoundingBoxAugmentOp";
constexpr char kDecodeOp[] = "DecodeOp";
#ifdef ENABLE_FFMPEG
constexpr char kDecodeVideoOp[] = "DecodeVideoOp";
#endif
constexpr char kCenterCropOp[] = "CenterCropOp";
constexpr char kConvertColorOp[] = "ConvertColorOp";
constexpr char kCutMixBatchOp[] = "CutMixBatchOp";
constexpr char kCutOutOp[] = "CutOutOp";
constexpr char kCropOp[] = "CropOp";
// Ascend310 DVPP just support C++ Interface API
constexpr char kDvppCropJpegOp[] = "DvppCropJpegOp";
constexpr char kDvppDecodeResizeCropJpegOp[] = "DvppDecodeResizeCropJpegOp";
constexpr char kDvppDecodeResizeJpegOp[] = "DvppDecodeResizeJpegOp";
constexpr char kDvppDecodeJpegOp[] = "DvppDecodeJpegOp";
constexpr char kDvppDecodePngOp[] = "DvppDecodePngOp";
constexpr char kDvppNormalizeOp[] = "DvppNormalizeOp";  // used by Ascend310 and Ascend910B
constexpr char kDvppResizeJpegOp[] = "DvppResizeJpegOp";
// Ascend910B DVPP used for Python Interface API
constexpr char kDvppAdjustBrightnessOp[] = "DvppAdjustBrightnessOp";
constexpr char kDvppAdjustContrastOp[] = "DvppAdjustContrastOp";
constexpr char kDvppAdjustHueOp[] = "DvppAdjustHueOp";
constexpr char kDvppAdjustSaturationOp[] = "DvppAdjustSaturationOp";
constexpr char kDvppAdjustSharpnessOp[] = "DvppAdjustSharpnessOp";
constexpr char kDvppAffineOp[] = "DvppAffineOp";
constexpr char kDvppAutoContrastOp[] = "DvppAutoContrastOp";
constexpr char kDvppConvertColorOp[] = "DvppConvertColorOp";
constexpr char kDvppCropOp[] = "DvppCropOp";
constexpr char kDvppDecodeOp[] = "DvppDecodeOp";
constexpr char kDvppEqualizeOp[] = "DvppEqualizeOp";
constexpr char kDvppEraseOp[] = "DvppEraseOp";
constexpr char kDvppGaussianBlurOp[] = "DvppGaussianBlurOp";
constexpr char kDvppHorizontalFlipOp[] = "DvppHorizontalFlipOp";
constexpr char kDvppInvertOp[] = "DvppInvertOp";
constexpr char kDvppPadOp[] = "DvppPadOp";
constexpr char kDvppPerspectiveOp[] = "DvppPerspectiveOp";
constexpr char kDvppPosterizeOp[] = "DvppPosterizeOp";
constexpr char kDvppResizeOp[] = "DvppResizeOp";
constexpr char kDvppResizedCropOp[] = "DvppResizedCropOp";
constexpr char kDvppRotateOp[] = "DvppRotateOp";
constexpr char kDvppSolarizeOp[] = "DvppSolarizeOp";
constexpr char kDvppVerticalFlipOp[] = "DvppVerticalFlipOp";
constexpr char kEqualizeOp[] = "EqualizeOp";
constexpr char kEraseOp[] = "EraseOp";
constexpr char kGaussianBlurOp[] = "GaussianBlurOp";
constexpr char kHorizontalFlipOp[] = "HorizontalFlipOp";
constexpr char kHwcToChwOp[] = "HWC2CHWOp";
constexpr char kInvertOp[] = "InvertOp";
constexpr char kMixUpBatchOp[] = "MixUpBatchOp";
constexpr char kNormalizeOp[] = "NormalizeOp";
constexpr char kNormalizePadOp[] = "NormalizePadOp";
constexpr char kPadOp[] = "PadOp";
constexpr char kPadToSizeOp[] = "PadToSizeOp";
constexpr char kPerspectiveOp[] = "PerspectiveOp";
constexpr char kPosterizeOp[] = "PosterizeOp";
constexpr char kRandAugmentOp[] = "RandAugmentOp";
constexpr char kRandomAdjustSharpnessOp[] = "RandomAdjustSharpnessOp";
constexpr char kRandomAffineOp[] = "RandomAffineOp";
constexpr char kRandomAutoContrastOp[] = "RandomAutoContrastOp";
constexpr char kRandomColorAdjustOp[] = "RandomColorAdjustOp";
constexpr char kRandomColorOp[] = "RandomColorOp";
constexpr char kRandomCropAndResizeOp[] = "RandomCropAndResizeOp";
constexpr char kRandomCropAndResizeWithBBoxOp[] = "RandomCropAndResizeWithBBoxOp";
constexpr char kRandomCropDecodeResizeOp[] = "RandomCropDecodeResizeOp";
constexpr char kRandomCropOp[] = "RandomCropOp";
constexpr char kRandomCropWithBBoxOp[] = "RandomCropWithBBoxOp";
constexpr char kRandomEqualizeOp[] = "RandomEqualizeOp";
constexpr char kRandomHorizontalFlipWithBBoxOp[] = "RandomHorizontalFlipWithBBoxOp";
constexpr char kRandomHorizontalFlipOp[] = "RandomHorizontalFlipOp";
constexpr char kRandomInvertOp[] = "RandomInvertOp";
constexpr char kRandomLightingOp[] = "RandomLightingOp";
constexpr char kRandomPosterizeOp[] = "RandomPosterizeOp";
constexpr char kRandomResizeOp[] = "RandomResizeOp";
constexpr char kRandomResizeWithBBoxOp[] = "RandomResizeWithBBoxOp";
constexpr char kRandomRotationOp[] = "RandomRotationOp";
constexpr char kRandomSolarizeOp[] = "RandomSolarizeOp";
constexpr char kRandomSharpnessOp[] = "RandomSharpnessOp";
constexpr char kRandomVerticalFlipOp[] = "RandomVerticalFlipOp";
constexpr char kRandomVerticalFlipWithBBoxOp[] = "RandomVerticalFlipWithBBoxOp";
constexpr char kRescaleOp[] = "RescaleOp";
constexpr char kResizeBilinearOp[] = "ResizeBilinearOp";
constexpr char kResizedCropOp[] = "ResizedCropOp";
constexpr char kResizeOp[] = "ResizeOp";
constexpr char kResizePreserveAROp[] = "ResizePreserveAROp";
constexpr char kResizeWithBBoxOp[] = "ResizeWithBBoxOp";
constexpr char kRgbaToBgrOp[] = "RgbaToBgrOp";
constexpr char kRgbaToRgbOp[] = "RgbaToRgbOp";
constexpr char kRgbToBgrOp[] = "RgbToBgrOp";
constexpr char kRgbToGrayOp[] = "RgbToGrayOp";
constexpr char kRotateOp[] = "RotateOp";
constexpr char kSharpnessOp[] = "SharpnessOp";
constexpr char kSlicePatchesOp[] = "SlicePatchesOp";
constexpr char kSolarizeOp[] = "SolarizeOp";
constexpr char kSwapRedBlueOp[] = "SwapRedBlueOp";
constexpr char kToTensorOp[] = "ToTensorOp";
constexpr char kUniformAugOp[] = "UniformAugmentOp";
constexpr char kVerticalFlipOp[] = "VerticalFlipOp";

// video
constexpr char kDvppDecodeVideoOp[] = "DvppDecodeVideoOp";

// text
constexpr char kAddTokenOp[] = "AddTokenOp";
constexpr char kBasicTokenizerOp[] = "BasicTokenizerOp";
constexpr char kBertTokenizerOp[] = "BertTokenizerOp";
constexpr char kCaseFoldOp[] = "CaseFoldOp";
constexpr char kFilterWikipediaXMLOp[] = "FilterWikipediaXMLOp";
constexpr char kJiebaTokenizerOp[] = "JiebaTokenizerOp";
constexpr char kLookupOp[] = "LookupOp";
constexpr char kNgramOp[] = "NgramOp";
constexpr char kSlidingWindowOp[] = "SlidingWindowOp";
constexpr char kNormalizeUTF8Op[] = "NormalizeUTF8Op";
constexpr char kRegexReplaceOp[] = "RegexReplaceOp";
constexpr char kRegexTokenizerOp[] = "RegexTokenizerOp";
constexpr char kToNumberOp[] = "ToNumberOp";
constexpr char kToVectorsOp[] = "ToVectorsOp";
constexpr char kTruncateOp[] = "TruncateOp";
constexpr char kTruncateSequencePairOp[] = "TruncateSequencePairOp";
constexpr char kUnicodeCharTokenizerOp[] = "UnicodeCharTokenizerOp";
constexpr char kUnicodeScriptTokenizerOp[] = "UnicodeScriptTokenizerOp";
constexpr char kWhitespaceTokenizerOp[] = "WhitespaceTokenizerOp";
constexpr char kWordpieceTokenizerOp[] = "WordpieceTokenizerOp";
constexpr char kRandomChoiceOp[] = "RandomChoiceOp";
constexpr char kRandomApplyOp[] = "RandomApplyOp";
constexpr char kComposeOp[] = "ComposeOp";
constexpr char kRandomSelectSubpolicyOp[] = "RandomSelectSubpolicyOp";
constexpr char kSentencepieceTokenizerOp[] = "SentencepieceTokenizerOp";

// audio
constexpr char kAllpassBiquadOp[] = "AllpassBiquadOp";
constexpr char kAmplitudeToDBOp[] = "AmplitudeToDBOp";
constexpr char kAngleOp[] = "AngleOp";
constexpr char kBandBiquadOp[] = "BandBiquadOp";
constexpr char kBandpassBiquadOp[] = "BandpassBiquadOp";
constexpr char kBandrejectBiquadOp[] = "BandrejectBiquadOp";
constexpr char kBassBiquadOp[] = "BassBiquadOp";
constexpr char kBiquadOp[] = "BiquadOp";
constexpr char kComplexNormOp[] = "ComplexNormOp";
constexpr char kComputeDeltasOp[] = "ComputeDeltasOp";
constexpr char kContrastOp[] = "ContrastOp";
constexpr char kDBToAmplitudeOp[] = " DBToAmplitudeOp";
constexpr char kDCShiftOp[] = "DCShiftOp";
constexpr char kDeemphBiquadOp[] = "DeemphBiquadOp";
constexpr char kDetectPitchFrequencyOp[] = "DetectPitchFrequencyOp";
constexpr char kDitherOp[] = "DitherOp";
constexpr char kEqualizerBiquadOp[] = "EqualizerBiquadOp";
constexpr char kFadeOp[] = "FadeOp";
constexpr char kFiltfiltOp[] = "FiltfiltOp";
constexpr char kFlangerOp[] = "FlangerOp";
constexpr char kFrequencyMaskingOp[] = "FrequencyMaskingOp";
constexpr char kGainOp[] = "GainOp";
constexpr char kGriffinLimOp[] = "GriffinLimOp";
constexpr char kHighpassBiquadOp[] = "HighpassBiquadOp";
constexpr char kInverseMelScaleOp[] = "InverseMelScaleOp";
constexpr char kInverseSpectrogramOp[] = "InverseSpectrogramOp";
constexpr char kLFCCOp[] = "LFCCOp";
constexpr char kLFilterOp[] = "LFilterOp";
constexpr char kLowpassBiquadOp[] = "LowpassBiquadOp";
constexpr char kMagphaseOp[] = "MagphaseOp";
constexpr char kMaskAlongAxisIIDOp[] = "MaskAlongAxisIIDOp";
constexpr char kMaskAlongAxisOp[] = "MaskAlongAxisOp";
constexpr char kMelScaleOp[] = "MelScaleOp";
constexpr char kMelSpectrogramOp[] = "MelSpectrogramOp";
constexpr char kMFCCOp[] = "MFCCOp";
constexpr char kMuLawDecodingOp[] = "MuLawDecodingOp";
constexpr char kMuLawEncodingOp[] = "MuLawEncodingOp";
constexpr char kOverdriveOp[] = "OverdriveOp";
constexpr char kPhaserOp[] = "PhaserOp";
constexpr char kPhaseVocoderOp[] = "PhaseVocoderOp";
constexpr char kPitchShiftOp[] = "PitchShiftOp";
constexpr char kResampleOp[] = "ResampleOp";
constexpr char kRiaaBiquadOp[] = "RiaaBiquadOp";
constexpr char kSlidingWindowCmnOp[] = "SlidingWindowCmnOp";
constexpr char kSpectralCentroidOp[] = "SpectralCentroidOp";
constexpr char kSpectrogramOp[] = "SpectrogramOp";
constexpr char kTimeMaskingOp[] = "TimeMaskingOp";
constexpr char kTimeStretchOp[] = "TimeStretchOp";
constexpr char kTrebleBiquadOp[] = "TrebleBiquadOp";
constexpr char kVadOp[] = "VadOp";
constexpr char kVolOp[] = "VolOp";

// data
constexpr char kConcatenateOp[] = "ConcatenateOp";
constexpr char kDuplicateOp[] = "DuplicateOp";
constexpr char kFillOp[] = "FillOp";
constexpr char kMaskOp[] = "MaskOp";
constexpr char kOneHotOp[] = "OneHotOp";
constexpr char kPadEndOp[] = "PadEndOp";
constexpr char kParseExampleOp[] = "ParseExampleOp";
constexpr char kSliceOp[] = "SliceOp";
constexpr char kToFloat16Op[] = "ToFloat16Op";
constexpr char kTypeCastOp[] = "TypeCastOp";
constexpr char kUniqueOp[] = "UniqueOp";

// other
constexpr char kCFuncOp[] = "CFuncOp";
constexpr char kPyFuncOp[] = "PyFuncOp";
constexpr char kPluginOp[] = "PluginOp";
constexpr char kNoOp[] = "NoOp";

// A class that does a computation on a Tensor
class TensorOp {
 public:
  TensorOp() = default;

  virtual ~TensorOp() = default;

  // A function that prints info about the tensor operation
  // @param out
  virtual void Print(std::ostream &out) const { out << Name() << std::endl; }

  // Provide stream operator for displaying it
  // @param output stream
  // @param so the TensorOp object to be printed
  // @return output stream
  friend std::ostream &operator<<(std::ostream &out, const TensorOp &so) {
    so.Print(out);
    return out;
  }

  // Perform an operation on one Tensor and produce one Tensor. This is for 1-to-1 column MapOp
  // @param input  shares the ownership of the Tensor (increase the ref count).
  // @param output the address to a shared_ptr where the result will be placed.
  // @return Status
  virtual Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

  // Perform an operation on Tensors from multiple columns, and produce multiple Tensors.
  // This is for m-to-n column MapOp.
  // @param input is a vector of shared_ptr to Tensor (pass by const reference).
  // @param output is the address to an empty vector of shared_ptr to Tensor.
  // @return Status
  virtual Status Compute(const TensorRow &input, TensorRow *output);

  // Perform an operation on one DeviceTensor and produce one DeviceTensor. This is for 1-to-1 column MapOp
  // @param input shares the ownership of the DeviceTensor (increase the ref count).
  // @param output the address to a shared_ptr where the result will be placed.
  // @return Status
  virtual Status Compute(const std::shared_ptr<DeviceTensor> &input, std::shared_ptr<DeviceTensor> *output);

#if !defined(BUILD_LITE) && defined(ENABLE_D)
  virtual Status Compute(const std::vector<std::shared_ptr<DeviceTensorAscend910B>> &input,
                         std::vector<std::shared_ptr<DeviceTensorAscend910B>> *output);

  // Perform an operation on one DeviceTensorAscend910B and produce one DeviceTensorAscend910B. This is for 1-to-1
  // column MapOp
  // @param input shares the ownership of the DeviceTensorAscned910B (increase the ref count).
  // @param output the address to a shared_ptr where the result will be placed.
  // @return Status
  virtual Status Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                         std::shared_ptr<DeviceTensorAscend910B> *output);
#endif

  // Returns true oif the TensorOp takes one input and returns one output.
  // @return true/false
  bool OneToOne() { return NumInput() == 1 && NumOutput() == 1; }

  // Returns true oif the TensorOp produces deterministic result.
  // @return true/false
  bool Deterministic() const { return is_deterministic_; }

  // Function to determine the number of inputs the TensorOp can take. 0: means undefined.
  // @return uint32_t
  virtual uint32_t NumInput() { return 1; }

  // Function to determine the number of output the TensorOp generates. 0: means undefined.
  // @return uint32_t
  virtual uint32_t NumOutput() { return 1; }

  // Function to determine the shapes of the output tensor given the input tensors' shapes.
  // If a subclass did not override this function, it means that the shape does not change.
  // @param inputs in: vector of the shapes of the input tensors.
  // @param outputs out: vector of the shapes of the output tensors to be filled.
  // @return Status
  virtual Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs);

  // Function to determine the types of the output tensor given the input tensor's types.
  // If a subclass did not override this function, it means that the type does not change.
  // @param inputs in: vector of the types of the input tensors.
  // @param outputs out: vector of the types of the output tensors to be filled.
  // @return Status
  virtual Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs);

  virtual std::string Name() const = 0;

  virtual Status to_json(nlohmann::json *out_json) { return Status::OK(); }

  virtual Status SetAscendResource(const std::shared_ptr<DeviceResource> &resource);

  virtual bool IsDvppOp() { return false; }

  virtual bool IsHWC() { return true; }  // the input of the op is HWC in default

  // Currently, it's used by PyFuncOp which can release global executor when map with thread/process mode
  virtual Status ReleaseResource() { return Status::OK(); }

  virtual void SetSeed(uint32_t seed) {}

 protected:
  bool is_deterministic_{true};
};

class RandomTensorOp : public TensorOp {
 public:
  RandomTensorOp();

  ~RandomTensorOp() override = default;

 protected:
  void SetSeed(uint32_t seed) override;

  std::mt19937 random_generator_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_TENSOR_OP_H_
