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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_CONSTANTS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_CONSTANTS_H_

#include <cstdint>
#include <limits>
#include <random>

#include "include/api/types.h"

namespace mindspore {
namespace dataset {
// Various type defines for convenience
using uchar = unsigned char;
using dsize_t = int64_t;

/// \brief The modulation in Flanger
enum class DATASET_API Modulation {
  kSinusoidal = 0,  ///< Use sinusoidal modulation.
  kTriangular = 1   ///< Use triangular modulation.
};

/// \brief The interpolation in Flanger
enum class DATASET_API Interpolation {
  kLinear = 0,    ///< Use linear for delay-line interpolation.
  kQuadratic = 1  ///< Use quadratic for delay-line interpolation.
};

/// \brief The dataset auto augment policy in AutoAugment
enum class DATASET_API AutoAugmentPolicy {
  kImageNet = 0,  ///< AutoAugment policy learned on the ImageNet dataset.
  kCifar10 = 1,   ///< AutoAugment policy learned on the Cifar10 dataset.
  kSVHN = 2       ///< AutoAugment policy learned on the SVHN dataset.
};

/// \brief The color conversion code
enum class DATASET_API ConvertMode {
  COLOR_BGR2BGRA = 0,                 ///< Add alpha channel to BGR image.
  COLOR_RGB2RGBA = COLOR_BGR2BGRA,    ///< Add alpha channel to RGB image.
  COLOR_BGRA2BGR = 1,                 ///< Remove alpha channel to BGR image.
  COLOR_RGBA2RGB = COLOR_BGRA2BGR,    ///< Remove alpha channel to RGB image.
  COLOR_BGR2RGBA = 2,                 ///< Convert BGR image to RGBA image.
  COLOR_RGB2BGRA = COLOR_BGR2RGBA,    ///< Convert RGB image to BGRA image.
  COLOR_RGBA2BGR = 3,                 ///< Convert RGBA image to BGR image.
  COLOR_BGRA2RGB = COLOR_RGBA2BGR,    ///< Convert BGRA image to RGB image.
  COLOR_BGR2RGB = 4,                  ///< Convert BGR image to RGB image.
  COLOR_RGB2BGR = COLOR_BGR2RGB,      ///< Convert RGB image to BGR image.
  COLOR_BGRA2RGBA = 5,                ///< Convert BGRA image to RGBA image.
  COLOR_RGBA2BGRA = COLOR_BGRA2RGBA,  ///< Convert RGBA image to BGRA image.
  COLOR_BGR2GRAY = 6,                 ///< Convert BGR image to GRAY image.
  COLOR_RGB2GRAY = 7,                 ///< Convert RGB image to GRAY image.
  COLOR_GRAY2BGR = 8,                 ///< Convert GRAY image to BGR image.
  COLOR_GRAY2RGB = COLOR_GRAY2BGR,    ///< Convert GRAY image to RGB image.
  COLOR_GRAY2BGRA = 9,                ///< Convert GRAY image to BGRA image.
  COLOR_GRAY2RGBA = COLOR_GRAY2BGRA,  ///< Convert GRAY image to RGBA image.
  COLOR_BGRA2GRAY = 10,               ///< Convert BGRA image to GRAY image.
  COLOR_RGBA2GRAY = 11                ///< Convert RGBA image to GRAY image.
};

/// \brief The mode for reading a image file.
enum class DATASET_API ImageReadMode {
  kUNCHANGED = 0,  ///< Remain the output in the original format.
  kGRAYSCALE = 1,  ///< Convert the output into one channel grayscale data.
  kCOLOR = 2,      ///< Convert the output into three channels RGB color data.
};

// \brief Possible density function in Dither.
enum DATASET_API DensityFunction {
  kTPDF = 0,  ///< Use triangular probability density function.
  kRPDF = 1,  ///< Use rectangular probability density function.
  kGPDF = 2   ///< Use gaussian probability density function.
};

/// \brief Values of norm in CreateDct.
enum class DATASET_API NormMode {
  kNone = 0,  ///< None type norm.
  kOrtho = 1  ///< Ortho type norm.
};

/// \brief Possible options for norm in MelscaleFbanks.
enum class DATASET_API NormType {
  kNone = 0,    ///< None type norm.
  kSlaney = 1,  ///< Slaney type norm.
};

/// \brief The mode for manual offload.
enum class DATASET_API ManualOffloadMode {
  kUnspecified,  ///< Not set, will use auto_offload setting instead.
  kDisabled,     ///< Do not perform offload.
  kEnabled       ///< Attempt to offload.
};

/// \brief Target devices to perform map operation.
enum class DATASET_API MapTargetDevice {
  kCpu = 0,     ///< CPU Device.
  kGpu,         ///< Gpu Device.
  kAscend310,   ///< Ascend310 Device.
  kAscend910B,  ///< Ascend910B Device.
  kInvalid = 100
};

/// \brief Possible options for mel_type in MelscaleFbanks.
enum class DATASET_API MelType {
  kHtk = 0,     ///< Htk scale type.
  kSlaney = 1,  ///< Slaney scale type.
};

/// \brief The initial type of tensor implementation.
enum class DATASET_API TensorImpl {
  kNone,      ///< None type tensor.
  kFlexible,  ///< Flexible type tensor, can be converted to any type.
  kCv,        ///< CV type tensor.
  kNP         ///< Numpy type tensor.
};

/// \brief The mode for shuffling data.
enum class DATASET_API ShuffleMode {
  kFalse = 0,   ///< No shuffling is performed.
  kFiles = 1,   ///< Shuffle files only.
  kGlobal = 2,  ///< Shuffle both the files and samples.
  kInfile = 3   ///< Shuffle data within each file.
};

/// \brief Possible scale for input audio.
enum class DATASET_API ScaleType {
  kMagnitude = 0,  ///< Audio scale is magnitude.
  kPower = 1,      ///< Audio scale is power.
};

/// \brief The scale for gain type.
enum class DATASET_API GainType {
  kAmplitude = 0,  ///< Audio gain type is amplitude.
  kPower = 1,      ///< Audio gain type is power.
  kDb = 2,         ///< Audio gain type is db.
};

/// \brief The method of padding.
enum class DATASET_API BorderType {
  kConstant = 0,  ///< Fill the border with constant values.
  kEdge = 1,      ///< Fill the border with the last value on the edge.
  kReflect = 2,   ///< Reflect the values on the edge omitting the last value of edge.
  kSymmetric = 3  ///< Reflect the values on the edge repeating the last value of edge.
};

/// \brief Possible fix rotation angle for Rotate Op.
enum class DATASET_API FixRotationAngle {
  k0Degree = 1,             ///< Rotate 0 degree.
  k0DegreeAndMirror = 2,    ///< Rotate 0 degree and apply horizontal flip.
  k180Degree = 3,           ///< Rotate 180 degree.
  k180DegreeAndMirror = 4,  ///< Rotate 180 degree and apply horizontal flip.
  k90DegreeAndMirror = 5,   ///< Rotate 90 degree and apply horizontal flip.
  k90Degree = 6,            ///< Rotate 90 degree.
  k270DegreeAndMirror = 7,  ///< Rotate 270 degree and apply horizontal flip.
  k270Degree = 8,           ///< Rotate 270 degree.
};

/// \brief Possible types for windows function.
enum class DATASET_API WindowType {
  kBartlett = 0,  ///< Bartlett window function.
  kBlackman = 1,  ///< Blackman window function.
  kHamming = 2,   ///< Hamming window function.
  kHann = 3,      ///< Hann window function.
  kKaiser = 4     ///< Kaiser window function.
};

/// \brief Possible options for Image format types in a batch.
enum class DATASET_API ImageBatchFormat {
  kNHWC = 0,  ///< Indicate the input batch is of NHWC format.
  kNCHW = 1   ///< Indicate the input batch is of NCHW format.
};

/// \brief Possible options for Image format types.
enum class DATASET_API ImageFormat {
  HWC = 0,  ///< Indicate the input batch is of NHWC format
  CHW = 1,  ///< Indicate the input batch is of NHWC format
  HW = 2    ///< Indicate the input batch is of NHWC format
};

/// \brief Possible options for interpolation method.
enum class DATASET_API InterpolationMode {
  kLinear = 0,            ///< Interpolation method is linear interpolation.
  kNearestNeighbour = 1,  ///< Interpolation method is nearest-neighbor interpolation.
  kCubic = 2,             ///< Interpolation method is bicubic interpolation.
  kArea = 3,              ///< Interpolation method is pixel area interpolation.
  kCubicPil = 4           ///< Interpolation method is bicubic interpolation like implemented in pillow.
};

/// \brief Possible formats for Vdec output image.
enum class DATASET_API VdecOutputFormat {
  kYuvSemiplanar420 = 1,  ///< Output image with PIXEL_FORMAT_YUV_SEMIPLANAR_420.
  kYvuSemiplanar420 = 2,  ///< Output image with PIXEL_FORMAT_YVU_SEMIPLANAR_420.
};

/// \brief Possible formats for Vdec input video.
enum class DATASET_API VdecStreamFormat {
  kH265MainLevel = 0,  ///< Input video with H265_MAIN_LEVEL
  kH264BaselineLevel,  ///< Input video with H264_BASELINE_LEVEL
  kH264MainLevel,      ///< Input video with H264_MAIN_LEVEL
  kH264HighLevel       ///< Input video with H264_HIGH_LEVEL
};

/// \brief Possible tokenize modes for JiebaTokenizer.
enum class DATASET_API JiebaMode {
  kMix = 0,  ///< Tokenize with MPSegment algorithm.
  kMp = 1,   ///< Tokenize with Hiddel Markov Model Segment algorithm.
  kHmm = 2   ///< Tokenize with a mix of MPSegment and HMMSegment algorithm.
};

/// \brief Possible options for SPieceTokenizerOutType.
enum class DATASET_API SPieceTokenizerOutType {
  kString = 0,  ///< Output of sentencepiece tokenizer is string type.
  kInt = 1      ///< Output of sentencepiece tokenizer is int type.
};

/// \brief Possible options for SPieceTokenizerLoadType.
enum class DATASET_API SPieceTokenizerLoadType {
  kFile = 0,  ///< Load sentencepiece tokenizer from local sentencepiece vocab file.
  kModel = 1  ///< Load sentencepiece tokenizer from sentencepiece vocab instance.
};

/// \brief Type options for SentencePiece Model.
enum class DATASET_API SentencePieceModel {
  kUnigram = 0,  ///< Based on Unigram model.
  kBpe = 1,      ///< Based on Byte Pair Encoding (BPE) model.
  kChar = 2,     ///< Based on Char model.
  kWord = 3      ///< Based on Word model.
};

/// \brief Possible options to specify a specific normalize mode.
enum class DATASET_API NormalizeForm {
  kNone = 0,  ///< Keep the input string tensor unchanged.
  kNfc,       ///< Normalize with Normalization Form C.
  kNfkc,      ///< Normalize with Normalization Form KC.
  kNfd,       ///< Normalize with Normalization Form D.
  kNfkd,      ///< Normalize with Normalization Form KD.
};

/// \brief Possible options for Mask.
enum class DATASET_API RelationalOp {
  kEqual = 0,     ///< equal to `==`
  kNotEqual,      ///< equal to `!=`
  kLess,          ///< equal to `<`
  kLessEqual,     ///< equal to `<=`
  kGreater,       ///< equal to `>`
  kGreaterEqual,  ///< equal to `>=`
};

/// \brief Possible modes for slice patches.
enum class DATASET_API SliceMode {
  kPad = 0,   ///< Pad some pixels before slice to patches.
  kDrop = 1,  ///< Drop remainder pixels before slice to patches.
};

/// \brief Possible options for SamplingStrategy.
enum class DATASET_API SamplingStrategy {
  kRandom = 0,     ///< Random sampling with replacement.
  kEdgeWeight = 1  ///< Sampling with edge weight as probability.
};

/// \brief Possible options for fade shape.
enum class DATASET_API FadeShape {
  kLinear = 0,       ///< Fade shape is linear mode.
  kExponential = 1,  ///< Fade shape is exponential mode.
  kLogarithmic = 2,  ///< Fade shape is logarithmic mode.
  kQuarterSine = 3,  ///< Fade shape is quarter_sine mode.
  kHalfSine = 4,     ///< Fade shape is half_sine mode.
};

/// \brief Sample method for audio resample.
enum class DATASET_API ResampleMethod {
  kSincInterpolation = 0,  ///< Resample audio by sinc interpolation method
  kKaiserWindow = 1,       ///< Resample audio by Kaiser window
};

/// \brief Possible configuration methods for processing error samples.
enum class DATASET_API ErrorSamplesMode {
  kReturn = 0,   ///< Erroneous sample results in error raised and returned
  kReplace = 1,  ///< Erroneous sample is replaced with an internally determined sample
  kSkip = 2      ///< Erroneous sample is skipped
};

/// \brief Convenience function to check bitmask for a 32bit int
/// \param[in] bits a 32bit int to be tested
/// \param[in] bitMask a 32bit int representing bit mask
/// \return bool Result for the check
inline bool DATASET_API BitTest(uint32_t bits, uint32_t bitMask) { return (bits & bitMask) == bitMask; }

/// \brief Convenience function to set bitmask for a 32bit int
/// \param[in] bits a 32bit int to deal with
/// \param[in] bitMask a 32bit int representing bit mask
inline void DATASET_API BitSet(uint32_t *bits, uint32_t bitMask) {
  if (bits == nullptr) {
    return;
  }
  *bits |= bitMask;
}

/// \brief Convenience function to clear bitmask from a 32bit int
/// \param[in] bits a 32bit int to deal with
/// \param[in] bitMask a 32bit int representing bit mask
inline void DATASET_API BitClear(uint32_t *bits, uint32_t bitMask) {
  if (bits == nullptr) {
    return;
  }
  *bits &= (~bitMask);
}

constexpr uint32_t kFrameWidthMax = 4096;
constexpr uint32_t kFrameHeightMax = 4096;
constexpr uint32_t kFrameWidthMin = 128;
constexpr uint32_t kFrameHeightMin = 128;

constexpr int64_t kDeMaxDim = std::numeric_limits<int64_t>::max();
constexpr int32_t kDeMaxRank = std::numeric_limits<int32_t>::max();
constexpr int64_t kDeMaxFreq = std::numeric_limits<int64_t>::max();  // 9223372036854775807 or 2^(64-1)
constexpr int64_t kDeMaxTopk = std::numeric_limits<int64_t>::max();

constexpr uint32_t kCfgRowsPerBuffer = 1;
constexpr uint32_t kCfgParallelWorkers = 8;
constexpr uint32_t kCfgWorkerConnectorSize = 16;
constexpr uint32_t kCfgOpConnectorSize = 16;
constexpr uint32_t kCfgSendingBatch = 0;
constexpr int32_t kCfgDefaultRankId = -1;
constexpr uint32_t kCfgDefaultSeed = std::mt19937::default_seed;
constexpr uint32_t kCfgMonitorSamplingInterval = 1000;        // timeout value for monitor sampling interval in
                                                              // milliseconds
constexpr uint32_t kCfgCallbackTimeout = 60;                  // timeout value for callback in seconds
constexpr uint32_t kCfgMultiprocessingTimeoutInterval = 300;  // timeout value for multiprocessing interval in seconds
constexpr int32_t kCfgDefaultCachePort = 50052;
constexpr char kCfgDefaultCacheHost[] = "127.0.0.1";
constexpr int32_t kDftCachePrefetchSize = 20;
constexpr int32_t kDftNumConnections = 12;
constexpr bool kDftAutoNumWorkers = false;
constexpr char kDftMetaColumnPrefix[] = "_meta-";
constexpr int32_t kDecimal = 10;  // used in strtol() to convert a string value according to decimal numeral system
constexpr int32_t kMinLegalPort = 1025;
constexpr int32_t kMaxLegalPort = 65535;

// Invalid OpenCV type should not be from 0 to 7 (opencv4/opencv2/core/hal/interface.h)
constexpr uint8_t kCVInvalidType = 255;

using connection_id_type = uint64_t;
using session_id_type = uint32_t;
using row_id_type = int64_t;

constexpr uint32_t kCfgAutoTuneInterval = 0;  // default number of steps
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_CONSTANTS_H_
