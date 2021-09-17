/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

namespace mindspore {
namespace dataset {
// Various type defines for convenience
using uchar = unsigned char;
using dsize_t = int64_t;

/// \brief The color conversion code
enum class ConvertMode {
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

/// \brief Target devices to perform map operation.
enum class MapTargetDevice {
  kCpu,       ///< CPU Device.
  kGpu,       ///< Gpu Device.
  kAscend310  ///< Ascend310 Device.
};

/// \brief The initial type of tensor implementation.
enum class TensorImpl {
  kNone,      ///< None type tensor.
  kFlexible,  ///< Flexible type tensor, can be converted to any type.
  kCv,        ///< CV type tensor.
  kNP         ///< Numpy type tensor.
};

/// \brief The mode for shuffling data.
enum class ShuffleMode {
  kFalse = 0,   ///< No shuffling is performed.
  kFiles = 1,   ///< Shuffle files only.
  kGlobal = 2,  ///< Shuffle both the files and samples.
  kInfile = 3   ///< Shuffle data within each file.
};

/// \brief Possible scale for input audio.
enum class ScaleType {
  kMagnitude = 0,  ///< Audio scale is magnitude.
  kPower = 1,      ///< Audio scale is power.
};

/// \brief The scale for gain type.
enum class GainType {
  kAmplitude = 0,  ///< Audio gain type is amplitude.
  kPower = 1,      ///< Audio gain type is power.
  kDb = 2,         ///< Audio gain type is db.
};

/// \brief The method of padding.
enum class BorderType {
  kConstant = 0,  ///< Fill the border with constant values.
  kEdge = 1,      ///< Fill the border with the last value on the edge.
  kReflect = 2,   ///< Reflect the values on the edge omitting the last value of edge.
  kSymmetric = 3  ///< Reflect the values on the edge repeating the last value of edge.
};

/// \brief Possible fix rotation angle for Rotate Op.
enum class FixRotationAngle {
  k0Degree = 1,             ///< Rotate 0 degree.
  k0DegreeAndMirror = 2,    ///< Rotate 0 degree and apply horizontal flip.
  k180Degree = 3,           ///< Rotate 180 degree.
  k180DegreeAndMirror = 4,  ///< Rotate 180 degree and apply horizontal flip.
  k90DegreeAndMirror = 5,   ///< Rotate 90 degree and apply horizontal flip.
  k90Degree = 6,            ///< Rotate 90 degree.
  k270DegreeAndMirror = 7,  ///< Rotate 270 degree and apply horizontal flip.
  k270Degree = 8,           ///< Rotate 270 degree.
};

/// \brief Possible options for Image format types in a batch.
enum class ImageBatchFormat {
  kNHWC = 0,  ///< Indicate the input batch is of NHWC format.
  kNCHW = 1   ///< Indicate the input batch is of NCHW format.
};

/// \brief Possible options for Image format types.
enum class ImageFormat {
  HWC = 0,  ///< Indicate the input batch is of NHWC format
  CHW = 1,  ///< Indicate the input batch is of NHWC format
  HW = 2    ///< Indicate the input batch is of NHWC format
};

/// \brief Possible options for interpolation method.
enum class InterpolationMode {
  kLinear = 0,            ///< Interpolation method is linear interpolation.
  kNearestNeighbour = 1,  ///< Interpolation method is nearest-neighbor interpolation.
  kCubic = 2,             ///< Interpolation method is bicubic interpolation.
  kArea = 3,              ///< Interpolation method is pixel area interpolation.
  kCubicPil = 4           ///< Interpolation method is bicubic interpolation like implemented in pillow.
};

/// \brief Possible tokenize modes for JiebaTokenizer.
enum class JiebaMode {
  kMix = 0,  ///< Tokenize with MPSegment algorithm.
  kMp = 1,   ///< Tokenize with Hiddel Markov Model Segment algorithm.
  kHmm = 2   ///< Tokenize with a mix of MPSegment and HMMSegment algorithm.
};

/// \brief Possible options for SPieceTokenizerOutType.
enum class SPieceTokenizerOutType {
  kString = 0,  ///< Output of sentencepiece tokenizer is string type.
  kInt = 1      ///< Output of sentencepiece tokenizer is int type.
};

/// \brief Possible options for SPieceTokenizerLoadType.
enum class SPieceTokenizerLoadType {
  kFile = 0,  ///< Load sentencepiece tokenizer from local sentencepiece vocab file.
  kModel = 1  ///< Load sentencepiece tokenizer from sentencepiece vocab instance.
};

/// \brief Type options for SentencePiece Model.
enum class SentencePieceModel {
  kUnigram = 0,  ///< Based on Unigram model.
  kBpe = 1,      ///< Based on Byte Pair Encoding (BPE) model.
  kChar = 2,     ///< Based on Char model.
  kWord = 3      ///< Based on Word model.
};

/// \brief Possible options to specify a specific normalize mode.
enum class NormalizeForm {
  kNone = 0,  ///< Keep the input string tensor unchanged.
  kNfc,       ///< Normalize with Normalization Form C.
  kNfkc,      ///< Normalize with Normalization Form KC.
  kNfd,       ///< Normalize with Normalization Form D.
  kNfkd,      ///< Normalize with Normalization Form KD.
};

/// \brief Possible options for Mask.
enum class RelationalOp {
  kEqual = 0,     ///< equal to `==`
  kNotEqual,      ///< equal to `!=`
  kLess,          ///< equal to `<`
  kLessEqual,     ///< equal to `<=`
  kGreater,       ///< equal to `>`
  kGreaterEqual,  ///< equal to `>=`
};

/// \brief Possible modes for slice patches.
enum class SliceMode {
  kPad = 0,   ///< Pad some pixels before slice to patches.
  kDrop = 1,  ///< Drop remainder pixels before slice to patches.
};

/// \brief Possible options for SamplingStrategy.
enum class SamplingStrategy {
  kRandom = 0,     ///< Random sampling with replacement.
  kEdgeWeight = 1  ///< Sampling with edge weight as probability.
};

/// \brief Possible values for output format in get all neighbors function of gnn dataset
enum class OutputFormat {
  kNormal = 0,  ///< Normal format.
  kCoo = 1,     ///< COO format.
  kCsr = 2      ///< CSR format.
};

/// \brief Possible options for fade shape.
enum class FadeShape {
  kLinear = 0,       ///< Fade shape is linear mode.
  kExponential = 1,  ///< Fade shape is exponential mode.
  kLogarithmic = 2,  ///< Fade shape is logarithmic mode.
  kQuarterSine = 3,  ///< Fade shape is quarter_sine mode.
  kHalfSine = 4,     ///< Fade shape is half_sine mode.
};

/// \brief Convenience function to check bitmask for a 32bit int
/// \param[in] bits a 32bit int to be tested
/// \param[in] bitMask a 32bit int representing bit mask
/// \return bool Result for the check
inline bool BitTest(uint32_t bits, uint32_t bitMask) { return (bits & bitMask) == bitMask; }

/// \brief Convenience function to set bitmask for a 32bit int
/// \param[in] bits a 32bit int to deal with
/// \param[in] bitMask a 32bit int representing bit mask
inline void BitSet(uint32_t *bits, uint32_t bitMask) {
  if (bits == nullptr) {
    return;
  }
  *bits |= bitMask;
}

/// \brief Convenience function to clear bitmask from a 32bit int
/// \param[in] bits a 32bit int to deal with
/// \param[in] bitMask a 32bit int representing bit mask
inline void BitClear(uint32_t *bits, uint32_t bitMask) {
  if (bits == nullptr) {
    return;
  }
  *bits &= (~bitMask);
}

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
constexpr uint32_t kCfgMonitorSamplingInterval = 1000;  // timeout value for sampling interval in milliseconds
constexpr uint32_t kCfgCallbackTimeout = 60;            // timeout value for callback in seconds
constexpr int32_t kCfgDefaultCachePort = 50052;
constexpr char kCfgDefaultCacheHost[] = "127.0.0.1";
constexpr int32_t kDftPrefetchSize = 20;
constexpr int32_t kDftNumConnections = 12;
constexpr int32_t kDftAutoNumWorkers = false;
constexpr char kDftMetaColumnPrefix[] = "_meta-";
constexpr int32_t kDecimal = 10;  // used in strtol() to convert a string value according to decimal numeral system
constexpr int32_t kMinLegalPort = 1025;
constexpr int32_t kMaxLegalPort = 65535;

// Invalid OpenCV type should not be from 0 to 7 (opencv4/opencv2/core/hal/interface.h)
constexpr uint8_t kCVInvalidType = 255;

using connection_id_type = uint64_t;
using session_id_type = uint32_t;
using row_id_type = int64_t;
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_CONSTANTS_H_
