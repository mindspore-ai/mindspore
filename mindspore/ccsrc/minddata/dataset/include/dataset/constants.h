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

/// \brief The method of padding.
enum class BorderType {
  kConstant = 0,  ///< Fill the border with constant values.
  kEdge = 1,      ///< Fill the border with the last value on the edge.
  kReflect = 2,   ///< Reflect the values on the edge omitting the last value of edge.
  kSymmetric = 3  ///< Reflect the values on the edge repeating the last value of edge.
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
  kNormal = 0,  ///< Normal format>
  kCoo = 1,     ///< COO format>
  kCsr = 2      ///< CSR format>
};

// convenience functions for 32bit int bitmask
inline bool BitTest(uint32_t bits, uint32_t bitMask) { return (bits & bitMask) == bitMask; }

inline void BitSet(uint32_t *bits, uint32_t bitMask) { *bits |= bitMask; }

inline void BitClear(uint32_t *bits, uint32_t bitMask) { *bits &= (~bitMask); }

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
