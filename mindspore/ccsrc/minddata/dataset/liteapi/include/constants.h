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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_CONSTANTS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_CONSTANTS_H_

#include <cstdint>
#include <limits>
#include <random>

namespace mindspore {
namespace dataset {
// Various type defines for convenience
using uchar = unsigned char;
using dsize_t = int64_t;

// Target devices to perform map operation
enum class MapTargetDevice { kCpu, kGpu, kAscend310 };

// Possible dataset types for holding the data and client type
enum class DatasetType { kUnknown, kArrow, kTf };

// Possible flavours of Tensor implementations
enum class TensorImpl { kNone, kFlexible, kCv, kNP };

// Possible values for shuffle
enum class ShuffleMode { kFalse = 0, kFiles = 1, kGlobal = 2 };

// Possible values for Border types
enum class BorderType { kConstant = 0, kEdge = 1, kReflect = 2, kSymmetric = 3 };

// Possible values for Image format types in a batch
enum class ImageBatchFormat { kNHWC = 0, kNCHW = 1 };

// Possible values for Image format types
enum class ImageFormat { HWC = 0, CHW = 1, HW = 2 };

// Possible interpolation modes
enum class InterpolationMode { kLinear = 0, kNearestNeighbour = 1, kCubic = 2, kArea = 3 };

// Possible JiebaMode modes
enum class JiebaMode { kMix = 0, kMp = 1, kHmm = 2 };

// Possible values for SPieceTokenizerOutType
enum class SPieceTokenizerOutType { kString = 0, kInt = 1 };

// Possible values for SPieceTokenizerLoadType
enum class SPieceTokenizerLoadType { kFile = 0, kModel = 1 };

// Possible values for SentencePieceModel
enum class SentencePieceModel { kUnigram = 0, kBpe = 1, kChar = 2, kWord = 3 };

// Possible values for NormalizeForm
enum class NormalizeForm {
  kNone = 0,
  kNfc,
  kNfkc,
  kNfd,
  kNfkd,
};

// Possible values for SamplingStrategy
enum class SamplingStrategy { kRandom = 0, kEdgeWeight = 1 };

// convenience functions for 32bit int bitmask
inline bool BitTest(uint32_t bits, uint32_t bitMask) { return (bits & bitMask) == bitMask; }

inline void BitSet(uint32_t *bits, uint32_t bitMask) { *bits |= bitMask; }

inline void BitClear(uint32_t *bits, uint32_t bitMask) { *bits &= (~bitMask); }

constexpr int32_t kDeMaxDim = std::numeric_limits<int32_t>::max();  // 2147483647 or 2^32 -1
constexpr int32_t kDeMaxRank = std::numeric_limits<int32_t>::max();
constexpr int64_t kDeMaxFreq = std::numeric_limits<int64_t>::max();  // 9223372036854775807 or 2^(64-1)
constexpr int64_t kDeMaxTopk = std::numeric_limits<int64_t>::max();

constexpr uint32_t kCfgRowsPerBuffer = 1;
constexpr uint32_t kCfgParallelWorkers = 8;
constexpr uint32_t kCfgWorkerConnectorSize = 16;
constexpr uint32_t kCfgOpConnectorSize = 16;
constexpr int32_t kCfgDefaultRankId = -1;
constexpr uint32_t kCfgDefaultSeed = std::mt19937::default_seed;
constexpr uint32_t kCfgMonitorSamplingInterval = 10;
constexpr uint32_t kCfgCallbackTimeout = 60;  // timeout value for callback in seconds
constexpr int32_t kCfgDefaultCachePort = 50052;
constexpr char kCfgDefaultCacheHost[] = "127.0.0.1";
constexpr int32_t kDftPrefetchSize = 20;
constexpr int32_t kDftNumConnections = 12;
constexpr int32_t kDftAutoNumWorkers = false;

// Invalid OpenCV type should not be from 0 to 7 (opencv4/opencv2/core/hal/interface.h)
constexpr uint8_t kCVInvalidType = 255;

using connection_id_type = uint64_t;
using session_id_type = uint32_t;
using row_id_type = int64_t;
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_CONSTANTS_H_
