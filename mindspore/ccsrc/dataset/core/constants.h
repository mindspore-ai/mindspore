/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef DATASET_CORE_CONSTANTS_H_
#define DATASET_CORE_CONSTANTS_H_

#include <cstdint>
#include <limits>
#include <random>

namespace mindspore {
namespace dataset {
// Various type defines for convenience
using uchar = unsigned char;
using dsize_t = int64_t;

// Possible dataset types for holding the data and client type
enum class DatasetType { kUnknown, kArrow, kTf };

// Possible flavours of Tensor implementations
enum class TensorImpl { kNone, kFlexible, kCv, kNP };

// convenience functions for 32bit int bitmask
inline bool BitTest(uint32_t bits, uint32_t bitMask) { return (bits & bitMask) == bitMask; }

inline void BitSet(uint32_t *bits, uint32_t bitMask) { *bits |= bitMask; }

inline void BitClear(uint32_t *bits, uint32_t bitMask) { *bits &= (~bitMask); }

constexpr int32_t kDeMaxDim = std::numeric_limits<int32_t>::max();  // 2147483647 or 2^32 -1
constexpr int32_t kDeMaxRank = std::numeric_limits<int32_t>::max();

constexpr uint32_t kCfgRowsPerBuffer = 1;
constexpr uint32_t kCfgParallelWorkers = 4;
constexpr uint32_t kCfgWorkerConnectorSize = 16;
constexpr uint32_t kCfgOpConnectorSize = 16;
constexpr uint32_t kCfgDefaultSeed = std::mt19937::default_seed;
constexpr uint32_t kCfgMonitorSamplingInterval = 10;

// Invalid OpenCV type should not be from 0 to 7 (opencv4/opencv2/core/hal/interface.h)
constexpr uint8_t kCVInvalidType = 255;

using connection_id_type = int64_t;
using row_id_type = int64_t;
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_CORE_CONSTANTS_H_
