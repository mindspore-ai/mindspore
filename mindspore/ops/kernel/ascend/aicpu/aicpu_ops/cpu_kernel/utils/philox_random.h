/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  https://github.com/tensorflow/tensorflow/blob/master/third_party/xla/xla/tsl/lib/random/philox_random.h

  Additional modifications made by Huawei Technologies Co., Ltd in 2020-2025.
*/

#ifndef _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_PHILOX_RANDOM_H
#define _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_PHILOX_RANDOM_H

#include <random>
#include <array>
#include <stdint.h>
#if __has_include("context/utils/status.h")
#include "context/utils/status.h"
#else
#include "context/common/status.h"
#endif

namespace aicpu {
namespace random {
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;

// The implementation of the Philox algorithm. More Details can be found in the paper:
// Salmon, John K., et al. (2011) "Parallel random numbers: As easy as 1, 2, 3."
// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
class PhiloxRandom {
 public:
  using ResultElementType = uint32_t;
  static constexpr int kKeyCount = 2;
  static constexpr int kResultElementCount = 4;
  static constexpr int kElementCost = 10;
  static constexpr int kMoveStepInBit = 32;
  using KeyArr = std::array<uint32_t, kKeyCount>;
  using ResultTypeArr = std::array<uint32_t, kResultElementCount>;

  PhiloxRandom() = default;

  explicit PhiloxRandom(uint64_t seed_low, uint64_t seed_high) {
    key_[kIndex0] = static_cast<uint32_t>(seed_low);
    key_[kIndex1] = static_cast<uint32_t>(seed_low >> kMoveStepInBit);
    counter_[kIndex2] = static_cast<uint32_t>(seed_high);
    counter_[kIndex3] = static_cast<uint32_t>(seed_high >> kMoveStepInBit);
  }

  PhiloxRandom(ResultTypeArr counter, KeyArr key) : counter_(counter), key_(key) {}

  void Skip(uint64_t count) {
    const uint32_t low = static_cast<uint32_t>(count);
    uint32_t high = static_cast<uint32_t>(count >> 32);

    counter_[kIndex0] += low;
    if (counter_[kIndex0] < low) ++high;

    counter_[kIndex1] += high;
    if (counter_[kIndex1] < high) {
      if (++counter_[kIndex2] == 0) ++counter_[kIndex3];
    }
  }

  // overload the directly call
  ResultTypeArr operator()() {
    ResultTypeArr tmp_counter = counter_;
    KeyArr tmp_key = key_;
    constexpr auto kTimes = 10;
    for (int i = 0; i < kTimes; i++) {
      tmp_counter = ComputeResult(tmp_counter, tmp_key);
      if (i < kTimes - 1) {
        tmp_key[kIndex0] += kPhiloxW32A;
        tmp_key[kIndex1] += kPhiloxW32B;
      } else {
        SkipNext();
      }
    }
    return tmp_counter;
  }

 private:
  ResultTypeArr counter_{};
  KeyArr key_{};

  // constants adopted by the original paper.
  static constexpr uint32_t kPhiloxW32A = 0x9E3779B9;
  static constexpr uint32_t kPhiloxW32B = 0xBB67AE85;
  static constexpr uint32_t kPhiloxM4x32A = 0xD2511F53;
  static constexpr uint32_t kPhiloxM4x32B = 0xCD9E8D57;

  void SkipNext() {
    if (++counter_[kIndex0] == 0 && ++counter_[kIndex1] == 0 && ++counter_[kIndex2] == 0) ++counter_[kIndex3];
  }

  static ResultTypeArr ComputeResult(const ResultTypeArr &counter, const KeyArr &key) {
    ResultTypeArr res;

    const uint64_t res0 = static_cast<uint64_t>(kPhiloxM4x32A) * counter[kIndex0];
    auto low0 = static_cast<uint32_t>(res0);
    auto high0 = static_cast<uint32_t>(res0 >> kMoveStepInBit);
    res[kIndex2] = high0 ^ counter[kIndex3] ^ key[kIndex1];
    res[kIndex3] = low0;

    const uint64_t res1 = static_cast<uint64_t>(kPhiloxM4x32B) * counter[kIndex2];
    auto low1 = static_cast<uint32_t>(res1);
    auto high1 = static_cast<uint32_t>(res1 >> kMoveStepInBit);
    res[kIndex0] = high1 ^ counter[kIndex1] ^ key[kIndex0];
    res[kIndex1] = low1;

    return res;
  }
};
}  // namespace random
}  // namespace aicpu
#endif  // _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_PHILOX_RANDOM_H_
