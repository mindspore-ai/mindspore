/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_PHILOX_RANDOM_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_PHILOX_RANDOM_H_

#include <iostream>
#include <random>
#include "runtime/hardware_abstract/visible.h"

namespace mindspore {
namespace kernel {
namespace random {
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;

template <typename T, int ElementCount>
class RUNTIME_HARDWARE_EXPORT Array {
 public:
  static constexpr int kElementCount = ElementCount;
  Array() {
    for (int i = 0; i < ElementCount; ++i) {
      data_[i] = T(0);
    }
  }

  const T &operator[](int index) const { return data_[index]; }

  T &operator[](int index) { return data_[index]; }

  size_t size() const { return ElementCount; }

 private:
  T data_[ElementCount];
};

// The implementation of the Philox algorithm. More Details can be found in the paper:
// Salmon, John K., et al. (2011) "Parallel random numbers: As easy as 1, 2, 3."
// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
class RUNTIME_HARDWARE_EXPORT PhiloxRandom {
 public:
  using ResultElementType = uint32_t;
  static constexpr int kKeyCount = 2;
  static constexpr int kResultElementCount = 4;
  static constexpr int kElementCost = 10;
  static constexpr int kMoveStepInBit = 32;
  using ResultType = Array<uint32_t, kResultElementCount>;
  using Key = Array<uint32_t, kKeyCount>;

  PhiloxRandom() {}

  explicit PhiloxRandom(uint64_t seed_lo, uint64_t seed_hi) {
    key_[kIndex0] = static_cast<uint32_t>(seed_lo);
    key_[kIndex1] = static_cast<uint32_t>(seed_lo >> kMoveStepInBit);
    counter_[kIndex2] = static_cast<uint32_t>(seed_hi);
    counter_[kIndex3] = static_cast<uint32_t>(seed_hi >> kMoveStepInBit);
  }

  PhiloxRandom(const ResultType &counter, const Key &key) : counter_(counter), key_(key) {}

  PhiloxRandom(int64_t seed, uint64_t offset) {
    const uint32_t seed_low_index = 0;
    const uint32_t seed_high_index = 1;
    const uint32_t offset_low_index = 2;
    const uint32_t offset_high_index = 3;
    key_[seed_low_index] = static_cast<uint32_t>(seed);
    key_[seed_high_index] = static_cast<uint32_t>(seed >> kMoveStepInBit);
    counter_[offset_low_index] = static_cast<uint32_t>(offset);
    counter_[offset_high_index] = static_cast<uint32_t>(offset >> kMoveStepInBit);
  }

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
  ResultType operator()() {
    ResultType tmp_counter = counter_;
    Key tmp_key = key_;
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
  ResultType counter_{};
  Key key_{};

  // constants adopted by the original paper.
  static constexpr uint32_t kPhiloxW32A = 0x9E3779B9;
  static constexpr uint32_t kPhiloxW32B = 0xBB67AE85;
  static constexpr uint32_t kPhiloxM4x32A = 0xD2511F53;
  static constexpr uint32_t kPhiloxM4x32B = 0xCD9E8D57;

  void SkipNext() {
    if (++counter_[kIndex0] == 0 && ++counter_[kIndex1] == 0 && ++counter_[kIndex2] == 0) ++counter_[kIndex3];
  }

  static ResultType ComputeResult(const ResultType &counter, const Key &key) {
    ResultType res;

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

RUNTIME_HARDWARE_EXPORT uint64_t GetSeed(const uint64_t &global_seed, const uint64_t &ops_seed);

}  // namespace random
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_PHILOX_RANDOM_H_
