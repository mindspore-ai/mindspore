/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_PHILOX_RANDOM_H_
#define MINDSPORE_CCSRC_KERNEL_PHILOX_RANDOM_H_

#include <iostream>
#include <random>
#include "include/backend/visible.h"

namespace mindspore {
namespace kernel {
namespace random {
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
/**
 * A class that represents an inline array.
 * Arguments:
 *   T: the array element type;
 *   ElementCount: the fixed size of the array;
 */
template <typename T, int ElementCount>
class BACKEND_EXPORT Array {
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

class BACKEND_EXPORT PhiloxRandom {
 public:
  using ResultElementType = uint32_t;
  // The number of elements that will be returned.
  static constexpr int kKeyCount = 2;
  static constexpr int kResultElementCount = 4;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 10;
  static constexpr int kMoveStepInBit = 32;
  /*
   * The type for the 64-bit key stored in the form of two 32-bit uint
   * that are used in the diffusion process.
   */
  using ResultType = Array<uint32_t, kResultElementCount>;
  using Key = Array<uint32_t, kKeyCount>;

  PhiloxRandom() {}

  explicit PhiloxRandom(uint64_t seed) {
    key_[kIndex0] = static_cast<uint32_t>(seed);
    key_[kIndex1] = static_cast<uint32_t>(seed >> kMoveStepInBit);
  }

  explicit PhiloxRandom(uint64_t seed_lo, uint64_t seed_hi) {
    key_[kIndex0] = static_cast<uint32_t>(seed_lo);
    key_[kIndex1] = static_cast<uint32_t>(seed_lo >> kMoveStepInBit);
    counter_[kIndex2] = static_cast<uint32_t>(seed_hi);
    counter_[kIndex3] = static_cast<uint32_t>(seed_hi >> kMoveStepInBit);
  }

  PhiloxRandom(ResultType counter, Key key) : counter_(counter), key_(key) {}

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

  ResultType const &counter() const { return counter_; }
  Key const &key() const { return key_; }

  // Skip the specified number of samples of 128-bits in the current stream.
  void Skip(uint64_t count) {
    const uint32_t count_lo = static_cast<uint32_t>(count);
    uint32_t count_hi = static_cast<uint32_t>(count >> kMoveStepInBit);

    counter_[kIndex0] += count_lo;
    if (counter_[kIndex0] < count_lo) {
      ++count_hi;
    }

    counter_[kIndex1] += count_hi;
    if (counter_[kIndex1] < count_hi) {
      if (++counter_[kIndex2] == 0) {
        ++counter_[kIndex3];
      }
    }
  }
  /*
   * Returns a group of four random numbers using the underlying Philox
   * algorithm.
   */
  ResultType operator()() {
    ResultType counter = counter_;
    Key key = key_;
    for (size_t i = 0; i < kElementCost; i++) {
      counter = ComputeSingleRound(counter, key);
      RaiseKey(&key);
    }
    SkipOne();
    return counter;
  }

 private:
  // We use the same constants as recommended by the original paper.
  static constexpr uint32_t kPhiloxW32A = 0x9E3779B9;
  static constexpr uint32_t kPhiloxW32B = 0xBB67AE85;
  static constexpr uint32_t kPhiloxM4x32A = 0xD2511F53;
  static constexpr uint32_t kPhiloxM4x32B = 0xCD9E8D57;

  // Helper function to skip the next sample of 128-bits in the current stream.
  void SkipOne() {
    if (++counter_[kIndex0] == 0) {
      if (++counter_[kIndex1] == 0) {
        if (++counter_[kIndex2] == 0) {
          ++counter_[kIndex3];
        }
      }
    }
  }
  /*
   * Helper function to return the lower and higher 32-bits from two 32-bit
   * integer multiplications.
   */
  static void MultiplyHighLow(uint32_t a, uint32_t b, uint32_t *result_low, uint32_t *result_high) {
    const uint64_t product = static_cast<uint64_t>(a) * b;
    *result_low = static_cast<uint32_t>(product);
    *result_high = static_cast<uint32_t>(product >> kMoveStepInBit);
  }

  // Helper function for a single round of the underlying Philox algorithm.
  static ResultType ComputeSingleRound(const ResultType &counter, const Key &key) {
    uint32_t lo0;
    uint32_t hi0;
    MultiplyHighLow(kPhiloxM4x32A, counter[kIndex0], &lo0, &hi0);

    uint32_t lo1;
    uint32_t hi1;
    MultiplyHighLow(kPhiloxM4x32B, counter[kIndex2], &lo1, &hi1);

    ResultType result;
    result[kIndex0] = hi1 ^ counter[kIndex1] ^ key[kIndex0];
    result[kIndex1] = lo1;
    result[kIndex2] = hi0 ^ counter[kIndex3] ^ key[kIndex1];
    result[kIndex3] = lo0;
    return result;
  }

  void RaiseKey(Key *key) {
    (*key)[kIndex0] += kPhiloxW32A;
    (*key)[kIndex1] += kPhiloxW32B;
  }

 private:
  ResultType counter_;
  Key key_;
};

BACKEND_EXPORT uint64_t GetSeed(const uint64_t &global_seed, const uint64_t &ops_seed);

}  // namespace random
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_KERNEL_PHILOX_RANDOM_H_
