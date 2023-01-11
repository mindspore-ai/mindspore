/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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

#ifndef _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_PHILOX_RANDOM_H
#define _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_PHILOX_RANDOM_H

#include <stdint.h>
#include "cpu_kernel/common/status.h"

/**
 * A class that represents an inline array.
 * Arguments:
 *   T: the array element type;
 *   ElementCount: the fixed size of the array;
 */
template <typename T, int ElementCount>
class Array {
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

class PhiloxRandom {
 public:
  using ResultType = Array<uint32_t, 4>;
  using ResultElementType = uint32_t;
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = 4;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 10;
  /*
   * The type for the 64-bit key stored in the form of two 32-bit uint
   * that are used in the diffusion process.
   */
  using Key = Array<uint32_t, 2>;

  PhiloxRandom() {}

  PhiloxRandom(int64_t seed, uint64_t offset) {
    const uint32_t seed_low_index = 0;
    const uint32_t seed_high_index = 1;
    const uint32_t offset_low_index = 2;
    const uint32_t offset_high_index = 3;
    key_[seed_low_index] = static_cast<uint32_t>(seed);
    key_[seed_high_index] = static_cast<uint32_t>(seed >> 32);
    counter_[offset_low_index] = static_cast<uint32_t>(offset);
    counter_[offset_high_index] = static_cast<uint32_t>(offset >> 32);
  }

  ResultType const &counter() const { return counter_; }

  Key const &key() const { return key_; }

  // Skip the specified number of samples of 128-bits in the current stream.
  void Skip(uint64_t count) {
    const uint32_t count_lo = static_cast<uint32_t>(count);
    uint32_t count_hi = static_cast<uint32_t>(count >> 32);

    counter_[0] += count_lo;
    if (counter_[0] < count_lo) {
      ++count_hi;
    }

    counter_[1] += count_hi;
    if (counter_[1] < count_hi) {
      if (++counter_[2] == 0) {
        ++counter_[3];
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
    /*
     * Run the single rounds for ten times. Manually unrolling the loop
     * for better performance.
     */
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
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
    if (++counter_[0] == 0) {
      if (++counter_[1] == 0) {
        if (++counter_[2] == 0) {
          ++counter_[3];
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
    *result_high = static_cast<uint32_t>(product >> 32);
  }

  // Helper function for a single round of the underlying Philox algorithm.
  static ResultType ComputeSingleRound(const ResultType &counter, const Key &key) {
    uint32_t lo0;
    uint32_t hi0;
    MultiplyHighLow(kPhiloxM4x32A, counter[0], &lo0, &hi0);

    uint32_t lo1;
    uint32_t hi1;
    MultiplyHighLow(kPhiloxM4x32B, counter[2], &lo1, &hi1);

    ResultType result;
    result[0] = hi1 ^ counter[1] ^ key[0];
    result[1] = lo1;
    result[2] = hi0 ^ counter[3] ^ key[1];
    result[3] = lo0;
    return result;
  }

  void RaiseKey(Key *key) {
    (*key)[0] += kPhiloxW32A;
    (*key)[1] += kPhiloxW32B;
  }

 private:
  ResultType counter_;
  Key key_;
};
#endif  // _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_PHILOX_RANDOM_H_
