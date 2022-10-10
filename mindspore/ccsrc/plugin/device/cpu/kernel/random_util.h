/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_GUARDED_PHILOX_RANDOM_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_GUARDED_PHILOX_RANDOM_H

#include <cstdint>
#include <mutex>
#include <string>
#include "Eigen/Core"

using mutex = std::mutex;
using mutex_lock = std::lock_guard<std::mutex>;

namespace mindspore {
namespace kernel {
namespace random {
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
template <typename T, size_t ElementCount>
class Array {
 public:
  Array() {
    for (size_t i = 0; i < ElementCount; ++i) {
      data_[i] = T(0);
    }
  }
  const T &operator[](size_t index) const { return data_[index]; }
  T &operator[](size_t index) { return data_[index]; }
  size_t size() const { return ElementCount; }

 private:
  T data_[ElementCount];
};

class MSPhiloxRandom {
 public:
  static constexpr size_t kKeyCount = 2;
  static constexpr size_t kResultElementCount = 4;
  static constexpr size_t loop_rounds = 10;
  /*
   * The type for the 64-bit key stored in the form of two 32-bit uint
   * that are used in the diffusion process.
   */
  using ResType = Array<uint32_t, kResultElementCount>;
  using Key = Array<uint32_t, kKeyCount>;

  MSPhiloxRandom() {}
  static constexpr int kMoveStepInBit = 32;
  explicit MSPhiloxRandom(uint64_t seed) {
    key_[kIndex0] = static_cast<uint32_t>(seed);
    key_[kIndex1] = static_cast<uint32_t>(seed >> kMoveStepInBit);
  }

  explicit MSPhiloxRandom(uint64_t seed_lo, uint64_t seed_hi) {
    key_[kIndex0] = static_cast<uint32_t>(seed_lo);
    key_[kIndex1] = static_cast<uint32_t>(seed_lo >> kMoveStepInBit);
    counter_[kIndex2] = static_cast<uint32_t>(seed_hi);
    counter_[kIndex3] = static_cast<uint32_t>(seed_hi >> kMoveStepInBit);
  }

  MSPhiloxRandom(const ResType &counter, const Key &key) : counter_(counter), key_(key) {}
  ResType const &counter() const { return counter_; }
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
  ResType operator()() {
    ResType counter = counter_;
    Key key = key_;
    for (size_t i = 0; i < loop_rounds; i++) {
      counter = SingleRoundCompute(counter, key);
      RaiseKey(&key);
    }
    SkipOne();
    return counter;
  }

 private:
  // We use the same constants as recommended by the original paper.
  static constexpr uint32_t kMSPhiloxW32A = 0x9E3779B9;
  static constexpr uint32_t kMSPhiloxW32B = 0xBB67AE85;
  static constexpr uint32_t kMSPhiloxM4x32A = 0xD2511F53;
  static constexpr uint32_t kMSPhiloxM4x32B = 0xCD9E8D57;

  void SkipOne() {
    if (++counter_[kIndex0] == 0) {
      if (++counter_[kIndex1] == 0) {
        if (++counter_[kIndex2] == 0) {
          ++counter_[kIndex3];
        }
      }
    }
  }

  static void HighLowMultiply(uint32_t a, uint32_t b, uint32_t *result_low, uint32_t *result_high) {
    const uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    *result_low = static_cast<uint32_t>(product);
    *result_high = static_cast<uint32_t>(product >> kMoveStepInBit);
  }

  static ResType SingleRoundCompute(const ResType &counter, const Key &key) {
    uint32_t low0;
    uint32_t high0;
    HighLowMultiply(kMSPhiloxM4x32A, counter[kIndex0], &low0, &high0);

    uint32_t low1;
    uint32_t high1;
    HighLowMultiply(kMSPhiloxM4x32B, counter[kIndex2], &low1, &high1);

    ResType result;
    result[kIndex0] = high1 ^ counter[kIndex1] ^ key[kIndex0];
    result[kIndex1] = low1;
    result[kIndex2] = high0 ^ counter[kIndex3] ^ key[kIndex1];
    result[kIndex3] = low0;
    return result;
  }

  void RaiseKey(Key *const key) {
    (*key)[kIndex0] += kMSPhiloxW32A;
    (*key)[kIndex1] += kMSPhiloxW32B;
  }

  ResType counter_;
  Key key_;
};

class GuardedPhiloxRandom {
 public:
  GuardedPhiloxRandom() : initialized_(false) {}

  void Init(uint64_t seed, uint64_t seed2);
  void Init(const random::MSPhiloxRandom::ResType &counter, const random::MSPhiloxRandom::Key &key);

  random::MSPhiloxRandom ReserveSamples128(uint64_t samples);

  random::MSPhiloxRandom ReserveRandomOutputs(int64_t output_count, int multiplier) {
    uint64_t conservative_sample_count = static_cast<uint64_t>(output_count * multiplier);
    return ReserveSamples128(conservative_sample_count);
  }

 private:
  mutex mu_;
  random::MSPhiloxRandom generator_;
  bool initialized_;
  uint64_t New64() const;

  GuardedPhiloxRandom(const GuardedPhiloxRandom &) = delete;
  void operator=(const GuardedPhiloxRandom &) = delete;
};

double Uint64ToDouble(uint32_t x0, uint32_t x1);

void BoxMullerDouble(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, double *data0, double *data1);

template <class T, typename RealType>
class MSNormalDistribution;

template <class T, typename RealType>
class MSUniformDistribution;

template <class T>
class MSUniformDistribution<T, double> {
 public:
  static const int kResultElementCount = T::kResultElementCount / kIndex2;
  using ResType = random::Array<double, kResultElementCount>;
  using ResultElementType = double;

  ResType operator()(T *gen) const {
    typename T::ResType sample = (*gen)();
    ResType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      result[i] = Uint64ToDouble(sample[kIndex2 * i], sample[kIndex2 * i + kIndex1]);
    }
    return result;
  }
};

template <class T>
class MSNormalDistribution<T, double> {
 public:
  static const int kResultElementCount = T::kResultElementCount / kIndex2;
  using ResType = random::Array<double, kResultElementCount>;
  using ResultElementType = double;

  ResType operator()(T *gen) const {
    typename T::ResType sample = (*gen)();
    ResType result;
    for (int i = 0; i < kResultElementCount; i += kIndex2) {
      const int i2 = kIndex2 * i;
      BoxMullerDouble(sample[i2], sample[i2 + kIndex1], sample[i2 + kIndex2], sample[i2 + kIndex3], &result[i],
                      &result[i + kIndex1]);
    }
    return result;
  }
};
}  // namespace random
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_GUARDED_PHILOX_RANDOM_H
