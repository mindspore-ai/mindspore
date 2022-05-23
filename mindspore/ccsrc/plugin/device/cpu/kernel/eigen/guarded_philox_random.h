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
#include "Eigen/Core"

#define PHILOX_DEVICE_INLINE

#define MS_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName &) = delete;        \
  void operator=(const TypeName &) = delete

using mutex = std::mutex;
using mutex_lock = std::lock_guard<std::mutex>;
using uint32 = uint32_t;
using uint64 = uint64_t;
using int64 = int64_t;
using int16 = int16_t;

namespace mindspore {
namespace random {
template <typename T, int ElementCount>
class Array {
 public:
  static const int kElementCount = ElementCount;
  PHILOX_DEVICE_INLINE Array() {
    for (int i = 0; i < ElementCount; ++i) {
      data_[i] = T(0);
    }
  }

  PHILOX_DEVICE_INLINE const T &operator[](int index) const { return data_[index]; }

  PHILOX_DEVICE_INLINE T &operator[](int index) { return data_[index]; }

  size_t size() const { return ElementCount; }

 private:
  T data_[ElementCount];
};

// Helper function to convert two 32-bit integers to a double between [0..1).
PHILOX_DEVICE_INLINE double Uint64ConvertToDouble(uint32 x0, uint32 x1);

void BoxMullerDouble(uint32 x0, uint32 x1, uint32 x2, uint32 x3, double *data0, double *data1);

uint64 New64();

class PhiloxRandom {
 public:
  using ResultType = Array<uint32, 4>;
  using ResultElementType = uint32;
  // The number of elements that will be returned.
  static const int kResultElementCount = 4;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 10;
  // The type for the 64-bit key stored in the form of two 32-bit uint
  // that are used in the diffusion process.
  using Key = Array<uint32, 2>;

  PHILOX_DEVICE_INLINE
  PhiloxRandom() {}

  PHILOX_DEVICE_INLINE
  explicit PhiloxRandom(uint64 seed) {
    key_[0] = static_cast<uint32>(seed);
    key_[1] = static_cast<uint32>(seed >> 32);
  }

  PHILOX_DEVICE_INLINE
  explicit PhiloxRandom(uint64 seed_lo, uint64 seed_hi) {
    key_[0] = static_cast<uint32>(seed_lo);
    key_[1] = static_cast<uint32>(seed_lo >> 32);
    counter_[2] = static_cast<uint32>(seed_hi);
    counter_[3] = static_cast<uint32>(seed_hi >> 32);
  }

  PHILOX_DEVICE_INLINE
  PhiloxRandom(ResultType counter, Key key) : counter_(counter), key_(key) {}

  PHILOX_DEVICE_INLINE
  ResultType const &counter() const { return counter_; }

  PHILOX_DEVICE_INLINE
  Key const &key() const { return key_; }

  // Skip the specified number of samples of 128-bits in the current stream.
  PHILOX_DEVICE_INLINE
  void Skip(uint64 count) {
    const uint32 count_lo = static_cast<uint32>(count);
    uint32 count_hi = static_cast<uint32>(count >> 32);

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

  // Returns a group of four random numbers using the underlying Philox
  // algorithm.
  PHILOX_DEVICE_INLINE ResultType operator()() {
    ResultType res_counter = counter_;
    Key res_key = key_;

    // Run the single rounds for ten times. Manually unrolling the loop
    // for better performance.
    res_counter = ComputeSingleRound(res_counter, res_key);
    RaiseKey(&res_key);
    res_counter = ComputeSingleRound(res_counter, res_key);
    RaiseKey(&res_key);
    res_counter = ComputeSingleRound(res_counter, res_key);
    RaiseKey(&res_key);
    res_counter = ComputeSingleRound(res_counter, res_key);
    RaiseKey(&res_key);
    res_counter = ComputeSingleRound(res_counter, res_key);
    RaiseKey(&res_key);
    res_counter = ComputeSingleRound(res_counter, res_key);
    RaiseKey(&res_key);
    res_counter = ComputeSingleRound(res_counter, res_key);
    RaiseKey(&res_key);
    res_counter = ComputeSingleRound(res_counter, res_key);
    RaiseKey(&res_key);
    res_counter = ComputeSingleRound(res_counter, res_key);
    RaiseKey(&res_key);
    res_counter = ComputeSingleRound(res_counter, res_key);

    SkipOne();

    return res_counter;
  }

 private:
  // We use the same constants as recommended by the original paper.
  static const uint32 kPhiloxW32A = 0x9E3779B9;
  static const uint32 kPhiloxW32B = 0xBB67AE85;
  static const uint32 kPhiloxM4x32A = 0xD2511F53;
  static const uint32 kPhiloxM4x32B = 0xCD9E8D57;

  // Helper function to skip the next sample of 128-bits in the current stream.
  PHILOX_DEVICE_INLINE void SkipOne() {
    if (++counter_[0] == 0) {
      if (++counter_[1] == 0) {
        if (++counter_[2] == 0) {
          ++counter_[3];
        }
      }
    }
  }

  // Helper function to return the lower and higher 32-bits from two 32-bit
  // integer multiplications.
  PHILOX_DEVICE_INLINE
  static void MultiplyHighLow(uint32 a, uint32 b, uint32 *result_low, uint32 *result_high) {
#ifndef __CUDA_ARCH__
    const uint64 product = static_cast<uint64>(a) * b;
    *result_low = static_cast<uint32>(product);
    *result_high = static_cast<uint32>(product >> 32);
#else
    *result_low = a * b;
    *result_high = __umulhi(a, b);
#endif
  }

  // Helper function for a single round of the underlying Philox algorithm.
  PHILOX_DEVICE_INLINE static ResultType ComputeSingleRound(const ResultType &counter, const Key &key) {
    uint32 lo0;
    uint32 hi0;
    MultiplyHighLow(kPhiloxM4x32A, counter[0], &lo0, &hi0);

    uint32 lo1;
    uint32 hi1;
    MultiplyHighLow(kPhiloxM4x32B, counter[2], &lo1, &hi1);

    ResultType result;
    result[0] = hi1 ^ counter[1] ^ key[0];
    result[1] = lo1;
    result[2] = hi0 ^ counter[3] ^ key[1];
    result[3] = lo0;
    return result;
  }

  PHILOX_DEVICE_INLINE void RaiseKey(Key *key) {
    (*key)[0] += kPhiloxW32A;
    (*key)[1] += kPhiloxW32B;
  }

 private:
  ResultType counter_;
  Key key_;
};

template <class Generator, typename RealType>
class NormalDistribution;

template <class Generator, typename RealType>
class UniformDistribution;

template <class Generator>
class UniformDistribution<Generator, double> {
 public:
  // The number of elements that will be returned.
  static const int kResultElementCount = Generator::kResultElementCount / 2;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static const bool kVariableSamplesPerOutput = false;
  using ResultType = Array<double, kResultElementCount>;
  using ResultElementType = double;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator *gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      result[i] = Uint64ConvertToDouble(sample[2 * i], sample[2 * i + 1]);
    }
    return result;
  }
};

template <class Generator>
class NormalDistribution<Generator, double> {
 public:
  // The number of elements that will be returned.
  static const int kResultElementCount = Generator::kResultElementCount / 2;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 70;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static const bool kVariableSamplesPerOutput = false;
  using ResultType = Array<double, kResultElementCount>;
  using ResultElementType = double;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator *gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; i += 2) {
      const int i2 = 2 * i;
      BoxMullerDouble(sample[i2], sample[i2 + 1], sample[i2 + 2], sample[i2 + 3], &result[i], &result[i + 1]);
    }
    return result;
  }
};
}  // namespace random

class GuardedPhiloxRandom {
 public:
  // Must call Init to finish initialization
  GuardedPhiloxRandom() : initialized_(false) {}

  // Initialize the generator from attributes "seed" and "seed2".
  // If both seeds are unspecified, use random seeds.
  // Must be called exactly once.
  //  Status Init(OpKernelConstruction* context);

  // Initialize with given seeds.
  void Init(int64 seed, int64 seed2);
  void Init(random::PhiloxRandom::ResultType counter, random::PhiloxRandom::Key key);

  // Reserve a certain number of 128-bit samples.
  // This function is thread safe.  The returned generator is valid for the
  // given number of samples, and can be used without a lock.
  random::PhiloxRandom ReserveSamples128(int64 samples);

  // Reserve a certain number of 32-bit samples.
  random::PhiloxRandom ReserveSamples32(int64 samples) { return ReserveSamples128((samples + 3) / 4); }

  // Reserve enough random samples in the generator for the given output count.
  random::PhiloxRandom ReserveRandomOutputs(int64 output_count, int multiplier) {
    int64 conservative_sample_count = output_count * multiplier;
    return ReserveSamples128(conservative_sample_count);
  }

 private:
  mutex mu_;
  random::PhiloxRandom generator_;  // GUARDED_BY(mu_);
  bool initialized_;

  MS_DISALLOW_COPY_AND_ASSIGN(GuardedPhiloxRandom);
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_GUARDED_PHILOX_RANDOM_H
