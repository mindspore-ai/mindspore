/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include <iostream>
#include <mutex>
#include <string>
#include "Eigen/Core"
#include "include/runtime/hardware_abstract/kernel_base/philox_random.h"

using mutex = std::mutex;
using mutex_lock = std::lock_guard<std::mutex>;

namespace mindspore {
namespace kernel {
namespace random {
class GuardedPhiloxRandom {
 public:
  GuardedPhiloxRandom() : initialized_(false) {}

  void Init(uint64_t seed, uint64_t seed2);
  void Init(const random::PhiloxRandom::ResultType &counter, const random::PhiloxRandom::Key &key);

  random::PhiloxRandom ReserveSamples128(uint64_t samples);

  random::PhiloxRandom ReserveSamples32(int64_t samples) { return ReserveSamples128((samples + 3) / 4); }

  random::PhiloxRandom ReserveRandomOutputs(int64_t output_count, int multiplier) {
    int64_t conservative_sample_count = output_count * multiplier;
    return ReserveSamples128(conservative_sample_count);
  }

 private:
  mutex mu_;
  random::PhiloxRandom generator_;
  bool initialized_;
  uint64_t New64();

  GuardedPhiloxRandom(const GuardedPhiloxRandom &) = delete;
  void operator=(const GuardedPhiloxRandom &) = delete;
};

double Uint64ToDouble(uint32_t x0, uint32_t x1);
float Uint32ToFloat(uint32_t x);
class SinglePhiloxRandom {
 public:
  explicit SinglePhiloxRandom(PhiloxRandom *gen)
      : generator_(gen), group_random_idx_(PhiloxRandom::kResultElementCount) {}
  uint32_t GenUint32() {
    if (group_random_idx_ == PhiloxRandom::kResultElementCount) {
      group_random_ = (*generator_)();
      group_random_idx_ = 0;
    }
    return group_random_[group_random_idx_++];
  }
  uint64_t GenUint64() {
    uint32_t lo = GenUint32(), hi = GenUint32();
    return lo | static_cast<uint64_t>(hi) << 32;
  }
  float GenFloat() {
    uint32_t u0 = GenUint32();
    return Uint32ToFloat(u0);
  }
  double GenDouble() {
    uint32_t lo = GenUint32(), hi = GenUint32();
    return Uint64ToDouble(lo, hi);
  }

 private:
  PhiloxRandom *generator_;
  PhiloxRandom::ResultType group_random_;
  int group_random_idx_ = 0;
};

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

  ResType operator()(T *gen) {
    typename T::ResultType sample = (*gen)();
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

  ResType operator()(T *gen) {
    typename T::ResultType sample = (*gen)();
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
