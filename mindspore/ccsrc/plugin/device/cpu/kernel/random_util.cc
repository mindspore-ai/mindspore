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

#include <random>
#include "securec/include/securec.h"
#include "mindspore/ccsrc/plugin/device/cpu/kernel/random_util.h"

namespace mindspore {
namespace kernel {
namespace random {
constexpr double M_PI_ = 3.14159265358979323846;
constexpr uint16_t kFp64ExpBias = 1023;
constexpr uint16_t kFp64ManLen = 52;
constexpr uint16_t kBitShift32 = 32;
uint64_t GuardedPhiloxRandom::New64() {
  std::random_device device;
  static std::mt19937_64 *rng = new std::mt19937_64(device());
  static mutex my_mutex;
  mutex_lock l(my_mutex);
  return (*rng)();
}

void GuardedPhiloxRandom::Init(uint64_t seed, uint64_t seed2) {
  if (seed == 0 && seed2 == 0) {
    seed = New64();
    seed2 = New64();
  }
  mutex_lock lock(mu_);
  generator_ = random::MSPhiloxRandom(seed, seed2);
  initialized_ = true;
}

void GuardedPhiloxRandom::Init(const random::MSPhiloxRandom::ResType &counter, const random::MSPhiloxRandom::Key &key) {
  mutex_lock lock(mu_);
  generator_ = random::MSPhiloxRandom(counter, key);
  initialized_ = true;
}

random::MSPhiloxRandom GuardedPhiloxRandom::ReserveSamples128(uint64_t samples) {
  mutex_lock lock(mu_);
  auto local = generator_;
  generator_.Skip(samples);
  return local;
}
double Uint64ToDouble(uint32_t x0, uint32_t x1) {
  const uint64_t ex = static_cast<uint64_t>(kFp64ExpBias);
  const uint32_t m_hi = x0 & 0xfffffu;
  const uint32_t m_lo = x1;
  const uint64_t mantissa = (static_cast<uint64_t>(m_hi) << kBitShift32) | m_lo;
  const uint64_t val = (ex << kFp64ManLen) | mantissa;
  double d_result;
  (void)memcpy_s(&d_result, sizeof(val), &val, sizeof(val));
  return d_result - 1.0;
}

void BoxMullerDouble(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, double *data0, double *data1) {
  const double epsilon = 1.0e-7;
  double u1 = Uint64ToDouble(x0, x1);
  const double val1 = 2 * M_PI_ * Uint64ToDouble(x2, x3);
  u1 = u1 < epsilon ? epsilon : u1;
  const double u2 = Eigen::numext::sqrt(-2.0 * Eigen::numext::log(u1));
#if !defined(__linux__)
  *data0 = Eigen::numext::sin(val1);
  *data1 = Eigen::numext::cos(val1);
#else
  sincos(val1, data0, data1);
#endif
  *data0 *= u2;
  *data1 *= u2;
}
}  // namespace random
}  // namespace kernel
}  // namespace mindspore
