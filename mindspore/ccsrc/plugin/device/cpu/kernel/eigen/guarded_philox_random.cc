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

#include "plugin/device/cpu/kernel/eigen/guarded_philox_random.h"
#include <random>

namespace mindspore {
#define CHECK(X)
constexpr double M_PI_ = 3.14159265358979323846;
namespace random {
uint64 New64() {
  std::random_device device("/dev/urandom");
  static std::mt19937_64 *rng = new std::mt19937_64(device());
  static mutex my_mutex;
  mutex_lock l(my_mutex);
  return (*rng)();
}

PHILOX_DEVICE_INLINE double Uint64ConvertToDouble(uint32 x0, uint32 x1) {
  const uint64 ex = static_cast<uint64>(1023);

  const uint32 m_hi = x0 & 0xfffffu;
  const uint32 m_lo = x1;
  const uint64 mantissa = (static_cast<uint64>(m_hi) << 32) | m_lo;
  const uint64 val = (ex << 52) | mantissa;
  double d_result;
  memcpy(&d_result, &val, sizeof(val));
  return d_result - 1.0;
}

void BoxMullerDouble(uint32 x0, uint32 x1, uint32 x2, uint32 x3, double *data0, double *data1) {
  // This function implements the Box-Muller transform:
  // http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Basic_form
  const double epsilon = 1.0e-7;
  double u1 = Uint64ConvertToDouble(x0, x1);
  if (u1 < epsilon) {
    u1 = epsilon;
  }
  const double val1 = 2 * M_PI_ * Uint64ConvertToDouble(x2, x3);
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

void GuardedPhiloxRandom::Init(int64 seed, int64 seed2) {
  if (seed == 0 && seed2 == 0) {
    // If both seeds are unspecified, use completely random seeds.
    seed = random::New64();
    seed2 = random::New64();
  }
  mutex_lock lock(mu_);
  generator_ = random::PhiloxRandom(seed, seed2);
  initialized_ = true;
}

void GuardedPhiloxRandom::Init(random::PhiloxRandom::ResultType counter, random::PhiloxRandom::Key key) {
  mutex_lock lock(mu_);
  generator_ = random::PhiloxRandom(counter, key);
  initialized_ = true;
}

random::PhiloxRandom GuardedPhiloxRandom::ReserveSamples128(int64 samples) {
  mutex_lock lock(mu_);
  auto local = generator_;
  generator_.Skip(samples);
  return local;
}
}  // namespace mindspore
