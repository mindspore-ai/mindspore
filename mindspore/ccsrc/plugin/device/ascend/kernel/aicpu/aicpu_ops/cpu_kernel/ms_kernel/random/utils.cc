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

#include "utils.h"
#include "utils/log.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include <random>
#include <securec.h>

namespace aicpu {
namespace random {
std::mt19937_64 *InitRngWithRandomSeed() {
  std::random_device device("/dev/urandom");
  return new std::mt19937_64(device());
}

uint64_t New64() {
  static std::mt19937_64 *const rng = InitRngWithRandomSeed();
  return (*rng)();
}

// This function implements the Box-Muller transform:
void BoxMullerFloat(const uint32_t &x0, const uint32_t &x1, float *f0, float *f1) {
  const float epsilon = 1.0e-7f;
  float u1 = Uint32ToFloat(x0);
  if (u1 < epsilon) {
    u1 = epsilon;
  }
  const float v1 = 2.0f * M_PI * Uint32ToFloat(x1);
  const float u2 = Eigen::numext::sqrt(-2.0f * Eigen::numext::log(u1));
  *f0 = Eigen::numext::sin(v1);
  *f1 = Eigen::numext::cos(v1);
  *f0 *= u2;
  *f1 *= u2;
}

// This function implements the Box-Muller transform:
void BoxMullerDouble(const uint32_t &x0, const uint32_t &x1, const uint32_t &x2, const uint32_t &x3, double *d0,
                     double *d1) {
  const double epsilon = 1.0e-7;
  double u1 = Uint64ToDouble(x0, x1);
  if (u1 < epsilon) {
    u1 = epsilon;
  }
  const double v1 = 2 * M_PI * Uint64ToDouble(x2, x3);
  const double u2 = Eigen::numext::sqrt(-2.0 * Eigen::numext::log(u1));
  *d0 = Eigen::numext::sin(v1);
  *d1 = Eigen::numext::cos(v1);
  *d0 *= u2;
  *d1 *= u2;
}

float Uint32ToFloat(const uint32_t &x) {
  const uint32_t man = x & 0x7fffffu;  // 23 bit mantissa
  const uint32_t exp = static_cast<uint32_t>(127);
  const uint32_t val = (exp << 23) | man;
  float result;
  auto ret = memcpy_s(&result, sizeof(val), &val, sizeof(val));
  if (ret != EOK) {
    KERNEL_LOG_ERROR("Memcpy failed for Uint32ToFloat.");
  }
  return result - 1.0f;
}

double Uint64ToDouble(const uint32_t &x0, const uint32_t &x1) {
  const uint32_t mhi = x0 & 0xfffffu;                             // upper 20 bits of mantissa
  const uint32_t mlo = x1;                                        // lower 32 bits of mantissa
  const uint64_t man = (static_cast<uint64_t>(mhi) << 32) | mlo;  // mantissa
  const uint64_t exp = static_cast<uint64_t>(1023);
  const uint64_t val = (exp << 52) | man;
  double result;
  auto ret = memcpy_s(&result, sizeof(val), &val, sizeof(val));
  if (ret != EOK) {
    KERNEL_LOG_ERROR("Memcpy failed for Uint64ToDouble.");
  }
  return result - 1.0;
}
}  // namespace random
}  // namespace aicpu
