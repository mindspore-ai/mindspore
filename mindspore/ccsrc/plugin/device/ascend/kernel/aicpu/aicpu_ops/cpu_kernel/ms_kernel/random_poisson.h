/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_RANDOM_UNIFORM_H_
#define AICPU_KERNELS_NORMALIZED_RANDOM_UNIFORM_H_
#define EIGEN_USE_THREADS
#define EIGEN_USE_SIMPLE_THREAD_POOL
#include "cpu_ops_kernel.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace aicpu {
class RandomPoissonCpuKernel : public CpuKernel {
 public:
  RandomPoissonCpuKernel() = default;
  ~RandomPoissonCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  /**
   * @brief generate data from Eigen
   * @param ctx cpu kernel context
   * @param output using to output data
   * @return status if success
   */
  template <typename T>
  void Generate(CpuKernelContext &ctx, Tensor *output);
};

namespace {
EIGEN_DEVICE_FUNC uint64_t get_random_seed() {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  uint64_t rnd = ::random() ^ ts.tv_nsec;
  return rnd;
}
static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t PCG_XSH_RS_state(uint64_t seed) {
  seed = seed ? seed : get_random_seed();
  return seed * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
}
}  // namespace

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T RandomToTypePoisson(uint64_t *state, double rate) {
  using Eigen::numext::exp;
  using Eigen::numext::log;
  using Eigen::numext::pow;
  T result;

  uint64_t m_stream = 0;

  // 若rate < 10,使用Knuth's algorithm
  if (rate < static_cast<double>(10.0)) {
    const double exp_neg_rate = Eigen::numext::exp(-rate);
    double prod = 1;
    double x = 0;

    // Keep trying until we surpass e^(-rate). This will take
    // expected time proportional to rate.
    while (true) {
      double u = Eigen::internal::RandomToTypeUniform<double>(state, m_stream);
      prod = prod * u;
      if (prod <= exp_neg_rate && x <= double(Eigen::NumTraits<T>::highest())) {
        result = static_cast<T>(x);
        return result;
      }
      x += 1;
    }
  }

  if (Eigen::numext::isinf(rate) && rate > double(0)) {
    T k = Eigen::NumTraits<T>::infinity();
    return k;
  } else {
    const double log_rate = log(rate);
    const double b = double(0.931) + double(2.53) * Eigen::numext::sqrt(rate);
    const double a = double(-0.059) + double(0.02483) * b;
    const double inv_alpha = double(1.1239) + double(1.1328) / (b - double(3.4));

    while (true) {
      double u = Eigen::internal::RandomToTypeUniform<double>(state, m_stream);
      u -= double(0.5);
      double v = Eigen::internal::RandomToTypeUniform<double>(state, m_stream);
      double u_shifted = double(0.5) - Eigen::numext::abs(u);
      double k = Eigen::numext::floor((double(2) * a / u_shifted + b) * u + rate + double(0.43));

      if (k > Eigen::NumTraits<double>::highest()) {
        continue;
      }

      // When alpha * f(G(U)) * G'(U) is close to 1, it is possible to
      // find a rectangle (-u_r, u_r) x (0, v_r) under the curve, such
      // that if v <= v_r and |u| <= u_r, then we can accept.
      // Here v_r = 0.9227 - 3.6224 / (b - 2) and u_r = 0.43.
      if (u_shifted >= double(0.07) && v <= double(0.9277) - double(3.6224) / (b - double(2))) {
        result = static_cast<T>(k);
        return result;
      }

      if (k < 0 || (u_shifted < double(0.013) && v > u_shifted)) {
        continue;
      }

      // The expression below is equivalent to the computation of step 2)
      // in transformed rejection (v <= alpha * F'(G(u)) * G'(u)).
      double s = log(v * inv_alpha / (a / (u_shifted * u_shifted) + b));
      double t = -rate + k * log_rate - Eigen::numext::lgamma(k + 1);
      if (s <= t) {
        result = static_cast<T>(k);
        return result;
      }
    }
  }
}

template <typename T>
class PoissonRandomGenerator {
 public:
  // Uses the given "seed" if non-zero, otherwise uses a random seed.
  PoissonRandomGenerator(double rate, uint64_t seed = 0) {
    m_state = PCG_XSH_RS_state(seed);
    m_rate = rate;
  }
  void setRate(double rate) { m_rate = rate; }
  T gen() const {
    T result = RandomToTypePoisson<T>(&m_state, m_rate);
    return result;
  }

 private:
  double m_rate;
  mutable uint64_t m_state;
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_RANDOM_UNIFORM_H_
