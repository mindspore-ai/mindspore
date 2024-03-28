/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_RANDOM_POISSON_H_
#define AICPU_KERNELS_NORMALIZED_RANDOM_POISSON_H_
#define EIGEN_USE_THREADS
#define EIGEN_USE_SIMPLE_THREAD_POOL
#include "inc/ms_cpu_kernel.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "random/utils.h"

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
  uint32_t Generate(CpuKernelContext &ctx, Tensor *output);
  uint64_t seed_ = 0;
  uint64_t seed2_ = 0;
  std::mt19937 rng_;
};

EIGEN_DEVICE_FUNC uint64_t get_random_seed() {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  uint64_t rnd = ::random() ^ ts.tv_nsec;
  return rnd;
}
static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t PCG_XSH_RS_state(uint64_t seed, uint64_t seed2) {
  if (seed == 0 && seed2 == 0) {
    seed = get_random_seed();
  } else {
    seed = random::GetSeed(seed, seed2);
  }
  return seed * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T RandomToTypePoisson(std::mt19937 &rng, double rate) {
  std::uniform_real_distribution<double> standard_uniform(0.0, 1.0);

  // if rate < 10, use Knuth's algorithm
  if (rate < static_cast<double>(10.0)) {
    double x, prod, u, exp_neg_rate;

    x = 0.0;
    prod = 1.0;
    exp_neg_rate = std::exp(-rate);
    // Keep trying until we surpass e^(-rate). This will take
    // expected time proportional to rate.
    while (true) {
      u = standard_uniform(rng);
      prod *= u;
      if (prod <= exp_neg_rate && x <= static_cast<double>(Eigen::NumTraits<T>::highest())) {
        return static_cast<T>(x);
      }
      x += 1;
    }
  }

  if (Eigen::numext::isinf(rate) && rate > static_cast<double>(0)) {
    T k = Eigen::NumTraits<T>::infinity();
    return k;
  } else {
    double k, u, v, a, b, inv_alpha, vr, us;

    double log_rate = std::log(rate);
    double sqrt_rate = std::sqrt(rate);
    b = static_cast<double>(0.931) + static_cast<double>(2.53) * sqrt_rate;
    a = static_cast<double>(-0.059) + static_cast<double>(0.02483) * b;
    inv_alpha = static_cast<double>(1.1239) + static_cast<double>(1.1328) / (b - static_cast<double>(3.4));
    vr = static_cast<double>(0.9277) - static_cast<double>(3.6224) / (b - static_cast<double>(2));

    while (true) {
      u = standard_uniform(rng) - static_cast<double>(0.5);
      v = standard_uniform(rng);
      us = static_cast<double>(0.5) - std::fabs(u);
      k = std::floor((static_cast<double>(2) * a / us + b) * u + rate + static_cast<double>(0.43));

      if (k > static_cast<double>(Eigen::NumTraits<double>::highest())) {
        continue;
      }

      // When alpha * f(G(U)) * G'(U) is close to 1, it is possible to
      // find a rectangle (-u_s, u_s) x (0, v_r) under the curve, such
      // that if v <= v_r and |u| <= u_s, then we can accept.
      // Here v_r = 0.9227 - 3.6224 / (b - 2) and u_s = 0.43.
      if ((us >= static_cast<double>(0.07)) && (v <= vr)) {
        return static_cast<T>(k);
      }

      if ((k < 0) || ((us < static_cast<double>(0.013)) && (v > us))) {
        continue;
      }

      // The expression below is equivalent to the computation of step 2)
      // in transformed rejection (v <= alpha * F'(G(u)) * G'(u)).
      double s = std::log(v * inv_alpha / (a / (us * us) + b));
      double t = -rate + k * log_rate - std::lgamma(k + 1);
      if (s <= t) {
        return static_cast<T>(k);
      }
    }
  }
}

template <typename T>
class PoissonRandomGenerator {
 public:
  explicit PoissonRandomGenerator(double rate) : m_rate(rate) {}
  void setRate(double rate) { m_rate = rate; }
  T gen(std::mt19937 &rng) const {
    T result = RandomToTypePoisson<T>(rng, m_rate);
    return result;
  }

 private:
  double m_rate;
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_RANDOM_POISSON_H_
