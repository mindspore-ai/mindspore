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
#include "kernel/cpu/eigen/random_poisson_cpu_kernel.h"
#include <limits>
#include <random>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/philox_random.h"

namespace mindspore {
namespace kernel {
namespace {
using KernelRunFunc = RandomPoissonCpuKernelMod::KernelRunFunc;
#define ADD_KERNEL(shape_dtype, rate_dtype, output_dtype, rate_type, output_type) \
  {KernelAttr()                                                                   \
     .AddInputAttr(kNumberType##shape_dtype)                                      \
     .AddInputAttr(kNumberType##rate_dtype)                                       \
     .AddOutputAttr(kNumberType##output_dtype),                                   \
   &RandomPoissonCpuKernelMod::LaunchKernel<rate_type, output_type>}

static unsigned int s_seed = static_cast<unsigned int>(time(nullptr));
#ifndef _MSC_VER
EIGEN_DEVICE_FUNC uint64_t get_random_seed() {
  auto rnd = rand_r(&s_seed);
  return IntToSize(rnd);
}
#else
EIGEN_DEVICE_FUNC uint64_t get_random_seed() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> distribution(0, std::numeric_limits<uint64_t>::max());
  return distribution(gen);
}
#endif

static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t PCG_XSH_RS_state(uint64_t seed, uint64_t seed2) {
  if (seed == 0 && seed2 == 0) {
    seed = get_random_seed();
  } else {
    seed = random::GetSeed(seed, seed2);
  }
  return seed * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
}
}  // namespace

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T RandomToTypePoisson(std::mt19937 *rng, double rate) {
  std::uniform_real_distribution<double> standard_uniform(0.0, 1.0);

  // if rate < 10, use Knuth's algorithm
  if (rate < static_cast<double>(10.0)) {
    double x, prod, exp_neg_rate;

    x = 0.0;
    prod = 1.0;
    exp_neg_rate = std::exp(-rate);
    // Keep trying until we surpass e^(-rate). This will take
    // expected time proportional to rate.
    while (true) {
      double u = standard_uniform(*rng);
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
    double a, b, inv_alpha, vr;

    double log_rate = std::log(rate);
    double sqrt_rate = std::sqrt(rate);
    b = static_cast<double>(0.931) + static_cast<double>(2.53) * sqrt_rate;
    a = static_cast<double>(-0.059) + static_cast<double>(0.02483) * b;
    inv_alpha = static_cast<double>(1.1239) + static_cast<double>(1.1328) / (b - static_cast<double>(3.4));
    vr = static_cast<double>(0.9277) - static_cast<double>(3.6224) / (b - static_cast<double>(2));

    while (true) {
      double u = standard_uniform(*rng) - static_cast<double>(0.5);
      double v = standard_uniform(*rng);
      double us = static_cast<double>(0.5) - std::fabs(u);
      double k = std::floor((static_cast<double>(2) * a / us + b) * u + rate + static_cast<double>(0.43));

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
  T gen(std::mt19937 *rng) const {
    T result = RandomToTypePoisson<T>(rng, m_rate);
    return result;
  }

 private:
  double m_rate;
};

bool AddrAlignedCheck(const void *addr, uint64_t alignment = 16) {
  if (alignment == 0) {
    return false;
  }
  return reinterpret_cast<uint64_t>(addr) % alignment == 0;
}

bool RandomPoissonCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  uint64_t seed = static_cast<uint64_t>(GetValue<int64_t>(primitive_->GetAttr("seed")));
  uint64_t seed2 = static_cast<uint64_t>(GetValue<int64_t>(primitive_->GetAttr("seed2")));
  uint64_t init_seed = PCG_XSH_RS_state(seed, seed2);
  rng_.seed(init_seed);
  return true;
}

template <typename Tin, typename T>
bool RandomPoissonCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &,
                                             const std::vector<kernel::KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[1]);
  MS_EXCEPTION_IF_NULL(outputs[0]);
  auto *rate_flat = reinterpret_cast<Tin *>(inputs[1]->device_ptr());
  auto *output = reinterpret_cast<T *>(outputs[0]->device_ptr());
  MS_EXCEPTION_IF_NULL(rate_flat);
  MS_EXCEPTION_IF_NULL(output);

  size_t num_of_rate = inputs[1]->size() / sizeof(Tin);
  size_t num_of_output = outputs[0]->size() / sizeof(T);

  if (AddrAlignedCheck(outputs[0]->device_ptr())) {
    Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> eigen_output(static_cast<T *>(output), num_of_output);
    PoissonRandomGenerator<T> m_generator(static_cast<double>(rate_flat[0]));
    for (size_t i = 0; i < num_of_rate; i++) {
      m_generator.setRate(static_cast<double>(rate_flat[i]));
      for (size_t j = i; j < num_of_output; j += num_of_rate) {
        eigen_output(j) = m_generator.gen(&rng_);
      }
    }
  } else {
    Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Unaligned> eigen_output(static_cast<T *>(output), num_of_output);
    PoissonRandomGenerator<T> m_generator(static_cast<double>(rate_flat[0]));
    for (size_t i = 0; i < num_of_rate; i++) {
      m_generator.setRate(static_cast<double>(rate_flat[i]));
      for (size_t j = i; j < num_of_output; j += num_of_rate) {
        eigen_output(j) = m_generator.gen(&rng_);
      }
    }
  }
  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &RandomPoissonCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    ADD_KERNEL(Int32, Float16, Float16, float16, float16), ADD_KERNEL(Int32, Float16, Float32, float16, float),
    ADD_KERNEL(Int32, Float16, Float64, float16, double),  ADD_KERNEL(Int32, Float16, Int32, float16, int),
    ADD_KERNEL(Int32, Float16, Int64, float16, int64_t),

    ADD_KERNEL(Int32, Float32, Float16, float, float16),   ADD_KERNEL(Int32, Float32, Float32, float, float),
    ADD_KERNEL(Int32, Float32, Float64, float, double),    ADD_KERNEL(Int32, Float32, Int32, float, int),
    ADD_KERNEL(Int32, Float32, Int64, float, int64_t),

    ADD_KERNEL(Int32, Float64, Float16, double, float16),  ADD_KERNEL(Int32, Float64, Float32, double, float),
    ADD_KERNEL(Int32, Float64, Float64, double, double),   ADD_KERNEL(Int32, Float64, Int32, double, int),
    ADD_KERNEL(Int32, Float64, Int64, double, int64_t),

    ADD_KERNEL(Int32, Int32, Float16, int, float16),       ADD_KERNEL(Int32, Int32, Float32, int, float),
    ADD_KERNEL(Int32, Int32, Float64, int, double),        ADD_KERNEL(Int32, Int32, Int32, int, int),
    ADD_KERNEL(Int32, Int32, Int64, int, int64_t),

    ADD_KERNEL(Int32, Int64, Float16, int64_t, float16),   ADD_KERNEL(Int32, Int64, Float32, int64_t, float),
    ADD_KERNEL(Int32, Int64, Float64, int64_t, double),    ADD_KERNEL(Int32, Int64, Int32, int64_t, int),
    ADD_KERNEL(Int32, Int64, Int64, int64_t, int64_t),

    ADD_KERNEL(Int64, Float16, Float16, float16, float16), ADD_KERNEL(Int64, Float16, Float32, float16, float),
    ADD_KERNEL(Int64, Float16, Float64, float16, double),  ADD_KERNEL(Int64, Float16, Int32, float16, int),
    ADD_KERNEL(Int64, Float16, Int64, float16, int64_t),

    ADD_KERNEL(Int64, Float32, Float16, float, float16),   ADD_KERNEL(Int64, Float32, Float32, float, float),
    ADD_KERNEL(Int64, Float32, Float64, float, double),    ADD_KERNEL(Int64, Float32, Int32, float, int),
    ADD_KERNEL(Int64, Float32, Int64, float, int64_t),

    ADD_KERNEL(Int64, Float64, Float16, double, float16),  ADD_KERNEL(Int64, Float64, Float32, double, float),
    ADD_KERNEL(Int64, Float64, Float64, double, double),   ADD_KERNEL(Int64, Float64, Int32, double, int),
    ADD_KERNEL(Int64, Float64, Int64, double, int64_t),

    ADD_KERNEL(Int64, Int32, Float16, int, float16),       ADD_KERNEL(Int64, Int32, Float32, int, float),
    ADD_KERNEL(Int64, Int32, Float64, int, double),        ADD_KERNEL(Int64, Int32, Int32, int, int),
    ADD_KERNEL(Int64, Int32, Int64, int, int64_t),

    ADD_KERNEL(Int64, Int64, Float16, int64_t, float16),   ADD_KERNEL(Int64, Int64, Float32, int64_t, float),
    ADD_KERNEL(Int64, Int64, Float64, int64_t, double),    ADD_KERNEL(Int64, Int64, Int32, int64_t, int),
    ADD_KERNEL(Int64, Int64, Int64, int64_t, int64_t)};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RandomPoisson, RandomPoissonCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
