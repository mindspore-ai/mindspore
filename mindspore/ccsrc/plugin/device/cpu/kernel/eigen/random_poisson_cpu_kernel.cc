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
#include "plugin/device/cpu/kernel/eigen/random_poisson_cpu_kernel.h"
#include <random>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
using KernelRunFunc = RandomPoissonCpuKernelMod::KernelRunFunc;
#define ADD_KERNEL(shape_dtype, rate_dtype, output_dtype, rate_type, output_type) \
  {                                                                               \
    KernelAttr()                                                                  \
      .AddInputAttr(kNumberType##shape_dtype)                                     \
      .AddInputAttr(kNumberType##rate_dtype)                                      \
      .AddOutputAttr(kNumberType##output_dtype),                                  \
      &RandomPoissonCpuKernelMod::LaunchKernel<rate_type, output_type>            \
  }

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

static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t PCG_XSH_RS_state(uint64_t seed) {
  seed = (seed == 0) ? get_random_seed() : seed;
  return seed * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
}
}  // namespace

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T RandomToTypePoisson(uint64_t *state, uint64_t m_stream, double rate) {
  using Eigen::numext::exp;
  using Eigen::numext::log;
  using Eigen::numext::pow;
  T result;

  // if rate < 10, use Knuth's algorithm
  if (rate < static_cast<double>(10.0)) {
    const double exp_neg_rate = Eigen::numext::exp(-rate);
    double prod = 1;
    double x = 0;

    // Keep trying until we surpass e^(-rate). This will take
    // expected time proportional to rate.
    while (true) {
      double u = Eigen::internal::RandomToTypeUniform<double>(state, m_stream);
      prod = prod * u;
      if (prod <= exp_neg_rate && x <= static_cast<double>(Eigen::NumTraits<T>::highest())) {
        result = static_cast<T>(x);
        return result;
      }
      x += 1;
    }
  }

  if (Eigen::numext::isinf(rate) && rate > static_cast<double>(0)) {
    T k = Eigen::NumTraits<T>::infinity();
    return k;
  } else {
    const double log_rate = log(rate);
    const double b = static_cast<double>(0.931) + static_cast<double>(2.53) * Eigen::numext::sqrt(rate);
    const double a = static_cast<double>(-0.059) + static_cast<double>(0.02483) * b;
    const double inv_alpha = static_cast<double>(1.1239) + static_cast<double>(1.1328) / (b - static_cast<double>(3.4));

    while (true) {
      double u = Eigen::internal::RandomToTypeUniform<double>(state, m_stream);
      u -= static_cast<double>(0.5);
      double v = Eigen::internal::RandomToTypeUniform<double>(state, m_stream);
      double u_shifted = static_cast<double>(0.5) - Eigen::numext::abs(u);
      double k =
        Eigen::numext::floor((static_cast<double>(2) * a / u_shifted + b) * u + rate + static_cast<double>(0.43));

      if (k > Eigen::NumTraits<double>::highest()) {
        continue;
      }

      // When alpha * f(G(U)) * G'(U) is close to 1, it is possible to
      // find a rectangle (-u_r, u_r) x (0, v_r) under the curve, such
      // that if v <= v_r and |u| <= u_r, then we can accept.
      // Here v_r = 0.9227 - 3.6224 / (b - 2) and u_r = 0.43.
      if (u_shifted >= static_cast<double>(0.07) &&
          v <= static_cast<double>(0.9277) - static_cast<double>(3.6224) / (b - static_cast<double>(2))) {
        result = static_cast<T>(k);
        return result;
      }

      if (k < 0 || (u_shifted < static_cast<double>(0.013) && v > u_shifted)) {
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
  PoissonRandomGenerator(double rate, uint64_t seed) : m_rate(rate), m_state(PCG_XSH_RS_state(seed)) {}
  void setRate(double rate) { m_rate = rate; }
  T gen() const {
    T result = RandomToTypePoisson<T>(&m_state, m_stream, m_rate);
    return result;
  }

 private:
  double m_rate;
  mutable uint64_t m_state;
  // Adapt Eigen 3.4.0 new api for stream random number generator.
  // Here's no parallel computing, so we set stream to any fixed value.
  static constexpr uint64_t m_stream = 0;
};

bool AddrAlignedCheck(const void *addr, uint64_t alignment = 16) {
  if (alignment == 0) {
    return false;
  }
  return reinterpret_cast<uint64_t>(addr) % alignment == 0;
}

bool RandomPoissonCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto prim = base_operator->GetPrim();
  kernel_name_ = base_operator->name();
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  MS_EXCEPTION_IF_NULL(prim);
  seed_ = GetValue<int64_t>(prim->GetAttr("seed"));
  seed2_ = GetValue<int64_t>(prim->GetAttr("seed2"));
  return true;
}

template <typename Tin, typename T>
bool RandomPoissonCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  auto attr_seed = LongToUlong(seed_);
  uint64_t final_seed = attr_seed;
  if (final_seed == 0) {
    auto attr_seed2 = seed2_;
    final_seed = LongToUlong(attr_seed2);
  }

  auto *rate_flat = reinterpret_cast<Tin *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(rate_flat);

  size_t num_of_rate = inputs[1]->size / sizeof(Tin);
  size_t num_of_output = outputs[0]->size / sizeof(T);

  auto *output = reinterpret_cast<T *>(outputs[0]->addr);

  if (AddrAlignedCheck(outputs[0]->addr)) {
    Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> eigen_output(static_cast<T *>(output), num_of_output);
    PoissonRandomGenerator<T> m_generator(static_cast<double>(rate_flat[0]), final_seed);
    for (size_t i = 0; i < num_of_rate; i++) {
      m_generator.setRate(static_cast<double>(rate_flat[i]));
      for (size_t j = i; j < num_of_output; j += num_of_rate) {
        eigen_output(j) = m_generator.gen();
      }
    }
  } else {
    Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Unaligned> eigen_output(static_cast<T *>(output), num_of_output);
    PoissonRandomGenerator<T> m_generator(static_cast<double>(rate_flat[0]), final_seed);
    for (size_t i = 0; i < num_of_rate; i++) {
      m_generator.setRate(static_cast<double>(rate_flat[i]));
      for (size_t j = i; j < num_of_output; j += num_of_rate) {
        eigen_output(j) = m_generator.gen();
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
