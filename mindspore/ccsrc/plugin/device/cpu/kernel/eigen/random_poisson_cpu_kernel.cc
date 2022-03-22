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
#include <cmath>
#include <ctime>
#include <random>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
static unsigned int seed = time(nullptr);
EIGEN_DEVICE_FUNC uint64_t get_random_seed() {
  uint64_t rnd = rand_r(&seed);
  return rnd;
}

static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t PCG_XSH_RS_state(uint64_t seed) {
  seed = seed ? seed : get_random_seed();
  return seed * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
}

const std::map<TypeId, size_t> rate_type_size_map = {{kNumberTypeInt32, sizeof(int32_t)},
                                                     {kNumberTypeInt64, sizeof(int64_t)},
                                                     {kNumberTypeFloat16, sizeof(float16)},
                                                     {kNumberTypeFloat32, sizeof(float)},
                                                     {kNumberTypeFloat64, sizeof(double)}};
}  // namespace

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T RandomToTypePoisson(uint64_t *state, double rate) {
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
      double u = Eigen::internal::RandomToTypeUniform<double>(state);
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
      double u = Eigen::internal::RandomToTypeUniform<double>(state);
      u -= static_cast<double>(0.5);
      double v = Eigen::internal::RandomToTypeUniform<double>(state);
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
  PoissonRandomGenerator(double rate, uint64_t seed) {
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

bool AddrAlignedCheck(const void *addr, uint64_t alignment = 16) {
  if (alignment == 0) {
    return false;
  }
  return reinterpret_cast<uint64_t>(addr) % alignment == 0;
}

void RandomPoissonCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  cnode_ptr_ = kernel_node;
  rate_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  ouput_type_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
  seed_ = static_cast<size_t>(common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "seed"));
  seed2_ = static_cast<size_t>(common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "seed2"));
}

template <typename T>
void RandomPoissonCpuKernelMod::Generate(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  int64_t final_seed = 0;
  auto attr_seed = seed_;
  final_seed = static_cast<int64_t>(attr_seed);
  if (final_seed == 0) {
    auto attr_seed2 = seed2_;
    final_seed = static_cast<int64_t>(attr_seed2);
  }

  auto *rate_flat = reinterpret_cast<double *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(rate_flat);

  auto rate_type_len = rate_type_size_map.find(rate_type_);
  size_t size_of_rate_type = rate_type_len->second;
  size_t num_of_rate = inputs[1]->size / size_of_rate_type;
  size_t num_of_output = outputs[0]->size / sizeof(T);

  auto *output = reinterpret_cast<T *>(outputs[0]->addr);

  if (AddrAlignedCheck(outputs[0]->addr)) {
    Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> eigen_output(static_cast<T *>(output), num_of_output);
    PoissonRandomGenerator<T> m_generator(rate_flat[0], final_seed);
    for (size_t i = 0; i < num_of_rate; i++) {
      m_generator.setRate(rate_flat[i]);
      for (size_t j = i; j < num_of_output; j += num_of_rate) {
        eigen_output(j) = m_generator.gen();
      }
    }
  } else {
    Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Unaligned> eigen_output(static_cast<T *>(output), num_of_output);
    PoissonRandomGenerator<T> m_generator(rate_flat[0], final_seed);
    for (size_t i = 0; i < num_of_rate; i++) {
      m_generator.setRate(rate_flat[i]);
      for (size_t j = i; j < num_of_output; j += num_of_rate) {
        eigen_output(j) = m_generator.gen();
      }
    }
  }
}

bool RandomPoissonCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                       const std::vector<AddressPtr> &outputs) {
  if (ouput_type_ == kNumberTypeFloat16) {
    Generate<float16>(inputs, outputs);
  } else if (ouput_type_ == kNumberTypeFloat32) {
    Generate<float>(inputs, outputs);
  } else if (ouput_type_ == kNumberTypeFloat64) {
    Generate<double>(inputs, outputs);
  } else if (ouput_type_ == kNumberTypeInt32) {
    Generate<int32_t>(inputs, outputs);
  } else if (ouput_type_ == kNumberTypeInt64) {
    Generate<int64_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "RandomPoisson kernel data type [%s] not support." << TypeIdToType(rate_type_)->ToString();
  }
  return true;
}

std::vector<KernelAttr> RandomPoissonCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RandomPoisson, RandomPoissonCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
