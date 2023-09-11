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

#include "plugin/device/cpu/kernel/polygamma_cpu_kernel.h"

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <map>
#include <string>

#include "utils/digamma_helper.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kOutputIndex = 0;
constexpr size_t kInputsNum = 2;
constexpr size_t kOutputsNum = 1;
}  // namespace

bool PolygammaCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  return true;
}

template <typename scalar_t>
static inline scalar_t zeta(scalar_t x, scalar_t q) {
  const scalar_t MACHEP = scalar_t{1.11022302462515654042E-16};
  constexpr scalar_t zero = scalar_t{0.0};
  constexpr scalar_t half = scalar_t{0.5};
  constexpr scalar_t one = scalar_t{1.0};
  constexpr scalar_t nine = scalar_t{9.0};
  constexpr int64_t NINE = 9;
  constexpr int64_t TWELVE = 12;
  static const scalar_t A[] = {12.0,
                               -720.0,
                               30240.0,
                               -1209600.0,
                               47900160.0,
                               -1.8924375803183791606e9,
                               7.47242496e10,
                               -2.950130727918164224e12,
                               1.1646782814350067249e14,
                               -4.5979787224074726105e15,
                               1.8152105401943546773e17,
                               -7.1661652561756670113e18};

  scalar_t a;
  scalar_t b;
  scalar_t k;
  scalar_t s;
  scalar_t t;
  scalar_t w;
  if (x == one) {
    return std::numeric_limits<scalar_t>::infinity();
  }

  if (x < one) {
    return std::numeric_limits<scalar_t>::quiet_NaN();
  }

  if (q <= zero) {
    if (q == ::floor(q)) {
      return std::numeric_limits<scalar_t>::infinity();
    }
    if (x != ::floor(x)) {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
  }

  int i = 0;
  s = ::pow(q, -x);
  a = q;
  b = zero;
  while ((i < NINE) || (a <= nine)) {
    i += 1;
    a += one;
    b = ::pow(a, -x);
    s += b;
    if ((-MACHEP * s < b) && (b < MACHEP * s)) {
      return static_cast<scalar_t>(s);
    }
  }

  w = a;
  s += b * w / (x - one);
  s -= half * b;
  a = one;
  k = zero;
  for (i = 0; i < TWELVE; i++) {
    a *= x + k;
    b /= w;
    t = a * b / A[i];
    s = s + t;
    if (s == 0) {
      t = ::fabs(t / s);
    } else {
      t = ::fabs(t / s);
    }
    if (t < MACHEP) {
      return static_cast<scalar_t>(s);
    }
    k += one;
    a *= x + k;
    b /= w;
    k += one;
  }
  return static_cast<scalar_t>(s);
}

template <typename T1, typename T2>
static inline T2 calc_polygamma(T1 a, T2 x) {
  if (a == static_cast<T1>(0)) {
    return CalcDigamma<T2>(x);
  }
  const auto one = T1{1};
  const auto two = T1{2};
  return ((a % two) ? one : -one) * ::exp(::lgamma(static_cast<T1>(a) + one)) * zeta<T2>(static_cast<T2>(a + 1), x);
}

template <typename T1, typename T2>
inline T2 ScalarPolygamma(T1 a, T2 x) {
  return calc_polygamma(a, x);
}

template <>
inline Eigen::half ScalarPolygamma(int32_t a, Eigen::half x) {
  const Eigen::half val{static_cast<Eigen::half>(calc_polygamma(a, static_cast<std::float_t>(x)))};
  return val;
}

template <>
inline Eigen::half ScalarPolygamma(int64_t a, Eigen::half x) {
  const Eigen::half val{static_cast<Eigen::half>(calc_polygamma(a, static_cast<std::float_t>(x)))};
  return val;
}

int PolygammaCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &others) {
  if (NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others) == KRET_RESIZE_FAILED) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return KRET_RESIZE_FAILED;
  }
  x_shape_ = inputs[kInputIndex1]->GetShapeVector();
  x_tensor_size_ = SizeOf(x_shape_);
  a_dtype_ = inputs[kInputIndex0]->GetDtype();
  x_dtype_ = inputs[kInputIndex1]->GetDtype();
  return 0;
}

bool PolygammaCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs) {
  if (a_dtype_ == kNumberTypeInt32) {
    if (x_dtype_ == kNumberTypeFloat16) {
      return LaunchKernel<int32_t, Eigen::half>(inputs, outputs);
    } else if (x_dtype_ == kNumberTypeFloat32) {
      return LaunchKernel<int32_t, float>(inputs, outputs);
    } else if (x_dtype_ == kNumberTypeFloat64) {
      return LaunchKernel<int32_t, double>(inputs, outputs);
    } else {
      MS_LOG(EXCEPTION) << "Data type of x is " << TypeIdLabel(x_dtype_) << " which is not supported.";
    }
  } else if (a_dtype_ == kNumberTypeInt64) {
    if (x_dtype_ == kNumberTypeFloat16) {
      return LaunchKernel<int64_t, Eigen::half>(inputs, outputs);
    } else if (x_dtype_ == kNumberTypeFloat32) {
      return LaunchKernel<int64_t, float>(inputs, outputs);
    } else if (x_dtype_ == kNumberTypeFloat64) {
      return LaunchKernel<int64_t, double>(inputs, outputs);
    } else {
      MS_LOG(EXCEPTION) << "Data type of x is " << TypeIdLabel(x_dtype_) << " which is not supported.";
    }
  } else {
    MS_LOG(EXCEPTION) << "Data type of a is " << TypeIdLabel(a_dtype_) << " which is not supported.";
  }
}

template <typename T1, typename T2>
bool PolygammaCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);

  auto input_a = reinterpret_cast<T1 *>(inputs[0]->addr);
  auto input_x = reinterpret_cast<T2 *>(inputs[1]->addr);
  auto output_y = reinterpret_cast<T2 *>(outputs[0]->addr);

  for (int64_t i = 0; i < x_tensor_size_; i++) {
    *(output_y + i) = ScalarPolygamma<T1, T2>(*input_a, *(input_x + i));
  }
  return true;
}

std::vector<KernelAttr> PolygammaCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Polygamma, PolygammaCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
