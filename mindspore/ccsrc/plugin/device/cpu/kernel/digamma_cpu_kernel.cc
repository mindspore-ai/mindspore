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

#include "plugin/device/cpu/kernel/digamma_cpu_kernel.h"

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <map>
#include <string>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputIndex = 0;
constexpr size_t kOutputIndex = 0;
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNum = 1;
}  // namespace

static inline double calc_digamma(double x);

bool DigammaCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  return true;
}

template <typename T>
inline T ScalarDigamma(T x) {
  return calc_digamma(x);
}

template <>
inline Eigen::half ScalarDigamma(Eigen::half x) {
  const Eigen::half val{static_cast<Eigen::half>(calc_digamma(static_cast<std::float_t>(x)))};
  return val;
}

template <typename T>
static inline T polevl(const T x, const T A[], size_t len) {
  T result = 0;
  for (size_t i = 0; i <= len; i++) {
    result = result * x + A[i];
  }
  return result;
}

static inline double calc_digamma(double x) {
  static constexpr double kPSI_10 = 2.25175258906672110764;
  static constexpr double kPI = 3.141592653589793238462;
  static constexpr int64_t kTEN = 10;
  static constexpr double kHALF = 0.5;
  static constexpr int64_t kSIX = 6;
  if (static_cast<double>(x) == 0) {
    return std::copysign(INFINITY, -x);
  }

  bool x_is_integer = static_cast<double>(x) == static_cast<double>(trunc(x));
  if (x < 0) {
    if (x_is_integer) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    double q, r = 0;
    r = std::modf(x, &q);
    return calc_digamma(1 - x) - kPI / tan(kPI * r);
  }

  double result = 0;
  while (x < kTEN) {
    result -= 1 / x;
    x += 1;
  }
  if (x == kTEN) {
    return result + kPSI_10;
  }

  static const double A[] = {
    8.33333333333333333333E-2, -2.10927960927960927961E-2, 7.57575757575757575758E-3, -4.16666666666666666667E-3,
    3.96825396825396825397E-3, -8.33333333333333333333E-3, 8.33333333333333333333E-2,
  };

  double y = 0;
  if (x < 1.0e17) {
    double z = 1.0 / (x * x);
    y = z * polevl(z, A, kSIX);
  }
  return result + log(x) - (kHALF / x) - y;
}

int DigammaCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &others) {
  if (NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others) == KRET_RESIZE_FAILED) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return KRET_RESIZE_FAILED;
  }
  input_shape_ = inputs[kInputIndex]->GetShapeVector();
  output_shape_ = outputs[kOutputIndex]->GetShapeVector();
  input_tensor_size_ = SizeToLong(SizeOf(input_shape_));
  dtype_ = inputs[kInputIndex]->GetDtype();
  return 0;
}

bool DigammaCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                 const std::vector<AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    return LaunchKernel<Eigen::half>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    return LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Data type is " << TypeIdLabel(dtype_) << " which is not supported.";
  }
}

template <typename T>
bool DigammaCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);

  auto input_x = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_y = reinterpret_cast<T *>(outputs[0]->addr);

  for (int64_t i = 0; i < input_tensor_size_; i++) {
    *(output_y + i) = ScalarDigamma<T>(*(input_x + i));
  }
  return true;
}

std::vector<KernelAttr> DigammaCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Digamma, DigammaCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
