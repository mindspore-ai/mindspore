/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <functional>
#include <map>
#include "plugin/device/cpu/kernel/bessel_k0_cpu_kernel.h"
#include "plugin/device/cpu/kernel/bessel_i0_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/bessel_k0.h"
#include "mindspore/core/ops/bessel_k0e.h"
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBesselK0InputsNum = 1;
constexpr size_t kBesselK0OutputsNum = 1;
constexpr size_t kBesselK0eInputsNum = 1;
constexpr size_t kBesselK0eOutputsNum = 1;

const double A[] = {1.37446543561352307156E-16, 4.25981614279661018399E-14, 1.03496952576338420167E-11,
                    1.90451637722020886025E-9,  2.53479107902614945675E-7,  2.28621210311945178607E-5,
                    1.26461541144692592338E-3,  3.59799365153615016266E-2,  3.44289899924628486886E-1,
                    -5.35327393233902768720E-1};

const double B[] = {
  5.30043377268626276149E-18, -1.64758043015242134646E-17, 5.21039150503902756861E-17, -1.67823109680541210385E-16,
  5.51205597852431940784E-16, -1.84859337734377901440E-15, 6.34007647740507060557E-15, -2.22751332699166985548E-14,
  8.03289077536357521100E-14, -2.98009692317273043925E-13, 1.14034058820847496303E-12, -4.51459788337394416547E-12,
  1.85594911495471785253E-11, -7.95748924447710747776E-11, 3.57739728140030116597E-10, -1.69753450938905987466E-9,
  8.57403401741422608519E-9,  -4.66048989768794782956E-8,  2.76681363944501510342E-7,  -1.83175552271911948767E-6,
  1.39498137188764993662E-5,  -1.28495495816278026384E-4,  1.56988388573005337491E-3,  -3.14481013119645005427E-2,
  2.44030308206595545468E0};
}  // namespace

double BesselK0CpuKernelMod::k0(double x) {
  double y, z;
  const double ZERO = 0.0;
  const double C1 = 8.0;
  const double C2 = 2.0;
  const double C3 = 5.0;
  const int DEG1 = 10;
  const int DEG2 = 25;

  if (x == ZERO) {
    return INFINITY;
  } else if (x < ZERO) {
    return NAN;
  }

  if (x <= C2) {
    y = x * x - C2;
    y = BesselI0CpuKernelMod::chbevl(y, A, DEG1) - log(C3 * x) * BesselI0CpuKernelMod::bessel_i0_func(x);
    return (y);
  }
  z = C1 / x - C2;
  y = exp(-x) * BesselI0CpuKernelMod::chbevl(z, B, DEG2) / sqrt(x);
  return y;
}

template <typename T>
void BesselK0CpuKernelMod::BesselK0Func(const T *input, T *output, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    double input_ = static_cast<double>(input[i]);
    double output_ = k0(input_);
    output[i] = static_cast<T>(output_);
  }
}

bool BesselK0CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BesselK0>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast BesselK0 ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kBesselK0InputsNum || outputs.size() != kBesselK0OutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "': input and output size should be " << kBesselK0InputsNum << " and "
                  << kBesselK0OutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  input_shape_ = inputs[0]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();
  input_dtype_ = inputs[0]->GetDtype();
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());

  switch (input_dtype_) {
    case kNumberTypeFloat64:
      kernel_func_ = &BesselK0CpuKernelMod::LaunchKernel<double>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &BesselK0CpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &BesselK0CpuKernelMod::LaunchKernel<float16>;
      break;
    default:
      MS_LOG(ERROR) << "BesselK0 kernel does not support " << TypeIdToString(input_dtype_);
      return false;
  }
  return true;
}

int BesselK0CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &others) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) != 0) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return ret;
  }
  return 0;
}

template <typename T>
bool BesselK0CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  auto end = inputs[0]->size / sizeof(T);
  auto task = std::bind(BesselK0Func<T>, input, output, 0, end);
  ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_);

  return true;
}

std::vector<KernelAttr> BesselK0CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BesselK0, BesselK0CpuKernelMod);

double BesselK0eCpuKernelMod::k0e(double x) {
  double y;
  const double ZERO = 0.0;
  const double C1 = 8.0;
  const double C2 = 2.0;
  const double C3 = 5.0;
  const int DEG1 = 10;
  const int DEG2 = 25;

  if (x == ZERO) {
    return INFINITY;
  } else if (x < ZERO) {
    return NAN;
  }

  if (x <= C2) {
    y = x * x - C2;
    y = BesselI0CpuKernelMod::chbevl(y, A, DEG1) - log(C3 * x) * BesselI0CpuKernelMod::bessel_i0_func(x);
    return (y * exp(x));
  }

  y = BesselI0CpuKernelMod::chbevl(C1 / x - C2, B, DEG2) / sqrt(x);
  return y;
}

template <typename T>
void BesselK0eCpuKernelMod::BesselK0eFunc(const T *input, T *output, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    double input_ = static_cast<double>(input[i]);
    double output_ = k0e(input_);
    output[i] = static_cast<T>(output_);
  }
}

bool BesselK0eCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BesselK0e>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast BesselK0 ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kBesselK0eInputsNum || outputs.size() != kBesselK0eOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "': input and output size should be " << kBesselK0eInputsNum << " and "
                  << kBesselK0eOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  input_shape_ = inputs[0]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();
  input_dtype_ = inputs[0]->GetDtype();
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());

  switch (input_dtype_) {
    case kNumberTypeFloat64:
      kernel_func_ = &BesselK0eCpuKernelMod::LaunchKernel<double>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &BesselK0eCpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &BesselK0eCpuKernelMod::LaunchKernel<float16>;
      break;
    default:
      MS_LOG(ERROR) << "BesselK0e kernel does not support " << TypeIdToString(input_dtype_);
      return false;
  }
  return true;
}

int BesselK0eCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &others) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) != 0) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return ret;
  }
  return 0;
}

template <typename T>
bool BesselK0eCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  auto end = inputs[0]->size / sizeof(T);
  auto task = std::bind(BesselK0eFunc<T>, input, output, 0, end);
  ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_);

  return true;
}

std::vector<KernelAttr> BesselK0eCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BesselK0e, BesselK0eCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
