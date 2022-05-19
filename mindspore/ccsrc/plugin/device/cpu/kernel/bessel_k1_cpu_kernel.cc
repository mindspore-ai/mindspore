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
#include "plugin/device/cpu/kernel/bessel_k1_cpu_kernel.h"
#include "plugin/device/cpu/kernel/bessel_i1_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/bessel_k1.h"
#include "mindspore/core/ops/bessel_k1e.h"
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBesselK1InputsNum = 1;
constexpr size_t kBesselK1OutputsNum = 1;
constexpr size_t kBesselK1eInputsNum = 1;
constexpr size_t kBesselK1eOutputsNum = 1;

const double A[] = {-7.02386347938628759343E-18, -2.42744985051936593393E-15, -6.66690169419932900609E-13,
                    -1.41148839263352776110E-10, -2.21338763073472585583E-8,  -2.43340614156596823496E-6,
                    -1.73028895751305206302E-4,  -6.97572385963986435018E-3,  -1.22611180822657148235E-1,
                    -3.53155960776544875667E-1,  1.52530022733894777053E0};

const double B[] = {
  -5.75674448366501715755E-18, 1.79405087314755922667E-17, -5.68946255844285935196E-17, 1.83809354436663880070E-16,
  -6.05704724837331885336E-16, 2.03870316562433424052E-15, -7.01983709041831346144E-15, 2.47715442448130437068E-14,
  -8.97670518232499435011E-14, 3.34841966607842919884E-13, -1.28917396095102890680E-12, 5.13963967348173025100E-12,
  -2.12996783842756842877E-11, 9.21831518760500529508E-11, -4.19035475934189648750E-10, 2.01504975519703286596E-9,
  -1.03457624656780970260E-8,  5.74108412545004946722E-8,  -3.50196060308781257119E-7,  2.40648494783721712015E-6,
  -1.93619797416608296024E-5,  1.95215518471351631108E-4,  -2.85781685962277938680E-3,  1.03923736576817238437E-1,
  2.72062619048444266945E0};
}  // namespace

double BesselK1CpuKernelMod::k1(double x) {
  double z;
  const double ZERO = 0.0;
  const double C1 = 8.0;
  const double C2 = 2.0;
  const double C3 = 0.5;
  const int DEG1 = 11;
  const int DEG2 = 25;

  if (x == ZERO) {
    return INFINITY;
  } else if (x < ZERO) {
    return NAN;
  }
  z = C3 * x;

  if (x <= C2) {
    double y = x * x - C2;
    y = log(z) * BesselI1CpuKernelMod::bessel_i1_func(x) + BesselI1CpuKernelMod::chbevl(y, A, DEG1) / x;
    return (y);
  }

  return (exp(-x) * BesselI1CpuKernelMod::chbevl(C1 / x - C2, B, DEG2) / sqrt(x));
}

template <typename T>
void BesselK1CpuKernelMod::BesselK1Func(const T *input, T *output, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    double input_ = static_cast<double>(input[i]);
    double output_ = k1(input_);
    output[i] = static_cast<T>(output_);
  }
}

bool BesselK1CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BesselK1>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast BesselK1 ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kBesselK1InputsNum || outputs.size() != kBesselK1OutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "': input and output size should be " << kBesselK1InputsNum << " and "
                  << kBesselK1OutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  input_shape_ = inputs[0]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();
  input_dtype_ = inputs[0]->GetDtype();
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());

  switch (input_dtype_) {
    case kNumberTypeFloat64:
      kernel_func_ = &BesselK1CpuKernelMod::LaunchKernel<double>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &BesselK1CpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &BesselK1CpuKernelMod::LaunchKernel<float16>;
      break;
    default:
      MS_LOG(ERROR) << "BesselK1 kernel does not support " << TypeIdToString(input_dtype_);
      return false;
  }
  return true;
}

int BesselK1CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
bool BesselK1CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  auto end = inputs[0]->size / sizeof(T);
  auto task = std::bind(BesselK1Func<T>, input, output, 0, end);
  ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_);

  return true;
}

std::vector<KernelAttr> BesselK1CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BesselK1, BesselK1CpuKernelMod);

double BesselK1eCpuKernelMod::k1e(double x) {
  const double ZERO = 0.0;
  const double C1 = 8.0;
  const double C2 = 2.0;
  const double C3 = 0.5;
  const int DEG1 = 11;
  const int DEG2 = 25;

  if (x == ZERO) {
    return INFINITY;
  } else if (x < ZERO) {
    return NAN;
  }

  if (x <= C2) {
    double y = x * x - C2;
    y = log(C3 * x) * BesselI1CpuKernelMod::bessel_i1_func(x) + BesselI1CpuKernelMod::chbevl(y, A, DEG1) / x;
    return (y * exp(x));
  }

  return (BesselI1CpuKernelMod::chbevl(C1 / x - C2, B, DEG2) / sqrt(x));
}

template <typename T>
void BesselK1eCpuKernelMod::BesselK1eFunc(const T *input, T *output, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    double input_ = static_cast<double>(input[i]);
    double output_ = k1e(input_);
    output[i] = static_cast<T>(output_);
  }
}

bool BesselK1eCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BesselK1e>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast BesselK1 ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kBesselK1eInputsNum || outputs.size() != kBesselK1eOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "': input and output size should be " << kBesselK1eInputsNum << " and "
                  << kBesselK1eOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  input_shape_ = inputs[0]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();
  input_dtype_ = inputs[0]->GetDtype();
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());

  switch (input_dtype_) {
    case kNumberTypeFloat64:
      kernel_func_ = &BesselK1eCpuKernelMod::LaunchKernel<double>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &BesselK1eCpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &BesselK1eCpuKernelMod::LaunchKernel<float16>;
      break;
    default:
      MS_LOG(ERROR) << "BesselK1e kernel does not support " << TypeIdToString(input_dtype_);
      return false;
  }
  return true;
}

int BesselK1eCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
bool BesselK1eCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  auto end = inputs[0]->size / sizeof(T);
  auto task = std::bind(BesselK1eFunc<T>, input, output, 0, end);
  ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_);

  return true;
}

std::vector<KernelAttr> BesselK1eCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BesselK1e, BesselK1eCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
