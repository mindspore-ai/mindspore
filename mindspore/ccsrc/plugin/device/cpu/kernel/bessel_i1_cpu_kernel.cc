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
#include <functional>
#include <map>
#include <cmath>
#include "plugin/device/cpu/kernel/bessel_i1_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/bessel_i1.h"
#include "mindspore/core/ops/bessel_i1e.h"
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBesselI1InputsNum = 1;
constexpr size_t kBesselI1OutputsNum = 1;
constexpr size_t kBesselI1eInputsNum = 1;
constexpr size_t kBesselI1eOutputsNum = 1;

static double A[] = {
  2.77791411276104639959E-18, -2.11142121435816608115E-17, 1.55363195773620046921E-16, -1.10559694773538630805E-15,
  7.60068429473540693410E-15, -5.04218550472791168711E-14, 3.22379336594557470981E-13, -1.98397439776494371520E-12,
  1.17361862988909016308E-11, -6.66348972350202774223E-11, 3.62559028155211703701E-10, -1.88724975172282928790E-9,
  9.38153738649577178388E-9,  -4.44505912879632808065E-8,  2.00329475355213526229E-7,  -8.56872026469545474066E-7,
  3.47025130813767847674E-6,  -1.32731636560394358279E-5,  4.78156510755005422638E-5,  -1.61760815825896745588E-4,
  5.12285956168575772895E-4,  -1.51357245063125314899E-3,  4.15642294431288815669E-3,  -1.05640848946261981558E-2,
  2.47264490306265168283E-2,  -5.29459812080949914269E-2,  1.02643658689847095384E-1,  -1.76416518357834055153E-1,
  2.52587186443633654823E-1};

static double B[] = {
  7.51729631084210481353E-18,  4.41434832307170791151E-18,  -4.65030536848935832153E-17, -3.20952592199342395980E-17,
  2.96262899764595013876E-16,  3.30820231092092828324E-16,  -1.88035477551078244854E-15, -3.81440307243700780478E-15,
  1.04202769841288027642E-14,  4.27244001671195135429E-14,  -2.10154184277266431302E-14, -4.08355111109219731823E-13,
  -7.19855177624590851209E-13, 2.03562854414708950722E-12,  1.41258074366137813316E-11,  3.25260358301548823856E-11,
  -1.89749581235054123450E-11, -5.58974346219658380687E-10, -3.83538038596423702205E-9,  -2.63146884688951950684E-8,
  -2.51223623787020892529E-7,  -3.88256480887769039346E-6,  -1.10588938762623716291E-4,  -9.76109749136146840777E-3,
  7.78576235018280120474E-1};
}  // namespace

double BesselI1CpuKernelMod::chbevl(double x, const double array[], int n) {
  const double C1 = 0.5;
  double b0, b1, b2;
  const double *p;
  size_t i;

  p = array;
  b0 = *p++;
  b1 = 0.0;
  i = n - 1;

  do {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + *p++;
  } while (--i);

  return (C1 * (b0 - b2));
}

double BesselI1CpuKernelMod::bessel_i1_func(double x) {
  const double ZERO = 0.0;
  const double BAR = 8.0;
  const double C1 = 2.0;
  const double C2 = 32.0;
  const int DEG1 = 29;
  const int DEG2 = 25;
  double z;

  z = fabs(x);
  if (z <= BAR) {
    double y = (z / C1) - C1;
    z = BesselI1CpuKernelMod::chbevl(y, A, DEG1) * z * exp(z);
  } else {
    z = exp(z) * BesselI1CpuKernelMod::chbevl(C2 / z - C1, B, DEG2) / sqrt(z);
  }
  if (x < ZERO) {
    z = -z;
  }
  return (z);
}

template <typename T>
void BesselI1CpuKernelMod::BesselI1Func(const T *input, T *output, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    output[i] = static_cast<T>(bessel_i1_func(static_cast<double>(input[i])));
  }
}

bool BesselI1CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BesselI1>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast BesselI1 ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kBesselI1InputsNum || outputs.size() != kBesselI1OutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "': input and output size should be " << kBesselI1InputsNum << " and "
                  << kBesselI1OutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  input_shape_ = inputs[0]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();
  input_dtype_ = inputs[0]->GetDtype();
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());

  switch (input_dtype_) {
    case kNumberTypeFloat64:
      kernel_func_ = &BesselI1CpuKernelMod::LaunchKernel<double>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &BesselI1CpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &BesselI1CpuKernelMod::LaunchKernel<float16>;
      break;
    default:
      MS_LOG(ERROR) << "BesselI1 kernel does not support " << TypeIdToString(input_dtype_);
      return false;
  }
  return true;
}

int BesselI1CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
bool BesselI1CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  auto end = inputs[0]->size / sizeof(T);
  auto task = std::bind(BesselI1Func<T>, input, output, 0, end);
  ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_);

  return true;
}

std::vector<KernelAttr> BesselI1CpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BesselI1, BesselI1CpuKernelMod);

double BesselI1eCpuKernelMod::bessel_i1e_func(double x) {
  const double ZERO = 0.0;
  const double BAR = 8.0;
  const double C1 = 2.0;
  const double C2 = 32.0;
  const int DEG1 = 29;
  const int DEG2 = 25;
  double z;

  z = fabs(x);
  if (z <= BAR) {
    double y = (z / C1) - C1;
    z = BesselI1CpuKernelMod::chbevl(y, A, DEG1) * z;
  } else {
    z = BesselI1CpuKernelMod::chbevl(C2 / z - C1, B, DEG2) / sqrt(z);
  }
  if (x < ZERO) {
    z = -z;
  }
  return (z);
}

template <typename T>
void BesselI1eCpuKernelMod::BesselI1eFunc(const T *input, T *output, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    output[i] = static_cast<T>(bessel_i1e_func(static_cast<double>(input[i])));
  }
}

bool BesselI1eCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BesselI1e>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast BesselI1e ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kBesselI1eInputsNum || outputs.size() != kBesselI1eOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "': input and output size should be " << kBesselI1eInputsNum << " and "
                  << kBesselI1eOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  input_shape_ = inputs[0]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();
  input_dtype_ = inputs[0]->GetDtype();
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());

  switch (input_dtype_) {
    case kNumberTypeFloat64:
      kernel_func_ = &BesselI1eCpuKernelMod::LaunchKernel<double>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &BesselI1eCpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &BesselI1eCpuKernelMod::LaunchKernel<float16>;
      break;
    default:
      MS_LOG(ERROR) << "BesselI1e kernel does not support " << TypeIdToString(input_dtype_);
      return false;
  }
  return true;
}

int BesselI1eCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
bool BesselI1eCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  auto end = inputs[0]->size / sizeof(T);
  auto task = std::bind(BesselI1eFunc<T>, input, output, 0, end);
  ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_);

  return true;
}

std::vector<KernelAttr> BesselI1eCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BesselI1e, BesselI1eCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
