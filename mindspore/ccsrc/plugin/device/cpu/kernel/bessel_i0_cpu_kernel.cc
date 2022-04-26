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

#include "plugin/device/cpu/kernel/bessel_i0_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include <cmath>
#include <tuple>
#include <type_traits>
#include "utils/ms_utils.h"

namespace {
constexpr size_t kBesselI0InputsNum = 1;
constexpr size_t kBesselI0OutputsNum = 1;

constexpr size_t kBesselI0eInputsNum = 1;
constexpr size_t kBesselI0eOutputsNum = 1;

static double A[] = {
  -4.41534164647933937950E-18, 3.33079451882223809783E-17, -2.43127984654795469359E-16, 1.71539128555513303061E-15,
  -1.16853328779934516808E-14, 7.67618549860493561688E-14, -4.85644678311192946090E-13, 2.95505266312963983461E-12,
  -1.72682629144155570723E-11, 9.67580903537323691224E-11, -5.18979560163526290666E-10, 2.65982372468238665035E-9,
  -1.30002500998624804212E-8,  6.04699502254191894932E-8,  -2.67079385394061173391E-7,  1.11738753912010371815E-6,
  -4.41673835845875056359E-6,  1.64484480707288970893E-5,  -5.75419501008210370398E-5,  1.88502885095841655729E-4,
  -5.76375574538582365885E-4,  1.63947561694133579842E-3,  -4.32430999505057594430E-3,  1.05464603945949983183E-2,
  -2.37374148058994688156E-2,  4.93052842396707084878E-2,  -9.49010970480476444210E-2,  1.71620901522208775349E-1,
  -3.04682672343198398683E-1,  6.76795274409476084995E-1};

static double B[] = {
  -7.23318048787475395456E-18, -4.83050448594418207126E-18, 4.46562142029675999901E-17,  3.46122286769746109310E-17,
  -2.82762398051658348494E-16, -3.42548561967721913462E-16, 1.77256013305652638360E-15,  3.81168066935262242075E-15,
  -9.55484669882830764870E-15, -4.15056934728722208663E-14, 1.54008621752140982691E-14,  3.85277838274214270114E-13,
  7.18012445138366623367E-13,  -1.79417853150680611778E-12, -1.32158118404477131188E-11, -3.14991652796324136454E-11,
  1.18891471078464383424E-11,  4.94060238822496958910E-10,  3.39623202570838634515E-9,   2.26666899049817806459E-8,
  2.04891858946906374183E-7,   2.89137052083475648297E-6,   6.88975834691682398426E-5,   3.36911647825569408990E-3,
  8.04490411014108831608E-1};
}  // namespace

namespace mindspore {
namespace kernel {
double chbevl(double x, double array[], size_t n) {
  const double C1 = 0.5;
  double b0, b1, b2, *p;
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

double BesselI0CpuKernelMod::bessel_i0_func(double x) {
  const double ZERO = 0.0;
  const double BAR = 8.0;
  const double C1 = 2.0;
  const double C2 = 32.0;
  const int DEG1 = 30;
  const int DEG2 = 25;
  if (x < ZERO) {
    x = -x;
  }
  if (x <= BAR) {
    double y = (x / C1) - C1;
    return (exp(x) * chbevl(y, A, DEG1));
  }
  return (exp(x) * chbevl(C2 / x - C1, B, DEG2) / sqrt(x));
}

template <typename T>
void BesselI0CpuKernelMod::BesselI0Func(const T *input, T *output, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    output[i] = static_cast<T>(bessel_i0_func(static_cast<double>(input[i])));
  }
}

void BesselI0CpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kBesselI0InputsNum, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kBesselI0OutputsNum, kernel_name_);
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());
}

bool BesselI0CpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                  const std::vector<AddressPtr> &outputs) {
  bool ret = true;
  if (input_dtype_ == kNumberTypeFloat16) {
    ret = LaunchKernel<float16>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat32) {
    ret = LaunchKernel<float>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat64) {
    ret = LaunchKernel<double>(inputs, outputs);
  } else {
    ret = false;
    MS_EXCEPTION(TypeError) << "Unsupported input data type for operator [" << kernel_name_
                            << "]: " << TypeIdToType(input_dtype_)->ToString();
  }
  return ret;
}

template <typename T>
bool BesselI0CpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  const auto input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  auto end = inputs[0]->size / sizeof(T);
  auto task = std::bind(BesselI0Func<T>, input, output, 0, end);
  ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_);

  return true;
}

std::vector<KernelAttr> BesselI0CpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BesselI0, BesselI0CpuKernelMod);

double BesselI0eCpuKernelMod::bessel_i0e_func(double x) {
  const double ZERO = 0.0;
  const double BAR = 8.0;
  const double C1 = 2.0;
  const double C2 = 32.0;
  const int DEG1 = 30;
  const int DEG2 = 25;
  if (x < ZERO) {
    x = -x;
  }
  if (x <= BAR) {
    double y = (x / C1) - C1;
    return (chbevl(y, A, DEG1));
  }
  return (chbevl(C2 / x - C1, B, DEG2) / sqrt(x));
}

template <typename T>
void BesselI0eCpuKernelMod::BesselI0eFunc(const T *input, T *output, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    output[i] = static_cast<T>(bessel_i0e_func(static_cast<double>(input[i])));
  }
}

void BesselI0eCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kBesselI0eInputsNum, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kBesselI0eOutputsNum, kernel_name_);
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());
}

bool BesselI0eCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &outputs) {
  bool ret = true;
  if (input_dtype_ == kNumberTypeFloat16) {
    ret = LaunchKernel<float16>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat32) {
    ret = LaunchKernel<float>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat64) {
    ret = LaunchKernel<double>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "Unsupported input data type for operator [" << kernel_name_
                            << "]: " << TypeIdToType(input_dtype_)->ToString();
  }
  return ret;
}

template <typename T>
bool BesselI0eCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  const auto input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  auto end = inputs[0]->size / sizeof(T);
  auto task = std::bind(BesselI0eFunc<T>, input, output, 0, end);
  ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_);

  return true;
}

std::vector<KernelAttr> BesselI0eCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BesselI0e, BesselI0eCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
