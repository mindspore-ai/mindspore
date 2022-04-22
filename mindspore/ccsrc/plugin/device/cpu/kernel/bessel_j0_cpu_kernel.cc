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
#include "plugin/device/cpu/kernel/bessel_j0_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/bessel_j0.h"
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBesselJ0InputsNum = 1;
constexpr size_t kBesselJ0OutputsNum = 1;
}  // namespace

double BesselJ0CpuKernelMod::polevl(double x, const double coef[], int N) {
  double ans;
  int i;
  const double *p;

  p = coef;
  ans = *p++;
  i = N;

  do {
    ans = ans * x + *p++;
  } while (--i);

  return (ans);
}

double BesselJ0CpuKernelMod::p1evl(double x, const double coef[], int N) {
  double ans;
  const double *p;
  int i;

  p = coef;
  ans = x + *p++;
  i = N - 1;

  do {
    ans = ans * x + *p++;
  } while (--i);

  return (ans);
}

double BesselJ0CpuKernelMod::j0(double x) {
  const double PP[] = {
    7.96936729297347051624E-4, 8.28352392107440799803E-2, 1.23953371646414299388E0,  5.44725003058768775090E0,
    8.74716500199817011941E0,  5.30324038235394892183E0,  9.99999999999999997821E-1,
  };

  const double PQ[] = {
    9.24408810558863637013E-4, 8.56288474354474431428E-2, 1.25352743901058953537E0, 5.47097740330417105182E0,
    8.76190883237069594232E0,  5.30605288235394617618E0,  1.00000000000000000218E0,
  };

  const double QP[] = {
    -1.13663838898469149931E-2, -1.28252718670509318512E0, -1.95539544257735972385E1, -9.32060152123768231369E1,
    -1.77681167980488050595E2,  -1.47077505154951170175E2, -5.14105326766599330220E1, -6.05014350600728481186E0,
  };

  const double QQ[] = {
    6.43178256118178023184E1, 8.56430025976980587198E2, 3.88240183605401609683E3, 7.24046774195652478189E3,
    5.93072701187316984827E3, 2.06209331660327847417E3, 2.42005740240291393179E2,
  };

  const double R1 = 5.78318596294678452118E0;

  const double R2 = 3.04712623436620863991E1;

  const double P3[] = {
    -4.79443220978201773821E9,
    1.95617491946556577543E12,
    -2.49248344360967716204E14,
    9.70862251047306323952E15,
  };

  const double Q8[] = {
    4.99563147152651017219E2,  1.73785401676374683123E5,  4.84409658339962045305E7,  1.11855537045356834862E10,
    2.11277520115489217587E12, 3.10518229857422583814E14, 3.18121955943204943306E16, 1.71086294081043136091E18,
  };

  const double PIO4 = .78539816339744830962;
  const double SQ2OPI = .79788456080286535588;
  const double EPS = 1.0e-5;
  const double BAR = 5.0;
  const double ZERO = 0.0;
  const double FOUR = 4.0;
  const double FIVE = 5.0;
  const double FIVE_SQUARED = 25.0;
  const int DEG_P = 6;
  const int DEG_Q = 7;
  const int DEG_3 = 3;
  const int DEG_8 = 8;

  if (x < ZERO) {
    x = -x;
  }

  if (x < EPS) {
    return (x * x) / FOUR;
  }

  double p, w, q, xn;
  if (x <= BAR) {
    double z = x * x;
    p = (z - R1) * (z - R2) * polevl(z, P3, DEG_3) / p1evl(z, Q8, DEG_8);
    return (p);
  }

  w = FIVE / x;
  q = FIVE_SQUARED / (x * x);
  p = polevl(q, PP, DEG_P) / p1evl(q, PQ, DEG_P);
  q = polevl(q, QP, DEG_Q) / p1evl(q, QQ, DEG_Q);
  xn = x - PIO4;
  p = p * cos(xn) - w * q * sin(xn);
  return (p * SQ2OPI / sqrt(x));
}

template <typename T>
void BesselJ0CpuKernelMod::BesselJ0Func(const T *input, T *output, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    double input_ = static_cast<double>(input[i]);
    double output_ = j0(input_);
    output[i] = static_cast<T>(output_);
  }
}

bool BesselJ0CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BesselJ0>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast BesselJ0 ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kBesselJ0InputsNum || outputs.size() != kBesselJ0OutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kBesselJ0InputsNum << " and "
                  << kBesselJ0OutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  input_shape_ = inputs[0]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();
  input_dtype_ = inputs[0]->GetDtype();
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());

  switch (input_dtype_) {
    case kNumberTypeFloat64:
      kernel_func_ = &BesselJ0CpuKernelMod::LaunchKernel<double>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &BesselJ0CpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &BesselJ0CpuKernelMod::LaunchKernel<float16>;
      break;
    default:
      MS_LOG(ERROR) << "BesselJ0 kernel does not support " << TypeIdToString(input_dtype_);
      return false;
  }
  return true;
}

bool BesselJ0CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &others) {
  if (!NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return false;
  }
  return true;
}

template <typename T>
bool BesselJ0CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  auto end = inputs[0]->size / sizeof(T);
  auto task = std::bind(BesselJ0Func<T>, input, output, 0, end);
  ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_);

  return true;
}

std::vector<KernelAttr> BesselJ0CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BesselJ0, BesselJ0CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
