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
#include <cmath>
#include <functional>
#include <map>
#include "plugin/device/cpu/kernel/bessel_y1_cpu_kernel.h"
#include "plugin/device/cpu/kernel/bessel_j1_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/bessel_y1.h"
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBesselY1InputsNum = 1;
constexpr size_t kBesselY1OutputsNum = 1;
}  // namespace

double BesselY1CpuKernelMod::polevl(double x, const double coef[], int N) {
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

double BesselY1CpuKernelMod::p1evl(double x, const double coef[], int N) {
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

double BesselY1CpuKernelMod::y1(double x) {
  const double PP[7] = {
    7.62125616208173112003E-4, 7.31397056940917570436E-2, 1.12719608129684925192E0, 5.11207951146807644818E0,
    8.42404590141772420927E0,  5.21451598682361504063E0,  1.00000000000000000254E0,
  };

  const double PQ[7] = {
    5.71323128072548699714E-4, 6.88455908754495404082E-2, 1.10514232634061696926E0,  5.07386386128601488557E0,
    8.39985554327604159757E0,  5.20982848682361821619E0,  9.99999999999999997461E-1,
  };

  const double QP[8] = {
    5.10862594750176621635E-2, 4.98213872951233449420E0, 7.58238284132545283818E1, 3.66779609360150777800E2,
    7.10856304998926107277E2,  5.97489612400613639965E2, 2.11688757100572135698E2, 2.52070205858023719784E1,
  };

  const double QQ[8] = {
    1.00000000000000000000E0, 7.42373277035675149943E1, 1.05644886038262816351E3, 4.98641058337653607651E3,
    9.56231892404756170795E3, 7.99704160447350683650E3, 2.82619278517639096600E3, 3.36093607810698293419E2,
  };

  const double YP[6] = {
    1.26320474790178026440E9,   -6.47355876379160291031E11, 1.14509511541823727583E14,
    -8.12770255501325109621E15, 2.02439475713594898196E17,  -7.78877196265950026825E17,
  };

  const double YQ[9] = {
    1.00000000000000000000E0,  5.94301592346128195359E2,  2.35564092943068577943E5,
    7.34811944459721705660E7,  1.87601316108706159478E10, 3.88231277496238566008E12,
    6.20557727146953693363E14, 6.87141087355300489866E16, 3.97270608116560655612E18,
  };

  const double NPY_2_PI = 0.6366197723675814;
  const double THPIO4 = 2.35619449019234492885;
  const double SQ2OPI = .79788456080286535588;
  const double NPY_INFINITY = 0X7f8000000UL;
  const double BAR = 5.0;
  const double ZERO = 0.0;
  const double ONE = 1.0;
  const int DEG_P = 6;
  const int DEG_Q = 7;
  const int DEG_5 = 5;
  const int DEG_8 = 8;

  double w, z, p, q, xn;

  if (x <= BAR) {
    if (x == ZERO) {
      return -NPY_INFINITY;
    } else if (x < ZERO) {
      return ZERO;
    }
    z = x * x;
    w = x * (polevl(z, YP, DEG_5) / polevl(z, YQ, DEG_8));
    w += NPY_2_PI * (BesselJ1CpuKernelMod::j1(x) * log(x) - ONE / x);
    return (w);
  }

  w = BAR / x;
  z = w * w;
  p = polevl(z, PP, DEG_P) / polevl(z, PQ, DEG_P);
  q = polevl(z, QP, DEG_Q) / p1evl(z, QQ, DEG_Q);
  xn = x - THPIO4;
  p = p * sin(xn) + w * q * cos(xn);
  return (p * SQ2OPI / sqrt(x));
}

template <typename T>
void BesselY1CpuKernelMod::BesselY1Func(const T *input, T *output, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    double input_ = static_cast<double>(input[i]);
    double output_ = y1(input_);
    output[i] = static_cast<T>(output_);
  }
}

bool BesselY1CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BesselY1>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "For 'BesselY1CpuKernelMod', BaseOperatorPtr can not dynamic cast to BesselY1 before initialize!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kBesselY1InputsNum || outputs.size() != kBesselY1OutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kBesselY1InputsNum << " and "
                  << kBesselY1OutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  input_shape_ = inputs[0]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();
  input_dtype_ = inputs[0]->GetDtype();
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());

  switch (input_dtype_) {
    case kNumberTypeFloat64:
      kernel_func_ = &BesselY1CpuKernelMod::LaunchKernel<double>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &BesselY1CpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &BesselY1CpuKernelMod::LaunchKernel<float16>;
      break;
    default:
      MS_LOG(ERROR) << "BesselY1 kernel does not support " << TypeIdToString(input_dtype_);
      return false;
  }
  return true;
}

bool BesselY1CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &others) {
  if (!NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return false;
  }
  return true;
}

template <typename T>
bool BesselY1CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  auto end = inputs[0]->size / sizeof(T);
  auto task = std::bind(BesselY1Func<T>, input, output, 0, end);
  ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_);

  return true;
}

std::vector<KernelAttr> BesselY1CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BesselY1, BesselY1CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
