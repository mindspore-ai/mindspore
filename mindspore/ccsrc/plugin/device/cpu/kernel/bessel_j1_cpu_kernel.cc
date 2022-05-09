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
#include "plugin/device/cpu/kernel/bessel_j1_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/bessel_j1.h"
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBesselJ1InputsNum = 1;
constexpr size_t kBesselJ1OutputsNum = 1;
}  // namespace

double BesselJ1CpuKernelMod::polevl(double x, const double coef[], int N) {
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

double BesselJ1CpuKernelMod::p1evl(double x, const double coef[], int N) {
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

double BesselJ1CpuKernelMod::j1(double x) {
  const double RP[4] = {
    -8.99971225705559398224E8,
    4.52228297998194034323E11,
    -7.27494245221818276015E13,
    3.68295732863852883286E15,
  };

  const double RQ[8] = {
    6.20836478118054335476E2,  2.56987256757748830383E5,  8.35146791431949253037E7,  2.21511595479792499675E10,
    4.74914122079991414898E12, 7.84369607876235854894E14, 8.95222336184627338078E16, 5.32278620332680085395E18,
  };

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

  const double QQ[7] = {
    7.42373277035675149943E1, 1.05644886038262816351E3, 4.98641058337653607651E3, 9.56231892404756170795E3,
    7.99704160447350683650E3, 2.82619278517639096600E3, 3.36093607810698293419E2,
  };

  const double Z1 = 1.46819706421238932572E1;
  const double Z2 = 4.92184563216946036703E1;

  const double SQ2OPI = .79788456080286535588;
  const double THPIO4 = 2.35619449019234492885;
  const double ZERO = 0.0;
  const double FIVE = 5.0;
  const int DEG_P = 6;
  const int DEG_Q = 7;
  const int DEG_3 = 3;
  const int DEG_8 = 8;

  double w, z, p, q, xn;

  w = x;
  if (x < ZERO) {
    return -j1(-x);
  }

  if (w <= FIVE) {
    z = x * x;
    w = polevl(z, RP, DEG_3) / p1evl(z, RQ, DEG_8);
    w = w * x * (z - Z1) * (z - Z2);
    return (w);
  }

  w = FIVE / x;
  z = w * w;
  p = polevl(z, PP, DEG_P) / polevl(z, PQ, DEG_P);
  q = polevl(z, QP, DEG_Q) / p1evl(z, QQ, DEG_Q);
  xn = x - THPIO4;
  p = p * cos(xn) - w * q * sin(xn);
  return (p * SQ2OPI / sqrt(x));
}

template <typename T>
void BesselJ1CpuKernelMod::BesselJ1Func(const T *input, T *output, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    double input_ = static_cast<double>(input[i]);
    double output_ = j1(input_);
    output[i] = static_cast<T>(output_);
  }
}

bool BesselJ1CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BesselJ1>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast BesselJ1 ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kBesselJ1InputsNum || outputs.size() != kBesselJ1OutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kBesselJ1InputsNum << " and "
                  << kBesselJ1OutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  input_shape_ = inputs[0]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();
  input_dtype_ = inputs[0]->GetDtype();
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());

  switch (input_dtype_) {
    case kNumberTypeFloat64:
      kernel_func_ = &BesselJ1CpuKernelMod::LaunchKernel<double>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &BesselJ1CpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &BesselJ1CpuKernelMod::LaunchKernel<float16>;
      break;
    default:
      MS_LOG(ERROR) << "BesselJ1 kernel does not support " << TypeIdToString(input_dtype_);
      return false;
  }
  return true;
}

int BesselJ1CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != 0) {
    return ret;
  }
  return 0;
}

template <typename T>
bool BesselJ1CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  auto end = inputs[0]->size / sizeof(T);
  auto task = std::bind(BesselJ1Func<T>, input, output, 0, end);
  ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_);

  return true;
}

std::vector<KernelAttr> BesselJ1CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BesselJ1, BesselJ1CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
