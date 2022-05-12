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
#include "plugin/device/cpu/kernel/bessel_y0_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/bessel_j0_cpu_kernel.h"
#include "mindspore/core/ops/bessel_y0.h"
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBesselY0InputsNum = 1;
constexpr size_t kBesselY0OutputsNum = 1;
}  // namespace

double BesselY0CpuKernelMod::polevl(double x, const double coef[], int N) {
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

double BesselY0CpuKernelMod::p1evl(double x, const double coef[], int N) {
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

double BesselY0CpuKernelMod::y0(double x) {
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

  const double YP[] = {
    1.55924367855235737965E4,  -1.46639295903971606143E7,  5.43526477051876500413E9,  -9.82136065717911466409E11,
    8.75906394395366999549E13, -3.46628303384729719441E15, 4.42733268572569800351E16, -1.84950800436986690637E16,
  };

  const double YQ[] = {
    1.04128353664259848412E3,  6.26107330137134956842E5,  2.68919633393814121987E8,  8.64002487103935000337E10,
    2.02979612750105546709E13, 3.17157752842975028269E15, 2.50596256172653059228E17,
  };

  const double PIO4 = .78539816339744830962;
  const double SQ2OPI = .79788456080286535588;
  const double NPY_2_PI = 0.6366197723675814;
  const double BAR = 5.0;
  const double ZERO = 0.0;
  const double FIVE = 5.0;
  const double FIVE_SQUARED = 25.0;
  const int DEG_P = 6;
  const int DEG_Q = 7;

  double z, p, w, q, xn;
  if (x <= BAR) {
    if (x == ZERO) {
      return -INFINITY;
    } else if (x < ZERO) {
      return NAN;
    }
    z = x * x;
    w = polevl(z, YP, DEG_Q) / p1evl(z, YQ, DEG_Q);
    w += NPY_2_PI * log(x) * BesselJ0CpuKernelMod::j0(x);
    return (w);
  }

  w = FIVE / x;
  z = FIVE_SQUARED / (x * x);
  p = polevl(z, PP, DEG_P) / polevl(z, PQ, DEG_P);
  q = polevl(z, QP, DEG_Q) / p1evl(z, QQ, DEG_Q);
  xn = x - PIO4;
  p = p * sin(xn) + w * q * cos(xn);
  return (p * SQ2OPI / sqrt(x));
}

template <typename T>
void BesselY0CpuKernelMod::BesselY0Func(const T *input, T *output, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    double input_ = static_cast<double>(input[i]);
    double output_ = y0(input_);
    output[i] = static_cast<T>(output_);
  }
}

bool BesselY0CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BesselY0>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "For 'BesselY0CpuKernelMod', BaseOperatorPtr can not dynamic cast to BesselY0 before initialize!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kBesselY0InputsNum || outputs.size() != kBesselY0OutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "': input and output size should be " << kBesselY0InputsNum << " and "
                  << kBesselY0OutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  input_shape_ = inputs[0]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();
  input_dtype_ = inputs[0]->GetDtype();
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());

  switch (input_dtype_) {
    case kNumberTypeFloat64:
      kernel_func_ = &BesselY0CpuKernelMod::LaunchKernel<double>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &BesselY0CpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &BesselY0CpuKernelMod::LaunchKernel<float16>;
      break;
    default:
      MS_LOG(ERROR) << "BesselY0 kernel does not support " << TypeIdToString(input_dtype_);
      return false;
  }
  return true;
}

int BesselY0CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
bool BesselY0CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  auto end = inputs[0]->size / sizeof(T);
  auto task = std::bind(BesselY0Func<T>, input, output, 0, end);
  ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_);

  return true;
}

std::vector<KernelAttr> BesselY0CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BesselY0, BesselY0CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
