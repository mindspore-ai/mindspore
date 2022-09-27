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

#include "plugin/device/cpu/kernel/erfinv_cpu_kernel.h"
#include "mindspore/core/ops/erfinv.h"
#include <cmath>
#include <limits>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kErfinvInputsNum = 1;
constexpr size_t kErfinvOutputsNum = 1;
constexpr double PI = 3.14159265358979323846264338327950288;
}  // namespace

bool ErfinvCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Erfinv>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast Erfinv ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int ErfinvCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
bool ErfinvCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &workspace,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kErfinvInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kErfinvOutputsNum, kernel_name_);
  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  size_t total = inputs[0]->size / sizeof(T);
  auto task = [&input, &output](size_t start, size_t end) {
    // coefficient of the rational approximation on range [-0,7, 0.7]
    const T a[4] = {T(0.886226899), T(-1.645349621), T(0.914624893), T(-0.140543331)};
    const T b[4] = {T(-2.118377725), T(1.442710462), T(-0.329097515), T(0.012229801)};
    // coefficient of the rational approximation on range (-1, -0,7) and (0.7, 1)
    const T c[4] = {T(-1.970840454), T(-1.624906493), T(3.429567803), T(1.641345311)};
    const T d[2] = {T(3.543889200), T(1.637067800)};

    for (size_t i = start; i < end; i++) {
      if (input[i] > 1.0 || input[i] < -1.0) {
        // special case when abs(y)>1, erfinv(y)=None
        output[i] = std::numeric_limits<T>::quiet_NaN();
      } else if (std::abs(input[i]) == 1) {
        // spaecail case when abs(y)=1: erfinv(y)=inf, y=1; erfinv(y)=-inf, y=-1.
        output[i] = std::copysign(std::numeric_limits<T>::infinity(), input[i]);
      } else {
        // Other case, using rational approximation on erfinv, firstly
        T w, nm, dm;
        if (input[i] <= static_cast<T>(0.7) || input[i] >= static_cast<T>(-0.7)) {
          // erfinv(y) = (a0*y + a1*y^3 + a2*y^5 + a3*y^7) / (1+ b0*y^2 + b1*y^4 + b2*y^6 + b3*y^8); -0.7<=y<=0.7
          w = input[i] * input[i];
          nm = (((a[3] * w + a[2]) * w + a[1]) * w + a[0]);
          dm = ((((b[3] * w + b[2]) * w + b[1]) * w + b[0]) * w + static_cast<T>(1.0));
          output[i] = input[i] * nm / dm;
        } else {
          // w = sqrt(-log(1-y)/2), erfinv(y) = (c0 + c1*w + c2*w^2 + c3*w^3) / (1 + d0*w+ d1*w^2); 0<=y<0.7
          // w = sqrt(-log(1+y)/2), erfinv(y) = (-c0 - c1*w - c2*w^2 - c3*w^3) / (1 + d0*w + d1*w^2); -0.7<y<0
          w = sqrt(-std::log((static_cast<T>(1.0) - std::abs(input[i])) / static_cast<T>(2.0)));
          nm = ((c[3] * w + c[2]) * w + c[1]) * w + c[0];
          dm = (d[1] * w + d[0]) * w + static_cast<T>(1.0);
          output[i] = std::copysign(nm, input[i]) / dm;
        }
        // Secondly, two steps of Newton-Raphson correction to obtain full accuracy, here x = erfinv(y), y=erf(x).
        // x = x-(erf(x)-y)/((2/sqrt(pi)*e^(-x^2))
        output[i] =
          output[i] - (std::erf(output[i]) - input[i]) /
                        ((static_cast<T>(2.0) / static_cast<T>(std::sqrt(PI))) * std::exp(-output[i] * output[i]));
        output[i] =
          output[i] - (std::erf(output[i]) - input[i]) /
                        ((static_cast<T>(2.0) / static_cast<T>(std::sqrt(PI))) * std::exp(-output[i] * output[i]));
      }
    }
  };
  ParallelLaunchAutoSearch(task, total, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, ErfinvCpuKernelMod::KernelRunFunc>> &ErfinvCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ErfinvCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &ErfinvCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &ErfinvCpuKernelMod::LaunchKernel<double>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Erfinv, ErfinvCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
