/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/hsigmoid_grad_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "mindspore/core/ops/grad/hsigmoid_grad.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore::kernel {
namespace {
constexpr auto kHSigmoidGrad = "HSigmoidGrad";
constexpr const size_t kHSigmoidGradInputsNum = 2;
constexpr const size_t kHSigmoidGradOutputsNum = 1;
using KernelRunFunc = HSigmoidGradCpuKernelMod::KernelRunFunc;
}  // namespace

template <typename T>
bool HSigmoidGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kHSigmoidGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kHSigmoidGradOutputsNum, kernel_name_);
  T *dy = static_cast<T *>(inputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(dy, false);
  T *x = static_cast<T *>(inputs[kIndex1]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(x, false);
  T *out = static_cast<T *>(outputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(out, false);

  auto zero = static_cast<T>(0);
  auto three = static_cast<T>(3);
  auto six = static_cast<T>(6);

  const size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  auto task = [&](size_t start, size_t end) {
    for (uint64_t i = start; i < end; ++i) {
      if (x[i] + three <= zero || x[i] >= three) {
        out[i] = zero;
      } else {
        out[i] = dy[i] / six;
      }
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &HSigmoidGradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &HSigmoidGradCpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &HSigmoidGradCpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &HSigmoidGradCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &HSigmoidGradCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &HSigmoidGradCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &HSigmoidGradCpuKernelMod::LaunchKernel<double>}};
  return func_list;
}

bool HSigmoidGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::HSigmoidGrad>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kHSigmoidGradInputsNum || outputs.size() != kHSigmoidGradOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kHSigmoidGradInputsNum << " and "
                  << kHSigmoidGradOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int HSigmoidGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }
  std::vector<int64_t> input_shape = inputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> input_shape_2 = inputs[kIndex1]->GetShapeVector();
  std::vector<int64_t> output_shape = outputs[kIndex0]->GetShapeVector();
  auto in_shape_size_1 = input_shape.size();
  if (in_shape_size_1 > max_dims_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of input should be less than or equal to max_dims 7, but got "
                      << in_shape_size_1 << ".";
    return KRET_RESIZE_FAILED;
  }
  auto in_shape_size_2 = input_shape_2.size();
  auto output_shape_size = output_shape.size();
  if (in_shape_size_1 != output_shape_size || in_shape_size_1 != in_shape_size_2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input one shape size should be the same as input two shape size and"
                  << " output shape size, but got input one shape size " << in_shape_size_1 << " input two shape size "
                  << in_shape_size_2 << " output shape size" << output_shape_size;
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, HSigmoidGrad,
                                 []() { return std::make_shared<HSigmoidGradCpuKernelMod>(kHSigmoidGrad); });
}  // namespace mindspore::kernel
