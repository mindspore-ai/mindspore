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

#include <algorithm>
#include <numeric>
#include <memory>
#include <iostream>
#include <vector>
#include <cmath>
#include <atomic>
#include "plugin/device/cpu/kernel/adaptive_max_pool_2d_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
#define F64 kNumberTypeFloat64
#define F32 kNumberTypeFloat32
#define F16 kNumberTypeFloat16
#define I32 kNumberTypeInt32
#define I64 kNumberTypeInt64

constexpr size_t kHWSize = 2;
}  // namespace

bool AdaptiveMaxPool2DGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int AdaptiveMaxPool2DGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_y_grad_shape_ = inputs.at(kIndex0)->GetShapeVector();
  input_x_shape_ = inputs.at(kIndex1)->GetShapeVector();
  input_argmax_shape_ = inputs.at(kIndex2)->GetShapeVector();
  ShapeVector output_shape = outputs.at(kIndex0)->GetShapeVector();

  outer_size_ = 1;
  inner_size_ = 1;
  output_stride_ = 1;
  output_size_ = 1;
  const size_t shape_size = input_argmax_shape_.size();
  for (size_t i = 0; i < shape_size; i++) {
    if (i < shape_size - kHWSize) {
      outer_size_ *= input_argmax_shape_[i];
    } else {
      inner_size_ *= input_argmax_shape_[i];
    }
  }

  for (size_t k = 0; k < shape_size; k++) {
    output_size_ *= output_shape[k];
    if (k >= shape_size - kHWSize) {
      output_stride_ *= output_shape[k];
    }
  }

  return KRET_OK;
}

template <typename T, typename S>
bool AdaptiveMaxPool2DGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                     const std::vector<kernel::AddressPtr> &outputs) {
  auto input_grad = GetDeviceAddress<T>(inputs, kIndex0);
  auto input_argmax = GetDeviceAddress<S>(inputs, kIndex2);
  auto output = GetDeviceAddress<T>(outputs, kIndex0);

  std::atomic_int memset_ret{EOK};
  auto output_int8 = GetDeviceAddress<int8_t>(outputs, kIndex0);
  auto init_task = [&](size_t start, size_t end) {
    size_t mem_size = end - start;
    while (mem_size > 0) {
      size_t real_mem_size = mem_size;
      if (real_mem_size > static_cast<size_t>(SECUREC_MEM_MAX_LEN)) {
        real_mem_size = static_cast<size_t>(SECUREC_MEM_MAX_LEN);
      }
      auto ret = memset_s(output_int8 + start, real_mem_size, 0, real_mem_size);
      if (ret != EOK) {
        memset_ret = ret;
        return;
      }
      mem_size -= real_mem_size;
      start += real_mem_size;
    }
  };
  ParallelLaunchAutoSearch(init_task, outputs[kIndex0]->size, this, &search_info_);
  if (memset_ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s failed, ret=" << memset_ret;
  }

  auto adaptive_max_pool_2d_grad = [&](int64_t start, int64_t end) {
    for (int64_t n = start; n < end; ++n) {
      for (int64_t i = 0; i < inner_size_; ++i) {
        int32_t maxp = input_argmax[i + n * inner_size_] + n * output_stride_;
        output[maxp] += static_cast<T>(input_grad[i + n * inner_size_]);
      }
    }
  };
  ParallelLaunchAutoSearch(adaptive_max_pool_2d_grad, LongToSize(outer_size_), this, &parallel_search_info_);

  return true;
}

std::vector<std::pair<KernelAttr, AdaptiveMaxPool2DGradCpuKernelMod::AdaptiveMaxPool2DGradLaunchFunc>>
  AdaptiveMaxPool2DGradCpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(I32).AddOutputAttr(F16),
     &AdaptiveMaxPool2DGradCpuKernelMod::LaunchKernel<float16, int32_t>},
    {KernelAttr().AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(I32).AddOutputAttr(F32),
     &AdaptiveMaxPool2DGradCpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr().AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(I32).AddOutputAttr(F64),
     &AdaptiveMaxPool2DGradCpuKernelMod::LaunchKernel<double, int32_t>},
    {KernelAttr().AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(I64).AddOutputAttr(F16),
     &AdaptiveMaxPool2DGradCpuKernelMod::LaunchKernel<float16, int64_t>},
    {KernelAttr().AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(I64).AddOutputAttr(F32),
     &AdaptiveMaxPool2DGradCpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr().AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(I64).AddOutputAttr(F64),
     &AdaptiveMaxPool2DGradCpuKernelMod::LaunchKernel<double, int64_t>}};

std::vector<KernelAttr> AdaptiveMaxPool2DGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, AdaptiveMaxPool2DGradCpuKernelMod::AdaptiveMaxPool2DGradLaunchFunc> &pair) {
      return pair.first;
    });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AdaptiveMaxPool2DGrad, AdaptiveMaxPool2DGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
