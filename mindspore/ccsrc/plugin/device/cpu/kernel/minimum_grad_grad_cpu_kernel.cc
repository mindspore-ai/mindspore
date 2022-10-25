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

#include "plugin/device/cpu/kernel/minimum_grad_grad_cpu_kernel.h"

#include <algorithm>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMinimumGradGradInputsNum = 4;
constexpr size_t kMinimumGradGradOutputsNum = 3;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kInputIndex3 = 3;
constexpr size_t kOutputIndex0 = 0;
constexpr size_t kOutputIndex1 = 1;
constexpr size_t kOutputIndex2 = 2;
}  // namespace

bool MinimumGradGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For MinimumGradGrad, data type: " << kernel_attr << " is not supported.";
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int MinimumGradGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  tensor_size_ = 1;

  x1_shape_ = inputs[kInputIndex0]->GetDeviceShapeAdaptively();
  x2_shape_ = inputs[kInputIndex1]->GetDeviceShapeAdaptively();
  grad_y1_shape_ = inputs[kInputIndex2]->GetDeviceShapeAdaptively();
  grad_y2_shape_ = inputs[kInputIndex3]->GetDeviceShapeAdaptively();

  output_shape_ = CPUKernelUtils::GetBroadcastShape(x1_shape_, x2_shape_);
  for (const uint64_t &d : output_shape_) {
    tensor_size_ *= d;
  }

  return KRET_OK;
}

template <typename T>
bool MinimumGradGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMinimumGradGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMinimumGradGradOutputsNum, kernel_name_);

  auto x1_addr = reinterpret_cast<T *>(inputs[kInputIndex0]->addr);
  auto x2_addr = reinterpret_cast<T *>(inputs[kInputIndex1]->addr);
  auto grad_y1_addr = reinterpret_cast<T *>(inputs[kInputIndex2]->addr);
  auto grad_y2_addr = reinterpret_cast<T *>(inputs[kInputIndex3]->addr);
  auto sopd_x1_addr = reinterpret_cast<T *>(outputs[kOutputIndex0]->addr);
  auto sopd_x2_addr = reinterpret_cast<T *>(outputs[kOutputIndex1]->addr);
  auto sopd_grads_addr = reinterpret_cast<T *>(outputs[kOutputIndex2]->addr);

  auto ret_sopd_x1 = memset_s(sopd_x1_addr, 1, 0, 1);
  if (ret_sopd_x1 != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output[0] failed. Error no: " << ret_sopd_x1;
  }
  auto ret_sopd_x2 = memset_s(sopd_x2_addr, 1, 0, 1);
  if (ret_sopd_x2 != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output[1] failed. Error no: " << ret_sopd_x2;
  }
  auto ret_sopd_grads = memset_s(sopd_grads_addr, tensor_size_, 0, tensor_size_);
  if (ret_sopd_grads != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output[2] failed. Error no: " << ret_sopd_grads;
  }

  if (x1_shape_ == x2_shape_) {
    auto task = [this, &x1_addr, &x2_addr, &grad_y1_addr, &grad_y2_addr, &sopd_grads_addr](size_t start, size_t end) {
      for (uint64_t i = start; i < end; ++i) {
        if (x1_addr[i] <= x2_addr[i]) {
          sopd_grads_addr[i] = grad_y1_addr[i];
        } else {
          sopd_grads_addr[i] = grad_y2_addr[i];
        }
      }
    };
    ParallelLaunchAutoSearch(task, tensor_size_, this, &parallel_search_info_);
  } else {
    BroadcastIterator base_iter(x1_shape_, x2_shape_, output_shape_);
    auto task = [&x1_addr, &x2_addr, &grad_y1_addr, &grad_y2_addr, &sopd_grads_addr, &base_iter](size_t start,
                                                                                                 size_t end) {
      auto iter = base_iter;
      iter.SetPos(start);
      for (uint64_t i = start; i < end; ++i) {
        if (x1_addr[iter.GetInputPosA()] <= x2_addr[iter.GetInputPosB()]) {
          sopd_grads_addr[i] = grad_y1_addr[iter.GetInputPosA()];
        } else {
          sopd_grads_addr[i] = grad_y2_addr[iter.GetInputPosB()];
        }
        iter.GenNextPos();
      }
    };
    output_size_ = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(output_shape_.size()); ++i) {
      output_size_ *= output_shape_[i];
    }
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  }
  return true;
}

std::vector<std::pair<KernelAttr, MinimumGradGradCpuKernelMod::MinimumGradGradCPUKernelFunc>>
  MinimumGradGradCpuKernelMod::func_list_ = {{KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddOutputAttr(kNumberTypeFloat32)
                                                .AddOutputAttr(kNumberTypeFloat32)
                                                .AddOutputAttr(kNumberTypeFloat32),
                                              &MinimumGradGradCpuKernelMod::LaunchKernel<float>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt32),
                                              &MinimumGradGradCpuKernelMod::LaunchKernel<int>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt32)
                                                .AddInputAttr(kNumberTypeUInt32)
                                                .AddInputAttr(kNumberTypeUInt32)
                                                .AddInputAttr(kNumberTypeUInt32)
                                                .AddOutputAttr(kNumberTypeUInt32)
                                                .AddOutputAttr(kNumberTypeUInt32)
                                                .AddOutputAttr(kNumberTypeUInt32),
                                              &MinimumGradGradCpuKernelMod::LaunchKernel<uint32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeInt64),
                                              &MinimumGradGradCpuKernelMod::LaunchKernel<int64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt64)
                                                .AddInputAttr(kNumberTypeUInt64)
                                                .AddInputAttr(kNumberTypeUInt64)
                                                .AddInputAttr(kNumberTypeUInt64)
                                                .AddOutputAttr(kNumberTypeUInt64)
                                                .AddOutputAttr(kNumberTypeUInt64)
                                                .AddOutputAttr(kNumberTypeUInt64),
                                              &MinimumGradGradCpuKernelMod::LaunchKernel<uint64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddOutputAttr(kNumberTypeFloat16)
                                                .AddOutputAttr(kNumberTypeFloat16)
                                                .AddOutputAttr(kNumberTypeFloat16),
                                              &MinimumGradGradCpuKernelMod::LaunchKernel<float16>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddOutputAttr(kNumberTypeFloat64)
                                                .AddOutputAttr(kNumberTypeFloat64)
                                                .AddOutputAttr(kNumberTypeFloat64),
                                              &MinimumGradGradCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> MinimumGradGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MinimumGradGradCPUKernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MinimumGradGrad, MinimumGradGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
