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

#include "plugin/device/cpu/kernel/maximum_grad_grad_cpu_kernel.h"
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaximumGradGradInputsNum = 6;
constexpr size_t kMaximumGradGradOutputsNum = 3;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kInputIndex3 = 3;
constexpr size_t kOutputIndex0 = 0;
constexpr size_t kOutputIndex1 = 1;
constexpr size_t kOutputIndex2 = 2;
}  // namespace

bool MaximumGradGradCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', kernel data type '" << kernel_attr << "' is not supported.";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int MaximumGradGradCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  x1_shape_ = inputs[kIndex0]->GetDeviceShapeVector();
  x2_shape_ = inputs[kIndex1]->GetDeviceShapeVector();
  grad_y1_shape_ = inputs[kIndex2]->GetDeviceShapeVector();
  grad_y2_shape_ = inputs[kIndex3]->GetDeviceShapeVector();

  tensor_size_ = 1;
  output_shape_ = CPUKernelUtils::GetBroadcastShape(x1_shape_, x2_shape_);
  for (const int64_t &d : output_shape_) {
    tensor_size_ *= static_cast<uint64_t>(d);
  }

  return KRET_OK;
}

template <typename T>
bool MaximumGradGradCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                               const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaximumGradGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaximumGradGradOutputsNum, kernel_name_);

  auto x1_addr = static_cast<T *>(inputs[kInputIndex0]->device_ptr());
  auto x2_addr = static_cast<T *>(inputs[kInputIndex1]->device_ptr());
  auto grad_y1_addr = static_cast<T *>(inputs[kInputIndex2]->device_ptr());
  auto grad_y2_addr = static_cast<T *>(inputs[kInputIndex3]->device_ptr());
  auto sopd_x1_addr = static_cast<T *>(outputs[kOutputIndex0]->device_ptr());
  auto sopd_x2_addr = static_cast<T *>(outputs[kOutputIndex1]->device_ptr());
  auto sopd_grads_addr = static_cast<T *>(outputs[kOutputIndex2]->device_ptr());

  auto tensor_bytes = tensor_size_ * sizeof(T);
  auto ret_sopd_x1 = memset_s(sopd_x1_addr, tensor_bytes, 0, tensor_bytes);
  if (ret_sopd_x1 != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output[0] failed. Error no: " << ret_sopd_x1;
  }
  auto ret_sopd_x2 = memset_s(sopd_x2_addr, tensor_bytes, 0, tensor_bytes);
  if (ret_sopd_x2 != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output[1] failed. Error no: " << ret_sopd_x2;
  }
  auto ret_sopd_grads = memset_s(sopd_grads_addr, tensor_bytes, 0, tensor_bytes);
  if (ret_sopd_grads != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output[2] failed. Error no: " << ret_sopd_grads;
  }
  if (x1_shape_ == x2_shape_) {
    auto task = [this, &x1_addr, &x2_addr, &grad_y1_addr, &grad_y2_addr, &sopd_grads_addr](size_t start, size_t end) {
      for (uint64_t i = start; i < end; ++i) {
        if (x1_addr[i] >= x2_addr[i]) {
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
        if (x1_addr[iter.GetInputPosA()] >= x2_addr[iter.GetInputPosB()]) {
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

std::vector<std::pair<KernelAttr, MaximumGradGradCpuKernelMod::MaximumGradGradCPUKernelFunc>>
  MaximumGradGradCpuKernelMod::func_list_ = {{KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                .AddOutputAttr(kNumberTypeFloat32)
                                                .AddOutputAttr(kNumberTypeFloat32)
                                                .AddOutputAttr(kNumberTypeFloat32),
                                              &MaximumGradGradCpuKernelMod::LaunchKernel<float>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                .AddOutputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt32),
                                              &MaximumGradGradCpuKernelMod::LaunchKernel<int>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt32)
                                                .AddInputAttr(kNumberTypeUInt32)
                                                .AddInputAttr(kNumberTypeUInt32)
                                                .AddInputAttr(kNumberTypeUInt32)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                .AddOutputAttr(kNumberTypeUInt32)
                                                .AddOutputAttr(kNumberTypeUInt32)
                                                .AddOutputAttr(kNumberTypeUInt32),
                                              &MaximumGradGradCpuKernelMod::LaunchKernel<uint32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                .AddOutputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeInt64),
                                              &MaximumGradGradCpuKernelMod::LaunchKernel<int64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt64)
                                                .AddInputAttr(kNumberTypeUInt64)
                                                .AddInputAttr(kNumberTypeUInt64)
                                                .AddInputAttr(kNumberTypeUInt64)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                .AddOutputAttr(kNumberTypeUInt64)
                                                .AddOutputAttr(kNumberTypeUInt64)
                                                .AddOutputAttr(kNumberTypeUInt64),
                                              &MaximumGradGradCpuKernelMod::LaunchKernel<uint64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                .AddOutputAttr(kNumberTypeFloat16)
                                                .AddOutputAttr(kNumberTypeFloat16)
                                                .AddOutputAttr(kNumberTypeFloat16),
                                              &MaximumGradGradCpuKernelMod::LaunchKernel<float16>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                .AddOutputAttr(kNumberTypeFloat64)
                                                .AddOutputAttr(kNumberTypeFloat64)
                                                .AddOutputAttr(kNumberTypeFloat64),
                                              &MaximumGradGradCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> MaximumGradGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaximumGradGradCPUKernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaximumGradGrad, MaximumGradGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
