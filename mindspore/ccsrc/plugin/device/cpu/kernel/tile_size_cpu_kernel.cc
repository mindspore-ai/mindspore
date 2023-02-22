/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/tile_size_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
bool TileSizeCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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

int TileSizeCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  return KRET_OK;
}

template <typename T>
bool TileSizeCpuKernelMod::LaunchKernel(const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::vector<AddressPtr> &workspace) {
  const auto shape_addr = reinterpret_cast<T *>(inputs[0]->GetData()->addr);
  const auto out_shape_addr = reinterpret_cast<T *>(inputs[1]->GetData()->addr);
  const auto ndim_addr = reinterpret_cast<T *>(inputs[2]->GetData()->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->GetData()->addr);
  auto output_size = outputs[0]->GetData()->size;

  std::vector<T> out(*ndim_addr, 1);
  auto shape_size = SizeOf(input_shapes_[0]);
  auto out_shape_size = SizeOf(input_shapes_[1]);
  auto it_num = std::min(shape_size, out_shape_size);

  for (size_t i = 0; i < it_num; i++) {
    if (shape_addr[i] != out_shape_addr[i]) {
      out[i] = out_shape_addr[i];
    }
  }
  auto cp_ret = memcpy_s(output_addr, output_size, out.data(), output_size);
  if (cp_ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
  }

  return true;
}

bool TileSizeCpuKernelMod::Launch(const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::vector<AddressPtr> &workspace) {
  return kernel_func_(this, inputs, outputs, workspace);
}

std::vector<std::pair<KernelAttr, TileSizeCpuKernelMod::TileSizeFunc>> TileSizeCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
   &TileSizeCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
   &TileSizeCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
   &TileSizeCpuKernelMod::LaunchKernel<int64_t>},
};

std::vector<KernelAttr> TileSizeCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, TileSizeFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TileSize, TileSizeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
