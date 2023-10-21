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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_COPY_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_COPY_CPU_KERNEL_H_

#include <unordered_map>
#include <vector>
#include <functional>
#include "ir/tensor_storage_info.h"
#include "kernel/kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
class CopyWithSliceCpuKernel : public NativeCpuKernelMod {
 public:
  CopyWithSliceCpuKernel() = default;
  ~CopyWithSliceCpuKernel() = default;

  bool LaunchCopyWithSlice(TypeId type_id, const TensorStorageInfoPtr &src_storage_info,
                           const kernel::AddressPtr &src_addr, const TensorStorageInfoPtr &dst_storage_info,
                           const kernel::AddressPtr &dst_addr);
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    MS_LOG(EXCEPTION) << "This api is not external";
  }

 private:
  using CopyWithSliceFunc =
    std::function<bool(CopyWithSliceCpuKernel *, const TensorStorageInfoPtr &, const kernel::AddressPtr &,
                       const TensorStorageInfoPtr &, const kernel::AddressPtr &)>;

  template <typename T>
  bool LaunchCopyWithSliceImpl(const TensorStorageInfoPtr &src_storage_info, const kernel::AddressPtr &src_addr,
                               const TensorStorageInfoPtr &dst_storage_info, const kernel::AddressPtr &dst_addr);
  static std::unordered_map<TypeId, CopyWithSliceFunc> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_COPY_CPU_KERNEL_H_
