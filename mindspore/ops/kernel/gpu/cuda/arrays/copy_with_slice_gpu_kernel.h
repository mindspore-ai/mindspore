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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_COPY_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_COPY_GPU_KERNEL_H_

#include <unordered_map>
#include "ir/tensor_storage_info.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "device_address/device_address.h"

namespace mindspore {
namespace kernel {
class CopyWithSliceGpuKernel {
 public:
  CopyWithSliceGpuKernel() = default;
  ~CopyWithSliceGpuKernel() = default;

  bool LaunchCopyWithSlice(TypeId type_id, const TensorStorageInfoPtr &src_storage_info,
                           const device::DeviceAddressPtr &src_addr, const TensorStorageInfoPtr &dst_storage_info,
                           const device::DeviceAddressPtr &dst_addr, void *stream_ptr);

 private:
  using CopyWithSliceFunc =
    std::function<bool(CopyWithSliceGpuKernel *, const TensorStorageInfoPtr &, const device::DeviceAddressPtr &,
                       const TensorStorageInfoPtr &, const device::DeviceAddressPtr &, void *)>;

  template <typename T>
  bool LaunchCopyWithSliceImpl(const TensorStorageInfoPtr &src_storage_info, const device::DeviceAddressPtr &src_addr,
                               const TensorStorageInfoPtr &dst_storage_info, const device::DeviceAddressPtr &dst_addr,
                               void *stream_ptr);
  static std::unordered_map<TypeId, CopyWithSliceFunc> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_COPY_GPU_KERNEL_H_
