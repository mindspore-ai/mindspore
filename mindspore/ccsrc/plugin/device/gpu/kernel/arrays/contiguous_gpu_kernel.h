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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_CONTIGUOUS_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_CONTIGUOUS_GPU_KERNEL_H_

#include <map>
#include <utility>
#include "ir/tensor_storage_info.h"
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
class ContiguousGpuKernel {
 public:
  ContiguousGpuKernel() = default;
  ~ContiguousGpuKernel() = default;

  bool LaunchContiguous(TypeId input_type_id, const kernel::AddressPtr &inputs,
                        const TensorStorageInfoPtr &input_storage_info, TypeId output_type_id,
                        const kernel::AddressPtr &outputs, void *stream_ptr);

 private:
  using ContiguousFunc =
    std::function<bool(ContiguousGpuKernel *, const kernel::AddressPtr &, const TensorStorageInfoPtr &,
                       const kernel::AddressPtr &, const int64_t &, void *)>;

  template <typename T>
  bool LaunchContiguousImpl(const kernel::AddressPtr &inputs, const TensorStorageInfoPtr &input_storage_info,
                            const kernel::AddressPtr &outputs, const int64_t &type_size, void *stream_ptr);
  static std::map<std::pair<TypeId, TypeId>, ContiguousFunc> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_CONTIGUOUS_GPU_KERNEL_H_
