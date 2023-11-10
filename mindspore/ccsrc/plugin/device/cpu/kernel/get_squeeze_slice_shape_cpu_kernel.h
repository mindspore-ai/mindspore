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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_GET_SQUEEZE_SLICE_SHAPE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_GET_SQUEEZE_SLICE_SHAPE_CPU_KERNEL_H_
#include <vector>
#include <map>
#include <utility>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class GetSqueezeSliceShapeCpuKernelMod : public NativeCpuKernelMod {
 public:
  GetSqueezeSliceShapeCpuKernelMod() = default;
  ~GetSqueezeSliceShapeCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  using GetSqueezeSliceShapeFunc =
    std::function<bool(GetSqueezeSliceShapeCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;

  static std::vector<std::pair<KernelAttr, GetSqueezeSliceShapeFunc>> func_list_;
  GetSqueezeSliceShapeFunc kernel_func_;

 private:
  std::vector<std::vector<int64_t>> data_shapes_;
  std::vector<int64_t> tuple_index_types_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_GET_SQUEEZE_SLICE_SHAPE_CPU_KERNEL_H_
