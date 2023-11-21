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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_REMOVE_EXPANDED_DIMS_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_REMOVE_EXPANDED_DIMS_CPU_KERNEL_H_
#include <vector>
#include <map>
#include <utility>
#include <functional>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class RemoveExpandedDimsCpuKernelMod : public NativeCpuKernelMod {
 public:
  RemoveExpandedDimsCpuKernelMod() = default;
  ~RemoveExpandedDimsCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override {
    for (size_t i = 0; i < out_shapes_.size(); i++) {
      outputs[i]->SetShapeVector(out_shapes_[i]);
      outputs[i]->set_size(
        LongToSize(std::accumulate(out_shapes_[i].begin(), out_shapes_[i].end(),
                                   UnitSizeInBytes(outputs[i]->dtype_id()), std::multiplies<int64_t>())));
    }
  }
  using RemoveExpandedDimsFunc =
    std::function<bool(RemoveExpandedDimsCpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, RemoveExpandedDimsFunc>> func_list_;
  RemoveExpandedDimsFunc kernel_func_;

 private:
  std::vector<std::vector<int64_t>> out_shapes_;
  std::vector<std::vector<int64_t>> data_shapes_;
  std::vector<int64_t> tuple_index_types_;
  bool has_true_ = false;
  bool has_sequence_ = false;
  bool empty_indices_out = false;
  int64_t rem_ndim_ = 0;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_REMOVE_EXPANDED_DIMS_CPU_KERNEL_H_
