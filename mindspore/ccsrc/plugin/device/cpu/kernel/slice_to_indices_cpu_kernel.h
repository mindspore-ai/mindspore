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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SLICE_TO_INDICES_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SLICE_TO_INDICES_CPU_KERNEL_H_
#include <vector>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "mindspore/core/ops/tile_size.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SliceToIndicesCpuKernelMod : public NativeCpuKernelMod {
 public:
  SliceToIndicesCpuKernelMod() = default;
  ~SliceToIndicesCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  void SyncOutputShape() override {
    for (size_t i = 0; i < out_shapes_.size(); i++) {
      outputs_[i]->SetShapeVector(out_shapes_[i]);
    }
  }

  using SliceToIndicesFunc =
    std::function<bool(SliceToIndicesCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &)>;

  static std::vector<std::pair<KernelAttr, SliceToIndicesFunc>> func_list_;
  SliceToIndicesFunc kernel_func_;

 private:
  std::vector<ShapeVector> out_shapes_;
  ShapeVector data_shape_;
  size_t index_axis_ = 0;
  int64_t expand_dims_mask_ = 0;
  std::vector<int64_t> tuple_index_types_;
  std::vector<int64_t> init_by_none_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SLICE_TO_INDICES_CPU_KERNEL_H_
