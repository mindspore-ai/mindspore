/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_GRID_SAMPLER_3D_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_GRID_SAMPLER_3D_CPU_KERNEL_H_
#include <map>
#include <algorithm>
#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/core/ops/ops_func_impl/grid_sampler_3d.h"

namespace mindspore {
namespace kernel {
class GridSampler3DCpuKernelMod : public NativeCpuKernelMod {
 public:
  GridSampler3DCpuKernelMod() = default;
  ~GridSampler3DCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  template <typename T>
  void LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  std::vector<KernelAttr> GetOpSupport() override {
    static const std::vector<KernelAttr> support_list = {KernelAttr()
                                                           .AddInputAttr(kNumberTypeFloat32)
                                                           .AddInputAttr(kNumberTypeFloat32)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeFloat32),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeFloat64)
                                                           .AddInputAttr(kNumberTypeFloat64)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeFloat64)};
    return support_list;
  }

 private:
  std::vector<int64_t> x_shape_;
  std::vector<int64_t> grid_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<size_t> x_stride_;
  std::vector<size_t> grid_stride_;
  std::vector<size_t> output_stride_;
  int64_t interpolation_mode_;
  int64_t padding_mode_;
  bool align_corners_;
  size_t output_number_;
  TypeId dtype_{kTypeUnknown};
  template <typename T>
  void ComputeTask(T *x_data_addr, T *grid_data_addr, T *output_data_addr, const size_t &seq);

  template <typename T>
  T grid_sampler_compute_source_index(T coord, int64_t size, int64_t padding_mode, bool align_corners);

  template <typename T>
  T reflect_coordinates(T coord, int64_t twice_low, int64_t twice_high) const;

  bool within_bounds_3d(int64_t d, int64_t h, int64_t w, int64_t D, int64_t H, int64_t W) const;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_GRID_SAMPLER_3D_CPU_KERNEL_H_
