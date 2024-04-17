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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_GRID_SAMPLER_3D_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_GRID_SAMPLER_3D_GRAD_CPU_KERNEL_H_
#include <map>
#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/core/ops/ops_func_impl/grid_sampler_3d_grad.h"

namespace mindspore {
namespace kernel {
class GridSampler3DGradCpuKernelMod : public NativeCpuKernelMod {
 public:
  GridSampler3DGradCpuKernelMod() = default;
  ~GridSampler3DGradCpuKernelMod() override = default;

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
                                                           .AddInputAttr(kNumberTypeFloat32)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeFloat32)
                                                           .AddOutputAttr(kNumberTypeFloat32),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeFloat64)
                                                           .AddInputAttr(kNumberTypeFloat64)
                                                           .AddInputAttr(kNumberTypeFloat64)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeFloat64)
                                                           .AddOutputAttr(kNumberTypeFloat64)};
    return support_list;
  }

 private:
  std::vector<int64_t> grad_shape_;
  std::vector<int64_t> x_shape_;
  std::vector<int64_t> grid_shape_;
  std::vector<int64_t> dx_shape_;
  std::vector<int64_t> dgrid_shape_;
  std::vector<size_t> grad_stride_;
  std::vector<size_t> x_stride_;
  std::vector<size_t> grid_stride_;
  std::vector<size_t> dx_stride_;
  std::vector<size_t> dgrid_stride_;
  int64_t interpolation_mode;
  int64_t padding_mode;
  bool align_corners_;
  size_t dx_size_;
  size_t grid_size_;
  TypeId dtype_{kTypeUnknown};
  template <typename T>
  void BilinearKernel(std::vector<T *> addr, std::vector<T> location, std::vector<T> mult,
                      std::vector<size_t> ptr) const;

  template <typename T>
  void ComputeTask(T *grad_addr, T *x_addr, T *grid_addr, T *dx_addr, T *dgrid_addr, const size_t &n) const;

  template <typename T>
  T grid_sampler_compute_source_index_set_grad(T coord, int64_t size, int64_t padding_mode, bool align_corners,
                                               T *grad_x) const;

  template <typename T>
  T reflect_coordinates_set_grad(T x, int64_t twice_low, int64_t twice_high, T *grad_x) const;

  template <typename T>
  T clip_coordinates_set_grad(T x, int64_t clip_limit, T *grad_x) const;

  template <typename T>
  void safe_add_3d(T *data, int64_t d, int64_t h, int64_t w, size_t sD, size_t sH, size_t sW, int64_t D, int64_t H,
                   int64_t W, T delta) const;

  bool within_bounds_3d(int64_t d, int64_t h, int64_t w, int64_t D, int64_t H, int64_t W) const;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_GRID_SAMPLER_3D_GRAD_CPU_KERNEL_H_
