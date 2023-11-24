/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_TILE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_TILE_CPU_KERNEL_H_

#include <complex>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "nnacl/base/tile_base.h"
#include "nnacl/kernel/tile.h"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

class TileCpuKernelMod : public NativeCpuKernelMod {
 public:
  TileCpuKernelMod() = default;
  ~TileCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  void LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  void TileMultipleCompute(void);

  ShapeVector x_shape_;
  ShapeVector y_shape_;
  ShapeVector multiple_shape;
  size_t input_num_;
  std::vector<int> multiples_;
  ShapeVector multiple_shape_;
  TypeId dtype_{kTypeUnknown};
  TypeId multiple_dtype_{kTypeUnknown};

  using TypeKernel = std::function<void(TileCpuKernelMod *, const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs)>;
  std::unordered_map<TypeId, TypeKernel> launch_map_;
  TypeKernel launch_func_;
  TileStruct tile_struct_;
  bool one_dim_tile_{false};
  size_t input_size_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_TILE_CPU_KERNEL_H_
