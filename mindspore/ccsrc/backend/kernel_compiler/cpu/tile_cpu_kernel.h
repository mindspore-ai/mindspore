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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TILE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TILE_CPU_KERNEL_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "nnacl/base/tile_base.h"

namespace mindspore {
namespace kernel {
class TileCPUKernel : public CPUKernel {
 public:
  TileCPUKernel() = default;
  ~TileCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  void TileTensorParamrInit(const CNodePtr &kernel_node);

  void TileMultipleCompute(void);

  std::vector<size_t> x_shape_;
  std::vector<size_t> y_shape_;
  std::vector<int> multiples_;
  TypeId dtype_{kTypeUnknown};
  using TypeKernel =
    std::function<void(TileCPUKernel *, const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs)>;
  std::unordered_map<TypeId, TypeKernel> launch_map_;
  TypeKernel launch_func_;
  TileParameter tile_parameter_;
  bool one_dim_tile_{false};
  size_t input_size_{0};
};

MS_REG_CPU_KERNEL(Tile, KernelAttr(), TileCPUKernel);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TILE_CPU_KERNEL_H_
