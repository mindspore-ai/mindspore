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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TILE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TILE_CPU_KERNEL_H_
#include <memory>
#include <unordered_map>
#include <vector>
#include <random>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "nnacl/base/tile_base.h"

namespace mindspore {
namespace kernel {
class MultinomialCpuKernel : public NativeCpuKernelMod {
 public:
  MultinomialCpuKernel() = default;
  ~MultinomialCpuKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32)};
    return support_list;
  }

 private:
  std::vector<size_t> input_shape_;
  int seed_{0};
  int seed2_{0};
  std::default_random_engine rng_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TILE_CPU_KERNEL_H_
