/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_SCATTER_ND_ARITHMETIC_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_SCATTER_ND_ARITHMETIC_CPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <set>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
class ScatterNdArithmeticCpuKernelMod : public NativeCpuKernelMod,
                                        public MatchKernelHelper<ScatterNdArithmeticCpuKernelMod> {
 public:
  ScatterNdArithmeticCpuKernelMod() = default;
  ~ScatterNdArithmeticCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);

  using ScatterNdSupportListType = std::vector<std::pair<KernelAttr, ScatterNdArithmeticCpuKernelMod::KernelRunFunc>>;

  template <typename T>
  using ComputeFunc = std::function<void(T *a, size_t a_index, T *b, size_t b_index)>;

  template <typename T>
  std::pair<bool, ComputeFunc<T>> InitComputeFunc();

  size_t slice_size_{1};
  size_t batch_size_{1};
  size_t inner_size_{1};
  std::vector<size_t> batch_strides_;
  std::vector<size_t> input_shape_;
  float block_size_{128.0};
  bool has_null_input_{false};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_SCATTER_ND_ARITHMETIC_CPU_KERNEL_H_
