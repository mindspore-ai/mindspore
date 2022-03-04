/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNPACK_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNPACK_CPU_KERNEL_H_

#include <algorithm>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>
#include <tuple>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "nnacl/base/unstack_base.h"

namespace mindspore {
namespace kernel {
class UnpackCpuKernelMod : public NativeCpuKernelMod {
 public:
  UnpackCpuKernelMod() = default;
  ~UnpackCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
  template <typename T>
  void InitIOSize(const CNodePtr &kernel_node);

  using UnstackFunc =
    std::function<bool(UnpackCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  using InitFunc = std::function<void(UnpackCpuKernelMod *, const CNodePtr &)>;
  static std::vector<std::tuple<KernelAttr, UnstackFunc, InitFunc>> func_list_;
  UnstackFunc kernel_func_;
  InitFunc init_io_func_;

  void InitInputOutputSize(const CNodePtr &kernel_node) override { init_io_func_(this, kernel_node); }

  UnstackParameter unstack_param_{};
  size_t output_num_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNPACK_CPU_KERNEL_H_
