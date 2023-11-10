/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MATMUL_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MATMUL_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
using LaunchEmptyTensorFunc = std::function<void(const std::vector<KernelTensor *> &)>;
constexpr auto kUnkown = "Unknown";
class MatMulCpuKernelMod : public NativeCpuKernelMod {
 public:
  MatMulCpuKernelMod() = default;
  explicit MatMulCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~MatMulCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    if (is_empty_tensor_) {
      launch_empty_tensor_func_(outputs);
      return true;
    }
    return func_obj_->RunFunc(inputs, workspace, outputs);
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::shared_ptr<CpuKernelFunc> func_obj_;
  std::string kernel_type_{kUnkown};

  bool is_empty_tensor_{false};
  LaunchEmptyTensorFunc launch_empty_tensor_func_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MATMUL_CPU_KERNEL_H_
