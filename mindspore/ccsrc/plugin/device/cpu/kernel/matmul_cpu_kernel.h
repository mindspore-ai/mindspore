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
constexpr auto kUnkown = "Unknown";
class MatMulCpuKernelMod : public NativeCpuKernelMod {
 public:
  MatMulCpuKernelMod() = default;
  explicit MatMulCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~MatMulCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return func_obj_->RunFunc(inputs, workspace, outputs);
  }

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::shared_ptr<CpuKernelFunc> func_obj_;
  std::string kernel_type_{kUnkown};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MATMUL_CPU_KERNEL_H_
