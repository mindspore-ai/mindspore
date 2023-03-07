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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_IN_PLACE_OP_V2_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_IN_PLACE_OP_V2_CPU_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
constexpr size_t kInplaceOpV2InputNum = 3;
constexpr size_t kInplaceOpV2OutputNum = 1;
class InPlaceOpV2CpuKernelMod : public NativeCpuKernelMod {
 public:
  InPlaceOpV2CpuKernelMod() = default;
  explicit InPlaceOpV2CpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~InPlaceOpV2CpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInplaceOpV2InputNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kInplaceOpV2OutputNum, kernel_name_);
    return func_obj_->RunFunc(inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::shared_ptr<CpuKernelFunc> func_obj_;
  std::string kernel_type_{"Unknown"};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_IN_PLACE_OP_V2_CPU_KERNEL_H_
