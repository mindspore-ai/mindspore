
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CAST_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CAST_CPU_KERNEL_H_

#include <functional>
#include <memory>
#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class CastCpuKernelMod : public NativeCpuKernelMod {
 public:
  CastCpuKernelMod() = default;
  ~CastCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    const size_t kCastInputsNum = 1;
    const size_t kCastOutputsNum = 1;
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCastInputsNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCastOutputsNum, kernel_name_);
    if (outputs[0]->size == 0) {
      MS_LOG(WARNING) << "For '" << kernel_name_ << "', the memory size of output must be greater than 0, but got 0.";
      return true;
    }
    return kernel_func_->RunFunc(inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  TypeId source_dtype_{kTypeUnknown};
  TypeId target_dtype_{kTypeUnknown};

  std::shared_ptr<CpuKernelFunc> kernel_func_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CAST_CPU_KERNEL_H_
