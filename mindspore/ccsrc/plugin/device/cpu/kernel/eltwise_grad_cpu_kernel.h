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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ELTWISE_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ELTWISE_GRAD_CPU_KERNEL_H_
#include <complex>
#include <memory>
#include <vector>
#include <map>
#include <limits>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

namespace mindspore {
namespace kernel {
constexpr size_t kInputMinNum = 2;
constexpr size_t kOutputNum = 1;
constexpr auto kUnKnown = "UnKnown";
class EltWiseGradCpuKernelMod : public NativeCpuKernelMod {
 public:
  EltWiseGradCpuKernelMod() = default;
  explicit EltWiseGradCpuKernelMod(const std::string &kernel_name) : kernel_name_(kernel_name) {}
  ~EltWiseGradCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    if (is_null_input_) {
      return true;
    }
    if (inputs.size() < kInputMinNum || outputs.size() != kOutputNum) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', it requires at least 2 inputs and 1 output, but got "
                    << inputs.size() << " input(s) and " << outputs.size() << " output(s).";
      return false;
    }
    if (outputs[0]->size == 0) {
      MS_LOG(WARNING) << "For '" << kernel_name_ << "', the memory size of output must be greater than 0, but got 0.";
      return true;
    }
    return func_obj_->RunFunc(inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::shared_ptr<CpuKernelFunc> func_obj_;
  bool is_null_input_{false};
  std::string kernel_name_{kUnKnown};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ELTWISE_GRAD_CPU_KERNEL_H_
