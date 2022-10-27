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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MULTI_MARGIN_LOSS_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MULTI_MARGIN_LOSS_CPU_KERNEL_H_
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class MultiMarginLossCPUKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<MultiMarginLossCPUKernelMod> {
 public:
  MultiMarginLossCPUKernelMod() = default;

  ~MultiMarginLossCPUKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    MS_EXCEPTION_IF_NULL(kernel_func_);
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

 private:
  template <typename T>
  void LaunchKernelFP32AndFP64(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename T>
  void LaunchKernelFP16(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  void CheckParam(const CNodePtr &kernel_node);
  size_t batch_size = 2;
  size_t dims = 1;
  std::string reduction = MEAN;
  float margin = 1.0;
  int64_t p = 1;
  size_t input_num = 1;
  TypeId dtype_{kTypeUnknown};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MULTI_MARGIN_LOSS_CPU_KERNEL_H_
