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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPSAMLE_TRILINEAR_3D_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPSAMLE_TRILINEAR_3D_CPU_KERNEL_H_

#include <algorithm>
#include <map>
#include <memory>
#include <vector>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "kernel/common_utils.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class UpsampleTrilinear3DCpuKernelMod : public NativeCpuKernelMod {
 public:
  UpsampleTrilinear3DCpuKernelMod() = default;
  ~UpsampleTrilinear3DCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename AccT>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  TypeId in_type_{kTypeUnknown};
  std::string kernel_name_;
  std::vector<int64_t> x_shape_;
  std::vector<int64_t> y_shape_;
  std::vector<float> attr_scales_;
  bool attr_align_corners_{false};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPSAMLE_TRILINEAR_3D_CPU_KERNEL_H_
