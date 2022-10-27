/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FRACTIONAL_MAX_POOL3D_GRAD_WITH_FIXED_KSIZE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FRACTIONAL_MAX_POOL3D_GRAD_WITH_FIXED_KSIZE_CPU_KERNEL_H_
#include <memory>
#include <vector>
#include <string>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class FractionalMaxPool3DGradWithFixedKsizeCPUKernelMod : public NativeCpuKernelMod {
 public:
  FractionalMaxPool3DGradWithFixedKsizeCPUKernelMod() = default;
  ~FractionalMaxPool3DGradWithFixedKsizeCPUKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename backprop_t, typename argmax_t>
  bool GradComputeTemplate(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename backprop_t>
  bool DoComputeWithArgmaxType(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs,
                               TypeId argmax_type);
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> out_backprop_shape_;
  std::vector<int64_t> argmax_shape_;
  std::string data_format_;
  TypeId out_backprop_type_;
  TypeId argmax_type_;
  int64_t c_dim_;
  int64_t d_dim_;
  int64_t h_dim_;
  int64_t w_dim_;
  int64_t inputN_;
  int64_t inputC_;
  int64_t inputD_;
  int64_t inputH_;
  int64_t inputW_;
  int64_t outputD_;
  int64_t outputH_;
  int64_t outputW_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FRACTIONAL_MAX_POOL3D_GRAD_WITH_FIXED_KSIZE_CPU_KERNEL_H_
