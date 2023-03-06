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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_INSTANCE_NORM_V2_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_INSTANCE_NORM_V2_CPU_KERNEL_H_

#include <set>
#include <vector>
#include <map>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class InstanceNormV2CpuKernelMod : public NativeCpuKernelMod {
 public:
  InstanceNormV2CpuKernelMod() = default;
  ~InstanceNormV2CpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  template <typename T>
  void CollectStatsKernel(const kernel::AddressPtr &x, float *_mean_, float *_var_sum) const;

  void CollectLinearAndConstant(const typename TTypes<float>::Vec &gamma, const typename TTypes<float>::Vec &beta,
                                const typename TTypes<float>::Vec &running_mean,
                                const typename TTypes<float>::Vec &running_var,
                                const typename TTypes<float>::Vec &save_mean,
                                const typename TTypes<float>::Vec &save_invstd, float *_alpha_, float *_beta_);

  template <typename T>
  void TransformInput(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  template <typename T, template <typename S> class VarTransform>
  void UpdateStatsTemplate(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> &outputs);

  TypeId in_type_{kTypeUnknown};
  bool is_training_ = true;
  float momentum_ = 0.1;
  float epsilon_ = 0.00001;
  std::vector<int64_t> x_shape_4d_;
  std::vector<int64_t> batch_channels_2d_;
  bool input_x_is_4d_ = true;
  int64_t instance_num_ = 0;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_INSTANCE_NORM_V2_CPU_KERNEL_H_
