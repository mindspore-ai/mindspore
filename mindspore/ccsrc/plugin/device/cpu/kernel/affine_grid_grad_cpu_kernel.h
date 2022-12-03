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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_AFFINEGRIDGRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_AFFINEGRIDGRAD_CPU_KERNEL_H_

#include <Eigen/Dense>
#include <unordered_map>
#include <utility>
#include <vector>
#include <map>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class AffineGridGradCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<AffineGridGradCpuKernelMod> {
 public:
  AffineGridGradCpuKernelMod() = default;
  ~AffineGridGradCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T, typename T0>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  template <typename T, typename T0>
  void LaunchKernel_3D(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  template <typename T, typename T0>
  void LaunchKernel_4D(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  template <typename T0>
  Eigen::MatrixXf make_base_grid_3D(const std::vector<kernel::AddressPtr> &inputs, Eigen::VectorXf vecX,
                                    Eigen::VectorXf vecY);
  template <typename T, typename T0>
  void DoCompute_3D(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs,
                    Eigen::MatrixXf all);
  template <typename T0>
  Eigen::MatrixXf make_base_grid_4D(const std::vector<kernel::AddressPtr> &inputs, Eigen::VectorXf vecX,
                                    Eigen::VectorXf vecY, Eigen::VectorXf vecZ);
  template <typename T, typename T0>
  void DoCompute_4D(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs,
                    Eigen::MatrixXf all);
  using AffineGridGradFunc =
    std::function<bool(AffineGridGradCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;

  std::vector<int64_t> x_size_dims_;
  bool align_corners_{false};
  std::vector<TypeId> input_info_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_AFFINEGRIDGRAD_CPU_KERNEL_H_
