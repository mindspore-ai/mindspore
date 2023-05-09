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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BITWISE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BITWISE_CPU_KERNEL_H_

#include <functional>
#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <complex>
#include <map>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/nnacl/arithmetic_parameter.h"

namespace mindspore {
namespace kernel {
constexpr size_t kBitwiseInitThreadNum = 50;
constexpr float kBitwiseInitBlockSize = 100000;
const size_t kBitwiseBigShapeNum = 5000000;
class BitwiseCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<BitwiseCpuKernelMod> {
 public:
  BitwiseCpuKernelMod() = default;
  explicit BitwiseCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~BitwiseCpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                    const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void InitFunc();

  using BitwiseParallelFunc = std::function<void(BitwiseCpuKernelMod *, const CTask &task)>;
  BitwiseParallelFunc bitwise_parallel_func_;

  using BitwiseLaunchFunc = std::function<bool(BitwiseCpuKernelMod *, const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &outputs)>;
  BitwiseLaunchFunc bitwise_launch_func_;
  void BitwiseParallelSearch(const CTask &task);
  void BitwiseParallelMaxThread(const CTask &task);

  template <typename T, typename BitwiseFunT>
  bool LaunchBroadcast(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  template <typename T, typename BitwiseFunT>
  bool LaunchNoBroadcast(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  std::string kernel_type_{"Unknown"};
  TypeId input_type_1_{kTypeUnknown};
  TypeId input_type_2_{kTypeUnknown};
  ShapeVector input_shape_1_;
  ShapeVector input_shape_2_;
  ShapeVector output_shape_;
  size_t output_size_ = 1;
  const size_t max_dims_{7};
  bool broadcast_ = false;
  size_t thread_num_{kBitwiseInitThreadNum};
  float block_size_{kBitwiseInitBlockSize};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BITWISE_CPU_KERNEL_H_
