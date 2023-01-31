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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PAD_V3_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PAD_V3_CPU_KERNEL_H_

#include <algorithm>
#include <array>
#include <complex>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class PadV3CpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<PadV3CpuKernelMod> {
 public:
  PadV3CpuKernelMod() = default;
  ~PadV3CpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    MS_EXCEPTION_IF_NULL(kernel_func_);
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

 private:
  template <typename S>
  bool GetPaddings(const std::vector<AddressPtr> &inputs);

  template <typename T>
  void ConstantModeCompute(T *input_ptr, T *output_ptr, T constant_values);

  template <typename T>
  void OtherModeCompute(T *input_ptr, T *output_ptr, int64_t p) const;

  template <typename T>
  void OtherModeCompute1D(T *input_ptr, T *output_ptr, int64_t p) const;

  template <typename T>
  void OtherModeCompute2D(T *input_ptr, T *output_ptr, int64_t p) const;

  template <typename T>
  void OtherModeCompute3D(T *input_ptr, T *output_ptr, int64_t p) const;

  int64_t IndexCalculate(int64_t pad_value, int64_t pad_end, int64_t now, int64_t input_value, int64_t o_start,
                         int64_t i_start) const;

  bool paddings_contiguous_;
  int64_t parallelSliceNum_{1};
  int64_t paddings_num_{0};
  int64_t input_dim_{0};
  std::string mode_ = "constant";
  std::vector<int64_t> paddings_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PAD_V3_CPU_KERNEL_H_
