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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PAD_V3_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PAD_V3_GRAD_CPU_KERNEL_H_

#include <algorithm>
#include <memory>
#include <string>
#include <array>
#include <complex>
#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

class PadV3GradCpuKernelMod : public NativeCpuKernelMod {
 public:
  PadV3GradCpuKernelMod() = default;
  ~PadV3GradCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename S>
  bool GetPaddings(const std::vector<AddressPtr> &inputs);

  template <typename T, typename S>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void PadV3GradCompute(T *input, T *output, int64_t p) const;

  template <typename T>
  void PadV3GradCompute1D(T *input, T *output, int64_t p) const;

  template <typename T>
  void PadV3GradCompute2D(T *input, T *output, int64_t p, int64_t i) const;

  template <typename T>
  void PadV3GradCompute3D(T *input, T *output, int64_t p, int64_t z) const;

  int64_t IndexCalculate(int64_t pad_value, int64_t pad_end, int64_t now, int64_t output_value, int64_t o_start,
                         int64_t i_start) const;

  using SelectFunc =
    std::function<bool(PadV3GradCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, SelectFunc>> func_list_;
  SelectFunc kernel_func_;

  TypeId dtype_{kTypeUnknown};
  TypeId pad_dtype_{kTypeUnknown};
  bool paddings_contiguous_;
  std::string mode_ = "reflect";
  std::vector<int64_t> paddings_{0, 0, 0, 0, 0, 0};
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  int64_t output_w_{0};
  int64_t output_h_{0};
  int64_t output_c_{0};
  int64_t input_w_{0};
  int64_t input_h_{0};
  int64_t input_c_{0};
  int64_t i_start_x_{0};
  int64_t i_start_y_{0};
  int64_t i_start_z_{0};
  int64_t o_start_x_{0};
  int64_t o_start_y_{0};
  int64_t o_start_z_{0};
  int64_t pad_l_{0};
  int64_t pad_t_{0};
  int64_t pad_f_{0};
  int64_t pad_r_{0};
  int64_t pad_d_{0};
  int64_t pad_b_{0};
  int64_t parallelSliceNum_{1};
  int64_t paddings_num_{0};
  int64_t input_dim_{0};
  int64_t input_num_{1};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PAD_V3_GRAD_CPU_KERNEL_H_
