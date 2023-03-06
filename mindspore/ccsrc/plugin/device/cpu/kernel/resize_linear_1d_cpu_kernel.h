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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RESIZE_LINEAR_1D_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RESIZE_LINEAR_1D_CPU_KERNEL_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore::kernel {
constexpr auto kUnknown = "Unknown";

class ResizeLinear1DCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<ResizeLinear1DCpuKernelMod> {
 public:
  ResizeLinear1DCpuKernelMod() = default;
  explicit ResizeLinear1DCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~ResizeLinear1DCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  void SetWorkSpaceSize(const std::vector<KernelTensorPtr> &inputs);

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  enum CoordinateTransformationMode { ALIGN_CORNERS = 0, HALF_PIXEL = 1, INVALID_MODE = 255 };
  template <typename T>
  using CoordinateTransformationFunc = std::function<T(const T &new_x, const int &old_length, const int &new_length)>;

  template <typename T>
  CoordinateTransformationFunc<T> ChooseCoordinateTransformationFunc(
    CoordinateTransformationMode coordinate_transformation_mode) const;

  template <typename T>
  void ComputeInterpolationCaches(const size_t out_size, const size_t in_size,
                                  const CoordinateTransformationFunc<T> &func, size_t *interp_lower,
                                  size_t *interp_upper, T *interp_lerp);

  std::string kernel_type_{kUnknown};
  size_t batch_{0};
  size_t channel_{0};
  size_t in_width_{0};
  size_t out_width_{0};
  CoordinateTransformationMode coordinate_transformation_mode_{ALIGN_CORNERS};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RESIZE_LINEAR_1D_CPU_KERNEL_H_
