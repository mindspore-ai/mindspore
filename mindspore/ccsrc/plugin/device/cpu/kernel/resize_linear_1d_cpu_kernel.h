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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "mindapi/base/types.h"

namespace mindspore::kernel {
constexpr auto kUnknown = "Unknown";

class ResizeLinear1DCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<ResizeLinear1DCpuKernelMod> {
 public:
  ResizeLinear1DCpuKernelMod() = default;
  explicit ResizeLinear1DCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~ResizeLinear1DCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  void SetWorkSpaceSize(const std::vector<KernelTensor *> &inputs);

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);

  template <typename T>
  using CoordinateTransformationFunc = std::function<T(const T &new_x, const int &old_length, const int &new_length)>;

  template <typename T>
  CoordinateTransformationFunc<T> ChooseCoordinateTransformationFunc(
    CoordinateTransformMode coordinate_transformation_mode) const;

  template <typename T>
  void ComputeInterpolationCaches(const size_t out_size, const size_t in_size,
                                  const CoordinateTransformationFunc<T> &func, size_t *interp_lower,
                                  size_t *interp_upper, T *interp_lerp);

  std::string kernel_type_{kUnknown};
  TypeId x_type_{kTypeUnknown};
  size_t batch_{0};
  size_t channel_{0};
  size_t in_width_{0};
  size_t out_width_{0};
  CoordinateTransformMode coordinate_transformation_mode_ = CoordinateTransformMode::ALIGN_CORNERS;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RESIZE_LINEAR_1D_CPU_KERNEL_H_
