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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_LPNORM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_LPNORM_GPU_KERNEL_H_
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class LpNormGpuKernelMod : public NativeGpuKernelMod {
 public:
  LpNormGpuKernelMod() { ResetResource(); }
  ~LpNormGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  std::vector<KernelTensorPtr> GetOutputs() override { return outputs_; }

  void ResetResource() noexcept {
    is_null_input_ = false;
    cuda_stream_ = nullptr;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();

    input_size_list_.emplace_back(input_elements_ * unit_size_);
    // The workspace for device input shape.
    size_t device_input_shape_size = input_shape_.size() * sizeof(size_t);
    // The workspace for device output shape.
    size_t device_output_shape_size = output_shape_.size() * sizeof(size_t);
    // The workspace for device output axis.
    size_t device_axis_shape_size = output_axis_.size() * sizeof(size_t);
    // The workspace for device output stride.
    size_t device_output_stride_size = output_stride_.size() * sizeof(size_t);

    workspace_size_list_.emplace_back(device_input_shape_size);
    workspace_size_list_.emplace_back(device_output_shape_size);
    workspace_size_list_.emplace_back(device_axis_shape_size);
    workspace_size_list_.emplace_back(device_output_stride_size);
    output_size_list_.emplace_back(output_elements_ * unit_size_);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  using LpNormFunc =
    std::function<bool(LpNormGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;

  void GetLpNormAttr();

 private:
  size_t unit_size_{1};
  float p_{2.0};
  float epsilon_{1e-12};
  std::vector<int64_t> axis_;
  void *cuda_stream_{nullptr};
  bool is_null_input_{false};

  std::optional<bool> is_input_dynamic_shape_{};
  BaseOperatorPtr kernel_ptr_{nullptr};
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> output_axis_;
  std::vector<size_t> output_stride_;
  size_t input_elements_{};
  size_t output_elements_{};
  std::vector<KernelTensorPtr> outputs_ = {};
  LpNormFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, LpNormFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_LPNORM_GPU_KERNEL_H_
