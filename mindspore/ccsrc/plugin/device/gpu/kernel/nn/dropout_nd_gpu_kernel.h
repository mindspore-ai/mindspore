/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT_ND_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT_ND_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class DropoutNDGpuKernelMod : public NativeGpuKernelMod {
 public:
  DropoutNDGpuKernelMod() { ResetResource(); }
  ~DropoutNDGpuKernelMod() override = default;

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

  bool Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
              const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  std::vector<KernelTensorPtr> GetOutputs() override { return outputs_; }

  std::vector<KernelAttr> GetOpSupport() override;

  void ResetResource() noexcept {
    cudnn_handle_ = nullptr;
    is_null_input_ = false;
    input_elements_ = 0;
    keep_prob_ = 0.0;
    n_ = 0;
    c_ = 0;
    num_chan_ = 0;
    num_per_chan_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();

    size_t input_size = input_elements_ * unit_size_;
    size_t mask_size = input_elements_ * sizeof(bool);
    input_size_list_.push_back(input_size);
    // For output size: the same as input size
    output_size_list_.push_back(input_size);
    output_size_list_.push_back(mask_size);

    // The workspace of rand_f for curandGen
    size_t workspace_size = num_chan_ * sizeof(float);
    workspace_size_list_.push_back(workspace_size);
  }

 private:
  void CheckDropOutNdShape();

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  using DropoutNdFunc =
    std::function<bool(DropoutNDGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;

 private:
  size_t unit_size_{1};
  bool is_null_input_{false};
  bool states_init_{false};
  size_t input_elements_{};
  size_t n_{1};
  size_t c_{1};
  size_t num_chan_{1};
  size_t num_per_chan_{1};
  float keep_prob_{0.5};
  std::optional<bool> is_input_dynamic_shape_{};
  BaseOperatorPtr kernel_ptr_{nullptr};
  void *cuda_stream_{nullptr};
  cudnnHandle_t cudnn_handle_{};
  curandGenerator_t curand_generator_{nullptr};
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  std::vector<KernelTensorPtr> outputs_ = {};
  DropoutNdFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, DropoutNdFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT_ND_GPU_KERNEL_H_
