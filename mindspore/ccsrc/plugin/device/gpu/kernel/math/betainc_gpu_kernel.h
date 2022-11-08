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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BETAINC_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BETAINC_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/betainc_impl.cuh"
namespace mindspore {
namespace kernel {
class BetaincGpuKernelMod : public NativeGpuKernelMod {
 public:
  BetaincGpuKernelMod() { ResetResource(); }
  ~BetaincGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  void ResetResource() noexcept {
    is_null_input_ = false;
    input_elements_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  void InitSizeLists() {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(input_size_);
    workspace_size_list_.push_back(input_size_);
    workspace_size_list_.push_back(input_size_);
    workspace_size_list_.push_back(input_size_);
    workspace_size_list_.push_back(input_size_);
    workspace_size_list_.push_back(input_size_);
    workspace_size_list_.push_back(input_size_);
  }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs,
                    const std::vector<AddressPtr> &workspace, void *cuda_stream);
  using KernelFunc =
    std::function<bool(BetaincGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;
  KernelFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, KernelFunc>> func_list_;
  bool is_null_input_;
  size_t input_elements_;
  size_t input_size_;
  size_t output_size;
  std::vector<size_t> a_shape_;
  std::vector<size_t> b_shape_;
  std::vector<size_t> x_shape_;
  std::vector<size_t> output_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BETAINC_GPU_KERNEL_H_
