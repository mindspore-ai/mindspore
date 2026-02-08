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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GRAD_GPU_KERNEL_ARRAYS_MASKED_SELECT_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GRAD_GPU_KERNEL_ARRAYS_MASKED_SELECT_GRAD_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <utility>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
constexpr size_t MAX_DIMS = 7;

class MaskedSelectGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  MaskedSelectGradGpuKernelMod() = default;
  ~MaskedSelectGradGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void ResetResource() noexcept;
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using MaskedSelectGradFunc =
    std::function<bool(MaskedSelectGradGpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, MaskedSelectGradFunc>> func_list_;
  MaskedSelectGradFunc kernel_func_;
  cudaStream_t cuda_stream_;

  size_t input_size_;
  size_t mask_size_;
  size_t broadcast_size_;
  size_t input_type_size_;  // sizeof(T)
  size_t mask_type_size_;
  size_t real_output_size_;  // Dynamic shape related.
  bool input_broadcast_;
  bool mask_broadcast_;
  std::vector<int64_t> input_shape_ = {1, 1, 1, 1, 1, 1, 1};
  std::vector<int64_t> mask_shape_ = {1, 1, 1, 1, 1, 1, 1};
  std::vector<int64_t> broadcast_shape_ = {1, 1, 1, 1, 1, 1, 1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GRAD_GPU_KERNEL_ARRAYS_MASKED_SELECT_GRAD_GPU_KERNEL_H_
