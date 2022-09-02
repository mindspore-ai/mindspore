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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_MASKED_SELECT_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_MASKED_SELECT_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
constexpr size_t MAX_DIMS = 7;

class MaskedSelectGpuKernelMod : public NativeGpuKernelMod {
 public:
  MaskedSelectGpuKernelMod() = default;
  ~MaskedSelectGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  void SyncData() override;
  std::vector<KernelTensorPtr> GetOutputs() override { return outputs_; }

 private:
  void ResetResource() noexcept;
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using MaskedSelectFunc =
    std::function<bool(MaskedSelectGpuKernelMod *, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, MaskedSelectFunc>> func_list_;
  MaskedSelectFunc kernel_func_;
  cudaStream_t cuda_stream_;

  size_t input_size_;
  size_t mask_size_;
  size_t broadcast_size_;
  size_t input_type_size_;  // sizeof(T)
  size_t mask_type_size_;
  size_t real_output_size_;  // Dynamic shape related.
  bool input_broadcast_;
  bool mask_broadcast_;
  std::vector<size_t> input_shape_ = {1, 1, 1, 1, 1, 1, 1};
  std::vector<size_t> mask_shape_ = {1, 1, 1, 1, 1, 1, 1};
  std::vector<size_t> broadcast_shape_ = {1, 1, 1, 1, 1, 1, 1};
  std::vector<KernelTensorPtr> outputs_{};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_MASKED_SELECT_GPU_KERNEL_H_
