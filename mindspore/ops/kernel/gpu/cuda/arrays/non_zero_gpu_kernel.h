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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_NON_ZERO_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_NON_ZERO_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <utility>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
class NonZeroGpuKernelMod : public NativeGpuKernelMod {
 public:
  NonZeroGpuKernelMod() = default;
  ~NonZeroGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void ResetResource() noexcept;
  template <typename DataType, typename IndexType>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using NonZeroLaunchFunc =
    std::function<bool(NonZeroGpuKernelMod *, const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, NonZeroLaunchFunc>> func_list_;
  NonZeroLaunchFunc kernel_func_;
  cudaStream_t cuda_stream_;
  size_t input_size_;
  size_t data_size_;         // That is, sizeof(DataType).
  size_t index_size_;        // That is, sizeof(IndexType)
  size_t real_output_size_;  // Dynamic shape related.
  std::vector<size_t> input_shape_{};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_NON_ZERO_GPU_KERNEL_H_
