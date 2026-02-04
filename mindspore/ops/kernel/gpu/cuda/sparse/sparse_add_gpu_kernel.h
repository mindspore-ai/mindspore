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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_ADD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_ADD_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <utility>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
class SparseAddGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseAddGpuKernelMod() = default;
  ~SparseAddGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void ResetResource() noexcept;
  void CalWorkSpace();
  template <typename T, typename S, typename K>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using SparseAddLaunchFunc =
    std::function<bool(SparseAddGpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, SparseAddLaunchFunc>> func_list_;
  SparseAddLaunchFunc kernel_func_;
  cudaStream_t cuda_stream_;
  size_t indices_column_ = 0;
  size_t a_indices_size_ = 0;
  size_t a_values_size_ = 0;
  size_t dense_shape_size_ = 0;
  size_t b_indices_size_ = 0;
  size_t b_values_size_ = 0;
  size_t indices_size_ = 0;       // That is, sizeof(T).
  size_t values_size_ = 0;        // That is, sizeof(S)
  size_t threshold_size_ = 0;     // That is, sizeof(K)
  int64_t real_output_size_ = 0;  // Dynamic shape related.
  std::vector<size_t> a_indices_shape_{};
  std::vector<size_t> a_values_shape_{};
  std::vector<size_t> dense_shape_{};
  std::vector<size_t> b_indices_shape_{};
  std::vector<size_t> b_values_shape_{};
  size_t rank_ = 0;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_ADD_GPU_KERNEL_H_
