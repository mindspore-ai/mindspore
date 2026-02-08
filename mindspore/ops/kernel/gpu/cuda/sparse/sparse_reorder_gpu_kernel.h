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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_REORDER_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_REORDER_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
class SparseReorderGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseReorderGpuKernelMod() {}
  ~SparseReorderGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  using SparseReorderFunc =
    std::function<bool(SparseReorderGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;

 private:
  void *cuda_stream_{nullptr};
  SparseReorderFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, SparseReorderFunc>> func_list_;
  size_t unit_size_{1};
  size_t values_unit_size_{1};
  size_t shape_unit_size_{1};
  size_t input_elements_{};
  size_t values_elements_{};
  size_t shape_elements_{};
  size_t output_indices_elements_{};
  size_t output_values_elements_{};
  int num_elems_;
  int num_dims_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_GPU_SPARSE_REORDER_GPU_KERNEL_H_
