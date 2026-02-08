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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_SPARSE_SPARSE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_SPARSE_SPARSE_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <map>
#include "mindspore/ops/infer/sparse_sparse_maximum.h"
#include "mindspore/ops/infer/sparse_sparse_minimum.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/sparse_sparse_maximum_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/sparse_sparse_minimum_impl.cuh"

namespace mindspore {
namespace kernel {
class SparseSparseGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseSparseGpuKernelMod() {}
  ~SparseSparseGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    return kernel_func_(this, inputs, workspace, outputs, cuda_stream);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  void ResetResource() noexcept {
    rank_ = 0;
    a_indices_num_ = 0;
    b_indices_num_ = 0;
    real_output_size_ = 0;
    is_null_input_ = false;
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  using SparseSparseFunc =
    std::function<bool(SparseSparseGpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &, void *)>;

 private:
  static std::vector<std::pair<KernelAttr, SparseSparseFunc>> func_list_;
  SparseSparseFunc kernel_func_{};
  cudaStream_t cuda_stream_;
  bool is_null_input_{false};
  size_t indices_size_ = 0;
  size_t values_size_ = 0;
  int64_t real_output_size_ = 0;
  int64_t rank_ = 0;
  int64_t a_indices_num_ = 0;
  int64_t b_indices_num_ = 0;
};
}  // namespace kernel
}  // namespace mindspore
#endif
