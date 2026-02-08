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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_SLICE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_SLICE_GPU_KERNEL_H_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <map>
#include <algorithm>
#include <functional>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "mindspore/ops/infer/sparse_slice.h"
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"
#include "kernel/gpu/cuda_impl/cuda_class/cuda_class_common.h"
#include "kernel/gpu/cuda_impl/cuda_ops/sparse_slice_impl.cuh"

namespace mindspore {
namespace kernel {
class SparseSliceGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseSliceGpuKernelMod() = default;
  ~SparseSliceGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using SparseSliceLaunchFunc = std::function<bool(
    SparseSliceGpuKernelMod *, const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &,
    const std::vector<kernel::KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, SparseSliceLaunchFunc>> func_list_;
  SparseSliceLaunchFunc kernel_func_;
  cudaStream_t cuda_stream;
  int64_t real_output_size_ = 0;
  size_t input_nnz_{0};
  size_t num_dim_{0};
  size_t out_size_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_SPARSE_SEGMENT_MEAN_GPU_KERNEL_H_
