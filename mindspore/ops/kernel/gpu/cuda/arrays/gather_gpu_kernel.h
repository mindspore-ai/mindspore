/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHER_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHER_GPU_KERNEL_H_

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "primitive/framework_ops.h"
#include "kernel/gpu/cuda_impl/cuda_ops/gather.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class GatherGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<GatherGpuKernelMod> {
 public:
  GatherGpuKernelMod() = default;
  ~GatherGpuKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (!kernel_func_) {
      MS_LOG(ERROR) << "Gather's kernel function is not initialized in gpu.";
      return false;
    }
    stream_ptr_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

 private:
  void Reshape() {
    if (axis_ < 0) {
      axis_ = axis_ + SizeToInt(input_shapes_.size());
    }
    size_t batch_size = 1;
    size_t batch_dims = LongToSize(batch_dims_);
    for (size_t i = 0; i < batch_dims; i++) {
      batch_size *= LongToSize(input_shapes_[i]);
    }
    size_t dim_before_axis = 1;
    for (size_t i = batch_dims; i < std::min(IntToSize(axis_), output_shapes_.size()); i++) {
      dim_before_axis *= LongToSize(output_shapes_[i]);
    }
    size_t dim_of_indices = 1;
    for (size_t i = batch_dims; i < indices_shapes_.size(); i++) {
      dim_of_indices *= LongToSize(indices_shapes_[i]);
    }
    size_t dim_after_indices = 1;
    for (size_t i = IntToSize(axis_) + 1; i < input_shapes_.size(); i++) {
      dim_after_indices *= LongToSize(input_shapes_[i]);
    }
    dims_[kIndex0] = batch_size;
    dims_[kIndex1] = dim_before_axis;
    dims_[kIndex2] = dim_of_indices;
    dims_[kIndex3] = dim_after_indices;
    return;
  }

 private:
  std::vector<int64_t> input_shapes_{};
  std::vector<int64_t> indices_shapes_{};
  std::vector<int64_t> output_shapes_{};
  size_t dims_[kIndex4] = {0};
  int64_t axis_ = 0;
  int64_t batch_dims_{0};
  bool is_null_input_ = false;
  size_t input_type_size_ = 0;
  size_t indices_type_size_ = 0;
  void *stream_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHER_GPU_KERNEL_H_
