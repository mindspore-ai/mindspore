/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_SPARSE_SEGMENT_MEAN_WITH_NUM_SEGMENTS_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_SPARSE_SEGMENT_MEAN_WITH_NUM_SEGMENTS_GPU_KERNEL_H

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <map>
#include <algorithm>
#include <functional>
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"
#include "kernel/gpu/cuda_impl/cuda_class/cuda_class_common.h"
#include "kernel/gpu/cuda_impl/cuda_ops/sparse_segment_mean_with_num_segments_impl.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class SparseSegmentMeanWithNumSegmentsGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseSegmentMeanWithNumSegmentsGpuKernelMod() = default;
  ~SparseSegmentMeanWithNumSegmentsGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using SparseSegmentMeanWithNumSegmentsLaunchFunc = std::function<bool(
    SparseSegmentMeanWithNumSegmentsGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
    const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, SparseSegmentMeanWithNumSegmentsLaunchFunc>> func_list_;
  SparseSegmentMeanWithNumSegmentsLaunchFunc kernel_func_;
  size_t outer_size_{1};
  size_t inner_size_{1};
  size_t indices_size_{1};
  size_t segment_size_{0};
  size_t x_size_{1};
  size_t y_size_{1};
  size_t batch_rank_{0};
  size_t batch_size_{1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_SPARSE_SEGMENT_MEAN_WITH_NUM_SEGMENTS_GPU_KERNEL_H
