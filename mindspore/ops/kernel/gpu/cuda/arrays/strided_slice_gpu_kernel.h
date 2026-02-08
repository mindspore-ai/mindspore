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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GPU_KERNEL_H_

#include <vector>
#include <map>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda/arrays/strided_slice_gpu_common.h"

namespace mindspore {
namespace kernel {
class StridedSliceGpuKernelMod : public NativeGpuKernelMod, public StridedSliceGpuCommon {
 public:
  StridedSliceGpuKernelMod() = default;
  ~StridedSliceGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (null_output_) {
      return true;
    }
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  std::vector<KernelAttr> GetOpSupport() override;
  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override {
    return {kBeginIndex_, kEndIndex_, kStrideIndex_};
  }

 protected:
  template <typename T, typename S = int64_t>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using StridedSliceFunc = std::function<bool(StridedSliceGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                              const std::vector<kernel::KernelTensor *> &,
                                              const std::vector<kernel::KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, StridedSliceFunc>> func_list_;
  StridedSliceFunc kernel_func_;

  bool is_null_input_{false};
  static constexpr size_t kBeginIndex_{1};
  static constexpr size_t kEndIndex_{2};
  static constexpr size_t kStrideIndex_{3};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GPU_KERNEL_H_
