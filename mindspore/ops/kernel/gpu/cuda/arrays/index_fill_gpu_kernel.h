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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_INDEX_FILL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_INDEX_FILL_GPU_KERNEL_H_

#include <limits>
#include <vector>
#include <functional>
#include <map>
#include <utility>
#include <memory>
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"
#include "kernel/gpu/cuda_impl/cuda_ops/index_fill_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_class/cuda_class_common.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class IndexFillGpuKernelMod : public NativeGpuKernelMod {
 public:
  IndexFillGpuKernelMod() = default;
  ~IndexFillGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  bool GetSizeInfo(KernelTensor *, int64_t &, int64_t &, int64_t &, cudaStream_t);
  template <typename DataType, typename IndexType>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using IndexFillLaunchFunc =
    std::function<bool(IndexFillGpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, IndexFillLaunchFunc>> func_list_;
  IndexFillLaunchFunc kernel_func_;
  int64_t x_num_{0};
  int64_t index_num_{0};
  std::vector<int64_t> x_shape_{};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_INDEX_FILL_GPU_KERNEL_H_
