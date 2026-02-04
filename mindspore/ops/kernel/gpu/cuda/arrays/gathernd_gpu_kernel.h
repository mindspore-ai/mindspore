/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_ARRAYS_GATHERND_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_ARRAYS_GATHERND_GPU_KERNEL_H_
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "kernel/gpu/cuda_impl/cuda_ops/gathernd.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class GatherNdFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  GatherNdFwdGpuKernelMod() = default;
  ~GatherNdFwdGpuKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (!kernel_func_) {
      MS_LOG(ERROR) << "GatherNd's kernel function is not initialized.";
      return false;
    }
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using GatherNdFwdFunc = std::function<bool(GatherNdFwdGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                             const std::vector<kernel::KernelTensor *> &,
                                             const std::vector<kernel::KernelTensor *> &, void *)>;

  GatherNdFwdFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, GatherNdFwdFunc>> func_list_;

  bool is_null_input_{false};
  int64_t dim_indices_last_{0};
  std::vector<size_t> dims_;
  std::vector<int64_t> input_shapes_;
  std::vector<int64_t> batch_strides_;
  std::vector<int64_t> batch_indices_;
  void *cuda_stream_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_ARRAYS_GATHERND_GPU_KERNEL_H_
