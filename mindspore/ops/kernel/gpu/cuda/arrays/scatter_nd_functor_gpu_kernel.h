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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SCATTER_ND_FUNCTOR_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SCATTER_ND_FUNCTOR_GPU_KERNEL_H_

#include <map>
#include <vector>
#include <string>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/scatter_nd_functor_impl.cuh"

namespace mindspore {
namespace kernel {
class ScatterNdFunctorGPUKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<ScatterNdFunctorGPUKernelMod> {
 public:
  ScatterNdFunctorGPUKernelMod() = default;
  ~ScatterNdFunctorGPUKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    stream_ptr_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

  ScatterNdFunctorType scatter_nd_functor_type_;

  size_t unit_size_{0};
  size_t num_units_{0};
  size_t index_depth_{0};
  // The out_strides_ would be int64 or int32
  std::vector<int32_t> out_strides_;
  std::vector<int32_t> work_shape_;

  bool is_null_input_{false};

  std::vector<KernelTensorPtr> outputs_{};

  void *stream_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SCATTER_ND_FUNCTOR_GPU_KERNEL_H_
