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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARGMAXANDMINWITHVALUE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARGMAXANDMINWITHVALUE_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <functional>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/general_reduction_impl.cuh"
namespace mindspore {
namespace kernel {
constexpr size_t kInputNum = 3;
constexpr size_t kOutputNum = 2;

class ArgMaxAndMinWithValueGpuKernelMod : public NativeGpuKernelMod {
 public:
  explicit ArgMaxAndMinWithValueGpuKernelMod(size_t index_output_idx = 0, size_t value_output_idx = 1)
      : index_output_idx(index_output_idx), value_output_idx(value_output_idx) {
    ResetResource();
  }
  ~ArgMaxAndMinWithValueGpuKernelMod() override = default;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    auto input_shape = inputs.at(kIndex0)->GetShapeVector();
    if (CheckNullInput(input_shape)) {
      MS_EXCEPTION(ValueError) << kernel_name_ << " cannot deal with empty input. Please try other inputs.";
    }
    int ret = KernelMod::Resize(inputs, outputs);
    if (ret != KRET_OK) {
      return ret;
    }
    if (!InitSize(inputs, outputs)) {
      return KRET_RESIZE_FAILED;
    }
    return KRET_OK;
  }

  std::vector<KernelAttr> GetOpSupport() override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, outputs, stream_ptr);
  }

  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                    void *stream_ptr);

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool InitSize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  void ResetResource() noexcept {
    kernel_name_ = "";
    axis_ = 0;
    bound_ = 0;
    outer_size_ = 0;
    inner_size_ = 0;
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 private:
  bool small_ = false;
  bool is_zero_dim_{false};
  int64_t axis_;
  size_t bound_;
  size_t outer_size_;
  size_t inner_size_;
  using ArgWithValueFunc = std::function<bool(ArgMaxAndMinWithValueGpuKernelMod *, const std::vector<KernelTensor *> &,
                                              const std::vector<KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, ArgWithValueFunc>> func_list_;
  ArgWithValueFunc kernel_func_;

  size_t index_output_idx{0};
  size_t value_output_idx{1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARGMAXANDMINWITHVALUE_GPU_KERNEL_H_
