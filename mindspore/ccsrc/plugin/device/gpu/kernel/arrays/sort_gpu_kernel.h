/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SORT_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SORT_GPU_KERNEL_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
#include <memory>
#include <map>

#include "mindspore/core/ops/sort.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/topk_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unary_op_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr int kSortInputsNum = 1;
constexpr int kSortOutputsNum = 2;

class SortGpuKernelMod : public NativeGpuKernelMod {
 public:
  SortGpuKernelMod() { ResetResource(); }
  ~SortGpuKernelMod() = default;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    auto kernel_name = base_operator->GetPrim()->name();
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSortInputsNum, kernel_name);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSortOutputsNum, kernel_name);
    auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
    if (!is_match) {
      return false;
    }

    input_shape_ = inputs[0]->GetShapeVector();
    is_null_input_ = CHECK_SHAPE_NULL(input_shape_, kernel_name, "input");
    if (is_null_input_) {
      return true;
    }

    input_rank_ = input_shape_.size();
    if (input_rank_ > TRANSPOSE_MAX_DIMENSION || input_rank_ < 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of input cannot be greater than "
                        << TRANSPOSE_MAX_DIMENSION << ", or less than 1"
                        << ", but got " << input_rank_;
    }

    input_size_ = 1;
    auto kernel_ptr = std::make_shared<ops::Sort>(base_operator->GetPrim());

    descending_ = static_cast<bool>(kernel_ptr->get_descending());
    axis_ = static_cast<int64_t>(kernel_ptr->get_axis());
    if (axis_ < 0) {
      axis_ += input_rank_;
    }
    if ((size_t)axis_ >= input_rank_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the value of 'axis' must be less than the dimension of input"
                        << ", but got the dimension of input: " << input_rank_
                        << ", got the value of 'axis': " << (size_t)axis_;
    }

    perm_.resize(input_rank_);
    std::iota(perm_.begin(), perm_.end(), 0);
    std::swap(perm_[input_rank_ - 1], perm_[axis_]);

    transposed_shape_ = input_shape_;
    std::swap(transposed_shape_[input_rank_ - 1], transposed_shape_[axis_]);

    inner_size_ = static_cast<size_t>(input_shape_[axis_]);
    outer_size_ = input_size_ / inner_size_;

    MS_LOG(DEBUG) << "In gpu kernel sort Init, axis_=" << axis_ << " descending_=" << descending_
                  << " input_rank_=" << input_rank_ << " input_size_=" << input_size_ << " inner_size_=" << inner_size_
                  << " outer_size_=" << outer_size_;
    (void)KernelMod::Resize(base_operator, inputs, outputs);
    if (input_size_list_.size() > 0) {
      size_t input_bytes = input_size_list_.at(kIndex0);
      size_t indices_bytes = input_size_ * sizeof(int32_t);
      workspace_size_list_.push_back(input_bytes);
      workspace_size_list_.push_back(indices_bytes);
      workspace_size_list_.push_back(input_rank_ * sizeof(size_t));
      workspace_size_list_.push_back(input_rank_ * sizeof(size_t));
      workspace_size_list_.push_back(input_rank_ * sizeof(size_t));
    }

    kernel_func_ = func_list_[index].second;
    return true;
  }

  void ResetResource() noexcept {
    input_size_ = 0;
    axis_ = 0;
    descending_ = false;
    is_null_input_ = false;
    input_shape_.clear();
    input_rank_ = 0;
    transposed_shape_.clear();
    perm_.clear();
    outer_size_ = 0;
    inner_size_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  std::vector<KernelTensorPtr> outputs_{};
  std::vector<KernelAttr> GetOpSupport() override;
  std::vector<KernelTensorPtr> GetOutputs() override { return outputs_; }

 private:
  size_t input_size_;
  int64_t axis_;
  bool descending_;
  bool is_null_input_;
  std::vector<int64_t> input_shape_;
  size_t input_rank_;

  // for transpose
  std::vector<int64_t> transposed_shape_;
  std::vector<size_t> perm_;

  // for topk
  size_t outer_size_;
  size_t inner_size_;

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using SortLaunchFunc = std::function<bool(SortGpuKernelMod *, const std::vector<AddressPtr> &,
                                            const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, SortLaunchFunc>> func_list_;
  SortLaunchFunc kernel_func_;
  cudaStream_t cuda_stream_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SORT_GPU_KERNEL_H_
