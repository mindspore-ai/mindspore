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
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "mindspore/ops/infer/sort.h"
#include "kernel/gpu/cuda/arrays/fast_sort_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/elementwise/eltwise_ops_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/topk_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/transpose_impl.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
constexpr int kSortInputsNum = 1;
constexpr int kSortOutputsNum = 2;

template <typename K, typename V>
class SortGpuKernelMod : public NativeGpuKernelMod {
 public:
  SortGpuKernelMod() { ResetResource(); }
  ~SortGpuKernelMod() { delete fast_sort_kernel_; }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    auto ret = KernelMod::Resize(inputs, outputs);
    if (ret != KRET_OK) {
      return ret;
    }

    input_shape_ = inputs[0]->GetShapeVector();

    is_null_input_ = CHECK_SHAPE_NULL(input_shape_, kernel_name_, "input");
    if (is_null_input_) {
      return KRET_OK;
    }

    input_rank_ = input_shape_.size();
    if (input_rank_ > transpose_max_dimension || input_rank_ < 1) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than "
                    << transpose_max_dimension << ", or less than 1"
                    << ", but got " << input_rank_;
      return KRET_RESIZE_FAILED;
    }
    descending_ = GetValue<bool>(primitive_->GetAttr("descending"));
    axis_ = GetValue<int64_t>(primitive_->GetAttr("axis"));
    if (axis_ < 0) {
      axis_ += input_rank_;
    }
    if (static_cast<size_t>(axis_) >= input_rank_) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the value of 'axis' must be less than the dimension of input"
                    << ", but got the dimension of input: " << input_rank_
                    << ", got the value of 'axis': " << static_cast<size_t>(axis_);
      return KRET_RESIZE_FAILED;
    }

    use_fast_ = input_shape_[axis_] > 0 && input_shape_[axis_] <= sort_dim_thres_;
    if (use_fast_) {
      return fast_sort_kernel_->Resize(inputs, outputs);
    } else {
      if (!old_kernel_support_) {
        auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
        MS_LOG(ERROR) << "Only support input datatype in [float16, float32] for sort kernel, but got "
                      << kernel_attr.GetInputAttr(0).dtype << " in KernelAttr.";
        return KRET_RESIZE_FAILED;
      }
    }

    perm_.resize(input_rank_);
    std::iota(perm_.begin(), perm_.end(), 0);
    std::swap(perm_[input_rank_ - 1], perm_[axis_]);

    input_size_ = 1;
    for (size_t i = 0; i < input_rank_; i++) {
      input_size_ *= static_cast<size_t>(input_shape_[i]);
    }

    transposed_shape_ = input_shape_;
    std::swap(transposed_shape_[input_rank_ - 1], transposed_shape_[axis_]);
    inner_size_ = static_cast<size_t>(input_shape_[axis_]);
    outer_size_ = input_size_ / inner_size_;
    MS_LOG(DEBUG) << "In gpu kernel sort Resize, axis_=" << axis_ << " descending_=" << descending_
                  << " input_rank_=" << input_rank_ << " input_size_=" << input_size_ << " inner_size_=" << inner_size_
                  << " outer_size_=" << outer_size_;

    if (inputs.size() > 0) {
      size_t input_bytes = inputs.at(kIndex0)->size();
      size_t indices_bytes = input_size_ * sizeof(int32_t);
      workspace_size_list_.push_back(input_bytes);
      workspace_size_list_.push_back(indices_bytes);
    }
    return KRET_OK;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSortInputsNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSortOutputsNum, kernel_name_);
    auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    KernelAttr fp16_kernel_attr;
    fp16_kernel_attr.AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32);
    KernelAttr fp32_kernel_attr;
    fp32_kernel_attr.AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32);
    std::vector<KernelAttr> support_list;
    support_list.emplace_back(fp16_kernel_attr);
    support_list.emplace_back(fp32_kernel_attr);
    old_kernel_support_ = MatchKernelAttr(kernel_attr, support_list).first;

    MS_LOG(DEBUG) << "In gpu kernel sort Init, axis_=" << axis_ << " descending_=" << descending_
                  << " input_rank_=" << input_rank_ << " input_size_=" << input_size_ << " inner_size_=" << inner_size_
                  << " outer_size_=" << outer_size_;
    (void)KernelMod::Resize(inputs, outputs);

    fast_sort_kernel_ = new FastSortGpuKernelMod<K, V>();
    if (fast_sort_kernel_ == nullptr) {
      MS_LOG(ERROR) << "Malloc FastSortGpuKernelMod failed while Init.";
      return false;
    }
    return fast_sort_kernel_->Init(primitive_, inputs, outputs);
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    if (use_fast_) {
      return fast_sort_kernel_->Launch(inputs, workspace, outputs, stream_ptr);
    }
    return LaunchKernel(inputs, workspace, outputs, stream_ptr);
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
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

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

  // fast sort
  FastSortGpuKernelMod<K, V> *fast_sort_kernel_{nullptr};
  bool use_fast_{false};
  constexpr static int64_t sort_dim_thres_ = 4096;
  bool old_kernel_support_{false};

  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  cudaStream_t cuda_stream_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SORT_GPU_KERNEL_H_
