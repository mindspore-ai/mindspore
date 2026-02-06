/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_FAST_SORT_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_FAST_SORT_GPU_KERNEL_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
#include <memory>
#include <map>

#include "mindspore/ops/infer/sort.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/tensor_layout_helper.cuh"
#include "kernel/gpu/cuda/arrays/sort_key_value_inplace.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {
constexpr int kFastSortInputsNum = 1;
constexpr int kFastSortOutputsNum = 2;

template <typename K, typename V>
class FastSortGpuKernelMod : public NativeGpuKernelMod {
 public:
  FastSortGpuKernelMod() = default;
  ~FastSortGpuKernelMod() {
    delete input_info_;
    delete output_index_info_;
    delete output_value_info_;
  }

  bool Init(const PrimitivePtr &primitive, const std::vector<KernelTensor *> &inputs,
            const std::vector<KernelTensor *> &outputs) {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kFastSortInputsNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kFastSortOutputsNum, kernel_name_);
    primitive_ = primitive;
    return true;
  }

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

    input_size_ = 1;
    for (int64_t i = 0; i < input_rank_; i++) {
      input_size_ *= input_shape_[i];
    }
    descending_ = GetValue<bool>(primitive_->GetAttr("descending"));
    axis_ = GetValue<int64_t>(primitive_->GetAttr("axis"));
    if (axis_ < 0) {
      axis_ += input_rank_;
    }
    if (axis_ >= input_rank_) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the value of 'axis' must be less than the dimension of input"
                    << ", but got the dimension of input: "
                    << ", got the value of 'axis': ";
      return KRET_RESIZE_FAILED;
    }

    constexpr int kMaxFixedSortSize = 4096;
    if (input_shape_[axis_] > kMaxFixedSortSize) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', only support sort dim less or equal to 4096, but got: ";
      return KRET_RESIZE_FAILED;
    }

    delete input_info_;
    delete output_index_info_;
    delete output_value_info_;

    if (input_rank_ > MAX_TENSORINFO_DIMS) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', only support sort dim less or equal to " << MAX_TENSORINFO_DIMS
                    << ", but got: " << input_rank_;
      return KRET_RESIZE_FAILED;
    }
    int shape[MAX_TENSORINFO_DIMS];
    for (int i = 0; i < input_rank_; i++) {
      shape[i] = input_shape_[i];
    }
    input_info_ = new TensorLayoutHelper(shape, input_rank_);
    if (input_info_ == nullptr) {
      MS_LOG(ERROR) << "Malloc TensorLayoutHelper for input failed while Resizing.";
      return KRET_RESIZE_FAILED;
    }
    output_index_info_ = new TensorLayoutHelper(shape, input_rank_);
    if (output_index_info_ == nullptr) {
      MS_LOG(ERROR) << "Malloc TensorLayoutHelper for output index failed while Resizing.";
      return KRET_RESIZE_FAILED;
    }
    output_value_info_ = new TensorLayoutHelper(shape, input_rank_);
    if (output_value_info_ == nullptr) {
      MS_LOG(ERROR) << "Malloc TensorLayoutHelper for output value failed while Resizing.";
      return KRET_RESIZE_FAILED;
    }
    return KRET_OK;
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return LaunchKernel(inputs, workspace, outputs, stream_ptr);
  }

  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
    cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (is_null_input_) {
      return true;
    }
    V *input_device = GetDeviceAddress<V>(inputs, 0);

    V *output_device = GetDeviceAddress<V>(outputs, 0);
    K *indices_device = GetDeviceAddress<K>(outputs, 1);

    auto ret = InitIndexBySlice<K>(*output_index_info_, axis_, indices_device, cuda_stream_);
    if (!ret) {
      MS_LOG(ERROR) << "InitIndexBySlice failed.";
      return false;
    }

    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(output_device, input_device, input_size_ * sizeof(V), cudaMemcpyDeviceToDevice, cuda_stream_),
      "cudaMemcpyAsync for output_device failed");
    return SortKeyValueInplace<V, K>(*output_value_info_, output_device, *output_index_info_, indices_device, axis_,
                                     descending_, cuda_stream_);
  }

 private:
  int64_t input_size_{0};
  int64_t axis_{0};
  bool descending_{false};
  bool is_null_input_{false};
  std::vector<int64_t> input_shape_;
  int64_t input_rank_{0};
  PrimitivePtr primitive_;

  TensorLayoutHelper *input_info_{nullptr};
  TensorLayoutHelper *output_index_info_{nullptr};
  TensorLayoutHelper *output_value_info_{nullptr};
  cudaStream_t cuda_stream_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_FAST_SORT_GPU_KERNEL_H_
