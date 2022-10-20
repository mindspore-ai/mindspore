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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_EMBEDDING_LOOKUP_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_EMBEDDING_LOOKUP_HELPER_H_
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <algorithm>
#include <memory>
#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/embedding_lookup_impl.cuh"

namespace mindspore {
namespace cukernel {
template <typename T>
size_t GetSize(const std::vector<int64_t> &shape) {
  if (shape.size() == 0) {
    return 0;
  }
  size_t result = sizeof(T);
  for (size_t i = 0; i < shape.size(); i++) {
    result *= static_cast<size_t>(shape[i]);
  }
  return result;
}

template <typename T, typename S, typename G>
class EmbeddingLookupHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit EmbeddingLookupHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }
  virtual ~EmbeddingLookupHelperGpuKernel() = default;

  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    auto input_params_shape_ = input_shapes[kIndex0];
    auto input_indices_shape_ = input_shapes[kIndex1];
    auto output_shape_ = output_shapes[kIndex0];
    is_null_input_ = CHECK_SHAPE_NULL(input_params_shape_, kernel_name_, "input_params_shape") ||
                     CHECK_SHAPE_NULL(input_indices_shape_, kernel_name_, "input_indices_shape");
    int64_t axis = 0;
    int64_t dim_before_axis = 1;
    for (size_t i = 0; i < LongToSize(axis); i++) {
      dim_before_axis *= output_shape_[i];
    }
    size_t dim_of_indices = 1;
    for (size_t i = 0; i < input_indices_shape_.size(); i++) {
      dim_of_indices *= input_indices_shape_[i];
    }
    int64_t dim_after_indices = 1;
    for (size_t i = LongToSize(axis) + input_indices_shape_.size(); i < output_shape_.size(); i++) {
      dim_after_indices *= output_shape_[i];
    }
    dims_[kIndex0] = dim_before_axis;
    dims_[kIndex1] = dim_of_indices;
    dims_[kIndex2] = dim_after_indices;
    input_dim1_ = input_params_shape_[0];
    input_size_list_.push_back(GetSize<T>(input_params_shape_));
    input_size_list_.push_back(GetSize<S>(input_indices_shape_));
    input_size_list_.push_back(sizeof(G));

    output_size_list_.push_back(GetSize<T>(output_shape_));
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }

    T *input_params_addr = nullptr;
    S *input_indices_addr = nullptr;
    T *output_addr = nullptr;

    int flag = GetDeviceAddress<T>(input_ptrs, kIndex0, kernel_name_, &input_params_addr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<S>(input_ptrs, kIndex1, kernel_name_, &input_indices_addr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(output_ptrs, kIndex0, kernel_name_, &output_addr);
    if (flag != 0) {
      return flag;
    }
    G *input_offset_addr = nullptr;
    flag = GetDeviceAddress<G>(input_ptrs, kIndex2, kernel_name_, &input_offset_addr);
    if (flag != 0) {
      return flag;
    }
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(&offset_, input_offset_addr, sizeof(G), cudaMemcpyDeviceToHost,
                                                      reinterpret_cast<cudaStream_t>(cuda_stream)),
                                      "cudaMemcpyAsync offset_ failed");
    CalEmbeddingLookup(input_params_addr, input_indices_addr, output_addr, dims_[kIndex0], dims_[kIndex1],
                       dims_[kIndex2], input_dim1_, static_cast<int64_t>(offset_),
                       reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void ResetResource() override {
    is_null_input_ = false;
    input_params_shape_.clear();
    input_indices_shape_.clear();
    input_offset_shape_.clear();
    output_shape_.clear();
    std::fill(dims_, dims_ + kIndex3, 0);
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
    offset_ = 0;
  }

 private:
  std::vector<int64_t> input_params_shape_;
  std::vector<int64_t> input_indices_shape_;
  std::vector<int64_t> input_offset_shape_;
  std::vector<int64_t> output_shape_;
  int64_t input_dim1_;
  bool is_null_input_;
  size_t dims_[kIndex3] = {};
  G offset_ = 0;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_EMBEDDING_LOOKUP_HELPER_H_
