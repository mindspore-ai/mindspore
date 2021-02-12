/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHERND_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHERND_GPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/gathernd.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class GatherNdGpuFwdKernel : public GpuKernel {
 public:
  GatherNdGpuFwdKernel() : dev_batch_strides_(nullptr), dev_batch_indices_(nullptr), memcpy_flag_(false) {}
  ~GatherNdGpuFwdKernel() {
    if (dev_batch_strides_ != nullptr) {
      device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(static_cast<void *>(dev_batch_strides_));
    }
    if (dev_batch_indices_ != nullptr) {
      device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(static_cast<void *>(dev_batch_indices_));
    }
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    S *indices_addr = GetDeviceAddress<S>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    if (!memcpy_flag_) {
      const size_t strides_len = sizeof(S) * batch_strides_.size();
      const size_t indices_len = sizeof(S) * batch_indices_.size();
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(dev_batch_strides_, &batch_strides_[0], strides_len,
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync failed in GatherNdGpuFwdKernel::Launch.");
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(dev_batch_indices_, &batch_indices_[0], indices_len,
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync failed in GatherNdGpuFwdKernel::Launch.");
      memcpy_flag_ = true;
    }

    GatherNd(input_addr, indices_addr, output_addr, dims_[0], dims_[1], dims_[2], dev_batch_strides_,
             dev_batch_indices_, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    InitResource();
    memcpy_flag_ = false;
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "Argument number is " << input_num << ", but GatherNdGpuFwdKernel needs 2.";
    }
    input_shapes_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    indices_shapes_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    output_shapes_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);

    Reshape();

    size_t dim_indices_last = dims_[dims_.size() - 1];
    batch_strides_.resize(dim_indices_last, 0);
    batch_indices_.resize(dim_indices_last, 0);

    if (dim_indices_last > 0) {
      batch_strides_[dim_indices_last - 1] = input_shapes_[dim_indices_last - 1];
      batch_indices_[dim_indices_last - 1] = dims_[1];
    }
    for (size_t i = dim_indices_last - 1; i > 0; --i) {
      batch_strides_[i - 1] = input_shapes_[i - 1];
      batch_indices_[i - 1] = batch_indices_[i] * input_shapes_[i];
    }

    const size_t strides_len = sizeof(S) * batch_strides_.size();
    void *dev_batch_strides_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(strides_len);
    if (dev_batch_strides_work == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to alloc dev_batch_strides_work, size: " << strides_len;
    }
    dev_batch_strides_ = static_cast<S *>(dev_batch_strides_work);

    const size_t indices_len = sizeof(S) * batch_indices_.size();
    void *dev_batch_indices_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(indices_len);
    if (dev_batch_indices_work == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to alloc dev_batch_indices_work, size: " << indices_len;
    }
    dev_batch_indices_ = static_cast<S *>(dev_batch_indices_work);

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    size_t size = GetSize(input_shapes_);
    input_size_list_.push_back(size);

    size = GetSize(indices_shapes_);
    input_size_list_.push_back(size);

    size = GetSize(output_shapes_);
    output_size_list_.push_back(size);
  }

 private:
  void Reshape() {
    size_t dim_of_indices = 1;
    for (size_t i = 0; i < indices_shapes_.size() - IntToSize(1); i++) {
      dim_of_indices *= indices_shapes_[i];
    }

    size_t dim_after_indices = 1;
    size_t dim_indices_last = indices_shapes_[indices_shapes_.size() - IntToSize(1)];
    for (size_t i = dim_indices_last; i < input_shapes_.size(); i++) {
      dim_after_indices *= input_shapes_[i];
    }
    dims_.emplace_back(dim_of_indices);
    dims_.emplace_back(dim_after_indices);
    dims_.emplace_back(dim_indices_last);
    return;
  }
  size_t GetSize(const std::vector<size_t> &shape) const {
    if (shape.size() == 0) {
      return 0;
    }
    size_t result = sizeof(T);
    for (size_t i = 0; i < shape.size(); i++) {
      result *= shape[i];
    }
    return result;
  }

  std::vector<size_t> input_shapes_;
  std::vector<size_t> indices_shapes_;
  std::vector<size_t> output_shapes_;

  std::vector<size_t> dims_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  std::vector<S> batch_strides_;
  std::vector<S> batch_indices_;

  S *dev_batch_strides_;
  S *dev_batch_indices_;
  bool memcpy_flag_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHERND_GPU_KERNEL_H_
