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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TOPK_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TOPK_GPU_KERNEL_H_

#include <limits>
#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/topk_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class TopKGpuKernel : public GpuKernel {
 public:
  TopKGpuKernel() : sorted_(false), outer_size_(1), inner_size_(1), k_(1), input_shape_size_(0) {}
  ~TopKGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    S *k = GetDeviceAddress<S>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    S *indices = GetDeviceAddress<S>(outputs, 1);
    const T init_k = std::numeric_limits<T>::lowest();
    S k_cut = 0;
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMemcpyAsync(&k_cut, k, sizeof(S), cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync k_cut failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaDeviceSynchronize(), "cudaDeviceSyncFailed - TopK");
    FastTopK(outer_size_, inner_size_, input_addr, k_cut, output_addr, indices, init_k,
             reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    auto input_shapes = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shapes = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    input_shape_size_ = input_shapes.size();
    for (size_t i = 0; i < input_shapes.size() - 1; i++) {
      outer_size_ *= input_shapes[i];
    }
    inner_size_ = input_shapes[input_shapes.size() - 1];
    k_ = output_shapes[output_shapes.size() - 1];

    sorted_ = GetAttr<bool>(kernel_node, "sorted");

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(outer_size_ * inner_size_ * sizeof(T));
    input_size_list_.push_back(sizeof(S));
    output_size_list_.push_back(outer_size_ * k_ * sizeof(T));
    output_size_list_.push_back(outer_size_ * k_ * sizeof(S));
  }

 private:
  bool sorted_;
  size_t outer_size_;
  size_t inner_size_;
  size_t k_;
  int input_shape_size_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TOPK_GPU_KERNEL_H_
