/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_TOPK_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_TOPK_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/topk_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class TopKGpuKernel : public GpuKernel {
 public:
  TopKGpuKernel() : sorted_(false), outer_size_(1), inner_size_(1), k_(1), use_share_mem_(true), ceil_power2_(0) {}
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
    T *data_buff = nullptr;
    S *index_buff = nullptr;
    if (use_share_mem_ == false) {
      data_buff = GetDeviceAddress<T>(workspaces, 0);
      index_buff = GetDeviceAddress<S>(workspaces, 1);
    }

    TopK(outer_size_, inner_size_, input_addr, k, output_addr, indices, data_buff, index_buff,
         reinterpret_cast<cudaStream_t>(stream_ptr));

    if (sorted_ == false) {
      BitonicSortByKey(outer_size_, k_, output_addr, indices, data_buff, index_buff,
                       reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto input_shapes = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shapes = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < input_shapes.size() - 1; i++) {
      outer_size_ *= input_shapes[i];
    }
    inner_size_ = input_shapes[input_shapes.size() - 1];
    k_ = output_shapes[output_shapes.size() - 1];

    sorted_ = GetAttr<bool>(kernel_node, "sorted");

    ceil_power2_ = RoundUpPower2(inner_size_);
    size_t buffer_size = ceil_power2_ * (sizeof(T) + sizeof(S));
    if (buffer_size > SHARED_MEM_PER_BLOCK) {
      use_share_mem_ = false;
      MS_LOG(INFO) << "CUDA share memory not enough, sort with RAM";
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(outer_size_ * inner_size_ * sizeof(T));
    input_size_list_.push_back(sizeof(S));
    output_size_list_.push_back(outer_size_ * k_ * sizeof(T));
    output_size_list_.push_back(outer_size_ * k_ * sizeof(S));
    if (use_share_mem_ == false) {
      workspace_size_list_.push_back(outer_size_ * ceil_power2_ * sizeof(T));
      workspace_size_list_.push_back(outer_size_ * ceil_power2_ * sizeof(S));
    }
  }

 private:
  bool sorted_;
  size_t outer_size_;
  size_t inner_size_;
  size_t k_;
  bool use_share_mem_;
  size_t ceil_power2_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // TopKpuKernel
