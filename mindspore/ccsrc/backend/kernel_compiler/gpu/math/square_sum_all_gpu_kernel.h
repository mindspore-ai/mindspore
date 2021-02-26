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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SQUARE_SUM_ALL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SQUARE_SUM_ALL_GPU_KERNEL_H_

#include <memory>
#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/square_sum_all_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class SquareSumAllGpuFwdKernel : public GpuKernel {
 public:
  SquareSumAllGpuFwdKernel() : input_size_(1), is_null_input_(false) {}
  ~SquareSumAllGpuFwdKernel() override {}
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr_0 = GetDeviceAddress<T>(inputs, 0);
    T *input_addr_1 = GetDeviceAddress<T>(inputs, 1);
    T *output_addr_0 = GetDeviceAddress<T>(outputs, 0);
    T *output_addr_1 = GetDeviceAddress<T>(outputs, 1);
    float *ws_addr_0 = GetDeviceAddress<float>(workspace, 0);
    float *ws_addr_1 = GetDeviceAddress<float>(workspace, 1);
    SquareSumAll(input_size_, input_addr_0, input_addr_1, output_addr_0, output_addr_1, ws_addr_0, ws_addr_1,
                 reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "SquareSumAllGpuFwdKernel input is null";
    }
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(sizeof(T));
    output_size_list_.push_back(sizeof(T));
    workspace_size_list_.push_back(sizeof(float));
    workspace_size_list_.push_back(sizeof(float));
  }

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  size_t input_size_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SQUARE_SUM_ALL_GPU_KERNEL_H_
