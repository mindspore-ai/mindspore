/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TILE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TILE_GPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/tile_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class TileGpuKernel : public GpuKernel {
 public:
  TileGpuKernel() { ResetResource(); }
  ~TileGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    size_t *input_shape_ptr = GetDeviceAddress<size_t>(workspace, 0);
    size_t *output_shape_ptr = GetDeviceAddress<size_t>(workspace, 1);
    T *output = GetDeviceAddress<T>(outputs, 0);

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(input_shape_ptr, &input_shape_[0], input_shape_.size() * sizeof(size_t),
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync input_shape_ failed");
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMemcpyAsync(output_shape_ptr, &output_shape_[0], output_shape_.size() * sizeof(size_t),
                      cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync output_shape_ failed");
    CalTile(output_size_, input_size_, shape_size_, input_shape_ptr, output_shape_ptr, input, output,
            reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but Tile needs 1 input.";
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but Tile has 1 output.";
    }
    input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape_) || CHECK_NULL_INPUT(output_shape_);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'TileGpuKernel', input or output is null";
      InitSizeLists();
      return true;
    }
    if (output_shape_.size() < 1) {
      MS_LOG(EXCEPTION) << "For 'TileGpuKernel', the rank of output cannot be less than 1, but got "
                        << output_shape_.size();
    }
    input_size_ = 1;
    for (size_t i = 0; i < input_shape_.size(); i++) {
      input_size_ *= input_shape_[i];
    }

    output_size_ = 1;
    if (output_shape_.size() > TILE_MAX_DIMENSION) {
      MS_LOG(EXCEPTION) << "Output is " << output_shape_.size() << "-D, but Tile supports up to " << TILE_MAX_DIMENSION
                        << "-D.";
    }
    shape_size_ = output_shape_.size();
    for (size_t i = 0; i < output_shape_.size(); i++) {
      output_size_ *= output_shape_[i];
    }
    std::vector<int64_t> multiples = GetAttr<std::vector<int64_t>>(kernel_node, "multiples");
    int64_t filling_value = static_cast<int64_t>(multiples.size()) - static_cast<int64_t>(input_shape_.size());
    // input_shape_.size() == output_shape_.size() == shape_size_
    (void)input_shape_.insert(input_shape_.begin(), LongToSize(filling_value), 1);
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 1;
    output_size_ = 1;
    shape_size_ = 1;
    is_null_input_ = false;
    input_shape_.clear();
    output_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    workspace_size_list_.push_back(input_shape_.size() * sizeof(size_t));
    workspace_size_list_.push_back(output_shape_.size() * sizeof(size_t));
    output_size_list_.push_back(output_size_ * sizeof(T));
  }

 private:
  size_t input_size_;
  size_t output_size_;
  size_t shape_size_;
  bool is_null_input_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TILE_GPU_KERNEL_H_
