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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_OTHER_CHECK_VALID_GPU_KERNEL_H
#define MINDSPORE_CCSRC_KERNEL_GPU_OTHER_CHECK_VALID_GPU_KERNEL_H

#include <vector>
#include "backend/kernel_compiler/gpu/cuda_impl/check_valid_impl.cuh"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class CheckValidGpuKernel : public GpuKernel {
 public:
  CheckValidGpuKernel() : anchor_boxes_size_(0), img_metas_size_(0), valid_size_(0), is_null_input_(false) {}

  ~CheckValidGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    VARIABLE_NOT_USED(workspace);
    T *anchor_boxes_addr = GetDeviceAddress<T>(inputs, 0);
    T *img_metas_addr = GetDeviceAddress<T>(inputs, 1);
    S *valid_addr = GetDeviceAddress<S>(outputs, 0);

    const size_t coordinate = 4;
    const size_t block_size = inputs[0]->size / sizeof(T);
    if ((block_size % coordinate) != 0) {
      MS_LOG(ERROR) << "The size of the box must be a multiple of 4.";
      return false;
    }

    const size_t size = block_size / coordinate;
    CheckValid(size, anchor_boxes_addr, img_metas_addr, valid_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    MS_EXCEPTION_IF_NULL(kernel_node);
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but CheckValid needs 2 inputs.";
      return false;
    }
    anchor_boxes_size_ = sizeof(T);
    img_metas_size_ = sizeof(T);
    valid_size_ = sizeof(S);

    auto anchor_boxes_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto img_metas_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto valid_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ =
      CHECK_NULL_INPUT(anchor_boxes_shape) || CHECK_NULL_INPUT(img_metas_shape) || CHECK_NULL_INPUT(valid_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'CheckValidGpuKernel', input or output is null";
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < anchor_boxes_shape.size(); i++) {
      anchor_boxes_size_ *= anchor_boxes_shape[i];
    }

    for (size_t i = 0; i < img_metas_shape.size(); i++) {
      img_metas_size_ *= img_metas_shape[i];
    }

    for (size_t i = 0; i < valid_shape.size(); i++) {
      valid_size_ *= valid_shape[i];
    }

    InitSizeLists();

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(anchor_boxes_size_);
    input_size_list_.push_back(img_metas_size_);
    output_size_list_.push_back(valid_size_);
  }

 private:
  size_t anchor_boxes_size_;
  size_t img_metas_size_;
  size_t valid_size_;
  bool is_null_input_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_OTHER_CHECK_VALID_GPU_KERNEL_H
