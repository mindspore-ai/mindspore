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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_OTHER_IOU_GPU_KERNEL_H
#define MINDSPORE_CCSRC_KERNEL_GPU_OTHER_IOU_GPU_KERNEL_H

#include <vector>
#include <string>
#include "backend/kernel_compiler/gpu/cuda_impl/iou_impl.cuh"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class IOUGpuKernel : public GpuKernel {
 public:
  IOUGpuKernel() : gt_boxes_size_(0), anchor_boxes_size_(0), iou_size_(0), mode_(0), is_null_input_(false) {}

  ~IOUGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    VARIABLE_NOT_USED(workspace);
    T *gt_boxes_addr = GetDeviceAddress<T>(inputs, 0);
    T *anchor_boxes_addr = GetDeviceAddress<T>(inputs, 1);
    T *iou_addr = GetDeviceAddress<T>(outputs, 0);

    const size_t coordinate = 4;
    const size_t block_size_0 = inputs[0]->size / sizeof(T);
    const size_t block_size_1 = inputs[1]->size / sizeof(T);
    if ((block_size_0 % coordinate) != 0 || (block_size_1 % coordinate) != 0) {
      MS_LOG(ERROR) << "The size of the box must be a multiple of 4.";
      return false;
    }

    const size_t input_len_0 = block_size_0 / coordinate;
    const size_t input_len_1 = block_size_1 / coordinate;
    IOU(input_len_0 * input_len_1, gt_boxes_addr, anchor_boxes_addr, iou_addr, mode_, input_len_0,
        reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    MS_EXCEPTION_IF_NULL(kernel_node);
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but IOU needs 2 inputs.";
      return false;
    }
    gt_boxes_size_ = sizeof(T);
    anchor_boxes_size_ = sizeof(T);
    iou_size_ = sizeof(T);

    auto gt_boxes_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto anchor_boxes_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto iou_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ =
      CHECK_NULL_INPUT(gt_boxes_shape) || CHECK_NULL_INPUT(anchor_boxes_shape) || CHECK_NULL_INPUT(iou_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'IOUGpuKernel', input or output is null";
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < gt_boxes_shape.size(); i++) {
      gt_boxes_size_ *= gt_boxes_shape[i];
    }

    for (size_t i = 0; i < anchor_boxes_shape.size(); i++) {
      anchor_boxes_size_ *= anchor_boxes_shape[i];
    }

    for (size_t i = 0; i < iou_shape.size(); i++) {
      iou_size_ *= iou_shape[i];
    }

    InitSizeLists();

    std::string mode = GetAttr<std::string>(kernel_node, "mode");

    if (mode == "iou") {
      mode_ = 0;
    } else if (mode == "iof") {
      mode_ = 1;
    } else {
      MS_LOG(ERROR) << "Mode only support 'iou' or 'iof'.";
      return false;
    }

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(gt_boxes_size_);
    input_size_list_.push_back(anchor_boxes_size_);
    output_size_list_.push_back(iou_size_);
  }

 private:
  size_t gt_boxes_size_;
  size_t anchor_boxes_size_;
  size_t iou_size_;
  size_t mode_;
  bool is_null_input_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_OTHER_IOU_GPU_KERNEL_H
