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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_OTHER_CHECK_VALID_GPU_KERNEL_H
#define MINDSPORE_CCSRC_KERNEL_GPU_OTHER_CHECK_VALID_GPU_KERNEL_H

#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/check_valid_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class CheckValidGpuKernelMod : public NativeGpuKernelMod {
 public:
  CheckValidGpuKernelMod() : anchor_boxes_size_(0), img_metas_size_(0), valid_size_(0), is_null_input_(false) {}
  ~CheckValidGpuKernelMod() override = default;

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
      MS_LOG(ERROR) << "For '" << kernel_name_ << ", the size of the box should be a multiple of 4.";
      return false;
    }

    const size_t size = block_size / coordinate;
    CheckValid(size, anchor_boxes_addr, img_metas_addr, valid_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    MS_EXCEPTION_IF_NULL(kernel_node);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 2, but got " << input_num;
    }
    anchor_boxes_size_ = sizeof(T);
    img_metas_size_ = sizeof(T);
    valid_size_ = sizeof(S);

    auto anchor_boxes_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto img_metas_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto valid_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(anchor_boxes_shape, kernel_name_, "bboxes") ||
                     CHECK_SHAPE_NULL(img_metas_shape, kernel_name_, "img_metas") ||
                     CHECK_SHAPE_NULL(valid_shape, kernel_name_, "output");
    if (is_null_input_) {
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
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_OTHER_CHECK_VALID_GPU_KERNEL_H
