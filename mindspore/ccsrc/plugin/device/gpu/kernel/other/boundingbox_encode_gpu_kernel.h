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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_OTHER_BOUNDINGBOX_ENCODE_GPU_KERNEL_H
#define MINDSPORE_CCSRC_KERNEL_GPU_OTHER_BOUNDINGBOX_ENCODE_GPU_KERNEL_H

#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/boundingbox_encode_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class BoundingBoxEncodeGpuKernelMod : public NativeGpuKernelMod {
 public:
  BoundingBoxEncodeGpuKernelMod() : anchor_size_(0), groundtruth_size_(0), deltas_size_(0), is_null_input_(false) {}
  ~BoundingBoxEncodeGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *anchor_addr = GetDeviceAddress<T>(inputs, 0);
    T *groundtruth_addr = GetDeviceAddress<T>(inputs, 1);
    T *deltas_addr = GetDeviceAddress<T>(outputs, 0);

    if (inputs[0]->size != inputs[1]->size) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', anchor box size must equal with groundtruth box size: " << inputs[1]->size << ", but got "
                    << inputs[0]->size;
      return false;
    }

    const size_t coordinate = 4;
    const size_t block_size = inputs[0]->size / sizeof(T);
    if ((block_size % coordinate) != 0) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << ", the size of the box should be a multiple of 4.";
      return false;
    }

    BoundingBoxEncode(block_size / coordinate, anchor_addr, groundtruth_addr, deltas_addr, means_[0], means_[1],
                      means_[2], means_[3], stds_[0], stds_[1], stds_[2], stds_[3],
                      reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    MS_EXCEPTION_IF_NULL(kernel_node);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 2, but got " << input_num;
    }
    anchor_size_ = sizeof(T);
    groundtruth_size_ = sizeof(T);
    deltas_size_ = sizeof(T);

    auto logits_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto labels_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(logits_shape, kernel_name_, "anchor_box") ||
                     CHECK_SHAPE_NULL(labels_shape, kernel_name_, "groundtruth_box") ||
                     CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < logits_shape.size(); i++) {
      anchor_size_ *= logits_shape[i];
    }

    for (size_t i = 0; i < labels_shape.size(); i++) {
      groundtruth_size_ *= labels_shape[i];
    }

    for (size_t i = 0; i < output_shape.size(); i++) {
      deltas_size_ *= output_shape[i];
    }

    InitSizeLists();

    const size_t coordinate_size = 4;
    auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    auto means = prim->GetAttr("means");
    MS_EXCEPTION_IF_NULL(means);
    if (means->isa<ValueTuple>() || means->isa<ValueList>()) {
      means_ = GetAttr<std::vector<float>>(kernel_node, "means");
    } else if (means->isa<FloatImm>()) {
      float mean = GetAttr<float>(kernel_node, "means");
      for (size_t i = 0; i < coordinate_size; i++) {
        means_.emplace_back(mean);
      }
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', attribute means type is invalid.";
    }
    auto stds = prim->GetAttr("stds");
    MS_EXCEPTION_IF_NULL(stds);
    if (stds->isa<ValueTuple>() || stds->isa<ValueList>()) {
      stds_ = GetAttr<std::vector<float>>(kernel_node, "stds");
    } else if (stds->isa<FloatImm>()) {
      float std = GetAttr<float>(kernel_node, "stds");
      for (size_t i = 0; i < coordinate_size; i++) {
        stds_.emplace_back(std);
      }
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', attribute stds type is invalid.";
    }

    if (means_.size() < coordinate_size || stds_.size() < coordinate_size) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the both size of means or stds cannot be less than 4, but got"
                        << " the size of means: " << means_.size() << ", the size of stds: " << stds_.size();
    }

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(anchor_size_);
    input_size_list_.push_back(groundtruth_size_);
    output_size_list_.push_back(deltas_size_);
  }

 private:
  size_t anchor_size_;
  size_t groundtruth_size_;
  size_t deltas_size_;
  std::vector<float> means_;
  std::vector<float> stds_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_OTHER_BOUNDINGBOX_ENCODE_GPU_KERNEL_H
