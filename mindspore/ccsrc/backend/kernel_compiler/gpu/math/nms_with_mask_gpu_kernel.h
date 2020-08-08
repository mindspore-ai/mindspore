/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NMS_WITH_MASK_IMPL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NMS_WITH_MASK_IMPL_H_

#include <vector>
#include <memory>
#include <iostream>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/nms_with_mask_impl.cuh"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class NMSWithMaskGpuFwdKernel : public GpuKernel {
 public:
  NMSWithMaskGpuFwdKernel() : num_input_(0), iou_value_(0.5), input_size_(0), output_size_(0), workspace_size_(0) {}
  ~NMSWithMaskGpuFwdKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *data_buff = GetDeviceAddress<T>(workspace, 0);  // sort buffer
    int *index_buff = GetDeviceAddress<int>(workspace, 1);
    T *area = GetDeviceAddress<T>(workspace, 2);  // store area values for all boxes
    T *output = GetDeviceAddress<T>(outputs, 0);
    int *sel_idx = GetDeviceAddress<int>(outputs, 1);
    bool *sel_boxes = GetDeviceAddress<bool>(outputs, 2);

    BitonicSortByKeyM(num_input_, num_input_, input, output, index_buff, data_buff, box_size_,
                      reinterpret_cast<cudaStream_t>(stream_ptr));
    CalPreprocess(num_input_, sel_idx, area, output, box_size_, reinterpret_cast<cudaStream_t>(stream_ptr));
    CalNMSWithMask(num_input_, iou_value_, output, area, sel_boxes, box_size_,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    CalFinalPass(num_input_, iou_value_, output, area, sel_boxes, box_size_,
                 reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    iou_value_ = GetAttr<float>(kernel_node, "iou_threshold");

    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but NMSWithMask needs 1 input.";
      return false;
    }

    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 3) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but NMSWithMask needs 3 output.";
      return false;
    }

    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (CHECK_NULL_INPUT(input_shape)) {
      MS_LOG(WARNING) << "NMSWithMask input is null";
      InitSizeLists();
      return true;
    }

    num_input_ = input_shape[0];  // Get N value in [N,5] data

    input_size_ = num_input_ * sizeof(T) * box_size_;  // 5 values per bbox
    output_size_ = (input_size_) + (num_input_ * sizeof(int)) + (num_input_ * sizeof(bool));
    workspace_size_ = (2 * num_input_ * sizeof(T)) + (1 * num_input_ * sizeof(int));

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    // N sized input/output data
    input_size_list_.push_back(num_input_ * sizeof(T) * box_size_);
    output_size_list_.push_back(num_input_ * sizeof(T) * box_size_);
    output_size_list_.push_back(num_input_ * sizeof(int));
    output_size_list_.push_back(num_input_ * sizeof(bool));

    // N sized workspace arrs
    workspace_size_list_.push_back(num_input_ * sizeof(T));
    workspace_size_list_.push_back(num_input_ * sizeof(int));
    workspace_size_list_.push_back(num_input_ * sizeof(T));
  }

 private:
  int num_input_;
  float iou_value_;
  static const int box_size_ = 5;  // pre_defined box width
  // int box_size__ = 5; // current size of bboxes
  // default values
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NMS_WITH_MASK_IMPL_H_
