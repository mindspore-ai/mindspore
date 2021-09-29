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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_CROP_AND_RESIZE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_CROP_AND_RESIZE_GPU_KERNEL_H_

#include <vector>
#include <string>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/crop_and_resize_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class CropAndResizeGpuKernel : public GpuKernel {
 public:
  CropAndResizeGpuKernel()
      : method_(0),
        extrapolation_value_(0),
        input_image_size_(0),
        input_boxes_size_(0),
        input_box_ind_size_(0),
        input_crop_size_(0),
        output_size_(0),
        batch_(0),
        input_height_(0),
        input_width_(0),
        final_height_(0),
        final_width_(0),
        channel_(0),
        is_null_input_(false) {}
  ~CropAndResizeGpuKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    VARIABLE_NOT_USED(workspace);
    T *input_image = GetDeviceAddress<T>(inputs, 0);
    float *input_boxes = GetDeviceAddress<float>(inputs, 1);
    int *input_box_index = GetDeviceAddress<int>(inputs, 2);
    float *output = GetDeviceAddress<float>(outputs, 0);
    size_t size = output_size_ / sizeof(float);
    CalCropAndResize(size, input_image, input_boxes, input_box_index, batch_, input_height_, input_width_,
                     final_height_, final_width_, channel_, method_, extrapolation_value_, output,
                     reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 4) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but CropAndResize needs 4 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but CropAndResize has 1 output.";
      return false;
    }
    // input image
    auto input_image_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto input_boxes_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto input_box_index_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto input_crop_size_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_image_shape) || CHECK_NULL_INPUT(input_boxes_shape) ||
                     CHECK_NULL_INPUT(input_box_index_shape) || CHECK_NULL_INPUT(input_crop_size_shape) ||
                     CHECK_NULL_INPUT(output_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'CropAndResizeGpuKernel', input or output is null.";
      InitSizeLists();
      return true;
    }
    size_t input_image_shape_len = input_image_shape.size();
    if (input_image_shape_len != 4) {
      MS_LOG(ERROR) << " image tensor is " << input_image_shape_len << "-D, but CropAndResize supports only " << 4
                    << "-D image tensors.";
      return false;
    }
    input_image_size_ = 1;
    for (size_t i = 0; i < input_image_shape_len; i++) {
      input_image_size_ *= input_image_shape[i];
    }
    input_image_size_ *= sizeof(T);
    input_height_ = input_image_shape[1];
    input_width_ = input_image_shape[2];
    // input boxes
    size_t input_boxes_shape_len = input_boxes_shape.size();
    if (input_boxes_shape_len != 2) {
      MS_LOG(ERROR) << "Boxes is rank" << input_boxes_shape_len << " but CropAndResize supports only rank " << 2
                    << " for boxes.";
      return false;
    }
    input_boxes_size_ = 1;
    for (size_t i = 0; i < input_boxes_shape_len; i++) {
      input_boxes_size_ *= input_boxes_shape[i];
    }
    input_boxes_size_ *= sizeof(float);
    // input box_index
    size_t input_box_index_shape_len = input_box_index_shape.size();
    if (input_box_index_shape_len != 1) {
      MS_LOG(ERROR) << "Box_index is rank " << input_box_index_shape_len << " but CropAndResize supports only rank "
                    << 1 << " for box_index.";
      return false;
    }
    input_box_ind_size_ = 1;
    input_box_ind_size_ *= input_box_index_shape[0];  // single dim required
    input_box_ind_size_ *= sizeof(int);
    // input crop_size
    size_t input_crop_size_shape_len = input_crop_size_shape.size();
    if (input_crop_size_shape_len != 1) {
      MS_LOG(ERROR) << "Crop_size is rank " << input_crop_size_shape_len << "-D, but CropAndResize supports only rank "
                    << 1 << " for Crop_size.";
      return false;
    }
    if (input_crop_size_shape[0] != 2) {
      MS_LOG(ERROR) << "Crop_size is size " << input_crop_size_shape[0] << "-D, but CropAndResize supports only size "
                    << 2 << " for Crop_size.";
      return false;
    }
    input_crop_size_ = 1;
    input_crop_size_ *= input_crop_size_shape[0];
    input_crop_size_ *= sizeof(int);
    // output
    auto output_shape_len = output_shape.size();
    if (output_shape_len != 4) {
      MS_LOG(ERROR) << "For 'CropAndResize', the rank of output should be 4, but got " << output_shape_len;
      return false;
    }
    output_size_ = 1;
    for (size_t i = 0; i < output_shape_len; i++) {
      output_size_ *= output_shape[i];
    }
    output_size_ *= sizeof(float);
    // set expected output params
    batch_ = output_shape[0];
    final_height_ = output_shape[1];
    final_width_ = output_shape[2];
    channel_ = output_shape[3];
    // get op parameters
    string method = GetAttr<string>(kernel_node, "method");
    if (method == "bilinear") {
      method_ = 1;
    } else if (method == "nearest") {
      method_ = 2;
    } else {  // bilinear-v2
      method_ = 3;
    }
    extrapolation_value_ = GetAttr<float>(kernel_node, "extrapolation_value");
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_image_size_);
    input_size_list_.push_back(input_boxes_size_);
    input_size_list_.push_back(input_box_ind_size_);
    input_size_list_.push_back(input_crop_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  int method_;
  float extrapolation_value_;
  int input_image_size_;
  int input_boxes_size_;
  int input_box_ind_size_;
  int input_crop_size_;
  int output_size_;
  int batch_;
  int input_height_;
  int input_width_;
  int final_height_;
  int final_width_;
  int channel_;
  bool is_null_input_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_CROP_AND_RESIZE_GPU_KERNEL_H_
