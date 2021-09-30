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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_ROI_ALIGN_GPU_KERNEL_H
#define MINDSPORE_CCSRC_KERNEL_GPU_ROI_ALIGN_GPU_KERNEL_H

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/roi_align_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class ROIAlignGpuFwdKernel : public GpuKernel {
 public:
  ROIAlignGpuFwdKernel()
      : pooled_height_(0),
        pooled_width_(0),
        spatial_scale_(),
        sample_num_(0),
        roi_end_mode_(0),
        roi_rows_(0),
        roi_cols_(0),
        batch_N_(0),
        channels_(0),
        height_(0),
        width_(0),
        is_null_input_(false),
        x_size_(0),
        rois_size_(0),
        output_size_(0) {}
  ~ROIAlignGpuFwdKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    const T *x = GetDeviceAddress<T>(inputs, 0);
    const T *rois = GetDeviceAddress<T>(inputs, 1);

    T *out_data = GetDeviceAddress<T>(outputs, 0);

    ROIAlign(x, rois, roi_rows_, roi_cols_, out_data, spatial_scale_, sample_num_, roi_end_mode_, channels_, height_,
             width_, pooled_height_, pooled_width_, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    // Get the number of input args
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but ROIAlign needs 2 input.";
      return false;
    }

    // Get the number of output args
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but ROIAlign needs 1 output.";
      return false;
    }

    // Get the input shapes
    auto x_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto rois_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    is_null_input_ = CHECK_NULL_INPUT(x_shape) || CHECK_NULL_INPUT(rois_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'ROIAlignGpuKernel', input is null.";
      InitSizeLists();
      return true;
    }

    auto x_shape_size = x_shape.size();
    if (x_shape_size != 4) {
      MS_LOG(ERROR) << "x shape size is " << x_shape_size << ", but should be 4.";
      return false;
    }

    // Get channels, height & width
    batch_N_ = x_shape[0];
    channels_ = x_shape[1];
    height_ = x_shape[2];
    width_ = x_shape[3];
    x_shape_ = {batch_N_, channels_, height_, width_};
    x_size_ = batch_N_ * channels_ * height_ * width_ * sizeof(T);

    if (rois_shape.size() < 2) {
      MS_LOG(EXCEPTION) << "For 'ROIAlignGpuKernel', the rank of rois_shape should be greater than or equal to 2, "
                        << "but got the rank of rois_shape: " << rois_shape.size();
    }
    // Get rois rows and cols
    roi_rows_ = rois_shape[0];
    roi_cols_ = rois_shape[1];
    rois_size_ = roi_rows_ * roi_cols_ * sizeof(T);
    rois_shape_ = {roi_rows_, roi_cols_};

    // Get primitive args
    pooled_height_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "pooled_height"));
    pooled_width_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "pooled_width"));
    spatial_scale_ = static_cast<T>(GetAttr<float>(kernel_node, "spatial_scale"));
    sample_num_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "sample_num"));
    roi_end_mode_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "roi_end_mode"));

    // Get output_shape
    output_shape_ = {roi_rows_, channels_, pooled_height_, pooled_width_};
    output_size_ = 1;
    for (size_t i = 0; i < 4; i++) {
      output_size_ *= output_shape_[i];
    }
    output_size_ *= sizeof(T);

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(x_size_);
    input_size_list_.push_back(rois_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  int pooled_height_;
  int pooled_width_;
  T spatial_scale_;
  int sample_num_;
  int roi_end_mode_;

  int roi_rows_;
  int roi_cols_;
  int batch_N_;
  int channels_;
  int height_;
  int width_;
  bool is_null_input_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  std::vector<int> x_shape_;
  std::vector<int> rois_shape_;
  std::vector<int> output_shape_;

  size_t x_size_;
  size_t rois_size_;
  size_t output_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_ROI_ALIGN_GPU_KERNEL_H
