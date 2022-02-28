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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_ROI_ALIGN_GRAD_GPU_KERNEL_H
#define MINDSPORE_CCSRC_KERNEL_GPU_ROI_ALIGN_GRAD_GPU_KERNEL_H

#include <vector>
#include <algorithm>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/roi_align_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class ROIAlignGradFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  ROIAlignGradFwdGpuKernelMod()
      : pooled_height_(0),
        pooled_width_(0),
        sample_num_(0),
        roi_end_mode_(0),
        roi_rows_(0),
        roi_cols_(0),
        batch_size_(0),
        channels_(0),
        width_(0),
        is_null_input_(false),
        dy_size_(0),
        rois_size_(0),
        output_size_(0) {}
  ~ROIAlignGradFwdGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    const T *dy = GetDeviceAddress<T>(inputs, 0);
    const T *rois = GetDeviceAddress<T>(inputs, 1);

    T *dx = GetDeviceAddress<T>(outputs, 0);

    ROIAlignGrad(dy, rois, batch_size_, roi_rows_, roi_cols_, dx, spatial_scale_, sample_num_, roi_end_mode_, channels_,
                 height_, width_, pooled_height_, pooled_width_, reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    // Get the number of input args
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 2, but got " << input_num;
    }

    // Get the number of output args
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
    }

    // Get the input shapes
    auto dy_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto rois_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    is_null_input_ = CHECK_SHAPE_NULL(dy_shape, kernel_name, "dy") || CHECK_SHAPE_NULL(rois_shape, kernel_name, "rois");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }

    auto dy_shape_size = dy_shape.size();
    if (dy_shape_size != 4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of dy should be equal to 4, but got "
                        << dy_shape_size;
    }

    // Parse y diff
    dy_shape_ = {static_cast<int>(dy_shape[0]), static_cast<int>(dy_shape[1]), static_cast<int>(dy_shape[2]),
                 static_cast<int>(dy_shape[3])};
    dy_size_ = dy_shape_[0] * dy_shape_[1] * dy_shape_[2] * dy_shape_[3] * sizeof(T);

    if (rois_shape.size() < 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of rois cannot be less than 2, but got "
                        << rois_shape.size();
    }
    // Get rois rows and cols
    roi_rows_ = rois_shape[0];
    roi_cols_ = rois_shape[1];
    rois_shape_ = {roi_rows_, roi_cols_};
    rois_size_ = roi_rows_ * roi_cols_ * sizeof(T);

    // Get primitive args
    std::vector<int64_t> xdiff_shape_me = GetAttr<std::vector<int64_t>>(kernel_node, "xdiff_shape");
    (void)std::transform(xdiff_shape_me.begin(), xdiff_shape_me.end(), std::back_inserter(xdiff_shape_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    pooled_height_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "pooled_height"));
    pooled_width_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "pooled_width"));
    spatial_scale_ = static_cast<T>(GetAttr<float>(kernel_node, "spatial_scale"));
    sample_num_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "sample_num"));
    roi_end_mode_ = 1;

    if (xdiff_shape_.size() < 4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the length of xdiff_shape cannot be less than 4, but got "
                        << xdiff_shape_.size();
    }
    // Get channels, height & width
    batch_size_ = xdiff_shape_[0];
    channels_ = xdiff_shape_[1];
    height_ = xdiff_shape_[2];
    width_ = xdiff_shape_[3];

    // Get output_shape
    output_shape_ = {batch_size_, channels_, height_, width_};
    output_size_ = batch_size_ * channels_ * height_ * width_ * sizeof(T);

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(dy_size_);
    input_size_list_.push_back(rois_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  std::vector<int> xdiff_shape_;
  int pooled_height_;
  int pooled_width_;
  T spatial_scale_;
  int sample_num_;
  int roi_end_mode_;

  int roi_rows_;
  int roi_cols_;
  int batch_size_;
  int channels_;
  int height_;
  int width_;
  bool is_null_input_;

  std::vector<int> dy_shape_;
  std::vector<int> rois_shape_;
  std::vector<int> output_shape_;

  size_t dy_size_;
  size_t rois_size_;
  size_t output_size_;
};  // namespace kernel
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_ROI_ALIGN_GRAD_GPU_KERNEL_H
