/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_PS_ROI_POOLING_GPU_KERNEL_H
#define MINDSPORE_CCSRC_KERNEL_GPU_PS_ROI_POOLING_GPU_KERNEL_H

#include <vector>
#include <functional>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/psroi_pooling_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr int OUT_PUT_SHAPE_SIZE = 4;
constexpr int X_SHAPE_SIZE = 4;
constexpr int ROI_SHAPE_SIZE = 2;
constexpr int X_SHAPE_INDEX0 = 0;
constexpr int X_SHAPE_INDEX1 = 1;
constexpr int X_SHAPE_INDEX2 = 2;
constexpr int X_SHAPE_INDEX3 = 3;
constexpr int ROI_SHAPE_INDEX0 = 0;
constexpr int ROI_SHAPE_INDEX1 = 1;

template <typename T>
class PsROIPoolingFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  PsROIPoolingFwdGpuKernelMod()
      : pooled_height_(0),
        pooled_width_(0),
        group_size_(0),
        spatial_scale_(),
        out_dim_(0),
        channels_(0),
        height_(0),
        width_(0),
        num_rois_(0),
        is_null_input_(false),
        x_size_(0),
        rois_size_(0),
        output_size_(0) {}
  ~PsROIPoolingFwdGpuKernelMod() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    const T *x = GetDeviceAddress<T>(inputs, 0);
    const T *rois = GetDeviceAddress<T>(inputs, 1);

    T *out_data = GetDeviceAddress<T>(outputs, 0);
    int *out_mapping_channel = GetDeviceAddress<int>(outputs, 1);
    MS_EXCEPTION_IF_NULL(x);
    MS_EXCEPTION_IF_NULL(rois);
    MS_EXCEPTION_IF_NULL(out_data);
    MS_EXCEPTION_IF_NULL(out_mapping_channel);

    auto status = PSROIPoolForwardLauncher(x, spatial_scale_, num_rois_, height_, width_, channels_, pooled_height_,
                                           pooled_width_, rois, group_size_, out_dim_, out_data, out_mapping_channel,
                                           reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    // Get primitive args
    pooled_height_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("pooled_height")));
    pooled_width_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("pooled_width")));
    num_rois_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("num_rois")));
    spatial_scale_ = static_cast<T>(GetValue<float>(primitive_->GetAttr("spatial_scale")));
    out_dim_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("out_dim")));
    group_size_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("group_size")));
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    output_size_list_.clear();
    workspace_size_list_.clear();
    // Get the input shapes
    const auto &x_shape = inputs[kIndex0]->GetShapeVector();
    const auto &rois_shape = inputs[kIndex1]->GetShapeVector();
    is_null_input_ =
      CHECK_SHAPE_NULL(x_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(rois_shape, kernel_name_, "roi");
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'PsROIPoolingFwdGpuKernelMod', input is null.";
      output_size_list_.push_back(output_size_);
      output_size_list_.push_back(out_mapping_channel_size_);
      return KRET_UNKNOWN_SHAPE;
    }

    auto x_shape_size = x_shape.size();
    if (x_shape_size != X_SHAPE_SIZE) {
      MS_LOG(ERROR) << "x shape size is " << x_shape_size << ", but must be 4.";
      return KRET_RESIZE_FAILED;
    }

    // Get channels, height & width
    int batch_size = x_shape[X_SHAPE_INDEX0];
    channels_ = x_shape[X_SHAPE_INDEX1];
    height_ = x_shape[X_SHAPE_INDEX2];
    width_ = x_shape[X_SHAPE_INDEX3];
    x_shape_ = {batch_size, channels_, height_, width_};
    x_size_ = batch_size * channels_ * height_ * width_ * sizeof(T);

    if (rois_shape.size() != ROI_SHAPE_SIZE) {
      MS_LOG(EXCEPTION) << "For 'PsROIPoolingFwdGpuKernelMod', the rank of rois_shape must be 2 "
                        << "(number_rois, (bs, xmin, ymin, xmax, ymax)), "
                        << "but got the rank of rois_shape: " << rois_shape.size();
    }
    rois_size_ = LongToSizeClipNeg(rois_shape[ROI_SHAPE_INDEX0] * rois_shape[ROI_SHAPE_INDEX1]) * sizeof(T);
    rois_shape_ = {LongToInt(rois_shape[ROI_SHAPE_INDEX0]), LongToInt(rois_shape[ROI_SHAPE_INDEX1])};

    // Get output_shape
    output_shape_ = {num_rois_, out_dim_, pooled_height_, pooled_width_};
    output_size_ = sizeof(T);
    for (size_t i = 0; i < OUT_PUT_SHAPE_SIZE; i++) {
      output_size_ *= output_shape_[i];
    }

    // map_channel的output size
    out_mapping_channel_shape_ = {num_rois_, out_dim_, pooled_height_, pooled_width_};
    out_mapping_channel_size_ = num_rois_ * out_dim_ * pooled_height_ * pooled_width_ * sizeof(int);

    output_size_list_.push_back(output_size_);
    output_size_list_.push_back(out_mapping_channel_size_);
    return KRET_OK;
  }

 private:
  int pooled_height_;
  int pooled_width_;
  int group_size_;
  T spatial_scale_;
  int out_dim_;
  int channels_;
  int height_;
  int width_;
  int num_rois_;
  bool is_null_input_;

  std::vector<int> x_shape_;
  std::vector<int> rois_shape_;
  std::vector<int> output_shape_;
  std::vector<int> out_mapping_channel_shape_;

  size_t x_size_;
  size_t rois_size_;
  size_t output_size_;
  size_t out_mapping_channel_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_PS_ROI_POOING_GPU_KERNEL_H
