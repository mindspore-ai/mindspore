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

#ifndef MINDSPORE_CCSRC_KERNEL_PS_ROI_POOLING_GRAD_GPU_KERNEL_H
#define MINDSPORE_CCSRC_KERNEL_PS_ROI_POOLING_GRAD_GPU_KERNEL_H

#include <vector>
#include <functional>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/psroi_pooling_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr int OUT_PUT_SHAPE_SIZE = 4;
constexpr int MAPPING_CHANNEL_SHAPE = 4;
constexpr int ROI_SHAPE_SIZE = 2;
constexpr int MAPPING_CHANNEL_SHAPE_INDEX0 = 0;
constexpr int MAPPING_CHANNEL_SHAPE_INDEX1 = 1;
constexpr int MAPPING_CHANNEL_SHAPE_INDEX2 = 2;
constexpr int MAPPING_CHANNEL_SHAPE_INDEX3 = 3;
constexpr int ROI_SHAPE_INDEX0 = 0;
constexpr int ROI_SHAPE_INDEX1 = 1;

template <typename T>
class PsROIPoolingBackGpuKernelMod : public NativeGpuKernelMod {
 public:
  PsROIPoolingBackGpuKernelMod()
      : batch_size_(0),
        num_rois_(0),
        spatial_scale_(),
        channels_(0),
        height_(0),
        width_(0),
        pooled_height_(0),
        pooled_width_(0),
        out_dim_(0),
        is_null_input_(false),
        dx_size_(0),
        rois_size_(0),
        mapping_channel_size_(0),
        output_size_(0) {}
  ~PsROIPoolingBackGpuKernelMod() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }

    const T *top_diff = GetDeviceAddress<T>(inputs, 0);
    const T *rois = GetDeviceAddress<T>(inputs, 1);
    const int *mapping_channel = GetDeviceAddress<int>(inputs, 2);
    T *bottom_diff = GetDeviceAddress<T>(outputs, 0);
    MS_EXCEPTION_IF_NULL(top_diff);
    MS_EXCEPTION_IF_NULL(rois);
    MS_EXCEPTION_IF_NULL(mapping_channel);
    MS_EXCEPTION_IF_NULL(bottom_diff);

    PSROIPoolBackwardLauncher(top_diff, mapping_channel, batch_size_, num_rois_, spatial_scale_, channels_, height_,
                              width_, pooled_width_, pooled_height_, out_dim_, bottom_diff, rois,
                              reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    // Get primitive args
    batch_size_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("batch_size")));
    num_rois_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("num_rois")));
    spatial_scale_ = static_cast<T>(GetValue<float>(primitive_->GetAttr("spatial_scale")));
    channels_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("channels")));
    height_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("height")));
    width_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("width")));
    pooled_height_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("pooled_height")));
    pooled_width_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("pooled_width")));
    out_dim_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("out_dim")));
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    output_size_list_.clear();
    workspace_size_list_.clear();
    // Get the input shapes
    const auto &dx_shape = inputs[kIndex0]->GetShapeVector();
    const auto &rois_shape = inputs[kIndex1]->GetShapeVector();
    const auto &mapping_channel_shape = inputs[kIndex2]->GetShapeVector();

    is_null_input_ = CHECK_SHAPE_NULL(dx_shape, kernel_name_, "input") ||
                     CHECK_SHAPE_NULL(rois_shape, kernel_name_, "rois") ||
                     CHECK_SHAPE_NULL(mapping_channel_shape, kernel_name_, "map");
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'PsROIPoolingBackGpuKernelMod', input is null.";
      output_size_list_.push_back(output_size_);
      return KRET_UNKNOWN_SHAPE;
    }

    dx_size_ = sizeof(T) * SizeOf(dx_shape);

    if (rois_shape.size() != ROI_SHAPE_SIZE) {
      MS_LOG(EXCEPTION) << "For 'PsROIPoolingFwdGpuKernelMod', the rank of rois_shape must be 2 "
                        << "(number_rois, (bs, xmin, ymin, xmax, ymax)), "
                        << "but got the rank of rois_shape: " << rois_shape.size();
    }
    rois_shape_ = {LongToInt(rois_shape[ROI_SHAPE_INDEX0]), LongToInt(rois_shape[ROI_SHAPE_INDEX1])};
    rois_size_ = LongToSizeClipNeg(rois_shape[ROI_SHAPE_INDEX0] * rois_shape[ROI_SHAPE_INDEX1]) * sizeof(T);

    if (mapping_channel_shape.size() != MAPPING_CHANNEL_SHAPE) {
      MS_LOG(EXCEPTION) << "For 'PsROIPoolingFwdGpuKernelMod', the rank of mapping_channel_shape must be"
                        << "(number_rois, out_dim, height_ width), "
                        << "but got the rank of rois_shape: " << rois_shape.size();
    }
    mapping_channel_shape_ = {static_cast<int>(mapping_channel_shape[MAPPING_CHANNEL_SHAPE_INDEX0]),
                              static_cast<int>(mapping_channel_shape[MAPPING_CHANNEL_SHAPE_INDEX1]),
                              static_cast<int>(mapping_channel_shape[MAPPING_CHANNEL_SHAPE_INDEX2]),
                              static_cast<int>(mapping_channel_shape[MAPPING_CHANNEL_SHAPE_INDEX3])};

    mapping_channel_size_ = static_cast<int>(mapping_channel_shape[MAPPING_CHANNEL_SHAPE_INDEX0]) *
                            static_cast<int>(mapping_channel_shape[MAPPING_CHANNEL_SHAPE_INDEX1]) *
                            static_cast<int>(mapping_channel_shape[MAPPING_CHANNEL_SHAPE_INDEX2]) *
                            static_cast<int>(mapping_channel_shape[MAPPING_CHANNEL_SHAPE_INDEX3]) * sizeof(T);

    // Get output_shape
    output_shape_ = {batch_size_, channels_, height_, width_};
    output_size_ = sizeof(T);
    for (size_t i = 0; i < OUT_PUT_SHAPE_SIZE; i++) {
      output_size_ *= output_shape_[i];
    }
    output_size_list_.push_back(output_size_);
    return KRET_OK;
  }

 private:
  int batch_size_;
  int num_rois_;
  T spatial_scale_;
  int channels_;
  int height_;
  int width_;
  int pooled_height_;
  int pooled_width_;
  int out_dim_;
  bool is_null_input_;

  std::vector<int> dx_shape_;
  std::vector<int> rois_shape_;
  std::vector<int> mapping_channel_shape_;
  std::vector<int> output_shape_;

  size_t dx_size_;
  size_t rois_size_;
  size_t mapping_channel_size_;
  size_t output_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_PS_ROI_POOLING_GRAD_GPU_KERNEL_H
