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
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/psroi_pooling_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr int INPUT_NUM = 3;
constexpr int OUTPUT_NUM = 1;
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
class PsROIPoolingBackGpuKernelMod : public DeprecatedNativeGpuKernelMod {
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

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
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

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    // Get the number of input args
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != INPUT_NUM) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but PsROIPoolingBackGpuKernelMod needs 3 input.";
      return false;
    }

    // Get the number of output args
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != OUTPUT_NUM) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but PsROIPoolingBackGpuKernelMod needs 1 output.";
      return false;
    }

    // Get the input shapes
    auto dx_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto rois_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto mapping_channel_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);

    is_null_input_ = CHECK_SHAPE_NULL(dx_shape, kernel_name, "input") ||
                     CHECK_SHAPE_NULL(rois_shape, kernel_name, "rois") ||
                     CHECK_SHAPE_NULL(mapping_channel_shape, kernel_name, "map");
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'PsROIPoolingBackGpuKernelMod', input is null.";
      InitSizeLists();
      return true;
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

    // Get primitive args
    batch_size_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "batch_size"));
    num_rois_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "num_rois"));
    spatial_scale_ = static_cast<T>(GetAttr<float>(kernel_node, "spatial_scale"));
    channels_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "channels"));
    height_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "height"));
    width_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "width"));
    pooled_height_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "pooled_height"));
    pooled_width_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "pooled_width"));
    out_dim_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "out_dim"));

    // Get output_shape
    output_shape_ = {batch_size_, channels_, height_, width_};
    output_size_ = sizeof(T);
    for (size_t i = 0; i < OUT_PUT_SHAPE_SIZE; i++) {
      output_size_ *= output_shape_[i];
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(dx_size_);
    input_size_list_.push_back(rois_size_);
    input_size_list_.push_back(mapping_channel_size_);
    output_size_list_.push_back(output_size_);
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
