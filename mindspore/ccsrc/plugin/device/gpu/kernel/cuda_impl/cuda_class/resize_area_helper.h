/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_RESIZE_AREA_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_RESIZE_AREA_HELPER_H_
#include <string>
#include <vector>
#include <memory>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/resize_area_impl.cuh"

namespace mindspore {
namespace cukernel {
class ResizeAreaAttr : public GpuKernelAttrBase {
 public:
  ResizeAreaAttr() = default;
  ~ResizeAreaAttr() override = default;
  bool align_corners;
};
constexpr size_t INPUT_NUM_T = 1;
constexpr size_t INPUT_NUM_SIZE = 1;
constexpr size_t OUTPUT_NUM = 1;
constexpr size_t WORK_NUM = 2;
constexpr size_t SHAPE_SIZE = 4;
constexpr int64_t kzero = 0;
constexpr int64_t kone = 1;
constexpr int64_t ktwo = 2;
constexpr int64_t kthree = 3;

template <typename T>
class ResizeAreaHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit ResizeAreaHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    align_corners_ = false;
    is_null_input_ = false;
  }
  virtual ~ResizeAreaHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    std::vector<std::vector<int64_t>> input_shapes_T, input_shapes_size;
    input_shapes_T.emplace_back(input_shapes[kzero]);
    input_shapes_size.emplace_back(input_shapes[kone]);
    int in_flag1 =
      CalShapesSizeInBytes<T>(input_shapes_T, INPUT_NUM_T, kernel_name_, "input_shapes_images", &input_size_list_);
    if (in_flag1 != 0) {
      return in_flag1;
    }
    int in_flag2 = CalShapesSizeInBytes<int32_t>(input_shapes_size, INPUT_NUM_SIZE, kernel_name_, "input_shapes_size",
                                                 &input_size_list_);
    if (in_flag2 != 0) {
      return in_flag2;
    }
    int out_flag =
      CalShapesSizeInBytes<float>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag != 0) {
      return out_flag;
    }
    is_null_input_ = (in_flag1 == 1 || in_flag2 == 1 || out_flag == 1);

    batch_size_ = input_shapes_T[kzero][kzero];
    in_height_ = input_shapes_T[kzero][kone];
    in_width_ = input_shapes_T[kzero][ktwo];
    channels_ = input_shapes_T[kzero][kthree];
    out_height_ = output_shapes[kzero][kone];
    out_width_ = output_shapes[kzero][ktwo];
    size_t workspace_x_size = out_width_ * sizeof(ResizeAreaCachedInterpolation);
    size_t workspace_y_size = out_height_ * sizeof(ResizeAreaCachedInterpolation);
    work_size_list_.emplace_back(workspace_x_size);
    work_size_list_.emplace_back(workspace_y_size);
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    T *image_ptr = nullptr;
    float *output_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &image_ptr);
    if (flag != 0) {
      return flag;
    }
    ResizeAreaCachedInterpolation *x_interps = nullptr;
    ResizeAreaCachedInterpolation *y_interps = nullptr;
    x_interps = reinterpret_cast<ResizeAreaCachedInterpolation *>(work_ptrs[0]);
    y_interps = reinterpret_cast<ResizeAreaCachedInterpolation *>(work_ptrs[1]);
    flag = GetDeviceAddress<float>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }
    align_corners_ = attr_ptr_->align_corners;
    CalResizeArea(image_ptr, x_interps, y_interps, output_ptr, batch_size_, channels_, out_height_, out_width_,
                  in_height_, in_width_, align_corners_, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<ResizeAreaAttr>(kernel_attr);
  }

 private:
  std::shared_ptr<ResizeAreaAttr> attr_ptr_;
  int post_output_size_;
  int32_t batch_size_;
  int32_t in_height_;
  int32_t in_width_;
  int32_t channels_;
  int32_t out_height_;
  int32_t out_width_;
  bool align_corners_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_RESIZE_AREA_HELPER_H_
