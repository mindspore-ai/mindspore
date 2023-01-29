/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_RESIZEBICUBICGRAD_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_RESIZEBICUBICGRAD_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/resize_bicubic_grad_impl.cuh"
#include "mindspore/ccsrc/kernel/common_utils.h"

namespace mindspore {
namespace cukernel {
class ResizeBicubicGradAttr : public GpuKernelAttrBase {
 public:
  ResizeBicubicGradAttr() = default;
  ~ResizeBicubicGradAttr() override = default;
  bool align_corners;
  bool half_pixel_centers;
};

template <typename T, typename S>
class ResizeBicubicGradHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit ResizeBicubicGradHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    align_corners_ = false;
    half_pixel_centers_ = false;
    batch_ = 0;
    channel_ = 0;
    input_grad_width_ = 0;
    input_grad_height_ = 0;
    origin_width_ = 0;
    origin_height_ = 0;
    is_null_resizebicubic_grad_input_ = false;
    h_scale_ = 0;
    w_scale_ = 0;
  }

  virtual ~ResizeBicubicGradHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t INPUT_NUM = 1;
    constexpr size_t OUTPUT_NUM = 1;
    constexpr int INPUT_C_ORDER = 1;
    constexpr int INPUT_H_ORDER = 2;
    constexpr int INPUT_W_ORDER = 3;
    ResetResource();
    align_corners_ = false;
    is_null_resizebicubic_grad_input_ = false;
    batch_ = 0;
    channel_ = 0;
    input_grad_height_ = 0;
    input_grad_width_ = 0;
    origin_height_ = 0;
    origin_width_ = 0;
    h_scale_ = 0;
    w_scale_ = 0;
    std::vector<std::vector<int64_t>> input_shape1;
    input_grad_shape_ = input_shapes[0];
    origin_shape_ = input_shapes[1];
    input_shape1.emplace_back(input_grad_shape_);
    int inp_flag = CalShapesSizeInBytes<T>(input_shape1, INPUT_NUM, kernel_name_, "input_shape1", &input_size_list_);
    if (inp_flag == -1) {
      return inp_flag;
    }
    std::vector<std::vector<int64_t>> shapes2;
    output_shape_ = output_shapes[0];
    shapes2.emplace_back(origin_shape_);
    inp_flag = CalShapesSizeInBytes<int32_t>(shapes2, INPUT_NUM, kernel_name_, "input_shape2", &input_size_list_);
    if (inp_flag == -1) {
      return inp_flag;
    }
    batch_ = input_grad_shape_[0];
    channel_ = input_grad_shape_[INPUT_C_ORDER];
    input_grad_height_ = input_grad_shape_[INPUT_H_ORDER];
    input_grad_width_ = input_grad_shape_[INPUT_W_ORDER];
    origin_height_ = origin_shape_[INPUT_H_ORDER];
    origin_width_ = origin_shape_[INPUT_W_ORDER];
    int out_flag =
      CalShapesSizeInBytes<S>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_resizebicubic_grad_input_ = (inp_flag == 1 || out_flag == 1);
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_resizebicubic_grad_input_) {
      return 0;
    }
    T *input_grad_ptr = nullptr;
    S *input_origin_image_ptr = nullptr;
    S *output_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_grad_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<S>(input_ptrs, 1, kernel_name_, &input_origin_image_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<S>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }
    h_scale_ = kernel::Scaling(origin_height_, input_grad_height_, align_corners_);
    w_scale_ = kernel::Scaling(origin_width_, input_grad_width_, align_corners_);
    // call cuda kernel
    CalResizeBicubicGrad(input_grad_ptr, batch_, channel_, input_grad_height_, input_grad_width_, origin_height_,
                         origin_width_, h_scale_, w_scale_, output_ptr, half_pixel_centers_, device_id_,
                         reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<ResizeBicubicGradAttr>(kernel_attr);
  }

 protected:
  int CheckKernelParam() override {
    constexpr int INPUT_SHAPE_DIMS = 4;
    constexpr int OUTPUT_SHAPE_DIMS = 4;
    align_corners_ = attr_ptr_->align_corners;
    half_pixel_centers_ = attr_ptr_->half_pixel_centers;
    if (align_corners_ && half_pixel_centers_) {
      MS_LOG(ERROR) << "The half_pixel_centers must be false when align_corners is true "
                    << ", but half_pixel_centers got True";
      return -1;
    }
    int64_t dims = static_cast<int64_t>(input_grad_shape_.size());
    if (dims != INPUT_SHAPE_DIMS) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', The grad images tensor must be a 4-D tensor,but got " << dims
                    << "dimension.";
      return -1;
    }
    int64_t dims2 = static_cast<int64_t>(origin_shape_.size());
    if (dims2 != INPUT_SHAPE_DIMS) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', The origin images tensor must be a 4-D tensor,but got " << dims2
                    << "dimension.";
      return -1;
    }
    int64_t dims3 = static_cast<int64_t>(output_shape_.size());
    if (dims3 != OUTPUT_SHAPE_DIMS) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', The output tensor must be a 4-D tensor,but got " << dims3
                    << " dimension.";
      return -1;
    }
    return 0;
  }

 private:
  bool align_corners_;
  bool half_pixel_centers_;
  std::shared_ptr<ResizeBicubicGradAttr> attr_ptr_;
  std::vector<int64_t> input_grad_shape_;
  std::vector<int64_t> origin_shape_;
  std::vector<int64_t> output_shape_;
  bool is_null_resizebicubic_grad_input_;
  int batch_;
  int channel_;
  int input_grad_width_;
  int input_grad_height_;
  int origin_width_;
  int origin_height_;
  float h_scale_;
  float w_scale_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_RESIZEBICUBICGRAD_HELPER_H_
