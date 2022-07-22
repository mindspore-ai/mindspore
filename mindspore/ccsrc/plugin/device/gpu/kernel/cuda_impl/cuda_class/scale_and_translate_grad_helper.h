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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SCALEANDTRANSLATEGRAD_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SCALEANDTRANSLATEGRAD_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/scale_and_translate_impl.cuh"
#include "mindspore/ccsrc/kernel/common_utils.h"

namespace mindspore {
namespace cukernel {
class ScaleAndTranslateGradAttr : public GpuKernelAttrBase {
 public:
  ScaleAndTranslateGradAttr() = default;
  ~ScaleAndTranslateGradAttr() override = default;
  string kernel_type;
  bool antialias;
};

template <typename T>
class ScaleAndTranslateGradHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit ScaleAndTranslateGradHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    kernel_type_ = "lanczos3";
    antialias_ = true;
    batch_ = 0;
    channel_ = 0;
    input_grad_width_ = 0;
    input_grad_height_ = 0;
    origin_width_ = 0;
    origin_height_ = 0;
    is_null_input_ = false;
    radius_ = 0;
  }

  virtual ~ScaleAndTranslateGradHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t INPUT_NUM = 4;
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();
    input_grad_shape_ = input_shapes[0];
    origin_shape_ = input_shapes[1];
    output_shape_ = output_shapes[0];
    int inp_flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    if (inp_flag == -1) {
      return inp_flag;
    }
    output_shape_ = output_shapes[0];
    batch_ = input_grad_shape_[0];
    input_grad_height_ = input_grad_shape_[1];
    input_grad_width_ = input_grad_shape_[kIndex2];
    channel_ = input_grad_shape_[kIndex3];
    origin_height_ = origin_shape_[1];
    origin_width_ = origin_shape_[kIndex2];
    int out_flag =
      CalShapesSizeInBytes<float>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_input_ = (inp_flag == 1 || out_flag == 1);
    // input_shape_
    size_t input_shape_size = (kIndex4) * sizeof(int64_t);
    work_size_list_.emplace_back(input_shape_size);
    // size
    size_t size_size = (kIndex2) * sizeof(int32_t);
    work_size_list_.emplace_back(size_size);
    // starts, span_size_, weight_size
    std::vector<std::vector<int64_t>> int32_shapes, weights_shapes, buffers_shapes;
    std::vector<int64_t> start_shape, weight_shape, spans_size_shape, buffer_shape, weight_size_shape;
    start_shape.push_back(input_grad_height_ + input_grad_width_ + origin_height_ + origin_width_);
    spans_size_shape.push_back(kIndex2);
    weight_size_shape.push_back(origin_height_ + origin_width_);
    int32_shapes.push_back(start_shape);
    int32_shapes.push_back(spans_size_shape);
    int32_shapes.push_back(weight_size_shape);
    int work_flag =
      CalShapesSizeInBytes<int32_t>(int32_shapes, kIndex3, kernel_name_, "int32_shapes", &work_size_list_);
    if (work_flag == -1) {
      return out_flag;
    }
    // forward_weights, grad_weights
    weight_shape.push_back(kIndex2 * (origin_height_ * input_grad_height_ + origin_width_ * input_grad_width_));
    weights_shapes.push_back(weight_shape);
    work_flag = CalShapesSizeInBytes<float>(weights_shapes, 1, kernel_name_, "weights_shapes", &work_size_list_);
    if (work_flag == -1) {
      return out_flag;
    }
    // intermediate_buffer
    buffer_shape.push_back(batch_ * origin_height_ * input_grad_width_ * channel_);
    buffers_shapes.push_back(buffer_shape);
    work_flag = CalShapesSizeInBytes<float>(buffers_shapes, 1, kernel_name_, "buffers_shapes", &work_size_list_);
    if (work_flag == -1) {
      return out_flag;
    }
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    T *input_grad_ptr = nullptr, *input_origin_image_ptr = nullptr;
    float *input_scale_ptr = nullptr, *input_translate_ptr = nullptr, *output_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_grad_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, 1, kernel_name_, &input_origin_image_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<float>(input_ptrs, kIndex2, kernel_name_, &input_scale_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<float>(input_ptrs, kIndex3, kernel_name_, &input_translate_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<float>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }
    int64_t *input_shape_ptr = nullptr;
    int32_t *forward_starts_ptr = nullptr, *grad_starts_ptr = nullptr, *spans_size_ptr = nullptr,
            *weight_size_ptr = nullptr, *size_ptr = nullptr;
    float *forward_weights_ptr = nullptr, *grad_weights_ptr = nullptr, *intermediate_ptr = nullptr;
    flag = GetDeviceAddress<int64_t>(work_ptrs, 0, kernel_name_, &input_shape_ptr);
    if (flag != 0) {
      return flag;
    }
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(input_shape_ptr, &origin_shape_[0], kIndex4 * sizeof(int64_t), cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream)),
      "cudaMemcpyAsync workspace failed");
    flag = GetDeviceAddress<int32_t>(work_ptrs, 1, kernel_name_, &size_ptr);
    if (flag != 0) {
      return flag;
    }
    std::vector<int32_t> size_;
    size_.push_back(input_grad_height_);
    size_.push_back(input_grad_width_);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(size_ptr, &size_[0], kIndex2 * sizeof(int32_t), cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream)),
      "cudaMemcpyAsync workspace failed");
    flag = GetDeviceAddress<int32_t>(work_ptrs, kIndex2, kernel_name_, &forward_starts_ptr);
    if (flag != 0) {
      return flag;
    }
    grad_starts_ptr = forward_starts_ptr + input_grad_height_ + input_grad_width_;
    flag = GetDeviceAddress<int32_t>(work_ptrs, kIndex3, kernel_name_, &spans_size_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int32_t>(work_ptrs, kIndex4, kernel_name_, &weight_size_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<float>(work_ptrs, kIndex5, kernel_name_, &forward_weights_ptr);
    if (flag != 0) {
      return flag;
    }
    grad_weights_ptr = forward_weights_ptr + origin_height_ * input_grad_height_ + origin_width_ * input_grad_width_;
    flag = GetDeviceAddress<float>(work_ptrs, kIndex6, kernel_name_, &intermediate_ptr);
    if (flag != 0) {
      return flag;
    }
    radius_ = GetKernelRadius();
    int64_t input_pix_per_batch = input_grad_height_ * input_grad_width_ * channel_;
    int64_t intermediate_pix_per_batch = input_grad_width_ * origin_height_ * channel_;
    int64_t output_pix_per_batch = origin_height_ * origin_width_ * channel_;
    thread_num_[0] = kIndex2 * (input_grad_height_ * origin_height_ + input_grad_width_ * origin_width_);
    thread_num_[1] = input_grad_height_ + input_grad_width_;
    thread_num_[kIndex2] = origin_height_ + origin_width_;
    thread_num_[kIndex3] = intermediate_pix_per_batch * batch_;
    thread_num_[kIndex4] = output_pix_per_batch * batch_;
    thread_num_[kIndex5] = batch_ * origin_height_;
    thread_num_[kIndex6] = batch_ * origin_height_ * origin_width_;
    CallScaleAndTranslateGrad(kernel_type_, input_grad_ptr, input_origin_image_ptr, radius_, input_shape_ptr, size_ptr,
                              input_scale_ptr, input_translate_ptr, antialias_, spans_size_ptr, forward_starts_ptr,
                              grad_starts_ptr, forward_weights_ptr, grad_weights_ptr, thread_num_, intermediate_ptr,
                              input_pix_per_batch, intermediate_pix_per_batch, output_pix_per_batch, output_ptr,
                              weight_size_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<ScaleAndTranslateGradAttr>(kernel_attr);
  }

 protected:
  int CheckKernelParam() override {
    constexpr int INPUT_SHAPE_DIMS = 4;
    constexpr int OUTPUT_SHAPE_DIMS = 4;
    kernel_type_ = attr_ptr_->kernel_type;
    antialias_ = attr_ptr_->antialias;
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

  float GetKernelRadius() {
    if (kernel_type_ == "lanczos1" || kernel_type_ == "box" || kernel_type_ == "triangle") {
      return 1.0f;
    } else if (kernel_type_ == "gaussian") {
      return 1.5f;
    } else if (kernel_type_ == "keyscubic" || kernel_type_ == "mitchellcubic") {
      return 2.0f;
    } else if (kernel_type_ == "lanczos3") {
      return 3.0f;
    } else {
      return 5.0f;
    }
  }

 private:
  string kernel_type_;
  bool antialias_;
  std::shared_ptr<ScaleAndTranslateGradAttr> attr_ptr_;
  std::vector<int64_t> input_grad_shape_;
  std::vector<int64_t> origin_shape_;
  std::vector<int64_t> output_shape_;
  size_t thread_num_[7];
  bool is_null_input_;
  int64_t batch_;
  int64_t channel_;
  int64_t input_grad_width_;
  int64_t input_grad_height_;
  int64_t origin_width_;
  int64_t origin_height_;
  float radius_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SCALEANDTRANSLATEGRAD_HELPER_H_
