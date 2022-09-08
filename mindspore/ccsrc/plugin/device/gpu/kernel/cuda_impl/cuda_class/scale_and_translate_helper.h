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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SCALE_AND_TRANSLATE_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SCALE_AND_TRANSLATE_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/scale_and_translate_impl.cuh"
#include "mindspore/core/mindapi/base/types.h"

namespace mindspore {
namespace cukernel {
constexpr size_t kImages = 0;
constexpr size_t kSSize = 1;
constexpr size_t kScale = 2;
constexpr size_t kTranslation = 3;
constexpr size_t kBatch = 0;
constexpr size_t kHeight = 1;
constexpr size_t kWidth = 2;
constexpr size_t kDepth = 3;
constexpr size_t kImageShapeLen = 4;
constexpr size_t kSize = 1;
constexpr size_t kSizeShapeLen = 1;
constexpr size_t kSizeElements = 2;
constexpr size_t kOutputIndex = 0;
constexpr size_t kOutputShapeLen = 4;
constexpr size_t INPUT_NUM = 4;
constexpr size_t OUTPUT_NUM = 1;
constexpr size_t ktwo = 2;
class ScaleAndTranslateAttr : public GpuKernelAttrBase {
 public:
  ScaleAndTranslateAttr() = default;
  ~ScaleAndTranslateAttr() override = default;
  std::string kernel_type_;
  bool antialias_;
};

template <typename T>
class ScaleAndTranslateHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit ScaleAndTranslateHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~ScaleAndTranslateHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    std::vector<int64_t> in_images_shape_ = input_shapes[kImages];
    std::vector<int64_t> in_size_shape_ = input_shapes[kSize];
    std::vector<int64_t> in_scale_shape_ = input_shapes[kScale];
    std::vector<int64_t> in_translation_shape_ = input_shapes[kTranslation];
    std::vector<int64_t> ou_output_shape_ = output_shapes[kOutputIndex];

    size_t cur_size_T = sizeof(T);
    for (const auto &val : in_images_shape_) {
      cur_size_T *= val;
    }
    input_size_list_.emplace_back(cur_size_T);
    size_t cur_size_int64 = sizeof(int64_t);
    for (const auto &val : in_size_shape_) {
      cur_size_int64 *= val;
    }
    input_size_list_.emplace_back(cur_size_int64);
    size_t cur_size_float = sizeof(float);
    for (const auto &val : in_scale_shape_) {
      cur_size_float *= val;
    }
    input_size_list_.emplace_back(cur_size_float);
    cur_size_float = sizeof(float);
    for (const auto &val : in_translation_shape_) {
      cur_size_float *= val;
    }
    input_size_list_.emplace_back(cur_size_float);
    int out_flag =
      CalShapesSizeInBytes<float>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_input_ = (out_flag == 1);

    if (is_null_input_) {
      return -1;
    }
    //  input image
    int64_t input_image_shape_len = in_images_shape_.size();
    if (input_image_shape_len != kImageShapeLen) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', ScaleAndTranslate supports only" << kImageShapeLen
                    << "-D for images tensor, but got " << input_image_shape_len << "-D.";
      return -1;
    }
    image_batch_ = in_images_shape_[kBatch];
    image_height_ = in_images_shape_[kHeight];
    image_width_ = in_images_shape_[kWidth];
    depth_ = in_images_shape_[kDepth];
    //  input size
    int64_t input_size_shape_len = in_size_shape_.size();
    if (input_size_shape_len != kSizeShapeLen) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', ScaleAndTranslate supports only" << kSizeShapeLen
                    << "-D for size, but got " << input_size_shape_len << "-D.";
      return -1;
    }
    if (in_size_shape_[0] != kSizeElements) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', ScaleAndTranslate supports the size of " << kSizeElements
                    << " for size, but got " << in_size_shape_[0] << " elements for size.";
      return -1;
    }
    //  output
    int64_t output_shape_len = ou_output_shape_.size();
    if (output_shape_len != kOutputShapeLen) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', ScaleAndTranslate supports only" << kOutputShapeLen
                    << "-D for output tensor, but got " << output_shape_len << "-D.";
      return -1;
    }
    output_height_ = ou_output_shape_[kHeight];
    output_width_ = ou_output_shape_[kWidth];
    // input_shape_
    size_t workspace_int64_size = INPUT_NUM * sizeof(int64_t);
    work_size_list_.emplace_back(workspace_int64_size);
    // starts
    std::vector<std::vector<int64_t>> int32_shapes, weights_shapes, buffers_shape;
    std::vector<int64_t> start_shape, weight_shape, spans_size_shape, buffer_shape;
    start_shape.push_back(output_height_ + output_width_);
    // span size
    spans_size_shape.push_back(ktwo);
    int32_shapes.push_back(start_shape);
    int32_shapes.push_back(spans_size_shape);
    int work_flag =
      CalShapesSizeInBytes<int32_t>(int32_shapes, kIndex2, kernel_name_, "starts_shapes", &work_size_list_);
    if (work_flag == -1) {
      return out_flag;
    }
    // weights
    weight_shape.push_back(image_height_ * output_height_ + image_width_ * output_width_);
    weights_shapes.push_back(weight_shape);
    work_flag = CalShapesSizeInBytes<float>(weights_shapes, 1, kernel_name_, "weights_shapes", &work_size_list_);
    if (work_flag == -1) {
      return out_flag;
    }
    // intermediate_buffer
    buffer_shape.push_back(image_batch_ * output_height_ * image_width_ * depth_);
    buffers_shape.push_back(buffer_shape);
    work_flag = CalShapesSizeInBytes<float>(buffers_shape, 1, kernel_name_, "buffers_shape", &work_size_list_);
    if (work_flag == -1) {
      return out_flag;
    }
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    T *images_ptr = nullptr;
    int32_t *size_ptr = nullptr;
    float *scale_ptr = nullptr;
    float *translation_ptr = nullptr;
    float *output_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, kImages, kernel_name_, &images_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int32_t>(input_ptrs, kSSize, kernel_name_, &size_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<float>(input_ptrs, kScale, kernel_name_, &scale_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<float>(input_ptrs, kTranslation, kernel_name_, &translation_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<float>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }
    kernel_type_ = attr_ptr_->kernel_type_;
    bool antialias = attr_ptr_->antialias_;
    int64_t *input_shape_ptr = nullptr;
    int32_t *forward_starts_ptr = nullptr, *spans_size_ptr = nullptr;
    float *forward_weights_ptr = nullptr, *intermediate_ptr = nullptr;
    flag = GetDeviceAddress<int64_t>(work_ptrs, kIndex0, kernel_name_, &input_shape_ptr);
    if (flag != 0) {
      return flag;
    }
    origin_shape_.push_back(image_batch_);
    origin_shape_.push_back(image_height_);
    origin_shape_.push_back(image_width_);
    origin_shape_.push_back(depth_);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(input_shape_ptr, &origin_shape_[0], INPUT_NUM * sizeof(int64_t), cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream)),
      "cudaMemcpyAsync workspace failed");
    flag = GetDeviceAddress<int32_t>(work_ptrs, kIndex1, kernel_name_, &forward_starts_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int32_t>(work_ptrs, kIndex2, kernel_name_, &spans_size_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<float>(work_ptrs, kIndex3, kernel_name_, &forward_weights_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<float>(work_ptrs, kIndex4, kernel_name_, &intermediate_ptr);
    if (flag != 0) {
      return flag;
    }
    float radius_ = GetKernelRadius();
    thread_num_[kIndex0] = image_height_ * output_height_ + image_width_ * output_width_;
    thread_num_[kIndex1] = output_height_ + output_width_;
    thread_num_[kIndex2] = image_batch_ * output_height_;
    thread_num_[kIndex3] = image_batch_ * output_height_ * output_width_;
    // call cuda kernel
    CalScaleAndTranslate(thread_num_, images_ptr, scale_ptr, translation_ptr, image_batch_, image_height_, image_width_,
                         depth_, output_height_, output_width_, kernel_type_, antialias, radius_, input_shape_ptr,
                         size_ptr, spans_size_ptr, forward_starts_ptr, forward_weights_ptr, intermediate_ptr,
                         output_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<ScaleAndTranslateAttr>(kernel_attr);
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
  std::string kernel_type_;
  size_t thread_num_[4];
  int64_t image_batch_;
  int64_t image_height_;
  int64_t image_width_;
  int64_t depth_;
  int64_t output_height_;
  int64_t output_width_;
  std::vector<int64_t> origin_shape_;
  std::shared_ptr<ScaleAndTranslateAttr> attr_ptr_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SCALE_AND_TRANSLATE_HELPER_H_
