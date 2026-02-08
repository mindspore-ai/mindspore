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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_CROP_AND_RESIZE_GRAD_BOXES_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_CROP_AND_RESIZE_GRAD_BOXES_HELPER_H_
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "kernel/gpu/cuda_impl/cuda_class/helper_base.h"
#include "kernel/gpu/cuda_impl/cuda_ops/crop_and_resize_grad_boxes_impl.cuh"
#include "mindapi/base/types.h"

namespace mindspore {
namespace cukernel {
constexpr size_t OUTPUT_NUM = 1;
constexpr size_t kGrads = 0;
constexpr size_t kGradsShapeLen = 4;
constexpr size_t kNumBoxes = 0;
constexpr size_t kHeight = 1;
constexpr size_t kWidth = 2;
constexpr size_t kDepth = 3;
constexpr size_t kBatch = 0;
constexpr size_t kImages = 1;
constexpr size_t kImageShapeLen = 4;
constexpr size_t kBoxes = 2;
constexpr size_t kBoxesShapeLen = 2;
constexpr size_t kCoordinateLen = 4;
constexpr size_t kBoxIndex = 3;
constexpr size_t kBoxIndexShapeLen = 1;
constexpr size_t kOutputIndex = 0;
constexpr size_t kOutputShapeLen = 2;
class CropAndResizeGradBoxesAttr : public GpuKernelAttrBase {
 public:
  CropAndResizeGradBoxesAttr() = default;
  ~CropAndResizeGradBoxesAttr() override = default;
  ResizeMethod method_;
};

template <typename T, typename G>
class CropAndResizeGradBoxesHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit CropAndResizeGradBoxesHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~CropAndResizeGradBoxesHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    std::vector<int64_t> in_grads_shape_ = input_shapes[kGrads];
    std::vector<int64_t> in_images_shape_ = input_shapes[kImages];
    std::vector<int64_t> in_boxes_shape_ = input_shapes[kBoxes];
    std::vector<int64_t> in_box_in_shape_ = input_shapes[kBoxIndex];
    std::vector<int64_t> ou_output_shape_ = output_shapes[kOutputIndex];

    int out_flag =
      CalShapesSizeInBytes<G>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_input_ = (out_flag == 1);

    if (is_null_input_) {
      return -1;
    }
    // input grads
    int64_t input_grads_shape_len = in_grads_shape_.size();
    if (input_grads_shape_len != kGradsShapeLen) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', CropAndResizeGradBoxes supports only" << kGradsShapeLen
                    << "-D for grads tensor, but got " << input_grads_shape_len << "-D.";
      return -1;
    }
    num_boxes_ = static_cast<int32_t>(in_grads_shape_[kNumBoxes]);
    crop_height_ = static_cast<int32_t>(in_grads_shape_[kHeight]);
    crop_width_ = static_cast<int32_t>(in_grads_shape_[kWidth]);
    crop_depth_ = static_cast<int32_t>(in_grads_shape_[kDepth]);
    //  input image
    int64_t input_image_shape_len = in_images_shape_.size();
    if (input_image_shape_len != kImageShapeLen) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', CropAndResizeGradBoxes supports only" << kImageShapeLen
                    << "-D for images tensor, but got " << input_image_shape_len << "-D.";
      return -1;
    }
    image_batch_ = static_cast<int32_t>(in_images_shape_[kBatch]);
    image_height_ = static_cast<int32_t>(in_images_shape_[kHeight]);
    image_width_ = static_cast<int32_t>(in_images_shape_[kWidth]);
    depth_ = static_cast<int32_t>(in_images_shape_[kDepth]);
    //  input boxes
    int64_t input_boxes_shape_len = in_boxes_shape_.size();
    if (input_boxes_shape_len != kBoxesShapeLen) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', CropAndResizeGradBoxes supports only" << kBoxesShapeLen
                    << "-D for boxes tensor, but got " << input_boxes_shape_len << "-D.";
      return -1;
    }
    if (in_boxes_shape_[1] != kCoordinateLen) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', CropAndResizeGradBoxes supports the size of " << kCoordinateLen
                    << " for boxes, but got " << in_boxes_shape_[1] << "for boxes.";
      return -1;
    }
    //  input box_index
    int64_t input_box_index_shape_len = in_box_in_shape_.size();
    if (input_box_index_shape_len != kBoxIndexShapeLen) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', CropAndResizeGradBoxes supports only" << kBoxIndexShapeLen
                    << "-D for box_index tensor, but got " << input_box_index_shape_len << "-D.";
      return -1;
    }
    //  output
    int64_t output_shape_len = ou_output_shape_.size();
    if (output_shape_len != kOutputShapeLen) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', CropAndResizeGradBoxes supports only" << kOutputShapeLen
                    << "-D for output tensor, but got " << output_shape_len << "-D.";
      return -1;
    }
    int64_t temp_result = static_cast<int64_t>(num_boxes_) * static_cast<int64_t>(crop_height_) *
                          static_cast<int64_t>(crop_width_) * static_cast<int64_t>(crop_depth_);
    if (temp_result > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', size calculation overflow: "
                        << "num_boxes_ = " << num_boxes_ << ", crop_height_ = " << crop_height_
                        << ", crop_width_ = " << crop_width_ << ", crop_depth_ = " << crop_depth_
                        << ", result would be " << temp_result
                        << ", max allowed = " << std::numeric_limits<int32_t>::max();
    }
    size = static_cast<int32_t>(temp_result);
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }

    G *grads_ptr = nullptr;
    T *images_ptr = nullptr;
    G *boxes_ptr = nullptr;
    int *box_in_ptr = nullptr;
    G *output_ptr = nullptr;
    int flag = GetDeviceAddress<G>(input_ptrs, kGrads, kernel_name_, &grads_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, kImages, kernel_name_, &images_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<G>(input_ptrs, kBoxes, kernel_name_, &boxes_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int>(input_ptrs, kBoxIndex, kernel_name_, &box_in_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<G>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    // call cuda kernel
    auto status = CalCropAndResizeGradBoxes(
      size, grads_ptr, images_ptr, boxes_ptr, box_in_ptr, num_boxes_, image_batch_, image_height_, image_width_,
      crop_height_, crop_width_, depth_, output_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<CropAndResizeGradBoxesAttr>(kernel_attr);
  }

 private:
  int32_t image_batch_;
  int32_t image_height_;
  int32_t image_width_;
  int32_t depth_;
  int32_t num_boxes_;
  int32_t crop_height_;
  int32_t crop_width_;
  int32_t crop_depth_;
  int32_t size;
  std::shared_ptr<CropAndResizeGradBoxesAttr> attr_ptr_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_CROP_AND_RESIZE_GRAD_BOXES_HELPER_H_
