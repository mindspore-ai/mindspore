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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ADAPTIVE_MAX_POOL2D_GRAD_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ADAPTIVE_MAX_POOL2D_GRAD_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adaptive_max_pool2d_grad_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr int64_t maxIndexIdx = 2;
constexpr int64_t dyDimSmall = 3;
constexpr int64_t hIdx = 2;

class AdaptiveMaxPool2DGradAttr : public GpuKernelAttrBase {
 public:
  AdaptiveMaxPool2DGradAttr() = default;
  ~AdaptiveMaxPool2DGradAttr() override = default;
};

template <typename T, typename S>
class AdaptiveMaxPool2DGradHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit AdaptiveMaxPool2DGradHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~AdaptiveMaxPool2DGradHelperGpuKernel() = default;

  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();

    // cal input_size_list_ (dy, x, index)
    size_t dy_size = sizeof(T);
    for (auto val : input_shapes[0]) {
      dy_size *= val;
    }
    input_size_list_.emplace_back(dy_size);

    size_t x_size = sizeof(T);
    for (auto val : input_shapes[1]) {
      x_size *= val;
    }
    input_size_list_.emplace_back(x_size);

    size_t index_size = sizeof(S);
    for (auto val : input_shapes[maxIndexIdx]) {
      index_size *= val;
    }
    input_size_list_.emplace_back(index_size);

    // cal output_size_list_ (dx)
    int out_flag = CalShapesSizeInBytes<T>(output_shapes, 1, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }

    input_shape_.emplace_back(input_shapes[0]);            // dy
    input_shape_.emplace_back(input_shapes[1]);            // x
    input_shape_.emplace_back(input_shapes[maxIndexIdx]);  // index
    output_shape_ = output_shapes[0];                      // dx

    is_null_input_ = (out_flag == 1);
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }

    // get device ptr input index output
    T *dy_ptr = nullptr;
    S *index_ptr = nullptr;
    T *dx_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &dy_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<S>(input_ptrs, maxIndexIdx, kernel_name_, &index_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &dx_ptr);
    if (flag != 0) {
      return flag;
    }

    // call cuda kernel
    const int shape_dim = output_shape_.size();  // dx grad dim 3 or 4
    auto input_shape = input_shape_[0];          // dy
    const int kMinDims = 3;
    if (shape_dim < kMinDims || SizeToInt(input_shape.size()) < kMinDims) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the rank of input and output can not less than " << kMinDims
                        << ", but got output shape: " << output_shape_ << ", input shape: " << input_shape_;
    }
    const int n = (shape_dim == dyDimSmall ? 1 : output_shape_[0]);
    const int c = (shape_dim == dyDimSmall ? output_shape_[0] : output_shape_[1]);
    const int in_h = input_shape[input_shape.size() - hIdx];
    const int in_w = input_shape[input_shape.size() - 1];
    const int out_h = output_shape_[output_shape_.size() - hIdx];
    const int out_w = output_shape_[output_shape_.size() - 1];

    CalAdaptiveMaxPool2DGrad(dy_ptr, index_ptr, n, c, in_h, in_w, out_h, out_w, dx_ptr, device_id_,
                             reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<AdaptiveMaxPool2DGradAttr>(kernel_attr);
  }

  void ResetResource() override {
    input_shape_.clear();
    output_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
  }

 private:
  std::shared_ptr<AdaptiveMaxPool2DGradAttr> attr_ptr_;
  std::vector<std::vector<int64_t>> input_shape_;  // 0:input_shape(y_grad) 2:index_shape(argmax)
  std::vector<int64_t> output_shape_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ADAPTIVE_MAX_POOL2D_GRAD_HELPER_H_
