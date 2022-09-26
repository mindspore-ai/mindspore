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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ARGMAX_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ARGMAX_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/argmax_impl.cuh"

namespace mindspore {
namespace cukernel {
class ArgMaxAttr : public GpuKernelAttrBase {
 public:
  ArgMaxAttr() = default;
  ~ArgMaxAttr() override = default;
  int64_t axis;
};

template <typename T, typename S>
class ArgMaxHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit ArgMaxHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    axis_ = 0;
    bound_ = 0;
    is_null_argmax_input_ = false;
  }

  virtual ~ArgMaxHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t INPUT_NUM = 1;
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();
    int inp_flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    if (inp_flag == -1) {
      return inp_flag;
    }
    input_shape_ = input_shapes[0];
    int out_flag =
      CalShapesSizeInBytes<S>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_argmax_input_ = (inp_flag == 1 || out_flag == 1);
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_argmax_input_) {
      return 0;
    }
    size_t outer_size = 1;
    for (int64_t i = axis_ - 1; i >= 0; i--) {
      outer_size *= input_shape_[i];
    }
    size_t inner_size = 1;
    for (int64_t i = axis_ + 1; i < static_cast<int64_t>(input_shape_.size()); i++) {
      inner_size *= input_shape_[i];
    }

    T *input_ptr = nullptr;
    S *output_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<S>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    // call cuda kernel
    CalArgmax(input_ptr, bound_, outer_size, inner_size, output_ptr, device_id_,
              reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<ArgMaxAttr>(kernel_attr);
  }

 protected:
  int CheckKernelParam() override {
    axis_ = attr_ptr_->axis;
    int64_t dims = static_cast<int64_t>(input_shape_.size());
    if (axis_ < -dims || axis_ >= dims) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'axis' should be in the range [-" << dims << "," << dims
                    << "), but got " << axis_;
      return -1;
    }
    if (axis_ < 0) {
      axis_ += dims;
    }
    bound_ = static_cast<S>(input_shape_[axis_]);
    if (input_shape_[axis_] != static_cast<int64_t>(bound_)) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the value of input_shape[axis] should be "
                    << static_cast<int64_t>(bound_) << ", but got " << input_shape_[axis_];
      return -1;
    }
    return 0;
  }

 private:
  int64_t axis_;
  std::shared_ptr<ArgMaxAttr> attr_ptr_;
  std::vector<int64_t> input_shape_;
  S bound_;
  bool is_null_argmax_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ARGMAX_HELPER_H_
