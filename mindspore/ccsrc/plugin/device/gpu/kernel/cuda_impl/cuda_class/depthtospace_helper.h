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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_DEPTHTOSPACE_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_DEPTHTOSPACE_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/depthtospace_impl.cuh"
namespace mindspore {
namespace cukernel {
class DepthToSpaceAttr : public GpuKernelAttrBase {
 public:
  DepthToSpaceAttr() = default;
  ~DepthToSpaceAttr() override = default;
  int64_t block_size;
};

template <typename T>
class DepthToSpaceHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit DepthToSpaceHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    kernel_size_ = 0;
    is_null_input_ = false;
  }
  virtual ~DepthToSpaceHelperGpuKernel() = default;
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
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag != 0) {
      return out_flag;
    }
    is_null_input_ = (inp_flag == 1 || out_flag == 1);
    kernel_size_ = output_size_list_[0] / sizeof(T);
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    size_t in = input_shape_[0];
    size_t ic = input_shape_[1];
    size_t ih = input_shape_[2];
    size_t iw = input_shape_[3];
    size_t on = in;
    size_t oc = ic / attr_ptr_->block_size / attr_ptr_->block_size;
    size_t oh = ih * attr_ptr_->block_size;
    size_t ow = iw * attr_ptr_->block_size;

    T *input_ptr = nullptr;
    T *output_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    // call cuda kernel
    CalDepthToSpace(kernel_size_, input_ptr, in, ic, ih, iw, on, oc, oh, ow, attr_ptr_->block_size, output_ptr,
                    device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<DepthToSpaceAttr>(kernel_attr);
  }

 protected:
  int CheckKernelParam() override {
    constexpr int BLOCK_SIZE_LOWEST = 2;
    if (attr_ptr_->block_size < BLOCK_SIZE_LOWEST) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'block_size' cannot be less than 2, but got "
                    << attr_ptr_->block_size;
      return -1;
    }
    return 0;
  }

 private:
  std::shared_ptr<DepthToSpaceAttr> attr_ptr_;
  std::vector<int64_t> input_shape_;
  size_t kernel_size_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_DEPTHTOSPACE_HELPER_H_
