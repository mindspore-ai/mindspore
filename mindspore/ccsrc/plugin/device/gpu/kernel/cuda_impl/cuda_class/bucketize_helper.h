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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_BUCKETIZE_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_BUCKETIZE_HELPER_H_
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bucketize_impl.cuh"

namespace mindspore {
namespace cukernel {
class BucketizeAttr : public GpuKernelAttrBase {
 public:
  BucketizeAttr() = default;
  ~BucketizeAttr() override = default;
  std::vector<float> boundaries;
};

template <typename T>
class BucketizeHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit BucketizeHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~BucketizeHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t INPUT_NUM = 1;
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();
    int inp_flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    if (inp_flag == -1) {
      return inp_flag;
    }
    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    size_t work_size = boundaries_size_ * sizeof(float);
    work_size_list_.emplace_back(work_size);
    is_null_input_ = (inp_flag == 1 || out_flag == 1);
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    T *input_ptr = nullptr;
    int32_t *output_ptr = nullptr;
    float *boundaries_ptr = nullptr;
    int size = input_size_list_[0] / sizeof(T);
    int M = work_size_list_[0] / sizeof(float);
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int32_t>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<float>(work_ptrs, 0, kernel_name_, &boundaries_ptr);
    if (flag != 0) {
      return flag;
    }
    cudaError_t ret = cudaMemcpyAsync(boundaries_ptr, &boundaries_[0], boundaries_size_ * sizeof(float),
                                      cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
    if (ret) {
      MS_LOG(ERROR) << "cudaMemcpyAsync error in BucketizeHelperGpuKernel::Process, error code is " << ret;
      return -1;
    }
    // call cuda kernel
    auto status = CalBucketize(size, M, boundaries_ptr, input_ptr, output_ptr, device_id_,
                               reinterpret_cast<cudaStream_t>(cuda_stream));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<BucketizeAttr>(kernel_attr);
    boundaries_ = attr_ptr_->boundaries;
    boundaries_size_ = boundaries_.size();
  }

 private:
  std::shared_ptr<BucketizeAttr> attr_ptr_;
  std::vector<float> boundaries_;
  int64_t boundaries_size_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_BUCKETIZE_HELPER_H_
