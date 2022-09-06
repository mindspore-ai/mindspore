/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ASSIGN_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ASSIGN_HELPER_H_

#include <vector>
#include <map>
#include <string>
#include <utility>
#include <algorithm>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"

namespace mindspore {
namespace cukernel {
template <typename T>
class AssignHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit AssignHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~AssignHelperGpuKernel() = default;

  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    auto input_shape = input_shapes[kIndex0];
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    input_size_ = sizeof(T) * SizeOf(input_shape);
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(input_size_);
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }

    // get device ptr input index output
    T *var = nullptr;
    T *value = nullptr;
    T *output = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, kIndex0, kernel_name_, &var);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, kIndex1, kernel_name_, &value);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(output_ptrs, kIndex0, kernel_name_, &output);
    if (flag != 0) {
      return flag;
    }
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(var, value, input_size_, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(cuda_stream)),
      "cudaMemcpyAsync failed.");
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(output, value, input_size_, cudaMemcpyDeviceToDevice,
                                                      reinterpret_cast<cudaStream_t>(cuda_stream)),
                                      "cudaMemcpyAsync failed.");
    return 0;
  }

  void ResetResource() override {
    input_size_ = 1;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
  }

 private:
  bool is_null_input_ = false;
  size_t input_size_ = 1;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_LAYER_NORM_GRAD_GRAD_HELPER_HELPER_H_
