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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_PARALLEL_CONCAT_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_PARALLEL_CONCAT_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/parallel_concat_impl.cuh"

namespace mindspore {
namespace cukernel {
template <typename T>
class ParallelConcatHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit ParallelConcatHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    axis_ = 0;
    input_num_ = 1;
    output_size_ = 0;
    inputs_host_ = nullptr;
  }

  virtual ~ParallelConcatHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    input_num_ = input_shapes.size();
    inputs_host_ = std::make_unique<T *[]>(input_num_);
    int current_dim = 0;
    for (int i = 0; i < input_num_; i++) {
      size_t input_size = 1;
      auto input_shape = input_shapes[i];
      for (size_t j = 0; j < input_shape.size(); j++) {
        input_size *= static_cast<size_t>(input_shape[j]);
      }

      if (input_size == 0) {
        input_num_--;
      } else {
        input_size_list_.push_back(input_size * sizeof(T));
        current_dim++;
      }
    }
    work_size_list_.push_back(sizeof(T *) * input_num_);
    work_size_list_.push_back(sizeof(int) * input_num_);

    auto output_shape = output_shapes[0];
    output_size_ = SizeOf(output_shape);
    output_size_list_.push_back(output_size_ * sizeof(T));
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (input_num_ == 0) {
      return 0;
    }
    T *output_ptr = nullptr;
    T **inputs_device = nullptr;
    int flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T *>(work_ptrs, 0, kernel_name_, &inputs_device);
    if (flag != 0) {
      return flag;
    }
    int current_dim = 0;
    for (size_t i = 0; i < input_ptrs.size(); i++) {
      T *input = nullptr;
      flag = GetDeviceAddress<T>(input_ptrs, i, kernel_name_, &input);
      if (flag != 0) {
        return flag;
      }
      if (input_size_list_[i] == 0) {
        input = nullptr;
      }
      if (input != nullptr) {
        inputs_host_[current_dim] = input;
        current_dim++;
      }
    }
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(inputs_device, inputs_host_.get(), sizeof(T *) * input_num_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream)),
      "ParallelConcat opt cudaMemcpyAsync inputs failed");

    // call cuda kernel
    ParallelConcatKernel(output_size_, input_num_, inputs_device, output_ptr, device_id_,
                         reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }
  void ResetResource() override {
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
    axis_ = 0;
    input_num_ = 1;
    output_size_ = 0;
    inputs_host_ = nullptr;
  }

 private:
  int axis_;
  int input_num_;
  size_t output_size_;
  std::unique_ptr<T *[]> inputs_host_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_PARALLEL_CONCAT_HELPER_H_
