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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_UNRAVEL_INDEX_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_UNRAVEL_INDEX_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unravel_index_impl.cuh"

namespace mindspore {
namespace cukernel {
template <typename T>
class UnravelIndexHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit UnravelIndexHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~UnravelIndexHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();

    constexpr size_t INPUT_NUM = 2;
    constexpr size_t OUTPUT_NUM = 1;
    int inp_flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    if (inp_flag == -1) {
      return inp_flag;
    }
    // get input shape vector
    input_indices_shape_ = input_shapes[0];
    input_dims_shape_ = input_shapes[1];

    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_input_ = (inp_flag == 1 || out_flag == 1);

    // emplace_back workspace_size
    size_t check_dims_ptr_workspace_size = sizeof(T);
    work_size_list_.emplace_back(check_dims_ptr_workspace_size);

    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    size_t indices_size = input_indices_shape_.size() == 0 ? 1 : input_indices_shape_[0];
    size_t dims_size = input_dims_shape_[0];

    T *input_indices_ptr = nullptr;
    T *input_dims_ptr = nullptr;
    T *output_ptr = nullptr;
    T *check_dims_ptr = nullptr;

    (void)GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_indices_ptr);
    (void)GetDeviceAddress<T>(input_ptrs, 1, kernel_name_, &input_dims_ptr);
    (void)GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output_ptr);
    (void)GetDeviceAddress<T>(work_ptrs, 0, kernel_name_, &check_dims_ptr);

    // call cuda kernel
    auto status = CalUnravelIndex(input_indices_ptr, input_dims_ptr, indices_size, dims_size, output_ptr, device_id_,
                                  reinterpret_cast<cudaStream_t>(cuda_stream));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return 0;
  }

 private:
  std::vector<int64_t> input_indices_shape_;
  std::vector<int64_t> input_dims_shape_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_UNRAVEL_INDEX_HELPER_H_
