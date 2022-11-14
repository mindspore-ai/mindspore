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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_LIST_DIFF_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_LIST_DIFF_HELPER_H_
#include <string>
#include <vector>

#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/list_diff_impl.cuh"

namespace mindspore {
namespace cukernel {
template <typename T, typename S>
class ListDiffHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit ListDiffHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    num_elements_x_ = 0;
    num_elements_y_ = 0;
    post_output_size_ = 0;
  }
  virtual ~ListDiffHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    input_size_list_.emplace_back(input_shapes[kIndex0][kIndex0] * sizeof(T));
    input_size_list_.emplace_back(input_shapes[kIndex1][kIndex0] * sizeof(T));
    num_elements_x_ = input_size_list_[kIndex0] / sizeof(T);
    num_elements_y_ = input_size_list_[kIndex1] / sizeof(T);
    output_size_list_.emplace_back(num_elements_x_ * sizeof(T));
    output_size_list_.emplace_back(num_elements_x_ * sizeof(S));
    work_size_list_.emplace_back(sizeof(int));
    work_size_list_.emplace_back(num_elements_x_ * sizeof(int));
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    T *x_ptr = nullptr;
    T *y_ptr = nullptr;
    T *out_ptr = nullptr;
    S *idx_ptr = nullptr;
    int *count_number = nullptr;
    int *worksapce_flag_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, kIndex0, kernel_name_, &x_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(input_ptrs, kIndex1, kernel_name_, &y_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, kIndex0, kernel_name_, &out_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<S>(output_ptrs, kIndex1, kernel_name_, &idx_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<int>(work_ptrs, kIndex0, kernel_name_, &count_number);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<int>(work_ptrs, kIndex1, kernel_name_, &worksapce_flag_ptr);
    if (flag != 0) {
      return flag;
    }

    post_output_size_ = ListDiff(count_number, num_elements_x_, num_elements_y_, x_ptr, y_ptr, out_ptr, idx_ptr,
                                 worksapce_flag_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  TensorInfo GetOutputTensorInfo() override {
    TensorInfo dyn_out;
    dyn_out.shapes.push_back({{post_output_size_}});
    return dyn_out;
  }

 private:
  size_t num_elements_x_;
  size_t num_elements_y_;
  int post_output_size_ = 0;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_LIST_DIFF_HELPER_H_
