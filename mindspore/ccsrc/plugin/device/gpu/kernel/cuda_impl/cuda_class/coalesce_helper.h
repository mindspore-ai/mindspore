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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_COALESCE_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_COALESCE_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/coalesce_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr size_t INPUT_NUM = 3;
constexpr int DIM0 = 0;
constexpr int DIM1 = 1;
constexpr int DIM2 = 2;

template <typename T>
class CoalesceHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit CoalesceHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {}

  virtual ~CoalesceHelperGpuKernel() = default;
  void ResetResource() override {
    values_num_ = 1;
    indices_num_ = 1;
    shape_elements_ = 1;
    input_size_list_.clear();
    work_size_list_.clear();
    output_size_list_.clear();
  }
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    int flag = CalShapesSizeInBytes<int64_t>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    if (flag != 0) {
      return flag;
    }
    for (const auto &val : input_shapes[DIM1]) {
      values_num_ *= val;
    }
    for (const auto &val : input_shapes[DIM0]) {
      indices_num_ *= val;
    }
    indices_num_ = indices_num_ / values_num_;
    for (const auto &val : input_shapes[DIM2]) {
      shape_elements_ *= val;
    }

    input_size_list_[1] = values_num_ * sizeof(T);
    size_t workspace_size = values_num_ * sizeof(int64_t);
    work_size_list_.emplace_back(workspace_size);
    work_size_list_.emplace_back(workspace_size);
    work_size_list_.emplace_back(workspace_size);
    output_size_list_.emplace_back(input_size_list_[DIM0]);
    output_size_list_.emplace_back(input_size_list_[DIM1]);
    output_size_list_.emplace_back(shape_elements_ * sizeof(int64_t));
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    int64_t *input_indices = nullptr;
    T *input_values = nullptr;
    int64_t *input_shape = nullptr;
    int64_t *flatten_input_indices = nullptr;
    int64_t *unique_indices = nullptr;
    int64_t *origin_indices = nullptr;
    int64_t *output_indices = nullptr;
    T *output_value = nullptr;
    int64_t *output_shape = nullptr;

    (void)GetDeviceAddress<int64_t>(input_ptrs, DIM0, kernel_name_, &input_indices);
    (void)GetDeviceAddress<T>(input_ptrs, DIM1, kernel_name_, &input_values);
    (void)GetDeviceAddress<int64_t>(input_ptrs, DIM2, kernel_name_, &input_shape);
    (void)GetDeviceAddress<int64_t>(work_ptrs, DIM0, kernel_name_, &flatten_input_indices);
    (void)GetDeviceAddress<int64_t>(work_ptrs, DIM1, kernel_name_, &unique_indices);
    (void)GetDeviceAddress<int64_t>(work_ptrs, DIM2, kernel_name_, &origin_indices);
    (void)GetDeviceAddress<int64_t>(output_ptrs, DIM0, kernel_name_, &output_indices);
    (void)GetDeviceAddress<T>(output_ptrs, DIM1, kernel_name_, &output_value);
    (void)GetDeviceAddress<int64_t>(output_ptrs, DIM2, kernel_name_, &output_shape);
    int ret_flag_host = 0;

    output_shape_num_ =
      Coalesce(origin_indices, unique_indices, shape_elements_, indices_num_, values_num_, &ret_flag_host,
               flatten_input_indices, input_indices, input_values, input_shape, output_indices, output_value,
               output_shape, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    return ret_flag_host;
  }
  TensorInfo GetOutputTensorInfo() override {
    TensorInfo dyn_out;
    dyn_out.shapes.push_back({{output_shape_num_}});
    return dyn_out;
  }

 private:
  size_t values_num_;
  int output_shape_num_;
  size_t indices_num_;
  size_t shape_elements_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_COALESCE_HELPER_H_
