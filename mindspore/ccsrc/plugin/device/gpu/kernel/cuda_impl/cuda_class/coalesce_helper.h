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
constexpr size_t WORK_NUM = 1;
constexpr size_t OUTPUT_NUM = 3;
constexpr int SHAPE = 2;

template <typename T>
class CoalesceHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit CoalesceHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    values_num = 1;
    output_shape_num = 0;
    indices_num = 1;
    shape_elements = 1;
    is_null_input_ = false;
  }

  virtual ~CoalesceHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    int flag = CalShapesSizeInBytes<int64_t>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    if (flag != 0) {
      return flag;
    }
    size_t workspace_size;
    for (const auto &val : input_shapes[1]) {
      values_num *= val;
    }
    for (const auto &val : input_shapes[0]) {
      indices_num *= val;
    }
    indices_num = indices_num / values_num;
    for (const auto &val : input_shapes[2]) {
      shape_elements *= val;
    }

    input_size_list_[1] = values_num * sizeof(T);
    workspace_size = values_num * sizeof(int64_t);
    work_size_list_.emplace_back(workspace_size);
    work_size_list_.emplace_back(workspace_size);
    work_size_list_.emplace_back(workspace_size);
    output_size_list_.emplace_back(input_size_list_[0]);
    output_size_list_.emplace_back(input_size_list_[1]);
    output_size_list_.emplace_back(shape_elements * sizeof(int64_t));
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    int64_t *input_indices = nullptr;
    T *input_values = nullptr;
    int64_t *input_shape = nullptr;
    int64_t *flatten_input_indices = nullptr;
    int64_t *unique_indices = nullptr;
    int64_t *origin_indices = nullptr;
    int64_t *output_indices = nullptr;
    T *output_value = nullptr;
    int64_t *output_shape = nullptr;

    int flag = GetDeviceAddress<int64_t>(input_ptrs, 0, kernel_name_, &input_indices);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, 1, kernel_name_, &input_values);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int64_t>(input_ptrs, SHAPE, kernel_name_, &input_shape);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int64_t>(work_ptrs, 0, kernel_name_, &flatten_input_indices);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int64_t>(work_ptrs, 1, kernel_name_, &unique_indices);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int64_t>(work_ptrs, SHAPE, kernel_name_, &origin_indices);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int64_t>(output_ptrs, 0, kernel_name_, &output_indices);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(output_ptrs, 1, kernel_name_, &output_value);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int64_t>(output_ptrs, SHAPE, kernel_name_, &output_shape);
    if (flag != 0) {
      return flag;
    }

    output_shape_num = Coalesce(origin_indices, unique_indices, shape_elements, indices_num, values_num,
                                flatten_input_indices, input_indices, input_values, input_shape, output_indices,
                                output_value, output_shape, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }
  TensorInfo GetOutputTensorInfo() override {
    TensorInfo dyn_out;
    dyn_out.shapes.push_back({{output_shape_num}});
    return dyn_out;
  }

 private:
  bool is_null_input_;
  size_t input_size_;
  size_t values_num;
  int output_shape_num;
  size_t indices_num;
  size_t shape_elements;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_COALESCE_HELPER_H_
