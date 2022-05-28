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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ADDCDIV_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ADDCDIV_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include <typeinfo>
#include <iostream>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/addcdiv_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr size_t MAX_SHAPE_SIZE = 7;
constexpr size_t kIdx2 = 2;
constexpr size_t kIdx3 = 3;
template <typename T, typename VT>
class AddcdivHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit AddcdivHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~AddcdivHelperGpuKernel() = default;

  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t OUTPUT_NUM = 1;

    ResetResource();

    input_data_shape_ = input_shapes[0];
    x1_shape_ = input_shapes[1];
    x2_shape_ = input_shapes[kIdx2];
    value_shape_ = input_shapes[kIdx3];
    size_t input_data_size = sizeof(T);
    for (auto val : input_shapes[0]) {
      input_data_size *= val;
    }
    input_size_list_.emplace_back(input_data_size);

    size_t x1_size = sizeof(T);
    for (auto val : input_shapes[1]) {
      x1_size *= val;
    }
    input_size_list_.emplace_back(x1_size);

    size_t x2_size = sizeof(T);
    for (auto val : input_shapes[kIdx2]) {
      x2_size *= val;
    }
    input_size_list_.emplace_back(x2_size);

    size_t value_size = sizeof(VT);
    for (auto val : input_shapes[kIdx3]) {
      value_size *= val;
    }
    input_size_list_.emplace_back(value_size);

    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }

    is_null_input_ = (out_flag == 1);

    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    T *input_data_ptr = nullptr;
    T *x1_ptr = nullptr;
    T *x2_ptr = nullptr;
    VT *value_ptr = nullptr;
    T *output_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_data_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, 1, kernel_name_, &x1_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, kIdx2, kernel_name_, &x2_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<VT>(input_ptrs, kIdx3, kernel_name_, &value_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    input_data_shape_.insert(input_data_shape_.begin(), MAX_SHAPE_SIZE - static_cast<int64_t>(input_data_shape_.size()),
                             1);
    x1_shape_.insert(x1_shape_.begin(), MAX_SHAPE_SIZE - static_cast<int64_t>(x1_shape_.size()), 1);
    x2_shape_.insert(x2_shape_.begin(), MAX_SHAPE_SIZE - static_cast<int64_t>(x2_shape_.size()), 1);
    value_shape_.insert(value_shape_.begin(), MAX_SHAPE_SIZE - static_cast<int64_t>(value_shape_.size()), 1);
    output_shape_.resize(MAX_SHAPE_SIZE, 1);

    for (size_t i = 0; i < MAX_SHAPE_SIZE; ++i) {
      if (input_data_shape_[i] != 1) {
        output_shape_[i] = input_data_shape_[i];
        continue;
      }
      if (x1_shape_[i] != 1) {
        output_shape_[i] = x1_shape_[i];
        continue;
      }
      if (x2_shape_[i] != 1) {
        output_shape_[i] = x2_shape_[i];
        continue;
      }
      if (value_shape_[i] != 1) {
        output_shape_[i] = value_shape_[i];
        continue;
      }
    }
    CalAddcdiv(input_data_shape_, x1_shape_, x2_shape_, value_shape_, output_shape_, input_data_ptr, x1_ptr, x2_ptr,
               value_ptr, output_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

 private:
  std::vector<int64_t> input_data_shape_;
  std::vector<int64_t> x1_shape_;
  std::vector<int64_t> x2_shape_;
  std::vector<int64_t> value_shape_;
  std::vector<int64_t> output_shape_;

  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ADDCDIV_HELPER_H_
