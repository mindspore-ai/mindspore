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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_UNIQUE_CONSECUTIVE_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_UNIQUE_CONSECUTIVE_HELPER_H_
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unique_consecutive_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr size_t INPUT_NUM = 1;

class UniqueConsecutiveHelperBase : public GpuKernelHelperBase {
 public:
  explicit UniqueConsecutiveHelperBase(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {}
  virtual ~UniqueConsecutiveHelperBase() = default;

  void set_return_idx(bool return_idx) { return_idx_ = return_idx; }
  void set_return_counts(bool return_counts) { return_counts_ = return_counts; }
  void set_is_flattend(bool is_flattend) { is_flattend_ = is_flattend; }
  void set_axis(int64_t axis) { axis_ = axis; }

  bool return_idx() { return return_idx_; }
  bool return_counts() { return return_counts_; }
  bool is_flattend() { return is_flattend_; }
  int64_t axis() { return axis_; }

 protected:
  bool return_idx_{false};
  bool return_counts_{false};
  bool is_flattend_{true};
  int64_t axis_;
};

template <typename T, typename S>
class UniqueConsecutiveHelperGpuKernel : public UniqueConsecutiveHelperBase {
 public:
  explicit UniqueConsecutiveHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : UniqueConsecutiveHelperBase(kernel_name, device_id) {
    num_elements_ = 1;
  }
  virtual ~UniqueConsecutiveHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    int flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    if (flag != 0) {
      return flag;
    }
    input_shape_ = input_shapes[0];
    num_elements_ = input_size_list_[0] / sizeof(T);
    size_t input_tensor_size = input_size_list_[0];
    size_t elements_size = num_elements_ * sizeof(S);
    size_t elements_plus_one_size = (num_elements_ + 1) * sizeof(S);
    // input_index workspace
    work_size_list_.emplace_back(elements_size);
    // sorted_index workspace
    work_size_list_.emplace_back(elements_size);
    // range_data workspace
    work_size_list_.emplace_back(elements_plus_one_size);
    // indices_data workspace
    work_size_list_.emplace_back(input_tensor_size);
    // Transpose scalar workspace
    work_size_list_.emplace_back(input_shape_.size() * sizeof(size_t));
    work_size_list_.emplace_back(input_shape_.size() * sizeof(size_t));

    output_size_list_.emplace_back(input_tensor_size);
    output_size_list_.emplace_back(elements_size);
    output_size_list_.emplace_back(elements_size);
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    T *t_input_ptr = nullptr;
    S *s_input_index = nullptr;
    S *s_sorted_index = nullptr;
    S *s_range_data = nullptr;
    T *t_indices_data = nullptr;
    size_t *dev_input_shape = nullptr;
    size_t *dev_input_axis = nullptr;
    T *t_output_ptr = nullptr;
    S *s_output_index = nullptr;
    S *s_output_counts = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, kIndex0, kernel_name_, &t_input_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<S>(work_ptrs, kIndex0, kernel_name_, &s_input_index);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<S>(work_ptrs, kIndex1, kernel_name_, &s_sorted_index);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<S>(work_ptrs, kIndex2, kernel_name_, &s_range_data);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(work_ptrs, kIndex3, kernel_name_, &t_indices_data);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<size_t>(work_ptrs, kIndex4, kernel_name_, &dev_input_shape);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<size_t>(work_ptrs, kIndex5, kernel_name_, &dev_input_axis);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(output_ptrs, kIndex0, kernel_name_, &t_output_ptr);
    if (flag != 0) {
      return flag;
    }
    if (return_idx()) {
      flag = GetDeviceAddress<S>(output_ptrs, kIndex1, kernel_name_, &s_output_index);
      if (flag != 0) {
        return flag;
      }
    }
    if (return_counts()) {
      flag = GetDeviceAddress<S>(output_ptrs, kIndex2, kernel_name_, &s_output_counts);
      if (flag != 0) {
        return flag;
      }
    }

    auto status = CalUniqueConsecutive(t_input_ptr, num_elements_, input_shape_, is_flattend(), axis(), s_input_index,
                                       s_sorted_index, s_range_data, t_indices_data, dev_input_shape, dev_input_axis,
                                       t_output_ptr, s_output_index, s_output_counts,
                                       reinterpret_cast<cudaStream_t>(cuda_stream), &post_output_size_);
    CHECK_CUDA_STATUS(status, kernel_name_);
    return 0;
  }

  TensorInfo GetOutputTensorInfo() override {
    TensorInfo dyn_out;
    dyn_out.shapes = post_output_size_;
    return dyn_out;
  }

 private:
  int num_elements_;
  std::vector<int64_t> input_shape_;
  std::vector<std::vector<int>> post_output_size_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_UNIQUE_CONSECUTIVE_HELPER_H_
