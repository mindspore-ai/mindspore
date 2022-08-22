/* Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_BINCOUNT_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_BINCOUNT_HELPER_H_

#include <map>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bincount_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kOutputIndex0 = 0;

template <typename T>
class BincountHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit BincountHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~BincountHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    constexpr size_t OUTPUT_NUM = 1;
    array_shape_ = input_shapes[kInputIndex0];
    size_shape_ = input_shapes[kInputIndex1];
    weights_shape_ = input_shapes[kInputIndex2];
    bins_shape_ = output_shapes[kOutputIndex0];

    int inp_flag = 0;
    size_t int32_size = sizeof(int32_t);
    size_t T_size = sizeof(T);

    size_t array_size = int32_size;
    for (const auto &val : array_shape_) {
      array_size *= val;
    }
    if (array_size == 0 && inp_flag == 0) {
      inp_flag = 1;
    }
    input_size_list_.emplace_back(array_size);

    size_t size_size = int32_size;
    for (const auto &val : size_shape_) {
      size_size *= val;
    }
    if (size_size == 0 && inp_flag == 0) {
      inp_flag = 1;
    }
    input_size_list_.emplace_back(size_size);

    size_t weights_size = T_size;
    for (const auto &val : weights_shape_) {
      weights_size *= val;
    }
    if (weights_shape_.size() == 0 || weights_size == 0) {
      has_weights_ = false;
    } else {
      input_size_list_.emplace_back(weights_size);
    }

    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }

    is_null_input_ = (inp_flag == 1 || out_flag == 1);
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    int32_t *array_ptr = nullptr;
    int32_t *size_ptr = nullptr;
    T *weight_ptr = nullptr;
    T *bins_ptr = nullptr;
    int flag = GetDeviceAddress<int32_t>(input_ptrs, kInputIndex0, kernel_name_, &array_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<int32_t>(input_ptrs, kInputIndex1, kernel_name_, &size_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(input_ptrs, kInputIndex2, kernel_name_, &weight_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, kOutputIndex0, kernel_name_, &bins_ptr);
    if (flag != 0) {
      return flag;
    }

    int64_t dims = static_cast<int64_t>(array_shape_.size());
    int64_t threads_size = 1;
    for (int64_t i = dims - 1; i >= 0; i--) {
      threads_size *= array_shape_[i];
    }

    CalBincount(array_ptr, size_ptr, weight_ptr, bins_ptr, has_weights_, threads_size, bins_shape_[0], device_id_,
                reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

 private:
  std::vector<int64_t> array_shape_;
  std::vector<int64_t> size_shape_;
  std::vector<int64_t> weights_shape_;
  std::vector<int64_t> bins_shape_;
  bool has_weights_{true};
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_BINCOUNT_HELPER_H_
