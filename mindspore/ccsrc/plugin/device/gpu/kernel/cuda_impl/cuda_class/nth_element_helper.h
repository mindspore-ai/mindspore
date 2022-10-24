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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_NTH_ELEMENT_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_NTH_ELEMENT_HELPER_H_
#include <utility>
#include <memory>
#include <string>
#include <vector>
#include <limits>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/nth_element_impl.cuh"

namespace mindspore {
namespace cukernel {
class NthElementAttr : public GpuKernelAttrBase {
 public:
  NthElementAttr() = default;
  ~NthElementAttr() override = default;
  bool reverse;
  int32_t n;
};

template <typename T>
class NthElementHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit NthElementHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    reverse_ = false;
    is_null_input_ = false;
  }

  virtual ~NthElementHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();

    input_shape_ = input_shapes[0];
    input_n_shape_ = input_shapes[1];
    int32_t last_dim_num = input_shape_[static_cast<int64_t>(input_shape_.size() - 1)];
    int64_t outer_size = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(input_shape_.size() - 1); i++) {
      outer_size *= input_shape_[i];
    }
    int64_t inner_size = outer_size * last_dim_num;
    int64_t inner_n_size = 1;
    input_size_list_.emplace_back(inner_size * sizeof(T));
    input_size_list_.emplace_back(inner_n_size * sizeof(int32_t));
    output_size_list_.emplace_back(outer_size * sizeof(T));
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    T *input_ptr = nullptr;
    int32_t *input_n_ptr = nullptr;
    int32_t input_n;
    T *output_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<int32_t>(input_ptrs, 1, kernel_name_, &input_n_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&input_n, input_n_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost,
                                                       reinterpret_cast<cudaStream_t>(cuda_stream)),
                                       "cudaMemcpyAsync input_n failed");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaDeviceSynchronize(), "cudaDeviceSyncFailed - NthElement");

    if (input_n < 0 || input_n >= static_cast<int>(input_shape_.back())) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of input n must be in [0, input.shape[-1]), "
                        << "but got " << input_n << ".";
    }
    size_t slices_number = 1;
    for (int i = 0; i < static_cast<int>(input_shape_.size() - 1); i++) {
      slices_number *= input_shape_[i];
    }
    const size_t slice_size = input_shape_[input_shape_.size() - 1];
    CalNthElement(slices_number, slice_size, input_ptr, input_n, output_ptr, reverse_, device_id_,
                  reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<NthElementAttr>(kernel_attr);
  }

 protected:
  int CheckKernelParam() override {
    reverse_ = attr_ptr_->reverse;
    if (input_n_shape_.size() != 0) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the input n must be a scalar or a 0-D tensor but got a "
                    << input_n_shape_.size() << "-D tensor.";
      return -1;
    }
    if (input_shape_.size() < 1) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', input size must be equal or greater than 1, "
                    << "but got " << input_shape_.size() << ".";
      return -1;
    }
    return 0;
  }

 private:
  bool reverse_{false};
  std::shared_ptr<NthElementAttr> attr_ptr_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> input_n_shape_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_NthElement_HELPER_H_
