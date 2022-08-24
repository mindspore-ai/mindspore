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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SCATTER_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SCATTER_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include <map>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/scatter_impl.cuh"

namespace mindspore {
namespace cukernel {
static const std::map<std::string, ScatterType> kScatterTypeMap = {
  {"ScatterMul", SCATTER_MUL},
  {"ScatterDiv", SCATTER_DIV},
};
class ScatterAttr : public GpuKernelAttrBase {
 public:
  ScatterAttr() = default;
  ~ScatterAttr() override = default;
};

template <typename T, typename S>
class ScatterHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit ScatterHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~ScatterHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t INPUT_NUM = 3;
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();
    int inp_flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    if (inp_flag == -1) {
      return inp_flag;
    }
    input_shape_ = input_shapes[0];
    indices_shape_ = input_shapes[1];
    first_dim_size_ = input_shape_[0];
    input_size_ = 1;
    inner_size_ = 1;
    for (int64_t i = 1; i < static_cast<int64_t>(input_shape_.size()); i++) {
      inner_size_ *= input_shape_[i];
    }
    input_size_ = input_shape_[0] * inner_size_;
    indices_size_ = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(indices_shape_.size()); i++) {
      indices_size_ *= indices_shape_[i];
    }
    updates_size_ = 1;
    updates_size_ = indices_size_ * inner_size_;
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
    auto iter = kScatterTypeMap.find(kernel_name_);
    if (iter == kScatterTypeMap.end()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "Only support these scatter functors: ScatterMul, ScatterDiv "
                    << " currently, but got " << kernel_name_;
    } else {
      scatter_type_ = iter->second;
    }
    T *input = nullptr;
    S *indices = nullptr;
    T *updates = nullptr;
    T *output = nullptr;
    S size_limit = static_cast<S>(first_dim_size_);

    int flag = GetDeviceAddress<T>(input_ptrs, kIndex0, kernel_name_, &input);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<S>(input_ptrs, kIndex1, kernel_name_, &indices);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, kIndex2, kernel_name_, &updates);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(output_ptrs, kIndex0, kernel_name_, &output);
    if (flag != 0) {
      return flag;
    }
    // call cuda kernel
    Scatter(scatter_type_, size_limit, inner_size_, indices_size_, indices, updates, input, device_id_,
            reinterpret_cast<cudaStream_t>(cuda_stream));

    cudaError_t status = (cudaMemcpyAsync(&output[0], &input[0], input_size_ * sizeof(T), cudaMemcpyDeviceToDevice,
                                          reinterpret_cast<cudaStream_t>(cuda_stream)));
    if (status != cudaSuccess) {
      MS_LOG(ERROR) << "CUDA Error: "
                    << "cudaMemcpyAsync output failed"
                    << " | Error Number: " << status << " " << cudaGetErrorString(status);
    }
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<ScatterAttr>(kernel_attr);
  }

  void ResetResource() noexcept override {
    input_size_ = 0;
    inner_size_ = 0;
    indices_size_ = 0;
    updates_size_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
  }

 private:
  ScatterType scatter_type_;
  std::shared_ptr<ScatterAttr> attr_ptr_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> indices_shape_;
  size_t first_dim_size_;
  size_t input_size_;
  size_t inner_size_;
  size_t indices_size_;
  size_t updates_size_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ARGMAX_HELPER_H_
