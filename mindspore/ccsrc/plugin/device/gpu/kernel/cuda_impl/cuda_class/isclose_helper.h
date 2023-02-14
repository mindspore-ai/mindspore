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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ISCLOSE_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ISCLOSE_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/isclose_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr size_t INPUT_NUM = 2;
constexpr size_t WORK_NUM = 1;
constexpr size_t OUTPUT_NUM = 1;
constexpr int MAX_DIMS = 7;

class IsCloseAttr : public GpuKernelAttrBase {
 public:
  IsCloseAttr() = default;
  ~IsCloseAttr() override = default;
  float rtol = 1e-05;
  float atol = 1e-08;
  bool equal_nan = false;
};

template <typename T>
class IsCloseHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit IsCloseHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    output_shape_num = 0;
    indices_num = 1;
    shape_elements = 1;
    is_null_input_ = false;
  }

  virtual ~IsCloseHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    int inp_flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    if (inp_flag == -1) {
      return inp_flag;
    }
    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_input_ = (inp_flag == 1 || out_flag == 1);

    auto inputx_shape = input_shapes[0];
    auto inputy_shape = input_shapes[1];
    auto output_shape = output_shapes[0];
    if (is_null_input_) {
      return 0;
    }

    for (size_t i = 0; i < inputx_shape.size(); i++) {
      if (inputx_shape[i] != inputy_shape[i]) {
        need_broadcast_ = true;
      }
    }

    lhs_shape_.resize(MAX_DIMS, 1);
    rhs_shape_.resize(MAX_DIMS, 1);
    output_shape_.resize(MAX_DIMS, 1);
    output_num_ = 1;
    for (size_t i = 0; i < output_shape.size(); i++) {
      if (need_broadcast_) {
        output_shape_[i] = output_shape[i];
      }
      output_num_ *= output_shape[i];
    }
    int lhs_offset = output_shape.size() - inputx_shape.size();
    for (size_t j = 0; j < inputx_shape.size(); j++) {
      if (need_broadcast_) {
        if ((j + lhs_offset) >= 0 && (j + lhs_offset) < MAX_DIMS) {
          lhs_shape_[j + lhs_offset] = inputx_shape[j];
        }
      }
    }
    int rhs_offset = output_shape.size() - inputy_shape.size();
    for (size_t k = 0; k < inputy_shape.size(); k++) {
      if (need_broadcast_) {
        if ((k + rhs_offset) >= 0 && (k + rhs_offset) < MAX_DIMS) {
          rhs_shape_[k + rhs_offset] = inputy_shape[k];
        }
      }
    }
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    atol = attr_ptr_->atol;
    rtol = attr_ptr_->rtol;
    equal_nan = attr_ptr_->equal_nan;

    T *inputx = nullptr;
    T *inputy = nullptr;
    bool *output = nullptr;

    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &inputx);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, 1, kernel_name_, &inputy);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<bool>(output_ptrs, 0, kernel_name_, &output);
    if (flag != 0) {
      return flag;
    }

    if (need_broadcast_) {
      BroadcastIsClose(lhs_shape_, rhs_shape_, output_shape_, inputx, inputy, rtol, atol, equal_nan, output, device_id_,
                       reinterpret_cast<cudaStream_t>(cuda_stream));
    } else {
      IsClose(output_num_, inputx, inputy, rtol, atol, equal_nan, output, device_id_,
              reinterpret_cast<cudaStream_t>(cuda_stream));
    }

    return 0;
  }
  TensorInfo GetOutputTensorInfo() override {
    TensorInfo dyn_out;
    dyn_out.shapes.push_back({{output_shape_num}});
    return dyn_out;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<IsCloseAttr>(kernel_attr);
  }

 private:
  std::vector<size_t> lhs_shape_;
  std::vector<size_t> rhs_shape_;
  std::vector<size_t> output_shape_;
  bool equal_nan;
  float atol;
  float rtol;
  std::shared_ptr<IsCloseAttr> attr_ptr_;
  bool is_null_input_;
  size_t input_size_;
  bool need_broadcast_;
  size_t output_num_;
  int output_shape_num;
  size_t indices_num;
  size_t shape_elements;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ISCLOSE_HELPER_H_
