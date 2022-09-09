/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_LAYER_NORM_GRAD_GRAD_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_LAYER_NORM_GRAD_GRAD_HELPER_H_

#include <vector>
#include <map>
#include <string>
#include <utility>
#include <algorithm>
#include <memory>

#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "mindspore/core/ops/grad/layer_norm_grad_grad.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/layer_norm_grad_grad_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr char float_type_id[6] = {"float"};
class LayerNormGradGradAttr : public GpuKernelAttrBase {
 public:
  LayerNormGradGradAttr() = default;
  ~LayerNormGradGradAttr() override = default;
  int begin_norm_axis;
  int begin_params_axis;
};

template <typename T>
class LayerNormGradGradHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit LayerNormGradGradHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~LayerNormGradGradHelperGpuKernel() = default;

  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    auto input_x_shape = input_shapes[kIndex0];
    is_null_input_ = CHECK_SHAPE_NULL(input_x_shape, kernel_name_, "input");
    if (begin_norm_axis_ < 0) {
      begin_norm_axis_ += input_x_shape.size();
    }

    if (begin_params_axis_ < 0) {
      begin_params_axis_ += input_x_shape.size();
    }

    if (IntToSize(begin_norm_axis_) > input_x_shape.size()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the value of 'begin_norm_axis' must be less than or equal "
                    << "to the dimension of input, but got begin_norm_axis: " << IntToSize(begin_norm_axis_)
                    << ", the dimension of input: " << input_x_shape.size();
    }
    for (size_t i = 0; i < IntToSize(begin_norm_axis_); i++) {
      input_row_ *= input_x_shape[i];
    }

    for (size_t i = begin_norm_axis_; i < input_x_shape.size(); i++) {
      input_col_ *= input_x_shape[i];
    }

    for (size_t i = begin_params_axis_; i < input_x_shape.size(); i++) {
      param_dim_ *= input_x_shape[i];
    }

    input_size_ = input_row_ * input_col_ * sizeof(T);
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(input_row_ * sizeof(T));
    input_size_list_.push_back(input_row_ * sizeof(T));
    input_size_list_.push_back(param_dim_ * sizeof(T));
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(param_dim_ * sizeof(T));
    input_size_list_.push_back(param_dim_ * sizeof(T));

    output_size_list_.push_back(input_size_);
    output_size_list_.push_back(input_size_);
    output_size_list_.push_back(param_dim_ * sizeof(T));

    work_size_list_.push_back(input_size_);
    work_size_list_.push_back(input_size_);

    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }

    // get device ptr input index output
    T *x = nullptr;
    T *dy = nullptr;
    T *var = nullptr;
    T *mean = nullptr;
    T *gamma = nullptr;
    T *grad_dx = nullptr;
    T *grad_dg = nullptr;
    T *grad_db = nullptr;
    T *d_x = nullptr;
    T *d_dy = nullptr;
    T *d_gamma = nullptr;
    T *global_sum1 = nullptr;
    T *global_sum2 = nullptr;

    int flag = GetDeviceAddress<T>(input_ptrs, kIndex0, kernel_name_, &x);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, kIndex1, kernel_name_, &dy);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, kIndex2, kernel_name_, &var);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, kIndex3, kernel_name_, &mean);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, kIndex4, kernel_name_, &gamma);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, kIndex5, kernel_name_, &grad_dx);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, kIndex6, kernel_name_, &grad_dg);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, kIndex7, kernel_name_, &grad_db);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(output_ptrs, kIndex0, kernel_name_, &d_x);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(output_ptrs, kIndex1, kernel_name_, &d_dy);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(output_ptrs, kIndex2, kernel_name_, &d_gamma);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(work_ptrs, kIndex0, kernel_name_, &global_sum1);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(work_ptrs, kIndex1, kernel_name_, &global_sum2);
    if (flag != 0) {
      return flag;
    }

    auto type_id = typeid(x).name();
    if (strcmp(float_type_id, type_id) != 0) {
      epsilon_ = 1e-7;
    }

    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemsetAsync(global_sum1, 0, input_size_, reinterpret_cast<cudaStream_t>(cuda_stream)),
      "Call cudaMemsetAsync global_sum1 failed");
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemsetAsync(global_sum2, 0, input_size_, reinterpret_cast<cudaStream_t>(cuda_stream)),
      "Call cudaMemsetAsync global_sum2 failed");
    CalLayerNormGradGrad(input_row_, input_col_, param_dim_, global_sum1, global_sum2, epsilon_, dy, x, mean, var,
                         gamma, grad_dx, grad_dg, grad_db, d_dy, d_x, d_gamma,
                         reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<LayerNormGradGradAttr>(kernel_attr);
    begin_norm_axis_ = attr_ptr_->begin_norm_axis;
    begin_params_axis_ = attr_ptr_->begin_params_axis;
  }

  void ResetResource() override {
    input_row_ = 1;
    input_col_ = 1;
    param_dim_ = 1;
    input_size_ = 1;
    epsilon_ = 1e-12;
    is_null_input_ = false;
    input_shape_.clear();
    output_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
  }

 private:
  std::shared_ptr<LayerNormGradGradAttr> attr_ptr_;
  std::vector<std::vector<int64_t>> input_shape_;  // 0:input_shape(y_grad) 2:index_shape(argmax)
  std::vector<int64_t> output_shape_;
  bool is_null_input_;
  int begin_norm_axis_;
  int begin_params_axis_;
  int input_row_ = 1;
  int input_col_ = 1;
  int param_dim_ = 1;
  int input_size_ = 1;
  T epsilon_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_LAYER_NORM_GRAD_GRAD_HELPER_HELPER_H_
