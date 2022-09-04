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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_CUMPROD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_CUMPROD_GPU_KERNEL_H_

#include <vector>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cumprod_impl.cuh"
#include "mindspore/core/ops/cumprod.h"

namespace mindspore {
namespace kernel {
constexpr int kMaxDimsSize = 3;
template <typename T>
class CumProdGpuKernelMod : public NativeGpuKernelMod {
 public:
  CumProdGpuKernelMod()
      : exclusive_(false),
        reverse_(false),
        is_null_input_(false),
        axis_(0),
        input_size_0_(0),
        stride_(0),
        stride2_(0) {}
  ~CumProdGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    T *ws_addr = GetDeviceAddress<T>(workspace, 0);
    CumProd(input_addr, output_addr, ws_addr, dims_[0], dims_[1], dims_[2], stride_, stride2_, exclusive_, reverse_,
            reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    auto kernel_ptr = std::dynamic_pointer_cast<ops::CumProd>(base_operator);
    MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
    kernel_name_ = kernel_ptr->name();
    exclusive_ = kernel_ptr->GetExclusive();
    reverse_ = kernel_ptr->GetReverse();
    axis_ = static_cast<int32_t>(kernel_ptr->GetAxis());

    if (inputs.size() != 1) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << inputs.size();
      return false;
    }
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override {
    int ret = KRET_OK;
    if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
      return ret;
    }
    input_size_0_ = sizeof(T);
    auto shape_signed = inputs[kIndex0]->GetShapeVector();
    shape_ = Convert2SizeTClipNeg(shape_signed);
    is_null_input_ = CHECK_SHAPE_NULL(shape_, kernel_name_, "input");
    if (is_null_input_) {
      workspace_size_list_.push_back(input_size_0_);
      return KRET_OK;
    }

    int input_dim_length = SizeToInt(shape_.size());
    if (axis_ >= input_dim_length) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the value of 'axis' should be less than " << input_dim_length
                    << ", but got " << axis_;
      return KRET_RESIZE_FAILED;
    }
    while (axis_ < 0) {
      axis_ += input_dim_length;
    }
    for (size_t i = 0; i < shape_.size(); i++) {
      input_size_0_ *= shape_[i];
    }
    Reshape();
    workspace_size_list_.push_back(input_size_0_);

    return KRET_OK;
  }

 private:
  void Reshape() {
    dims_[0] = 1;
    dims_[1] = shape_[IntToSize(axis_)];
    dims_[2] = 1;
    for (size_t i = 0; i < IntToSize(axis_); i++) {
      dims_[0] *= shape_[i];
    }
    for (size_t i = IntToSize(axis_) + 1; i < shape_.size(); i++) {
      dims_[2] *= shape_[i];
    }
    stride_ = dims_[1] * dims_[2];
    stride2_ = dims_[2];
    return;
  }
  bool exclusive_;
  bool reverse_;
  bool is_null_input_;
  int axis_;
  size_t input_size_0_;
  size_t stride_;
  size_t stride2_;
  size_t dims_[kMaxDimsSize] = {};
  std::vector<size_t> shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_CUMPROD_GPU_KERNEL_H_
