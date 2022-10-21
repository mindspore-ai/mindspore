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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARGMAXANDMINWITHVALUE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARGMAXANDMINWITHVALUE_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include "mindspore/core/ops/argmax_with_value.h"
#include "mindspore/core/ops/argmin_with_value.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/general_reduction_impl.cuh"
namespace mindspore {
namespace kernel {
constexpr size_t kInputNum = 1;
constexpr size_t kOutputNum = 2;

template <typename T, typename S>
class ArgMaxAndMinWithValueGpuKernelMod : public NativeGpuKernelMod {
 public:
  ArgMaxAndMinWithValueGpuKernelMod() { ResetResource(); }
  ~ArgMaxAndMinWithValueGpuKernelMod() override = default;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override {
    if (!InitSize(base_operator, inputs, outputs)) {
      return KRET_RESIZE_FAILED;
    }
    return KRET_OK;
  }

  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
      KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
      KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
      KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16)};
    return support_list;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 1);
    S *index = GetDeviceAddress<S>(outputs, 0);
    CalGeneralReduction(small_, input, bound_, outer_size_, inner_size_, index, output,
                        reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    MS_EXCEPTION_IF_NULL(base_operator);
    kernel_name_ = base_operator->name();
    if (kernel_name_ != "ArgMaxWithValue" && kernel_name_ != "ArgMinWithValue") {
      MS_EXCEPTION(ArgumentError) << "The kernel must be either ArgMaxWithValue or ArgMinWithValue.";
    }

    // Check inputs and outputs size.
    if (inputs.size() != kInputNum) {
      MS_EXCEPTION(ArgumentError)
        << "For kernel mod[ArgMaxAndMinWithValueGpuKernelMod], the size of input should be 1, but got "
        << inputs.size();
    }
    if (outputs.size() != kOutputNum) {
      MS_EXCEPTION(ArgumentError)
        << "For kernel mod[ArgMaxAndMinWithValueGpuKernelMod], the size of output should be 2, but got "
        << outputs.size();
    }

    if (kernel_name_ == "ArgMinWithValue") {
      auto kernel_ptr = std::dynamic_pointer_cast<ops::ArgMinWithValue>(base_operator);
      MS_EXCEPTION_IF_NULL(kernel_ptr);
      axis_ = kernel_ptr->axis();
    } else {
      auto kernel_ptr = std::dynamic_pointer_cast<ops::ArgMaxWithValue>(base_operator);
      MS_EXCEPTION_IF_NULL(kernel_ptr);
      axis_ = kernel_ptr->axis();
    }
    small_ = (kernel_name_ == "ArgMinWithValue") ? true : false;
    return InitSize(base_operator, inputs, outputs);
  }

  bool InitSize(const BaseOperatorPtr &, const std::vector<KernelTensorPtr> &inputs,
                const std::vector<KernelTensorPtr> &outputs) {
    MS_EXCEPTION_IF_NULL(inputs[0]);
    auto shape = Convert2SizeTClipNeg(inputs[0]->GetShapeVector());
    MS_EXCEPTION_IF_NULL(outputs[0]);
    auto output_shape = Convert2SizeTClipNeg(outputs[0]->GetShapeVector());
    int64_t dims = SizeToLong(shape.size());

    // If the rank is uncertain, do not update the axis.
    ShapeVector dynamic_rank_shape = {-2};
    if (inputs[0]->GetShapeVector() != dynamic_rank_shape) {
      if (axis_ < -dims || axis_ >= dims) {
        MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the 'axis' must be in the range [-" << dims << ","
                                 << dims << "), but got " << axis_;
      }
      if (axis_ < 0) {
        axis_ += dims;
      }
    }

    input_size_ = sizeof(T);
    for (auto x : shape) {
      input_size_ *= x;
    }
    output_size_ = sizeof(S);
    for (auto x : output_shape) {
      output_size_ *= x;
    }
    bound_ = static_cast<S>(shape[axis_]);
    if (static_cast<S>(shape[axis_]) != bound_) {
      MS_EXCEPTION(ArgumentError) << "For '" << kernel_name_ << "', the value of shape[axis] must be "
                                  << static_cast<size_t>(bound_) << ", but got " << shape[axis_];
    }
    outer_size_ = 1;
    for (int64_t i = axis_ - 1; i >= 0; i--) {
      outer_size_ *= shape[i];
    }
    inner_size_ = 1;
    for (int64_t i = axis_ + 1; i < dims; i++) {
      inner_size_ *= shape[i];
    }

    input_size_list_.clear();
    output_size_list_.clear();
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    output_size_list_.push_back(output_size_ / sizeof(S) * sizeof(T));
    return true;
  }

  void ResetResource() noexcept {
    kernel_name_ = "";
    axis_ = 0;
    input_size_ = 0;
    output_size_ = 0;
    bound_ = 0;
    outer_size_ = 0;
    inner_size_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 private:
  bool small_ = false;
  int64_t axis_;
  size_t input_size_;
  size_t output_size_;
  S bound_;
  size_t outer_size_;
  size_t inner_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARGMAXANDMINWITHVALUE_GPU_KERNEL_H_
