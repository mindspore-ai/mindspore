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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_ADAPTIVEMAXPOOL2D_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_ADAPTIVEMAXPOOL2D_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include "mindspore/core/ops/adaptive_max_pool2d.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adaptive_max_pool2d_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class AdaptiveMaxPool2DKernelMod : public NativeGpuKernelMod {
 public:
  AdaptiveMaxPool2DKernelMod()
      : input_size_(0),
        output_size_(0),
        len_(0),
        input_height_(0),
        input_width_(0),
        output_height_(0),
        output_width_(0),
        size_(0),
        kernel_name_("AdaptiveMaxPool2D") {}
  ~AdaptiveMaxPool2DKernelMod() override = default;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override {
    if (!InitSize(base_operator, inputs, outputs)) {
      return KRET_RESIZE_FAILED;
    }
    return KRET_OK;
  }

  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
      KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64)};
    return support_list;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    int64_t *indices_addr = nullptr;
    indices_addr = GetDeviceAddress<int64_t>(outputs, 1);

    auto status = ApplyAdaptiveMaxPool2D(size_, input_height_, input_width_, output_height_, output_width_, input_addr,
                                         output_addr, indices_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    auto kernel_ptr = std::dynamic_pointer_cast<ops::AdaptiveMaxPool2D>(base_operator);
    if (kernel_ptr == nullptr) {
      MS_EXCEPTION(ValueError)
        << "For primitive[AdaptiveMaxPool2D], cast op from BaseOperator to AdaptiveMaxPool2D failed.";
      return false;
    }
    kernel_name_ = kernel_ptr->name();
    return InitSize(base_operator, inputs, outputs);
  }

  bool InitSize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                const std::vector<KernelTensorPtr> &outputs) {
    int ret = KernelMod::Resize(base_operator, inputs, outputs);
    if (ret != KRET_OK) {
      return ret;
    }
    auto kernel_ptr = std::dynamic_pointer_cast<ops::AdaptiveMaxPool2D>(base_operator);
    if (kernel_ptr == nullptr) {
      MS_EXCEPTION(ValueError)
        << "For primitive[AdaptiveMaxPool2D], cast op from BaseOperator to AdaptiveMaxPool2D failed.";
      return false;
    }

    // Check the parameters valid.
    if (inputs.size() != 1) {
      MS_EXCEPTION(ValueError) << "For primitive[AdaptiveMaxPool2D], the size of input should be 1, but got "
                               << inputs.size();
      return false;
    }
    MS_EXCEPTION_IF_NULL(inputs[0]);
    auto input_shape = inputs[0]->GetShapeVector();
    len_ = static_cast<size_t>(input_shape.size());
    if (len_ == 1 && input_shape[0] == ops::kDynamicRankValue) {
      return true;
    }
    if (len_ != ops::kFormatCHWShapeSize && len_ != ops::kFormatNCHWShapeSize) {
      MS_EXCEPTION(ValueError) << "For primitive[AdaptiveMaxPool2D], the shape size of input argument[input_x] must "
                                  "be 3 or 4, but got:"
                               << len_;
      return false;
    }

    input_height_ = static_cast<size_t>(input_shape[len_ - ops::kOutputSizeAttrSize]);
    input_width_ = static_cast<size_t>(input_shape[len_ - ops::kOutputSizeAttrSize + 1]);
    size_ = static_cast<size_t>(len_ == ops::kFormatCHWShapeSize ? input_shape[0] : input_shape[0] * input_shape[1]);
    input_size_ = sizeof(T);
    for (size_t i = 0; i < len_; i++) {
      input_size_ *= input_shape[i];
    }

    auto output_size = kernel_ptr->output_size();
    if (output_size.size() == ops::kOutputSizeAttrSize) {
      // If the output size is none, the output shapes should be same as the input.
      output_height_ = (output_size[0] != ops::kPyValueNone ? static_cast<size_t>(output_size[0]) : input_height_);
      output_width_ = (output_size[1] != ops::kPyValueNone ? static_cast<size_t>(output_size[1]) : input_width_);
    } else {
      MS_EXCEPTION(ValueError)
        << "For primitive[AdaptiveMaxPool2D], the size of attr[output_size] should be 2, but got:"
        << output_size.size();
      return false;
    }
    return true;
  }

 private:
  // The size of input memory.
  size_t input_size_;
  // The size of output memory.
  size_t output_size_;
  // The size of input shape.
  size_t len_;
  size_t input_height_;
  size_t input_width_;
  size_t output_height_;
  size_t output_width_;
  // The number of H*W.
  size_t size_;
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_ADAPTIVEMAXPOOL2D_GPU_KERNEL_H_
