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

#include "plugin/device/cpu/kernel/adaptive_max_pool2d_cpu_kernel.h"
#include <algorithm>
#include <random>
#include <utility>
#include <set>
#include <map>
#include <functional>
#include "abstract/utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/adaptive_max_pool2d.h"

namespace mindspore {
namespace kernel {
bool AdaptiveMaxPool2dCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::AdaptiveMaxPool2D>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "For primitive[AdaptiveMaxPool2D], cast op from BaseOperator to AdaptiveMaxPool2D failed.";
    return false;
  }
  kernel_name_ = base_operator->name();
  // (H_out, W_out)
  auto output_size = kernel_ptr->output_size();
  (void)std::copy(output_size.begin(), output_size.end(), std::back_inserter(attr_output_size_));
  return MatchKernelFunc(base_operator, inputs, outputs);
}

bool AdaptiveMaxPool2dCpuKernelMod::ResizedInputSize(const std::vector<KernelTensorPtr> &inputs) {
  auto input_shape = inputs[0]->GetShapeVector();
  size_t rank = static_cast<size_t>(input_shape.size());
  if (rank != ops::kFormatCHWShapeSize && rank != ops::kFormatNCHWShapeSize) {
    MS_LOG(ERROR) << "For primitive[AdaptiveMaxPool2D], the shape size of input argument[input_x] must "
                     "be 3 or 4, but got:"
                  << rank;
    return false;
  }

  input_height_ = static_cast<size_t>(input_shape[rank - ops::kOutputSizeAttrSize]);
  input_width_ = static_cast<size_t>(input_shape[rank - ops::kOutputSizeAttrSize + 1]);
  channel_size_ =
    static_cast<size_t>(rank == ops::kFormatCHWShapeSize ? input_shape[0] : input_shape[0] * input_shape[1]);
  input_hw_ = input_height_ * input_width_;
  return true;
}

bool AdaptiveMaxPool2dCpuKernelMod::ResizedOutputSize() {
  if (attr_output_size_.size() != ops::kOutputSizeAttrSize) {
    MS_LOG(ERROR) << "For primitive[AdaptiveMaxPool2D], the size of attr[output_size] should be 2, but got:"
                  << attr_output_size_.size();
    return false;
  }
  // If the output_size is none, the output shapes should be same as the input.
  output_height_ =
    (attr_output_size_[0] != ops::kPyValueNone ? static_cast<size_t>(attr_output_size_[0]) : input_height_);
  output_width_ =
    (attr_output_size_[1] != ops::kPyValueNone ? static_cast<size_t>(attr_output_size_[1]) : input_width_);
  if (output_height_ == 0 || output_width_ == 0) {
    MS_LOG(ERROR) << "Output range should not be zero.";
    return false;
  }
  output_hw_ = output_height_ * output_width_;
  return true;
}

int AdaptiveMaxPool2dCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::AdaptiveMaxPool2D>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "For primitive[AdaptiveMaxPool2D], cast op from BaseOperator to AdaptiveMaxPool2D failed.";
    return KRET_RESIZE_FAILED;
  }

  // Check the parameters valid.
  if (inputs.size() != 1) {
    MS_LOG(ERROR) << "For primitive[AdaptiveMaxPool2D], the size of input should be 1, but got " << inputs.size();
    return KRET_RESIZE_FAILED;
  }

  if (!ResizedInputSize(inputs)) {
    return KRET_RESIZE_FAILED;
  }

  if (!ResizedOutputSize()) {
    return KRET_RESIZE_FAILED;
  }

  return KRET_OK;
}

namespace {
struct LocalWindow {
  size_t h_begin;
  size_t h_end;
  size_t w_begin;
  size_t w_end;
};

template <typename T>
void ComputeLocalMax(size_t *max_indice, T *max_val, const LocalWindow &lw, size_t input_width, const T *input_ptr) {
  for (size_t h_index = lw.h_begin; h_index < lw.h_end; ++h_index) {
    for (size_t w_index = lw.w_begin; w_index < lw.w_end; ++w_index) {
      size_t indice = h_index * input_width + w_index;
      T val = input_ptr[indice];
      if (val > (*max_val)) {
        (*max_indice) = indice;
        (*max_val) = val;
      }
    }
  }
}
}  // namespace

template <typename T>
bool AdaptiveMaxPool2dCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                                 const std::vector<AddressPtr> &outputs) {
  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  int64_t *indices_addr = GetDeviceAddress<int64_t>(outputs, kIndex1);

  auto task = [this, &input_addr, &output_addr, &indices_addr](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      size_t input_offset = i * input_hw_;
      size_t output_offset = i * output_hw_;
      T *input_ptr = input_addr + input_offset;
      T *output_ptr = output_addr + output_offset;
      int64_t *indices_ptr = indices_addr + output_offset;

      for (size_t oh_index = 0; oh_index < output_height_; ++oh_index) {
        size_t h_begin = start_index(oh_index, output_height_, input_height_);
        size_t h_end = end_index(oh_index, output_height_, input_height_);

        for (size_t ow_index = 0; ow_index < output_width_; ++ow_index) {
          size_t w_begin = start_index(ow_index, output_width_, input_width_);
          size_t w_end = end_index(ow_index, output_width_, input_width_);

          // compute local max.
          size_t max_indice = h_begin * input_width_ + w_begin;
          T max_val = input_ptr[max_indice];

          LocalWindow lw;
          lw.h_begin = h_begin;
          lw.h_end = h_end;
          lw.w_begin = w_begin;
          lw.w_end = w_end;

          ComputeLocalMax(&max_indice, &max_val, lw, input_width_, input_ptr);
          size_t output_index = oh_index * output_width_ + ow_index;
          output_ptr[output_index] = max_val;
          indices_ptr[output_index] = SizeToLong(max_indice);
        }
      }
    }
  };

  ParallelLaunch(task, channel_size_, 0, this, pool_);
  return true;
}

const AdaptiveMaxPool2dCpuKernelMod::FuncList &AdaptiveMaxPool2dCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, AdaptiveMaxPool2dCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &AdaptiveMaxPool2dCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &AdaptiveMaxPool2dCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &AdaptiveMaxPool2dCpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
     &AdaptiveMaxPool2dCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
     &AdaptiveMaxPool2dCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
     &AdaptiveMaxPool2dCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AdaptiveMaxPool2D, AdaptiveMaxPool2dCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
