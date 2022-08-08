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

#include "plugin/device/cpu/kernel/max_pool_grad_with_argmax_cpu_kernel.h"
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/grad/max_pool_grad_with_argmax.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxPoolGradWithArgmaxInputsNum = 3;
constexpr size_t kMaxPoolGradWithArgmaxOutputsNum = 1;
constexpr size_t kDimLowerLimit = 4;
constexpr size_t kInputDims = 4;
}  // namespace

bool MaxPoolGradWithArgmaxCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MaxPoolGradWithArgmax>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Cast op from BaseOperator to MaxPoolingGradWithArgmax failed.";
    return false;
  }

  stride_height_ = LongToInt(kernel_ptr->get_strides()[kDim2]);
  stride_width_ = LongToInt(kernel_ptr->get_strides()[kDim3]);
  if (stride_height_ < 1 || stride_width_ < 1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', expected strides to be Union[int, tuple[int]] with value no less than 1 "
                                "but got the window height: "
                             << stride_height_ << ", and the window width: " << stride_height_;
  }
  pad_mode_ = kernel_ptr->get_pad_mode();
  // pair = [is_match, index]
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[pair.second].second;
  return true;
}

void MaxPoolGradWithArgmaxCpuKernelMod::ResizedInputSize(const std::vector<KernelTensorPtr> &inputs) {
  auto x_shape = inputs[kDim0]->GetShapeVector();
  if (x_shape.size() != kInputDims) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the input 'x' must be 4-dimensional.";
  }
  for (size_t i = 0; i < x_shape.size(); i++) {
    if (x_shape[i] <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', expected input 'x' has non-empty spatial dimensions, "
                                  "but 'x' has sizes "
                               << x_shape[i] << " wit the dimension " << i << " being empty.";
    }
  }
  batch_ = LongToInt(x_shape[kDim0]);
  channel_ = LongToInt(x_shape[kDim1]);
  x_height_ = LongToInt(x_shape[kDim2]);
  x_width_ = LongToInt(x_shape[kDim3]);

  auto dy_shape = inputs[kDim1]->GetShapeVector();
  // check the spatial dimensions of dy if needed
  for (size_t i = 0; i < dy_shape.size(); i++) {
    if (dy_shape[i] <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', expected input 'dy' has non-empty spatial dimensions, "
                                  "but 'dy' has sizes "
                               << dy_shape[i] << " wit the dimension " << i << " being empty.";
    }
  }
  dy_height_ = LongToInt(dy_shape[kDim2]);
  dy_width_ = LongToInt(dy_shape[kDim3]);
  auto index_shape = inputs[kDim2]->GetShapeVector();
  for (size_t i = 0; i < index_shape.size(); i++) {
    if (index_shape[i] <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', expected input 'index' has non-empty spatial dimensions, "
                                  "but 'index' has sizes "
                               << index_shape[i] << " wit the dimension " << i << " being empty.";
    }
  }

  if (x_shape.size() < kDimLowerLimit || dy_shape.size() < kDimLowerLimit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'x' and 'dy' cannot be less than 4, but got "
                      << "the dimension of 'x': " << x_shape.size() << ", the dimension of 'dy': " << dy_shape.size();
  }
}

int MaxPoolGradWithArgmaxCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaxPoolGradWithArgmaxInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaxPoolGradWithArgmaxOutputsNum, kernel_name_);
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MaxPoolGradWithArgmax>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Cast op from BaseOperator to MaxPoolingGradWithArgmax failed.";
    return KRET_RESIZE_FAILED;
  }
  ResizedInputSize(inputs);
  return KRET_OK;
}

template <typename T, typename S>
bool MaxPoolGradWithArgmaxCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                     const std::vector<kernel::AddressPtr> &outputs) {
  auto *input = reinterpret_cast<T *>(inputs.at(kDim0)->addr);
  MS_EXCEPTION_IF_NULL(input);
  auto *grad = reinterpret_cast<T *>(inputs.at(kDim1)->addr);
  MS_EXCEPTION_IF_NULL(grad);
  auto *index = reinterpret_cast<S *>(inputs.at(kDim2)->addr);
  MS_EXCEPTION_IF_NULL(index);
  auto *output = reinterpret_cast<T *>(outputs.at(kDim0)->addr);
  MS_EXCEPTION_IF_NULL(output);
  const int c = channel_;
  const int xCHW = c * x_height_ * x_width_;
  const int dyCHW = c * dy_height_ * dy_width_;
  const size_t outputLength = IntToSize(batch_ * xCHW);
  auto init = [output](size_t start, size_t end) {
    const T zero = static_cast<T>(0);
    for (size_t i = start; i < end; ++i) {
      output[i] = zero;
    }
  };
  ParallelLaunchAutoSearch(init, outputLength, this, &parallel_search_info_);
  const size_t length = IntToSize(batch_ * dyCHW);
  auto task = [input, output, grad, index, &xCHW, &dyCHW](size_t start, size_t end) {
    for (int i = SizeToInt(start); i < SizeToInt(end); ++i) {
      const int idx = static_cast<int>(index[i]);
      const int posn = i / dyCHW;
      output[posn * xCHW + idx] += grad[i];
    }
  };
  ParallelLaunchAutoSearch(task, length, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, MaxPoolGradWithArgmaxCpuKernelMod::MaxPoolGradWithArgmaxFunc>>
  MaxPoolGradWithArgmaxCpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &MaxPoolGradWithArgmaxCpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &MaxPoolGradWithArgmaxCpuKernelMod::LaunchKernel<float, int64_t>},
};

std::vector<KernelAttr> MaxPoolGradWithArgmaxCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaxPoolGradWithArgmaxFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaxPoolGradWithArgmax, MaxPoolGradWithArgmaxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
