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

#include "plugin/device/cpu/kernel/median_grad_cpu_kernel.h"

#include <functional>
#include <algorithm>
#include <type_traits>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/grad/median_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMedianGradInputsNum = 4;
constexpr size_t kMedianGradOutputsNum = 1;
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kHalf = 2;
}  // namespace

bool MedianGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMedianGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMedianGradOutputsNum, kernel_name_);
  kernel_name_ = base_operator->GetPrim()->name();
  input0_type_ = inputs[kIndex0]->GetDtype();
  input1_type_ = inputs[kIndex1]->GetDtype();

  auto kernel_ptr = std::dynamic_pointer_cast<ops::MedianGrad>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  global_median_ = kernel_ptr->get_global_median();
  axis_ = kernel_ptr->get_axis();
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int MedianGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  std::vector<size_t> src_shape;
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(src_shape), LongToSize);
  size_t input_element_num = std::accumulate(src_shape.begin(), src_shape.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_element_num == 0);
  if (is_null_input_) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', input tensor[0] got 'shapes[" << kIndex0 << "]' is "
                    << input_element_num;
    return KRET_OK;
  }
  input0_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  input1_shape_ = inputs[kIndex1]->GetDeviceShapeAdaptively();
  input2_shape_ = inputs[kIndex2]->GetDeviceShapeAdaptively();
  input0_dim_ = input0_shape_.size();
  input1_dim_ = input1_shape_.size();
  input2_dim_ = input2_shape_.size();
  input0_num_elements_ = 1;
  input1_num_elements_ = 1;
  axis_ = axis_ >= 0 ? axis_ : axis_ + static_cast<int>(input1_dim_);

  for (size_t i = 0; i < input1_dim_; i++) {
    input1_num_elements_ *= static_cast<size_t>(input1_shape_[i]);
  }
  for (size_t i = 0; i < input0_dim_; i++) {
    input0_num_elements_ *= static_cast<size_t>(input0_shape_[i]);
  }
  if (input0_type_ != input1_type_) {
    MS_LOG(ERROR) << "For " << kernel_name_ << ", the dtype of y_grad should be same with x, but got " << input0_type_
                  << ".";
    return KRET_RESIZE_FAILED;
  }
  if (input0_dim_ != input2_dim_) {
    MS_LOG(ERROR) << "For " << kernel_name_ << ", the shape of y_grad should be same with y, but got " << input0_shape_
                  << ".";
    return KRET_RESIZE_FAILED;
  }
  for (size_t i = 0; i < input2_dim_; i++) {
    if (input0_shape_[i] != input2_shape_[i]) {
      MS_LOG(ERROR) << "For " << kernel_name_ << ", the shape of y_grad should be same with y, but got "
                    << input0_shape_ << ".";
      return KRET_RESIZE_FAILED;
    }
  }
  return KRET_OK;
}

const std::vector<std::pair<KernelAttr, MedianGradCpuKernelMod::KernelRunFunc>> &MedianGradCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, MedianGradCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &MedianGradCpuKernelMod::LaunchKernel<int16_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &MedianGradCpuKernelMod::LaunchKernel<int32_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &MedianGradCpuKernelMod::LaunchKernel<int64_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &MedianGradCpuKernelMod::LaunchKernel<float, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &MedianGradCpuKernelMod::LaunchKernel<double, double>},
  };
  return func_list;
}

template <typename T1, typename T2>
bool MedianGradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  if (global_median_ == false) {
    return MedianGradCompute<T1, T2>(inputs, outputs);
  } else {
    return GlobalMedianGradCompute<T1, T2>(inputs, outputs);
  }
  return true;
}

template <typename T1, typename T2>
bool MedianGradCpuKernelMod::GlobalMedianGradCompute(const std::vector<AddressPtr> &inputs,
                                                     const std::vector<AddressPtr> &outputs) const {
  auto y_grad = static_cast<T1 *>(inputs[0]->addr);
  auto x = static_cast<T1 *>(inputs[1]->addr);
  auto y = static_cast<T1 *>(inputs[2]->addr);
  auto x_grad = static_cast<T2 *>(outputs[0]->addr);

  int64_t count_repeat = 0;
  for (size_t i = 0; i < input1_num_elements_; i++) {
    bool is_equal = false;
    if constexpr (std::is_same_v<T1, double>) {
      is_equal = common::IsDoubleEqual(*(x + i), *y);
    } else if constexpr (std::is_same_v<T1, float>) {
      is_equal = common::IsFloatEqual(*(x + i), *y);
    }

    count_repeat += is_equal ? 1 : 0;
  }
  auto sharder_mediangrad = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      bool is_equal = false;
      if constexpr (std::is_same_v<T1, double>) {
        is_equal = common::IsDoubleEqual(*(x + i), *y);
      } else if constexpr (std::is_same_v<T1, float>) {
        is_equal = common::IsFloatEqual(*(x + i), *y);
      }

      *(x_grad + i) = is_equal ? static_cast<T2>(*y_grad / count_repeat) : 0;
    }
  };
  CPUKernelUtils::ParallelFor(sharder_mediangrad, input1_num_elements_);
  return true;
}

template <typename T1, typename T2>
bool MedianGradCpuKernelMod::MedianGradCompute(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &outputs) {
  auto y_grad = static_cast<T1 *>(inputs[0]->addr);
  auto indices = static_cast<int64_t *>(inputs[3]->addr);
  auto x_grad = static_cast<T2 *>(outputs[0]->addr);

  for (size_t i = 0; i < input1_num_elements_; i++) {
    *(x_grad + i) = 0;
  }
  std::vector<int64_t> shape_keepdim;
  for (size_t i = 0; i < input1_dim_; i++) {
    if (i == static_cast<size_t>(axis_)) {
      shape_keepdim.push_back(1);
    } else {
      shape_keepdim.push_back(input1_shape_[i]);
    }
  }

  std::vector<int64_t> element_num_each_dim_x;
  std::vector<int64_t> element_num_each_dim_y;
  int64_t element_num_y = 1;
  int64_t element_num_x = 1;
  for (size_t i = 0; i < shape_keepdim.size(); i++) {
    (void)element_num_each_dim_x.insert(element_num_each_dim_x.begin(), element_num_x);
    element_num_x *= input1_shape_[shape_keepdim.size() - 1 - i];
    (void)element_num_each_dim_y.insert(element_num_each_dim_y.begin(), element_num_y);
    element_num_y *= shape_keepdim[shape_keepdim.size() - 1 - i];
  }

  auto sharder_mediangrad = [&](int64_t start, int64_t end) {
    std::vector<int64_t> dim_vec;
    for (size_t i = 0; i < input1_dim_; i++) {
      dim_vec.push_back(0);
    }
    for (int64_t nth_element = start; nth_element < end; nth_element++) {
      int64_t elements_remain = nth_element;
      for (size_t i = 0; i < input1_dim_; i++) {
        dim_vec[i] = elements_remain / element_num_each_dim_y[i];
        elements_remain %= element_num_each_dim_y[i];
      }
      int64_t update_element_pos = 0;
      for (size_t i = 0; i < input1_dim_; i++) {
        if (i == static_cast<size_t>(axis_)) {
          update_element_pos += *(indices + nth_element) * element_num_each_dim_x[i];
        } else {
          update_element_pos += dim_vec[i] * element_num_each_dim_x[i];
        }
      }
      *(x_grad + update_element_pos) = static_cast<T2>(*(y_grad + nth_element));
    }
  };
  CPUKernelUtils::ParallelFor(sharder_mediangrad, input0_num_elements_);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MedianGrad, MedianGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
