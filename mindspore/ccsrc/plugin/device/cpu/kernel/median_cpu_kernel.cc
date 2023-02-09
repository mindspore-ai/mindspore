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

#include "plugin/device/cpu/kernel/median_cpu_kernel.h"

#include <functional>
#include <algorithm>
#include <type_traits>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/median.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMedianInputsNum = 1;
constexpr size_t kMedianOutputsNum = 2;
constexpr size_t kIndex0 = 0;
constexpr size_t kHalf = 2;
}  // namespace

bool MedianCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMedianInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMedianOutputsNum, kernel_name_);
  kernel_name_ = base_operator->GetPrim()->name();
  input_type_ = inputs[kIndex0]->GetDtype();

  auto kernel_ptr = std::dynamic_pointer_cast<ops::Median>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  global_median_ = kernel_ptr->get_global_median();
  axis_ = kernel_ptr->get_axis();
  keepdim_ = kernel_ptr->get_keep_dims();
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int MedianCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  input_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  input_dim_ = input_shape_.size();
  input_num_elements_ = 1;
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
  if (global_median_) {
    if (axis_ != 0) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', when 'global_median' is True, the 'axis' must be 0, but got "
                    << axis_;
      return KRET_RESIZE_FAILED;
    }
    if (keepdim_) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', when 'global_median' is True, the 'keep_dims' must be False, but got " << keepdim_;
      return KRET_RESIZE_FAILED;
    }
  }
  if (input_dim_ != 0) {
    if (axis_ > static_cast<int>(input_dim_ - 1) || axis_ < static_cast<int>(-input_dim_)) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the axis must be in [" << -input_dim_ << "," << input_dim_
                    << "), but got " << axis_ << ".";
      return KRET_RESIZE_FAILED;
    }
    for (size_t i = 0; i < input_dim_; i++) {
      input_num_elements_ *= static_cast<size_t>(input_shape_[i]);
    }
  } else {
    if (axis_ > 0 || axis_ < -1) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the axis must be in [" << -1 << "," << 1 << "), but got " << axis_
                    << ".";
      return KRET_RESIZE_FAILED;
    }
  }
  return KRET_OK;
}

const std::vector<std::pair<KernelAttr, MedianCpuKernelMod::KernelRunFunc>> &MedianCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, MedianCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
     &MedianCpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
     &MedianCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &MedianCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
     &MedianCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
     &MedianCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}

template <typename T>
bool MedianCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  if (global_median_ == false) {
    return MedianCompute<T>(inputs, outputs);
  } else {
    return GlobalMedianCompute<T>(inputs, outputs);
  }
  return true;
}

template <typename T>
bool MedianCpuKernelMod::GlobalMedianCompute(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &outputs) {
  auto *input0 = static_cast<T *>(inputs[0]->addr);
  auto *output0 = static_cast<T *>(outputs[0]->addr);
  output_num_elements_ = 1;
  int64_t median_pos = static_cast<int64_t>((input_num_elements_ - 1) / kHalf);
  std::nth_element(input0, input0 + median_pos, input0 + static_cast<int64_t>(input_num_elements_));
  *output0 = *(input0 + median_pos);
  return true;
}

template <typename T>
bool MedianCpuKernelMod::MedianCompute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto *input0 = static_cast<T *>(inputs[0]->addr);
  auto *output0 = static_cast<T *>(outputs[0]->addr);
  auto *output1 = static_cast<int64_t *>(outputs[1]->addr);
  if (input_dim_ == 0) {
    output_num_elements_ = 1;
    *output0 = *input0;
    *output1 = 0;
    return true;
  }
  if (axis_ < 0) {
    axis_ += static_cast<int>(input_dim_);
  }
  size_t dim_data_num = static_cast<size_t>(input_shape_[axis_]);
  T *temp_median_vec = new T[dim_data_num];
  int64_t *temp_median_index_vec = new int64_t[dim_data_num];
  size_t group = 1;
  size_t jump = 1;
  int64_t median_pos = static_cast<int64_t>((dim_data_num - 1) / kHalf);
  if (axis_ != 0) {
    for (size_t i = 0; i < static_cast<size_t>(axis_); i++) {
      group *= static_cast<size_t>(input_shape_[i]);
    }
  }
  if (axis_ != static_cast<int>(input_dim_ - 1)) {
    for (size_t i = static_cast<size_t>(axis_ + 1); i < input_dim_; i++) {
      jump *= static_cast<size_t>(input_shape_[i]);
    }
  }
  auto start = input0;
  for (size_t i = 0; i < group; i++) {
    for (size_t j = 0; j < jump; j++) {
      for (size_t k = 0; k < dim_data_num; k++) {
        auto num_index = start + k * jump + j;
        temp_median_index_vec[k] = static_cast<int64_t>(k);
        temp_median_vec[k] = *num_index;
      }
      std::nth_element(temp_median_index_vec, temp_median_index_vec + median_pos, temp_median_index_vec + dim_data_num,
                       [&temp_median_vec, dim_data_num](size_t pos1, size_t pos2) {
                         bool is_equal = false;
                         if constexpr (std::is_same_v<T, double>) {
                           is_equal = common::IsDoubleEqual(*(temp_median_vec + pos1), *(temp_median_vec + pos2));
                         } else if constexpr (std::is_same_v<T, float>) {
                           is_equal = common::IsFloatEqual(*(temp_median_vec + pos1), *(temp_median_vec + pos2));
                         }
                         return (*(temp_median_vec + pos1) < *(temp_median_vec + pos2)) ||
                                (pos1 < dim_data_num && is_equal && pos1 < pos2);
                       });
      std::nth_element(temp_median_vec, temp_median_vec + median_pos, temp_median_vec + dim_data_num);
      *(output0 + i * jump + j) = *(temp_median_vec + median_pos);
      *(output1 + i * jump + j) = *(temp_median_index_vec + median_pos);
    }
    if (i != group - 1) {
      start += jump * dim_data_num;
    }
  }
  delete[] temp_median_vec;
  delete[] temp_median_index_vec;
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Median, MedianCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
