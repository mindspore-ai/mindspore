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

#include "plugin/device/cpu/kernel/glu_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "mindspore/core/ops/glu.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore::kernel {
namespace {
constexpr const size_t kGLUInputsNum = 1;
constexpr const size_t kGLUOutputsNum = 1;
constexpr const int64_t kParallelDataNum = 16 * 1024;
const int64_t kEvenNum = 2;
}  // namespace

template <typename T>
bool GLUCpuKernelMod::SplitWithDimZero(T *input_data_ptr, T *output_data_ptr) {
  int64_t copy_num = shape_value_ / value_shape_vec_[0];
  T *input_copy_ptr = input_data_ptr;
  if (value_shape_vec_[0] % kEvenNum != 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', x.shape[0] must be even, but got "
                  << value_shape_vec_[0] << ".";
    return false;
  }
  int64_t size_split = value_shape_vec_[0] / kEvenNum;
  int64_t copy_size_per = size_split * copy_num;
  input_copy_ptr += copy_size_per;
  auto task = [&](size_t start, size_t end) {
    for (size_t k = start; k < end; k++) {
      T val = *(input_copy_ptr + k);
      *(output_data_ptr + k) = (*(input_data_ptr + k)) * (exp(val) / (T(1) + exp(val)));
    }
  };
  if (copy_size_per < kParallelDataNum) {
    task(0, copy_size_per);
  } else {
    ParallelLaunchAutoSearch(task, copy_size_per, this, &parallel_search_info_);
  }

  return true;
}

template <typename T>
bool GLUCpuKernelMod::SplitCompute(T *input_data_ptr, T *output_data_ptr) {
  int64_t prefix = 1;
  for (int32_t i = 0; i < split_dim_; i++) {
    prefix *= value_shape_vec_[i];
  }
  int64_t midfix = value_shape_vec_[split_dim_];
  if (midfix % kEvenNum != 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', x.shape[" << split_dim_ << "] must be even, but got " << midfix << ".";
    return false;
  }
  int64_t size_split = midfix / kEvenNum;
  int64_t subfix = 1;
  for (size_t i = split_dim_ + 1; i < value_shape_vec_.size(); i++) {
    subfix *= value_shape_vec_[i];
  }
  int64_t offset = 0;
  T *input_copy_ptr = input_data_ptr;
  int64_t copy_num = subfix * size_split;
  offset += copy_num;
  input_data_ptr = input_data_ptr + offset;
  auto task = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      auto sharder_input_data = input_data_ptr + (i) * (subfix * midfix);
      auto sharder_input_copy = input_copy_ptr + (i) * (subfix * midfix);
      auto sharder_output = output_data_ptr + (i)*copy_num;
      for (int64_t k = 0; k < copy_num; k++) {
        T val = *(sharder_input_data + k);
        *(sharder_output + k) = (*(sharder_input_copy + k)) * (exp(val) / (T(1) + exp(val)));
      }
    }
  };
  if (prefix < kParallelDataNum) {
    task(0, prefix);
  } else {
    ParallelLaunchAutoSearch(task, prefix, this, &parallel_search_info_);
  }

  return true;
}

template <typename T>
bool GLUCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGLUInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGLUOutputsNum, kernel_name_);
  auto *input_ptr = static_cast<T *>(inputs[kIndex0]->addr);
  auto *output_ptr = static_cast<T *>(outputs[kIndex0]->addr);
  if (split_dim_ == 0) {
    return SplitWithDimZero<T>(input_ptr, output_ptr);
  } else {
    return SplitCompute<T>(input_ptr, output_ptr);
  }
}

const std::vector<std::pair<KernelAttr, GLUCpuKernelMod::KernelRunFunc>> &GLUCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, GLUCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &GLUCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &GLUCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &GLUCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}

bool GLUCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                           const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::GLU>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();
  split_dim_ = kernel_ptr->get_axis();
  value_shape_vec_ = inputs.at(kIndex0)->GetShapeVector();
  int64_t dim_value = SizeToLong(value_shape_vec_.size());
  for (auto &k : value_shape_vec_) {
    shape_value_ *= k;
  }

  if (split_dim_ < -dim_value || split_dim_ >= dim_value) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'axis' must be in range [" << -dim_value << ", " << dim_value
                  << "), but got " << split_dim_ << ".";
  }
  if (split_dim_ < 0) {
    split_dim_ += dim_value;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, GLU, GLUCpuKernelMod);
}  // namespace mindspore::kernel
