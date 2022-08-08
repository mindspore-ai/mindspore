/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/pad_cpu_kernel.h"
#include <complex>
#include <map>
#include <utility>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kPadInputsNum = 1;
constexpr size_t kPadOutputsNum = 1;
constexpr size_t kPadElemSize = 2;
}  // namespace

bool PadCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                           const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.size() != kPadInputsNum || outputs.size() != kPadOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output tensor number must be " << kPadInputsNum << " and "
                  << kPadOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int PadCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs,
                            const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  auto paddings = base_operator->GetAttr(kAttrPaddings);
  MS_EXCEPTION_IF_NULL(paddings);
  paddings_ = GetValue<std::vector<std::vector<int64_t>>>(paddings);
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  input_shape_ = Convert2SizeTClipNeg(inputs[kIndex0]->GetShapeVector());
  is_null_input_ =
    (std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>()) == 0);
  if (is_null_input_) {
    return static_cast<int>(KRET_OK);
  }

  input_rank_ = input_shape_.size();
  if (paddings_.size() != input_rank_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of 'paddings' must be equal to the rank of the input, but got the "
                         "dimension of 'paddings': "
                      << paddings_.size() << ", and the rank of the input: " << input_rank_;
  }

  for (size_t i = 0; i < paddings_.size(); i++) {
    if (paddings_[i].size() != kPadElemSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', each element in 'paddings' must have size 2, but got: " << paddings_[i].size();
    }
    flattened_paddings_.push_back(LongToSize(paddings_[i][0]));
    flattened_paddings_.push_back(LongToSize(paddings_[i][1]));
  }

  input_size_ = 1;
  output_size_ = 1;
  for (size_t i = 0; i < input_rank_; i++) {
    input_size_ *= input_shape_[i];
    output_size_ *=
      (input_shape_[i] + flattened_paddings_[kPadElemSize * i] + flattened_paddings_[(kPadElemSize * i) + 1]);
  }

  if (input_rank_ < 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the rank of input must be greater than or equal to 1, but got the rank of input: "
                      << input_rank_;
  }
  if (output_shape.size() != input_rank_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the rank of input must be equal to the rank of output, but got the rank of input: "
                      << input_rank_ << ", and the rank of output: " << output_shape.size();
  }
  strides_.resize(input_rank_);
  strides_[input_rank_ - 1] = 1;
  for (int32_t i = SizeToInt(input_rank_) - 2; i >= 0; i--) {
    size_t ind = IntToSize(i);
    strides_[ind] = static_cast<size_t>(output_shape[ind + 1]) * strides_[ind + 1];
  }
  return static_cast<int>(KRET_OK);
}

template <typename T>
bool PadCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kPadInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPadOutputsNum, kernel_name_);
  const auto *inputs_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *outputs_addr = reinterpret_cast<T *>(outputs[0]->addr);
  if (memset_s(outputs_addr, outputs[0]->size, 0, outputs[0]->size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }

  auto task = [&inputs_addr, &outputs_addr, this](size_t start, size_t end) {
    for (size_t gt_id = start; gt_id < end; ++gt_id) {
      size_t linear_index = gt_id;
      size_t padded_linear_index = 0;
      for (size_t i = input_rank_; i >= 1; i--) {
        size_t unravel_dimension = input_shape_[i - 1];
        size_t unraveled_index = linear_index % unravel_dimension;
        padded_linear_index += ((unraveled_index + flattened_paddings_[kPadElemSize * (i - 1)]) * strides_[i - 1]);
        linear_index -= unraveled_index;
        linear_index /= unravel_dimension;
      }
      outputs_addr[padded_linear_index] = inputs_addr[gt_id];
    }
  };
  ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, PadCpuKernelMod::KernelRunFunc>> &PadCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, PadCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), &PadCpuKernelMod::LaunchKernel<bool>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), &PadCpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &PadCpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &PadCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &PadCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &PadCpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &PadCpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &PadCpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &PadCpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &PadCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &PadCpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &PadCpuKernelMod::LaunchKernel<std::complex<float>>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &PadCpuKernelMod::LaunchKernel<std::complex<double>>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Pad, PadCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
