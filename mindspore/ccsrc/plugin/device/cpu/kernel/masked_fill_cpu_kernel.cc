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

#include "plugin/device/cpu/kernel/masked_fill_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <functional>
#include <complex>
#include "mindspore/core/ops/masked_fill.h"

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
constexpr size_t kMaskedFillInputsNum = 3;
constexpr size_t kMaskedFillOutputsNum = 1;
}  // namespace

bool MaskedFillCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  batch_rank_ = base_operator->get_batch_rank();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "MaskedFill does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int MaskedFillCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KRET_OK;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }
  ShapeVector input_shape = inputs.at(kIndex0)->GetShapeVector();
  ShapeVector mask_shape = inputs.at(kIndex1)->GetShapeVector();
  ShapeVector value_shape = inputs.at(kIndex2)->GetShapeVector();
  ShapeVector output_shape = outputs.at(kIndex0)->GetShapeVector();
  need_broadcast_ = (input_shape == mask_shape) ? false : true;
  size_t batch_size = value_shape.size();
  if (LongToSize(batch_rank_) != batch_size) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the value shape size should equal to " << batch_rank_
                  << ", but got " << batch_size;
    return KRET_RESIZE_FAILED;
  }
  if (input_shape.size() < batch_size || mask_shape.size() < batch_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of input and mask should not be less than value's, but got input: "
                      << input_shape.size() << ", mask: " << mask_shape.size() << ", value:" << value_shape.size();
  }
  for (size_t i = 0; i < batch_size; i++) {
    if (input_shape[i] != mask_shape[i] && input_shape[i] != value_shape[i]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the first " << batch_size
                        << " shape should be the same for 'input', 'mask' and 'value', but got input shape: "
                        << input_shape << ", mask shape: " << mask_shape << ", value shape: " << value_shape;
    }
  }

  output_size_ = std::accumulate(output_shape.begin(), output_shape.end(), size_t(1), std::multiplies<size_t>());
  value_size_ =
    LongToSize(std::accumulate(value_shape.begin(), value_shape.end(), int64_t(1), std::multiplies<int64_t>()));
  MS_EXCEPTION_IF_ZERO("value_size", value_size_);
  inner_size_ = output_size_ / value_size_;
  MS_EXCEPTION_IF_ZERO("inner_size", inner_size_);
  if (need_broadcast_) {
    mask_index_.clear();
    input_index_.clear();
    mask_index_.resize(output_size_);
    input_index_.resize(output_size_);
    BroadcastIterator base_iter(input_shape, mask_shape, output_shape);
    base_iter.SetPos(0);
    for (size_t i = 0; i < output_size_; i++) {
      mask_index_[i] = base_iter.GetInputPosB();
      input_index_[i] = base_iter.GetInputPosA();
      base_iter.GenNextPos();
    }
  }

  return ret;
}

template <typename T>
bool MaskedFillCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaskedFillInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaskedFillOutputsNum, kernel_name_);

  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  auto mask = reinterpret_cast<bool *>(inputs[1]->addr);
  auto value = reinterpret_cast<T *>(inputs[2]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  if (need_broadcast_) {
    auto task = [this, input, mask, output, value](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        output[i] = mask[mask_index_[i]] ? value[i / inner_size_] : input[input_index_[i]];
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
    return true;
  }

  if (value_size_ == 1) {
    auto task = [this, input, mask, output, value](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        output[i] = mask[i] ? value[0] : input[i];
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  } else {
    auto task = [this, input, mask, output, value](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        output[i] = mask[i] ? value[i / inner_size_] : input[i];
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  }

  return true;
}

std::vector<std::pair<KernelAttr, MaskedFillCpuKernelMod::MaskedFillFunc>> MaskedFillCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &MaskedFillCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &MaskedFillCpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &MaskedFillCpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &MaskedFillCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &MaskedFillCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &MaskedFillCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &MaskedFillCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &MaskedFillCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeUInt16),
   &MaskedFillCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeUInt32)
     .AddOutputAttr(kNumberTypeUInt32),
   &MaskedFillCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeUInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   &MaskedFillCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   &MaskedFillCpuKernelMod::LaunchKernel<bool>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeComplex64)
     .AddOutputAttr(kNumberTypeComplex64),
   &MaskedFillCpuKernelMod::LaunchKernel<complex64>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeComplex128)
     .AddOutputAttr(kNumberTypeComplex128),
   &MaskedFillCpuKernelMod::LaunchKernel<complex128>},
};

std::vector<KernelAttr> MaskedFillCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaskedFillFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaskedFill, MaskedFillCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
