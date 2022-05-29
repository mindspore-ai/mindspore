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
#include "mindspore/core/ops/masked_fill.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaskedFillInputsNum = 3;
constexpr size_t kMaskedFillOutputsNum = 1;
}  // namespace

bool MaskedFillCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
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
  std::vector<int64_t> input_shape = inputs.at(kIndex0)->GetShapeVector();
  std::vector<int64_t> mask_shape = inputs.at(kIndex1)->GetShapeVector();
  std::vector<int64_t> value_shape = inputs.at(kIndex2)->GetShapeVector();
  std::vector<int64_t> output_shape = outputs.at(kIndex0)->GetShapeVector();
  std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToSize);
  std::transform(mask_shape.begin(), mask_shape.end(), std::back_inserter(mask_shape_), LongToSize);
  std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(output_shape_), LongToSize);
  need_broadcast_ = (input_shape_ == mask_shape_) ? false : true;
  size_t batch_size = value_shape.size();
  if (input_shape.size() <= batch_size || mask_shape.size() <= batch_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of input and mask should be greater than value's, but got input: "
                      << input_shape.size() << ", mask: " << mask_shape.size() << ", value:" << value_shape.size();
  }
  for (size_t i = 0; i < batch_size; i++) {
    if (input_shape[i] != mask_shape[i] && input_shape[i] != value_shape[i]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the first " << batch_size
                        << " shape should be the same for 'input', 'mask' and 'value', but got input shape: "
                        << input_shape << ", mask shape: " << mask_shape << ", value shape: " << value_shape;
    }
  }

  output_size_ = std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<size_t>());
  size_t rank_size = LongToSize(std::accumulate(value_shape.begin(), value_shape.end(), 1, std::multiplies<int64_t>()));
  inner_size_ = output_size_ / rank_size;
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
    BroadcastIterator base_iter(input_shape_, mask_shape_, output_shape_);
    auto task = [this, &base_iter, input, mask, output, value](size_t start, size_t end) {
      auto iter = base_iter;
      iter.SetPos(start);
      for (size_t i = start; i < end; i++) {
        output[i] = mask[iter.GetInputPosB()] ? value[i / inner_size_] : input[iter.GetInputPosA()];
        iter.GenNextPos();
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
    return true;
  }

  auto task = [this, input, mask, output, value](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      output[i] = mask[i] ? value[i / inner_size_] : input[i];
    }
  };

  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
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
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &MaskedFillCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &MaskedFillCpuKernelMod::LaunchKernel<int32_t>},
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
