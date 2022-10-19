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

#include <functional>
#include <limits>
#include "mindspore/core/ops/edit_distance.h"
#include "plugin/device/cpu/kernel/edit_distance_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr char normalize[] = "normalize";
constexpr size_t kMaximumInputsNum = 6;
constexpr size_t kMaximumOutputsNum = 1;
}  // namespace

bool EditDistanceCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);

  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "EditDistance does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int EditDistanceCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  for (size_t i = 0; i < kMaximumInputsNum; ++i) {
    shapes_.push_back(inputs[i]->GetShapeVector());
  }
  shapes_.push_back(outputs[0]->GetShapeVector());
  auto kernel_ptr = std::make_shared<ops::EditDistance>(base_operator->GetPrim());
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  normalize_ = kernel_ptr->normalize();
  return KRET_OK;
}

template <typename T1, typename T2>
bool EditDistanceCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaximumInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaximumOutputsNum, kernel_name_);

  std::vector<int64_t> &output_shape = shapes_[kIndex6];
  const size_t rank = output_shape.size();
  if (rank < 1) {
    MS_EXCEPTION(RuntimeError) << "For '" << kernel_name_ << "', output's shape size must be greater than 0.";
  }

  const auto *hypothesis_indices_addr = reinterpret_cast<T1 *>(inputs[0]->addr);
  const auto *hypothesis_values_addr = reinterpret_cast<T2 *>(inputs[1]->addr);
  const auto *truth_indices_addr = reinterpret_cast<T1 *>(inputs[3]->addr);
  const auto *truth_values_addr = reinterpret_cast<T2 *>(inputs[4]->addr);
  auto *output_addr = reinterpret_cast<float *>(outputs[0]->addr);

  const size_t hypothesis_values_length = inputs[1]->size / sizeof(T2);
  const size_t truth_values_length = inputs[4]->size / sizeof(T2);
  const size_t output_length = outputs[0]->size / sizeof(float);

  std::vector<size_t> output_strides(rank);
  output_strides[rank - 1] = 1;
  for (size_t d = rank - 1; d >= 1; d--) {
    output_strides[d - 1] = output_strides[d] * LongToSize(output_shape[d]);
  }

  for (size_t output_index = 0; output_index < output_length; output_index++) {
    // produce hypothesis sequence
    std::vector<T2> hypothesis_seq;
    for (size_t i = 0; i < hypothesis_values_length; i++) {
      size_t loc = 0;
      for (size_t j = 0; j < rank; j++) {
        int64_t index = hypothesis_indices_addr[i * (rank + 1) + j];
        if (index >= output_shape[j] || index < 0) {
          MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the " << i << "th hypothesis_value in " << j
                                   << "th dimension index: " << index << " of 'output' out of bounds: [0, "
                                   << output_shape[j] << "). ";
        }
        loc += LongToSize(index) * output_strides[j];
      }
      if (loc == output_index) {
        hypothesis_seq.push_back(hypothesis_values_addr[i]);
      }
    }
    size_t hypothesis_seq_size = hypothesis_seq.size();
    // produce truth sequence
    std::vector<T2> truth_seq;
    for (size_t i = 0; i < truth_values_length; i++) {
      size_t loc = 0;
      for (size_t j = 0; j < rank; j++) {
        int64_t index = truth_indices_addr[i * (rank + 1) + j];
        if (index >= output_shape[j] || index < 0) {
          MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the " << i << "th truth_value in " << j
                                   << "th dimension index: " << index << " of 'output' out of bounds: [0, "
                                   << output_shape[j] << "). ";
        }
        loc += LongToSize(index) * output_strides[j];
      }
      if (loc == output_index) {
        truth_seq.push_back(truth_values_addr[i]);
      }
    }
    size_t truth_seq_size = truth_seq.size();
    // calculate distance
    if (normalize_ && truth_seq_size == 0) {
      output_addr[output_index] =
        (hypothesis_seq_size != 0 ? std::numeric_limits<float>::infinity() : static_cast<float>(0));
      continue;
    }
    auto cmp = std::equal_to<T2>();
    size_t dis = LevenshteinDistance<T2>(truth_seq, hypothesis_seq, cmp);
    output_addr[output_index] = (normalize_ ? SizeToFloat(dis) / SizeToFloat(truth_seq_size) : SizeToFloat(dis));
  }
  return true;
}

std::vector<std::pair<KernelAttr, EditDistanceCpuKernelMod::EditDistanceFunc>> EditDistanceCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &EditDistanceCpuKernelMod::LaunchKernel<int64_t, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &EditDistanceCpuKernelMod::LaunchKernel<int64_t, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &EditDistanceCpuKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &EditDistanceCpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &EditDistanceCpuKernelMod::LaunchKernel<int64_t, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &EditDistanceCpuKernelMod::LaunchKernel<int64_t, uint16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &EditDistanceCpuKernelMod::LaunchKernel<int64_t, uint32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &EditDistanceCpuKernelMod::LaunchKernel<int64_t, uint64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &EditDistanceCpuKernelMod::LaunchKernel<int64_t, float16>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &EditDistanceCpuKernelMod::LaunchKernel<int64_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &EditDistanceCpuKernelMod::LaunchKernel<int64_t, double>}};

std::vector<KernelAttr> EditDistanceCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, EditDistanceFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, EditDistance, EditDistanceCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
