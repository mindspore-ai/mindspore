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

#include "plugin/device/cpu/kernel/dynamic_stitch_cpu_kernel.h"
#include <functional>
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
int64_t GetShapeSize(const ShapeVector &shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());
}

template <typename T>
bool DynamicStitchCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                             const std::vector<kernel::KernelTensor *> &outputs) {
  int first_dim_size = 0;
  size_t input_count = inputs.size();
  input_tuple_num_ = input_count / 2;
  int max_index = -1;
  for (size_t i = 0; i < input_tuple_num_; ++i) {
    auto indice = reinterpret_cast<int32_t *>(inputs[i]->device_ptr());
    auto shape_size = GetShapeSize(inputs[i]->GetShapeVector());
    for (auto j = 0; j < shape_size; ++j) {
      max_index = std::max(indice[j], max_index);
    }
  }
  first_dim_size = max_index + 1;

  std::vector<TypeId> dtypes{outputs[kIndex0]->dtype_id()};
  result_shape_.push_back(first_dim_size);
  const auto &data0_shape = inputs[input_tuple_num_]->GetShapeVector();
  auto indice_dims = inputs[kIndex0]->GetShapeVector().size();
  for (size_t d = indice_dims; d < data0_shape.size(); ++d) {
    result_shape_.emplace_back(data0_shape[d]);
  }

  size_t num_out_dims = 2;
  ShapeVector out_dims(num_out_dims, 0);
  for (size_t out_dim = 0; out_dim <= num_out_dims - 1; ++out_dim) {
    out_dims[out_dim] = out_dim >= result_shape_.size() ? 1 : result_shape_[out_dim];
  }
  for (size_t in_dim = num_out_dims; in_dim < result_shape_.size(); ++in_dim) {
    out_dims[num_out_dims - 1] *= result_shape_[in_dim];
  }

  auto merged = reinterpret_cast<T *>(outputs[kIndex0]->device_ptr());
  size_t slice_size = LongToSize(out_dims[1]);
  size_t slice_bytes = slice_size * sizeof(T);
  for (size_t i = 0; i < input_tuple_num_; i++) {
    auto indice = reinterpret_cast<int32_t *>(inputs[i]->device_ptr());
    auto data = reinterpret_cast<T *>(inputs[i + input_tuple_num_]->device_ptr());
    auto shape_size = GetShapeSize(inputs[i]->GetShapeVector());
    for (auto j = 0; j < shape_size; ++j) {
      auto ret = memcpy_s(merged + indice[j] * slice_size, slice_bytes, data + j * slice_size, slice_bytes);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s error. Error no: " << ret;
      }
    }
  }
  return true;
}

void DynamicStitchCpuKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                         const std::vector<KernelTensor *> &outputs) {
  outputs[kIndex0]->SetShapeVector(result_shape_);
  auto data_dtype = inputs[kIndex1]->dtype_id();
  auto data_dtype_size = GetTypeByte(TypeIdToType(data_dtype));
  size_t batch_size = std::accumulate(result_shape_.cbegin(), result_shape_.cend(), 1, std::multiplies<size_t>());
  outputs[kIndex0]->set_size(batch_size * data_dtype_size);
}

std::vector<std::pair<KernelAttr, DynamicStitchCpuKernelMod::DynamicStitchFunc>> DynamicStitchCpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeFloat32),
    &DynamicStitchCpuKernelMod::LaunchKernel<float>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt8)
      .AddOutputAttr(kNumberTypeInt8),
    &DynamicStitchCpuKernelMod::LaunchKernel<int8_t>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt16)
      .AddOutputAttr(kNumberTypeInt16),
    &DynamicStitchCpuKernelMod::LaunchKernel<int16_t>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt32),
    &DynamicStitchCpuKernelMod::LaunchKernel<int32_t>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64),
    &DynamicStitchCpuKernelMod::LaunchKernel<int64_t>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeUInt8)
      .AddOutputAttr(kNumberTypeUInt8),
    &DynamicStitchCpuKernelMod::LaunchKernel<uint8_t>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeUInt16)
      .AddOutputAttr(kNumberTypeUInt16),
    &DynamicStitchCpuKernelMod::LaunchKernel<uint16_t>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeUInt32)
      .AddOutputAttr(kNumberTypeUInt32),
    &DynamicStitchCpuKernelMod::LaunchKernel<uint32_t>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeUInt64)
      .AddOutputAttr(kNumberTypeUInt64),
    &DynamicStitchCpuKernelMod::LaunchKernel<uint64_t>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeBool)
      .AddOutputAttr(kNumberTypeBool),
    &DynamicStitchCpuKernelMod::LaunchKernel<bool>}};

int DynamicStitchCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DynamicStitchFunc> &pair) { return pair.first; });
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(EXCEPTION) << "DynamicStitch does not support this kernel data type: " << kernel_attr;
  }

  kernel_func_ = func_list_[index].second;
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DynamicStitch, DynamicStitchCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
