/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/remove_expanded_dims_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <string>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"
#include "ops/op_name.h"
#include "mindspore/core/ops/remove_expanded_dims.h"

namespace mindspore {
namespace kernel {
bool RemoveExpandedDimsCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::RemoveExpandedDims>(base_operator);
  tuple_index_types_ = GetValue<std::vector<int64_t>>(kernel_ptr->GetAttr(kAttrTupleIndexTypes));
  has_true_ = GetValue<bool>(kernel_ptr->GetAttr(kAttrHasTrue));
  has_sequence_ = GetValue<bool>(kernel_ptr->GetAttr(kAttrHasSequence));
  rem_ndim_ = GetValue<int64_t>(kernel_ptr->GetAttr(kAttrExpandDimsCnt));
  empty_indices_out = GetValue<bool>(kernel_ptr->GetAttr(kAttrEmptyIndicesOut));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  is_need_retrieve_output_shape_ = true;
  return true;
}

static inline void CheckCopy(void *dest, size_t destMax, const void *src, size_t count, const string &kernel_name) {
  if (destMax == 0) {
    if (memset_s(dest, sizeof(int64_t), 0, sizeof(int64_t)) != EOK) {
      MS_LOG(EXCEPTION) << kernel_name << " memset error";
    }
    return;
  }
  if (memcpy_s(dest, destMax, src, count) != EOK) {
    MS_LOG(EXCEPTION) << "For " << kernel_name << ", memcpy error";
  }
}

int RemoveExpandedDimsCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_UNKNOWN_OUT_SHAPE && ret != KRET_OK) {
    return ret;
  }
  data_shapes_ = GetShapes(inputs);
  return KRET_OK;
}

bool RemoveExpandedDimsCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &workspace,
                                                  const std::vector<AddressPtr> &outputs) {
  const auto has_false_val_addr = reinterpret_cast<size_t *>(inputs[kIndex2]->addr);
  const auto broadcast_shape_val_addr = reinterpret_cast<int64_t *>(inputs[kIndex3]->addr);
  const auto idx_advanced_val_addr = reinterpret_cast<int64_t *>(inputs[kIndex4]->addr);

  bool has_false = has_false_val_addr[0] > 0;
  ShapeVector broadcast_shape;
  if (!data_shapes_[kIndex3].empty()) {
    for (size_t i = 0; i < LongToSize(data_shapes_[kIndex3][0]); i++) {
      (void)broadcast_shape.emplace_back(broadcast_shape_val_addr[i]);
    }
  }

  int64_t idx_advanced = idx_advanced_val_addr[0];
  auto indices_output_addr = reinterpret_cast<int64_t *>(outputs[kIndex0]->addr);
  auto new_value_shape_output_addr = reinterpret_cast<int64_t *>(outputs[kIndex1]->addr);
  auto new_idx_output_addr = reinterpret_cast<int64_t *>(outputs[kIndex2]->addr);
  ShapeVector data_shape = data_shapes_[0];
  ShapeVector value_shape = data_shapes_[1];
  size_t valid_tensor_nums = 0;
  for (size_t i = 0; i < tuple_index_types_.size(); i++) {
    if (tuple_index_types_[i] == kMetaTypeEllipsis) {
      valid_tensor_nums = data_shape.size() + rem_ndim_;
      break;
    } else if (tuple_index_types_[i] != kTypeUnknown) {
      valid_tensor_nums += 1;
    }
  }
  size_t rem_dim = SizeToLong(data_shape.size()) - (valid_tensor_nums - rem_ndim_);
  auto [indices_out, new_value_shape, new_idx_advanced] = ops::RemoveExpandedDims::ConstRemoveExpandedDims(
    has_true_, has_false, has_sequence_, broadcast_shape, rem_dim, value_shape, data_shape, empty_indices_out,
    idx_advanced, tuple_index_types_, rem_ndim_);

  CheckCopy(indices_output_addr, sizeof(int64_t), &indices_out, sizeof(int64_t), kernel_name_);
  CheckCopy(new_value_shape_output_addr, sizeof(int64_t) * new_value_shape.size(), new_value_shape.data(),
            sizeof(int64_t) * new_value_shape.size(), kernel_name_);
  CheckCopy(new_idx_output_addr, sizeof(int64_t), &new_idx_advanced, sizeof(int64_t), kernel_name_);
  out_shapes_ = std::vector<ShapeVector>(outputs.size(), ShapeVector());
  if (!new_value_shape.empty()) {
    out_shapes_[kIndex1] = ShapeVector{SizeToLong(new_value_shape.size())};
  }
  return true;
}

bool RemoveExpandedDimsCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs) {
  return kernel_func_(this, inputs, workspace, outputs);
}

std::vector<std::pair<KernelAttr, RemoveExpandedDimsCpuKernelMod::RemoveExpandedDimsFunc>>
  RemoveExpandedDimsCpuKernelMod::func_list_ = {};

std::vector<KernelAttr> RemoveExpandedDimsCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::vector<TypeId> data_type_ids = {kNumberTypeFloat16,   kNumberTypeFloat32,   kNumberTypeFloat64, kNumberTypeInt8,
                                       kNumberTypeInt16,     kNumberTypeInt32,     kNumberTypeInt64,   kNumberTypeUInt8,
                                       kNumberTypeUInt16,    kNumberTypeUInt32,    kNumberTypeUInt64,  kNumberTypeBool,
                                       kNumberTypeComplex64, kNumberTypeComplex128};
  std::transform(data_type_ids.begin(), data_type_ids.end(), std::back_inserter(func_list_),
                 [](TypeId data_type_id) -> std::pair<KernelAttr, RemoveExpandedDimsFunc> {
                   return {KernelAttr()
                             .AddInputAttr(data_type_id)
                             .AddInputAttr(data_type_id)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64),
                           &RemoveExpandedDimsCpuKernelMod::LaunchKernel};
                 });
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, RemoveExpandedDimsFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RemoveExpandedDims, RemoveExpandedDimsCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
