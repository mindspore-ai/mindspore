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

#include "plugin/device/cpu/kernel/remake_tuple_index_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <string>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "mindspore/core/ops/remake_tuple_index.h"
#include "ops/op_name.h"

namespace mindspore {
namespace kernel {
bool RemakeTupleIndexCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto kernel_ptr = std::dynamic_pointer_cast<ops::RemakeTupleIndex>(base_operator);
  tuple_index_types_ = GetValue<std::vector<int64_t>>(kernel_ptr->GetAttr(kAttrTupleIndexTypes));
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int RemakeTupleIndexCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  valid_tensor_num_ = GetShapes(inputs)[0].size();
  return KRET_OK;
}

bool RemakeTupleIndexCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                                const std::vector<AddressPtr> &outputs) {
  auto output_attr = reinterpret_cast<char *>(outputs[kIndex0]->addr);
  size_t ellipse_position = 0;
  size_t not_ellipsis_position_cnt = 0;
  for (size_t i = 0; i < 8; i++) {
    if (tuple_index_types_[i] == kMetaTypeEllipsis) {
      ellipse_position = i;
    } else if (tuple_index_types_[i] != kTypeUnknown) {
      not_ellipsis_position_cnt += 1;
    }
  }
  std::vector<char *> inputs_host;

  for (size_t i = 0; i < ellipse_position; i++) {
    (void)inputs_host.emplace_back(reinterpret_cast<char *>(inputs[kIndex1 + i]->addr));
  }
  size_t ellipse_count = valid_tensor_num_ - not_ellipsis_position_cnt;
  for (size_t i = 0; i < ellipse_count; i++) {
    (void)inputs_host.emplace_back(reinterpret_cast<char *>(inputs[kIndex1 + not_ellipsis_position_cnt + i]->addr));
  }
  size_t remain_dims = valid_tensor_num_ - inputs_host.size();
  for (size_t i = 0; i < remain_dims; i++) {
    (void)inputs_host.emplace_back(reinterpret_cast<char *>(inputs[kIndex1 + ellipse_position + i]->addr));
  }
  // multi-threading
  size_t copy_time = output_size_list_[0] / sizeof(int64_t);
  auto task = [&](size_t start, size_t end) {
    for (size_t pos = start; pos < end; ++pos) {
      size_t cur_input_index = pos % valid_tensor_num_;
      size_t local_idx = pos / valid_tensor_num_;
      (void)memcpy_s(output_attr + sizeof(int64_t) * pos, sizeof(int64_t),
                     inputs_host[cur_input_index] + sizeof(int64_t) * local_idx, sizeof(int64_t));
    }
  };
  CPUKernelUtils::ParallelForAutoSearch(task, copy_time, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, RemakeTupleIndexCpuKernelMod::RemakeTupleIndexFunc>>
  RemakeTupleIndexCpuKernelMod::func_list_ = {};

std::vector<KernelAttr> RemakeTupleIndexCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::vector<TypeId> data_type_ids = {kNumberTypeFloat16,   kNumberTypeFloat32,   kNumberTypeFloat64, kNumberTypeInt8,
                                       kNumberTypeInt16,     kNumberTypeInt32,     kNumberTypeInt64,   kNumberTypeUInt8,
                                       kNumberTypeUInt16,    kNumberTypeUInt32,    kNumberTypeUInt64,  kNumberTypeBool,
                                       kNumberTypeComplex64, kNumberTypeComplex128};
  std::transform(data_type_ids.begin(), data_type_ids.end(), std::back_inserter(func_list_),
                 [](TypeId data_type_id) -> std::pair<KernelAttr, RemakeTupleIndexFunc> {
                   return {KernelAttr()
                             .AddInputAttr(data_type_id)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64),
                           &RemakeTupleIndexCpuKernelMod::LaunchKernel};
                 });

  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, RemakeTupleIndexFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RemakeTupleIndex, RemakeTupleIndexCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
