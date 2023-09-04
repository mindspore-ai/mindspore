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

#include "plugin/device/cpu/kernel/normalize_dim_index_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <string>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"
#include "ops/op_name.h"
#include "mindspore/core/ops/normalize_dim_index.h"

namespace mindspore {
namespace kernel {
bool NormalizeDimIndexCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::NormalizeDimIndex>(base_operator);
  tuple_index_types_ = GetValue<std::vector<int64_t>>(kernel_ptr->GetAttr(kAttrTupleIndexTypes));
  dim_index_ = LongToSize(GetValue<int64_t>(kernel_ptr->GetAttr(kAttrTupleIndexAxis)));
  expand_dims_cnt_ = LongToSize(GetValue<int64_t>(kernel_ptr->GetAttr(kAttrExpandDimsCnt)));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

static inline void CheckCopy(void *dest, size_t destMax, const void *src, size_t count, const string &kernel_name) {
  auto cp_ret = memcpy_s(dest, destMax, src, count);
  if (cp_ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kernel_name << ", memcpy error, errorno: " << cp_ret;
  }
}

int NormalizeDimIndexCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  data_shapes_ = GetShapes(inputs);
  return KRET_OK;
}

bool NormalizeDimIndexCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &outputs) {
  auto output_addr = static_cast<int64_t *>(outputs[kIndex0]->addr);
  ShapeVector data_shape = data_shapes_[0];
  auto new_dim_index = ops::NormalizeDimIndex::ConstNormalizeDimIndex(data_shape.size() + expand_dims_cnt_, dim_index_,
                                                                      tuple_index_types_, 0);
  CheckCopy(output_addr, sizeof(int64_t), &new_dim_index, sizeof(int64_t), kernel_name_);
  return true;
}

bool NormalizeDimIndexCpuKernelMod::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                           const std::vector<AddressPtr> &outputs) {
  return kernel_func_(this, outputs);
}

std::vector<std::pair<KernelAttr, NormalizeDimIndexCpuKernelMod::NormalizeDimIndexFunc>>
  NormalizeDimIndexCpuKernelMod::func_list_ = {};

std::vector<KernelAttr> NormalizeDimIndexCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::vector<TypeId> data_type_ids = {kNumberTypeFloat16,   kNumberTypeFloat32,   kNumberTypeFloat64, kNumberTypeInt8,
                                       kNumberTypeInt16,     kNumberTypeInt32,     kNumberTypeInt64,   kNumberTypeUInt8,
                                       kNumberTypeUInt16,    kNumberTypeUInt32,    kNumberTypeUInt64,  kNumberTypeBool,
                                       kNumberTypeComplex64, kNumberTypeComplex128};
  std::transform(data_type_ids.begin(), data_type_ids.end(), std::back_inserter(func_list_),
                 [](TypeId data_type_id) -> std::pair<KernelAttr, NormalizeDimIndexFunc> {
                   return {KernelAttr().AddInputAttr(data_type_id).AddOutputAttr(kNumberTypeInt64),
                           &NormalizeDimIndexCpuKernelMod::LaunchKernel};
                 });
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, NormalizeDimIndexFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NormalizeDimIndex, NormalizeDimIndexCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
