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

#include "plugin/device/cpu/kernel/get_squeeze_slice_shape_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <string>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"
#include "ops/op_name.h"
#include "mindspore/core/ops/get_squeeze_slice_shape.h"

namespace mindspore {
namespace kernel {
bool GetSqueezeSliceShapeCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::GetSqueezeSliceShape>(base_operator);
  tuple_index_types_ = GetValue<std::vector<int64_t>>(kernel_ptr->GetAttr(kAttrTupleIndexTypes));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  outputs_ = outputs;
  return true;
}

static inline void CheckCopy(void *dest, size_t destMax, const void *src, size_t count, const string &kernel_name) {
  auto cp_ret = memcpy_s(dest, destMax, src, count);
  if (cp_ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kernel_name << ", memcpy error, errorno: " << cp_ret;
  }
}

int GetSqueezeSliceShapeCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  data_shapes_ = GetShapes(inputs);
  return KRET_OK;
}

bool GetSqueezeSliceShapeCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &workspace,
                                                    const std::vector<AddressPtr> &outputs) {
  auto output_addr = reinterpret_cast<int64_t *>(outputs[kIndex0]->addr);

  ShapeVector data_shape = data_shapes_[0];
  std::vector<size_t> ini_index;
  size_t ellipse_position = 0;
  size_t not_ellipse_occupy_dims = 0;
  const size_t max_indices_num = 8;
  for (size_t j = 0; j < max_indices_num; j++) {
    if (tuple_index_types_[j] == kMetaTypeEllipsis) {
      ellipse_position = j;
    } else if (tuple_index_types_[j] != kTypeUnknown) {
      not_ellipse_occupy_dims += 1;
    }
  }
  std::vector<int64_t> new_data_shape;
  size_t ellipsis_range_size = data_shape.size() - not_ellipse_occupy_dims;
  for (size_t j = 0; j < max_indices_num; j++) {
    if (tuple_index_types_[j] == kObjectTypeTensorType) {
      if (j > ellipse_position) {
        (void)ini_index.emplace_back(j + ellipsis_range_size);
      } else {
        (void)ini_index.emplace_back(j);
      }
    }
  }
  for (size_t i = 0; i < data_shape.size(); i++) {
    if (!std::any_of(ini_index.begin(), ini_index.end(), [i](size_t x) { return x == i; })) {
      (void)new_data_shape.emplace_back(data_shape[i]);
    }
  }
  CheckCopy(output_addr, sizeof(int64_t) * new_data_shape.size(), new_data_shape.data(),
            sizeof(int64_t) * new_data_shape.size(), kernel_name_);
  return true;
}

bool GetSqueezeSliceShapeCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &workspace,
                                              const std::vector<AddressPtr> &outputs) {
  return kernel_func_(this, inputs, workspace, outputs);
}

std::vector<std::pair<KernelAttr, GetSqueezeSliceShapeCpuKernelMod::GetSqueezeSliceShapeFunc>>
  GetSqueezeSliceShapeCpuKernelMod::func_list_ = {};

std::vector<KernelAttr> GetSqueezeSliceShapeCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::vector<TypeId> data_type_ids = {kNumberTypeFloat16,   kNumberTypeFloat32,   kNumberTypeFloat64, kNumberTypeInt8,
                                       kNumberTypeInt16,     kNumberTypeInt32,     kNumberTypeInt64,   kNumberTypeUInt8,
                                       kNumberTypeUInt16,    kNumberTypeUInt32,    kNumberTypeUInt64,  kNumberTypeBool,
                                       kNumberTypeComplex64, kNumberTypeComplex128};
  std::transform(data_type_ids.begin(), data_type_ids.end(), std::back_inserter(func_list_),
                 [](TypeId data_type_id) -> std::pair<KernelAttr, GetSqueezeSliceShapeFunc> {
                   return {KernelAttr().AddInputAttr(data_type_id).AddOutputAttr(kNumberTypeInt64),
                           &GetSqueezeSliceShapeCpuKernelMod::LaunchKernel};
                 });
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, GetSqueezeSliceShapeFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, GetSqueezeSliceShape, GetSqueezeSliceShapeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
