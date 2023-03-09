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

#include "plugin/device/cpu/kernel/sequence/list_insert_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kInputsNum = 3;
constexpr int kOutputsNum = 1;
constexpr int kTargetIndex = 2;
}  // namespace
bool ListInsertCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int ListInsertCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  list_shape_ = inputs[0]->GetShapeVector();
  element_shape_ = inputs[kTargetIndex]->GetShapeVector();
  if (list_shape_.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " the input list size must greater 0";
  }
  return KRET_OK;
}

template <typename T, typename S>
bool ListInsertCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs) {
  const auto input_addr = GetDeviceAddress<T>(inputs, 0);
  const auto index_addr = GetDeviceAddress<S>(inputs, 1);
  const auto target_addr = GetDeviceAddress<T>(inputs, kTargetIndex);
  auto output_addr = GetDeviceAddress<T>(outputs, 0);
  auto len_list = list_shape_[0];
  auto output_size = outputs[0]->size;
  auto target_size = inputs[kTargetIndex]->size;
  int64_t index = *index_addr;

  if (index < -len_list) {
    index = 0;
  }
  if (index > len_list) {
    index = len_list;
  }
  index = index < 0 ? index + len_list : index;
  size_t element_index_size =
    std::accumulate(element_shape_.begin(), element_shape_.end(), 1, std::multiplies<int64_t>());
  size_t output_offset = element_index_size * index;
  size_t input_tail = element_index_size * (len_list - index);

  if (output_size < output_offset * sizeof(T) + target_size + input_tail * sizeof(T)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " the output_size[" << output_size << "] must greater than input["
                      << len_list * element_index_size << "] + target[" << target_size << "]";
  }

  if (output_offset != 0) {
    auto cp_ret = memcpy_s(output_addr, output_size, input_addr, output_offset * sizeof(T));
    if (cp_ret != EOK) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
    }
    output_size -= output_offset * sizeof(T);
  }

  if (target_size != 0) {
    auto cp_ret = memcpy_s(output_addr + output_offset, output_size, target_addr, element_index_size * sizeof(T));
    if (cp_ret != EOK) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
    }
    output_size -= element_index_size * sizeof(T);
  }

  if (input_tail != 0) {
    auto cp_ret = memcpy_s(output_addr + output_offset + element_index_size, output_size, input_addr + output_offset,
                           input_tail * sizeof(T));
    if (cp_ret != EOK) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
    }
  }
  return true;
}

#define ADD_KERNEL(x_dtype, idx_dtype, x_type, idx_type)      \
  {                                                           \
    KernelAttr()                                              \
      .AddInputAttr(kObjectTypeTuple, x_dtype)                \
      .AddInputAttr(kObjectTypeNumber, idx_dtype)             \
      .AddInputAttr(x_dtype)                                  \
      .AddOutputAttr(kObjectTypeTuple, x_dtype),              \
      &ListInsertCpuKernelMod::LaunchKernel<x_type, idx_type> \
  }

#define ADD_KERNEL0(x_dtype, idx_dtype, x_type, idx_type)     \
  {                                                           \
    KernelAttr()                                              \
      .AddInputAttr(kObjectTypeTuple, x_dtype)                \
      .AddInputAttr(kObjectTypeNumber, idx_dtype)             \
      .AddInputAttr(kObjectTypeNumber, x_dtype)               \
      .AddOutputAttr(kObjectTypeTuple, x_dtype),              \
      &ListInsertCpuKernelMod::LaunchKernel<x_type, idx_type> \
  }

const std::vector<std::pair<KernelAttr, ListInsertCpuKernelMod::KernelRunFunc>> &ListInsertCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, ListInsertCpuKernelMod::KernelRunFunc>> func_list = {
    ADD_KERNEL(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t),
    ADD_KERNEL(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t),
    ADD_KERNEL(kNumberTypeInt32, kNumberTypeInt64, int, int64_t),
    ADD_KERNEL(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t),
    ADD_KERNEL(kNumberTypeFloat32, kNumberTypeInt32, float, int),
    ADD_KERNEL(kNumberTypeFloat64, kNumberTypeInt32, double, int),
    ADD_KERNEL(kNumberTypeInt32, kNumberTypeInt32, int, int),
    ADD_KERNEL(kNumberTypeInt64, kNumberTypeInt32, int64_t, int),
    ADD_KERNEL0(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t),
    ADD_KERNEL0(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t),
    ADD_KERNEL0(kNumberTypeInt32, kNumberTypeInt64, int, int64_t),
    ADD_KERNEL0(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t),
    ADD_KERNEL0(kNumberTypeFloat32, kNumberTypeInt32, float, int),
    ADD_KERNEL0(kNumberTypeFloat64, kNumberTypeInt32, double, int),
    ADD_KERNEL0(kNumberTypeInt32, kNumberTypeInt32, int, int),
    ADD_KERNEL0(kNumberTypeInt64, kNumberTypeInt32, int64_t, int)};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ListInsert, ListInsertCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
