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

#include "plugin/device/cpu/kernel/sequence/list_append_and_insert_grad_cpu_kernel.h"
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
constexpr int kInputsNum = 2;
constexpr int kOutputsNum = 1;
}  // namespace
bool ListAppendAndInsertGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int ListAppendAndInsertGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs,
                                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  list_shape_ = inputs[0]->GetShapeVector();
  if (list_shape_.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " the input list size must greater 0";
  }
  return KRET_OK;
}

template <typename T, typename S>
bool ListAppendAndInsertGradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                       const std::vector<AddressPtr> &workspace,
                                                       const std::vector<AddressPtr> &outputs) {
  const auto input_addr = GetDeviceAddress<T>(inputs, 0);
  const auto index_addr = GetDeviceAddress<S>(inputs, 1);
  auto output_addr = GetDeviceAddress<T>(outputs, 0);
  auto len_list = list_shape_[0];
  auto output_size = outputs[0]->size;
  auto input_size = inputs[0]->size;
  int64_t index = *index_addr;

  if (index < -len_list || index >= len_list) {
    MS_EXCEPTION(ValueError) << "The prim '" << kernel_name_ << "', pop index[" << index << "] out of range.";
  }
  index = index < 0 ? index + len_list : index;

  if (input_size == 0) {
    MS_EXCEPTION(ValueError) << "The prim '" << kernel_name_ << "', pop from empty list";
  }
  if (output_size == 0) {
    return true;
  }

  size_t element_index_size = 1;
  if (list_shape_.size() > 1) {
    element_index_size = std::accumulate(list_shape_.begin() + 1, list_shape_.end(), 1, std::multiplies<int64_t>());
  }
  size_t output_offset = element_index_size * index;
  size_t input_tail = element_index_size * (len_list - index - 1);

  if (index != 0) {
    auto cp_ret = memcpy_s(output_addr, output_size, input_addr, output_offset * sizeof(T));
    if (cp_ret != EOK) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
    }
    output_size -= output_offset * sizeof(T);
  }

  if (input_tail != 0) {
    auto cp_ret = memcpy_s(output_addr + output_offset, output_size, input_addr + output_offset + element_index_size,
                           input_tail * sizeof(T));
    if (cp_ret != EOK) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
    }
  }
  return true;
}

const std::vector<std::pair<KernelAttr, ListAppendAndInsertGradCpuKernelMod::KernelRunFunc>>
  &ListAppendAndInsertGradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ListAppendAndInsertGradCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
     &ListAppendAndInsertGradCpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat64),
     &ListAppendAndInsertGradCpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
     &ListAppendAndInsertGradCpuKernelMod::LaunchKernel<int, int64_t>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
     &ListAppendAndInsertGradCpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
     &ListAppendAndInsertGradCpuKernelMod::LaunchKernel<float, int>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat64),
     &ListAppendAndInsertGradCpuKernelMod::LaunchKernel<double, int>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
     &ListAppendAndInsertGradCpuKernelMod::LaunchKernel<int, int>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
     &ListAppendAndInsertGradCpuKernelMod::LaunchKernel<int64_t, int>}};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ListAppendAndInsertGrad, ListAppendAndInsertGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
