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

#include "plugin/device/cpu/kernel/sequence/sequence_getitem_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "kernel/kernel.h"
#include "ops/op_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/common/thread_pool.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kInputsNum = 2;
constexpr int kOutputsNum = 1;
}  // namespace
bool SequenceGetItemCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int SequenceGetItemCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (inputs.empty() || inputs[0] == nullptr) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " the input is invalid, input size:" << inputs.size();
  }

  auto tuple_shape = inputs[0]->GetShapeVector();
  if (tuple_shape.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " the input tuple size must greater 0";
  }
  auto length = tuple_shape[0];
  if (length <= 0) {
    MS_EXCEPTION(ValueError) << "For RealTupleGetItem, the element size of tuple input should great than 0, but got "
                             << length << ".";
  }

  auto index_value = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  if (index_value >= length || index_value < -length) {
    MS_EXCEPTION(ValueError) << "For RealTupleGetItem, index is out of range: [" << -length << ", " << length
                             << "), but got " << index_value << ".";
  }
  if (index_value < 0) {
    index_value += length;
  }
  auto index = LongToSize(index_value);
  auto user_data = inputs[0]->user_data();
  if (user_data == nullptr || !user_data->has(kRealElementsSize)) {
    offset_size_ = inputs[0]->size() / LongToSize(length) * index;
    return KRET_OK;
  }

  // The input tuple contain different inner element size.
  auto real_elements_size = user_data->get<std::vector<size_t>>(kRealElementsSize);
  MS_EXCEPTION_IF_NULL(real_elements_size);
  output_size_list_ = {(*real_elements_size)[index]};
  offset_size_ = (*std::max_element(real_elements_size->begin(), real_elements_size->end())) * index;

  return KRET_OK;
}

template <typename T>
bool SequenceGetItemCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &,
                                               const std::vector<KernelTensor *> &outputs) {
  const auto input_addr = GetDeviceAddress<T>(inputs, 0);
  MS_EXCEPTION_IF_NULL(input_addr);
  auto output_addr = GetDeviceAddress<T>(outputs, 0);
  MS_EXCEPTION_IF_NULL(output_addr);
  auto output_size = output_size_list_[0];
  auto target_addr_base = reinterpret_cast<char *>(input_addr) + offset_size_;
  auto cp_ret = memcpy_s(output_addr, output_size, target_addr_base, output_size);
  if (cp_ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
  }

  return true;
}

const std::vector<std::pair<KernelAttr, SequenceGetItemCpuKernelMod::KernelRunFunc>>
  &SequenceGetItemCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SequenceGetItemCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &SequenceGetItemCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &SequenceGetItemCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SequenceGetItemCpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &SequenceGetItemCpuKernelMod::LaunchKernel<int>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SequenceGetItemCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat16),
     &SequenceGetItemCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat32),
     &SequenceGetItemCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat64),
     &SequenceGetItemCpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeNumber, kNumberTypeInt32),
     &SequenceGetItemCpuKernelMod::LaunchKernel<int>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
     &SequenceGetItemCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
     &SequenceGetItemCpuKernelMod::LaunchKernel<int64_t>}};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RealTupleGetItem, SequenceGetItemCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
