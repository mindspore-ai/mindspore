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

#include "plugin/device/cpu/kernel/triu_indices_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
bool TriuIndicesCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);

  row_ = GetValue<int64_t>(prim->GetAttr("row"));
  col_ = GetValue<int64_t>(prim->GetAttr("col"));
  offset_ = GetValue<int64_t>(prim->GetAttr("offset"));
  if (row_ < 0) {
    MS_EXCEPTION(ValueError) << "For TriuIndices, row is " << row_ << ", but row should be greater than or equal to 0.";
  }
  if (col_ < 0) {
    MS_EXCEPTION(ValueError) << "For TriuIndices, col is " << col_ << ", but col should be greater than or equal to 0.";
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "TriuIndices does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int TriuIndicesCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  return KernelMod::Resize(base_operator, inputs, outputs);
}

template <typename T>
bool TriuIndicesCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  auto offset1_ = offset_ - 1;
  auto m_first_row = offset1_ > 0 ? std::min<int64_t>(col_, 1 + offset1_) : row_ + offset1_ > 0;
  auto m_last_row = std::max<int64_t>(0, std::min<int64_t>(col_, row_ + offset1_));
  auto n_row_all = std::max<int64_t>(0, std::min<int64_t>(row_, row_ + offset1_));
  auto n_row_trapezoid = (m_last_row - m_first_row + 1);
  auto tril_size = ((m_first_row + m_last_row) * n_row_trapezoid) >> 1;
  auto diff_row = n_row_all - n_row_trapezoid;
  if (diff_row > 0) {
    tril_size += diff_row * col_;
  }
  auto triu_size = row_ * col_ - tril_size;
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  int64_t i = 0;
  int64_t c = std::max<int64_t>(0, offset_), r = 0;
  while (i < triu_size) {
    output_addr[i] = r;
    output_addr[triu_size + i++] = c;
    c += 1;
    if (c >= col_) {
      r += 1;
      c = std::max<int64_t>(0, r + offset_);
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, TriuIndicesCpuKernelMod::TriuIndicesFunc>> TriuIndicesCpuKernelMod::func_list_ = {
  {KernelAttr().AddOutputAttr(kNumberTypeInt32), &TriuIndicesCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddOutputAttr(kNumberTypeInt64), &TriuIndicesCpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> TriuIndicesCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, TriuIndicesFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TriuIndices, TriuIndicesCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
