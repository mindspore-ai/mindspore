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

#include "plugin/device/cpu/kernel/matmul_cpu_kernel.h"
#include "plugin/device/cpu/kernel/eigen/matmul_double_cpu_kernel_func.h"
#include "plugin/device/cpu/kernel/mkldnn/matmul_cpu_kernel_func.h"
#include <utility>
#include <algorithm>
#include <functional>
#include <map>

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kMatMul = "MatMul";
constexpr auto kBatchMatMul = "BatchMatMul";

using MatMulFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
static std::map<std::string, std::vector<std::pair<KernelAttr, MatMulFuncCreator>>> support_list_map = {
  {kMatMul,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     []() { return std::make_shared<MatMulCpuKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }}}},
  {kBatchMatMul,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     []() { return std::make_shared<MatMulCpuKernelFunc>(); }}}}};
}  // namespace

std::vector<KernelAttr> MatMulCpuKernelMod::GetOpSupport() {
  auto iter = support_list_map.find(kernel_type_);
  if (iter == support_list_map.end()) {
    MS_LOG(EXCEPTION) << "Does not support " << kernel_type_ << "!";
  }

  std::vector<KernelAttr> support_list;
  std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, MatMulFuncCreator> &pair) { return pair.first; });
  return support_list;
}

void MatMulCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "Suppose to be " << kernel_type_ << " but got " << kernel_name_;
  }

  auto iter = support_list_map.find(kernel_type_);
  if (iter == support_list_map.end()) {
    MS_LOG(EXCEPTION) << "MatMul cpu does not support " << kernel_type_;
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "MatMul does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = support_list_map[kernel_type_][index].second();
  func_obj_->InitFunc(kernel_node);
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, MatMul,
                                 []() { return std::make_shared<MatMulCpuKernelMod>(kMatMul); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, BatchMatMul,
                                 []() { return std::make_shared<MatMulCpuKernelMod>(kBatchMatMul); });
}  // namespace kernel
}  // namespace mindspore
