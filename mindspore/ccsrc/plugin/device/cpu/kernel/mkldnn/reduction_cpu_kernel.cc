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

#include "plugin/device/cpu/kernel/mkldnn/reduction_cpu_kernel.h"
#include <map>
#include <utility>
#include <string>
#include <set>
#include <algorithm>
#include "utils/ms_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/lp_norm.h"

namespace mindspore {
namespace kernel {
namespace {
struct ReductionDescParam {
  dnnl::algorithm algorithm{dnnl::algorithm::undef};
  float p_{2.0f};
  float eps_{0.0f};
};
}  // namespace

dnnl::reduction::desc ReductionCpuKernelMod::GetReductionDesc(const dnnl::memory::desc &src_desc,
                                                              const dnnl::memory::desc &dst_desc) {
  static const std::map<std::string, ReductionDescParam> reduction_op_desc_map{
    {prim::kPrimLpNorm->name(), ReductionDescParam{dnnl::algorithm::reduction_norm_lp_sum, p_, eps_}}};
  const auto desc_pair = reduction_op_desc_map.find(kernel_name_);
  if (desc_pair == reduction_op_desc_map.end()) {
    MS_LOG(EXCEPTION) << "ReductionCpuKernelMod does not support " << kernel_name_;
  }
  auto desc = CreateDesc<dnnl::reduction::desc>(desc_pair->second.algorithm, src_desc, dst_desc, desc_pair->second.p_,
                                                desc_pair->second.eps_);
  return desc;
}

void ReductionCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  const std::string p = "p";
  if (!common::AnfAlgo::HasNodeAttr(p, kernel_node)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' has no kernel attribute: " << p;
  }
  p_ = LongToFloat(common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, p));
  const std::string eps = "epsilon";
  if (!common::AnfAlgo::HasNodeAttr(eps, kernel_node)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' has no kernel attribute: " << eps;
  }
  eps_ = common::AnfAlgo::GetNodeAttr<float>(kernel_node, eps);
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  std::vector<size_t> input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex0);
  std::vector<size_t> output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, kIndex0);
  // For Reduction kernel required at least 4d data shape, extend it to 4d.
  while (input_shape.size() < kIndex4) {
    input_shape.insert(input_shape.begin(), 1);
  }
  while (output_shape.size() < kIndex4) {
    output_shape.insert(output_shape.begin(), 1);
  }
  dnnl::memory::desc src_desc = GetDefaultMemDesc(input_shape);
  dnnl::memory::desc dst_desc = GetDefaultMemDesc(output_shape);
  auto desc = GetReductionDesc(src_desc, dst_desc);
  auto prim_desc = CreateDesc<dnnl::reduction::primitive_desc>(desc, engine_);
  primitive_ = CreatePrimitive<dnnl::reduction>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, dst_desc);
}

template <typename T>
bool ReductionCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  auto input = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  SetArgumentHandle(DNNL_ARG_SRC, input);
  SetArgumentHandle(DNNL_ARG_DST, output);
  ExecutePrimitive();
  return true;
}

std::vector<std::pair<KernelAttr, ReductionCpuKernelMod::ReductionFunc>> ReductionCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &ReductionCpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> ReductionCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ReductionFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, LpNorm,
                                 []() { return std::make_shared<ReductionCpuKernelMod>(prim::kPrimLpNorm->name()); });
}  // namespace kernel
}  // namespace mindspore
