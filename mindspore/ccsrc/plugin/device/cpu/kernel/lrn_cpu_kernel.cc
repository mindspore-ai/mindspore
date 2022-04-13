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
#include "plugin/device/cpu/kernel/lrn_cpu_kernel.h"
#include <vector>
#include <algorithm>
#include <utility>
#include <memory>
#include <string>

namespace mindspore {
namespace kernel {
void LrnCpuKernelMod::GetLrnAttr(const CNodePtr &kernel_node) {
  const std::string depth_radius = "depth_radius";
  if (!common::AnfAlgo::HasNodeAttr(depth_radius, kernel_node)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' has no kernel attribute: " << depth_radius;
  }
  depth_radius_ = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "depth_radius");
  const std::string bias = "bias";
  if (!common::AnfAlgo::HasNodeAttr(bias, kernel_node)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' has no kernel attribute: " << bias;
  }
  bias_ = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "bias");
  const std::string alpha = "alpha";
  if (!common::AnfAlgo::HasNodeAttr(alpha, kernel_node)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' has no kernel attribute: " << alpha;
  }
  alpha_ = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "alpha");

  const std::string beta = "beta";
  if (!common::AnfAlgo::HasNodeAttr(beta, kernel_node)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' has no kernel attribute: " << beta;
  }
  beta_ = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "beta");

  const std::string norm_region = "norm_region";
  if (!common::AnfAlgo::HasNodeAttr(norm_region, kernel_node)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' has no kernel attribute: " << norm_region;
  }
  norm_region_ = common::AnfAlgo::GetNodeAttr<string>(kernel_node, "norm_region");

  if (norm_region_ == "ACROSS_CHANNELS") {
    dnnl_algorithm_ = dnnl::algorithm::lrn_across_channels;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "''s attribute 'norm_region' should be ACROSS_CHANNELS but got "
                      << norm_region;
  }
}

void LrnCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  GetLrnAttr(kernel_node);
  input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex0);
  dnnl::memory::desc src_desc = GetDefaultMemDesc(input_shape_);
  const auto lrn_multiple = 2;
  dnnl::memory::dim local_size = lrn_multiple * depth_radius_ + 1;
  const auto dnnl_alpha = static_cast<float>(local_size) * alpha_;
  auto desc = CreateDesc<dnnl::lrn_forward::desc>(dnnl::prop_kind::forward_training, dnnl_algorithm_, src_desc,
                                                  local_size, dnnl_alpha, beta_, bias_);
  auto prim_desc = CreateDesc<dnnl::lrn_forward::primitive_desc>(desc, engine_);
  primitive_ = CreatePrimitive<dnnl::lrn_forward>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, src_desc);
}

template <typename T>
bool LrnCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  auto input = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  SetArgumentHandle(DNNL_ARG_SRC, input);
  SetArgumentHandle(DNNL_ARG_DST, output);
  ExecutePrimitive();
  return true;
}

std::vector<std::pair<KernelAttr, LrnCpuKernelMod::LrnFunc>> LrnCpuKernelMod::func_list_ = {
  // For kNumberTypeFloat16 input data type will cast to kNumberTypeFloat32 from frontend to backend.
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &LrnCpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> LrnCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LrnFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LRN, LrnCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
