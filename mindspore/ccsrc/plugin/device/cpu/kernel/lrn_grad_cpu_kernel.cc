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
#include "plugin/device/cpu/kernel/lrn_grad_cpu_kernel.h"
#include <vector>
#include <algorithm>
#include <utility>
#include <memory>
#include <string>

namespace mindspore {
namespace kernel {
void LrnGradCpuKernelMod::GetLrnAttr(const CNodePtr &kernel_node) {
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
  dnnl_algorithm_ = dnnl::algorithm::lrn_across_channels;
}

void LrnGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
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
  // Backward description
  auto backward_desc =
    CreateDesc<dnnl::lrn_backward::desc>(dnnl_algorithm_, src_desc, src_desc, local_size, dnnl_alpha, beta_, bias_);
  auto backward_prim_desc = CreateDesc<dnnl::lrn_backward::primitive_desc>(backward_desc, engine_, prim_desc);
  primitive_ = CreatePrimitive<dnnl::lrn_backward>(backward_prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, src_desc);
  AddArgument(DNNL_ARG_DIFF_SRC, src_desc);
  AddArgument(DNNL_ARG_DIFF_DST, src_desc);
}

template <typename T>
bool LrnGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  // The input order is dout, x, out.
  auto dout = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto input = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
  auto out = reinterpret_cast<T *>(inputs.at(kIndex2)->addr);
  auto grad_x = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  SetArgumentHandle(DNNL_ARG_SRC, input);
  SetArgumentHandle(DNNL_ARG_DST, out);
  SetArgumentHandle(DNNL_ARG_DIFF_DST, dout);
  SetArgumentHandle(DNNL_ARG_DIFF_SRC, grad_x);
  ExecutePrimitive();
  return true;
}

std::vector<std::pair<KernelAttr, LrnGradCpuKernelMod::LrnGradFunc>> LrnGradCpuKernelMod::func_list_ = {
  // For kNumberTypeFloat16 input data type will cast to kNumberTypeFloat32 from frontend to backend.
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LrnGradCpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> LrnGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LrnGradFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LRNGrad, LrnGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
