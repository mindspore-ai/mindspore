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

#include "plugin/device/cpu/kernel/mkldnn/lrn_grad_cpu_kernel.h"
#include <vector>
#include <algorithm>
#include <utility>
#include <memory>
#include <string>
#include <map>
#include "mindspore/core/ops/grad/lrn_grad.h"

namespace mindspore {
namespace kernel {
bool LrnGradCpuKernelMod::GetLrnGradAttr(const BaseOperatorPtr &base_operator) {
  if (kernel_name_ != ops::kNameLRNGrad) {
    MS_LOG(ERROR) << "For 'LRNGrad' kernel name get failed, but got " << kernel_name_;
    return false;
  }
  auto kernel_ptr = std::make_shared<ops::LRNGrad>(base_operator->GetPrim());
  MS_ERROR_IF_NULL(kernel_ptr);
  depth_radius_ = kernel_ptr->get_depth_radius();
  bias_ = kernel_ptr->get_bias();
  alpha_ = kernel_ptr->get_alpha();
  beta_ = kernel_ptr->get_beta();
  dnnl_algorithm_ = dnnl::algorithm::lrn_across_channels;
  return true;
}

bool LrnGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!GetLrnGradAttr(base_operator)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got GetReductionAttr failed.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int LrnGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &) {
  MS_EXCEPTION_IF_NULL(base_operator);
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != KRET_OK) {
    return ret;
  }
  constexpr size_t kInputsNum = 3;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  dnnl::memory::desc src_desc = GetDefaultMemDesc(input_shape);
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
  return ret;
}

template <typename T>
bool LrnGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  // The input order is dout, x, out.
  constexpr size_t kInputsNum = 3;
  constexpr size_t kOutputsNum = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
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
     .AddAllSameAttr(true)
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
