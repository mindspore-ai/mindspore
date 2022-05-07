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
  float epsilon_{0.0f};
};
}  // namespace

dnnl::reduction::desc ReductionCpuKernelMod::GetReductionDesc(const dnnl::memory::desc &src_desc,
                                                              const dnnl::memory::desc &dst_desc) {
  static const std::map<std::string, ReductionDescParam> reduction_op_desc_map{
    {prim::kPrimLpNorm->name(), ReductionDescParam{dnnl::algorithm::reduction_norm_lp_sum, p_, epsilon_}}};
  const auto desc_pair = reduction_op_desc_map.find(kernel_name_);
  if (desc_pair == reduction_op_desc_map.end()) {
    MS_LOG(EXCEPTION) << "ReductionCpuKernelMod does not support " << kernel_name_;
  }
  auto desc = CreateDesc<dnnl::reduction::desc>(desc_pair->second.algorithm, src_desc, dst_desc, desc_pair->second.p_,
                                                desc_pair->second.epsilon_);
  return desc;
}

bool ReductionCpuKernelMod::GetReductionAttr(const BaseOperatorPtr &base_operator) {
  if (kernel_name_ != ops::kNameLpNorm) {
    MS_LOG(ERROR) << "For 'LpNorm' kernel name get failed, but got " << kernel_name_;
    return false;
  }
  auto kernel_ptr = std::make_shared<ops::LpNorm>(base_operator->GetPrim());
  p_ = LongToFloat(kernel_ptr->get_p());
  epsilon_ = kernel_ptr->get_epsilon();
  return true;
}

bool ReductionCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!GetReductionAttr(base_operator)) {
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

int ReductionCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KRET_OK;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs)) != KRET_OK) {
    return ret;
  }
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  auto output_shape = outputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToSize);
  (void)std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(output_shape_), LongToSize);
  // For Reduction kernel required at least 4d data shape, extend it to 4d.
  while (input_shape_.size() < kIndex4) {
    input_shape_.insert(input_shape_.begin(), 1);
  }
  while (output_shape_.size() < kIndex4) {
    output_shape_.insert(output_shape_.begin(), 1);
  }
  dnnl::memory::desc src_desc = GetDefaultMemDesc(input_shape_);
  dnnl::memory::desc dst_desc = GetDefaultMemDesc(output_shape_);
  auto desc = GetReductionDesc(src_desc, dst_desc);
  auto prim_desc = CreateDesc<dnnl::reduction::primitive_desc>(desc, engine_);
  primitive_ = CreatePrimitive<dnnl::reduction>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, dst_desc);
  return ret;
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

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LpNorm, ReductionCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
