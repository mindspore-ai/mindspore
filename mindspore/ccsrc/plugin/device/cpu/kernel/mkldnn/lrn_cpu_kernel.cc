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

#include "plugin/device/cpu/kernel/mkldnn/lrn_cpu_kernel.h"
#include <vector>
#include <algorithm>
#include <utility>
#include <memory>
#include <string>
#include <map>
#include "mindspore/core/ops/lrn.h"

namespace mindspore {
namespace kernel {
bool LrnCpuKernelMod::GetLrnAttr() {
  if (kernel_name_ != ops::kNameLRN) {
    MS_LOG(ERROR) << "For 'LRN' kernel name get failed, but got " << kernel_name_;
    return false;
  }
  depth_radius_ = GetValue<int64_t>(KernelMod::primitive_->GetAttr(ops::kDepthRadius));
  bias_ = GetValue<double_t>(KernelMod::primitive_->GetAttr(ops::kBias));
  alpha_ = GetValue<double_t>(KernelMod::primitive_->GetAttr(ops::kAlpha));
  beta_ = GetValue<int64_t>(KernelMod::primitive_->GetAttr(ops::kDepthRadius));
  norm_region_ = GetValue<std::string>(KernelMod::primitive_->GetAttr(ops::kNormRegion));
  if (norm_region_ != "ACROSS_CHANNELS") {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "''s attribute 'norm_region' must be ACROSS_CHANNELS but got "
                  << norm_region_;
    return false;
  }
  dnnl_algorithm_ = dnnl::algorithm::lrn_across_channels;
  return true;
}

bool LrnCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!GetLrnAttr()) {
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

int LrnCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  constexpr size_t kInputsNum = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  TypeId ms_type_id = inputs.at(kIndex0)->dtype_id();
  auto dnnl_type_id = GetDnnlDataType(ms_type_id);
  if (dnnl_type_id == dnnl::memory::data_type::undef) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', LrnCpuKernelMod::Resize failed, dnnl do not support data type:" << TypeIdToString(ms_type_id);
    return KRET_RESIZE_FAILED;
  }
  std::vector<size_t> input_shape_;
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToSize);
  dnnl::memory::desc src_desc = GetExactMemDesc(input_shape_, dnnl_type_id);
  const auto lrn_multiple = 2;
  dnnl::memory::dim local_size = lrn_multiple * depth_radius_ + 1;
  const auto dnnl_alpha = static_cast<float>(local_size) * alpha_;
  auto desc = CreateDesc<dnnl::lrn_forward::desc>(dnnl::prop_kind::forward_training, dnnl_algorithm_, src_desc,
                                                  local_size, dnnl_alpha, beta_, bias_);
  auto prim_desc = CreateDesc<dnnl::lrn_forward::primitive_desc>(desc, engine_);
  primitive_ = CreatePrimitive<dnnl::lrn_forward>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, src_desc);
  return KRET_OK;
}

bool LrnCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                   const std::vector<kernel::KernelTensor *> &outputs) {
  constexpr size_t kInputsNum = 1;
  constexpr size_t kOutputsNum = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  SetArgumentHandle(DNNL_ARG_SRC, inputs.at(kIndex0)->device_ptr());
  SetArgumentHandle(DNNL_ARG_DST, outputs.at(kIndex0)->device_ptr());
  ExecutePrimitive();
  return true;
}

std::vector<std::pair<KernelAttr, LrnCpuKernelMod::LrnFunc>> LrnCpuKernelMod::func_list_ = {
  // For kNumberTypeFloat16 input data type will cast to kNumberTypeFloat32 from frontend to backend.
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), &LrnCpuKernelMod::LaunchKernel}};

std::vector<KernelAttr> LrnCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LrnFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LRN, LrnCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
