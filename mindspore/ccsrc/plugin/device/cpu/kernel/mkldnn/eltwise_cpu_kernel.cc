/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/mkldnn/eltwise_cpu_kernel.h"
#include <functional>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNum = 1;

struct DescParam {
  dnnl::algorithm algorithm{dnnl::algorithm::undef};
  float alpha{0.0f};
  float beta{0.0f};
};
}  // namespace

dnnl::eltwise_forward::desc EltWiseCpuKernelMod::GetForwardEltwiseDesc(const dnnl::memory::desc src_desc) {
  static const std::unordered_map<std::string, DescParam> eltwise_op_desc_map{
    {prim::kPrimRelu->name(), DescParam{dnnl::algorithm::eltwise_relu}},
    {prim::kPrimRelu6->name(), DescParam{dnnl::algorithm::eltwise_clip, 0.0f, 6.0f}},
    {prim::kPrimAbs->name(), DescParam{dnnl::algorithm::eltwise_abs}},
    {prim::kPrimExp->name(), DescParam{dnnl::algorithm::eltwise_exp}},
    {prim::kPrimLog->name(), DescParam{dnnl::algorithm::eltwise_log}},
    {prim::kPrimSigmoid->name(), DescParam{dnnl::algorithm::eltwise_logistic}},
    {prim::kPrimSqrt->name(), DescParam{dnnl::algorithm::eltwise_sqrt}},
    {prim::kPrimTanh->name(), DescParam{dnnl::algorithm::eltwise_tanh}},
    {prim::kPrimElu->name(), DescParam{dnnl::algorithm::eltwise_elu, 1.0f, 0.0f}},
    {prim::kPrimSoftplus->name(), DescParam{dnnl::algorithm::eltwise_soft_relu}},
    {prim::kPrimMish->name(), DescParam{dnnl::algorithm::eltwise_mish}},
  };
  const auto desc_pair = eltwise_op_desc_map.find(kernel_name_);
  if (desc_pair == eltwise_op_desc_map.end()) {
    MS_LOG(EXCEPTION) << "For 'EltWise Op', it does not support " << kernel_name_;
  }
  auto desc = CreateDesc<dnnl::eltwise_forward::desc>(dnnl_forward_, desc_pair->second.algorithm, src_desc,
                                                      desc_pair->second.alpha, desc_pair->second.beta);
  return desc;
}

bool EltWiseCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR) << "For 'EltWise Op', the kernel name must be in " << kernel::Map2Str(kernel_attr_map_)
                  << ", but got " << kernel_name_;
  }
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = iter->second[index].second;
  return true;
}

int EltWiseCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  src_shape_.clear();
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(src_shape_), LongToSize);
  input_element_num_ = std::accumulate(src_shape_.begin(), src_shape_.end(), 1, std::multiplies<size_t>());
  is_null_input_ = (input_element_num_ == 0);
  if (is_null_input_) {
    return KRET_OK;
  }
  TypeId input_type_id = inputs.at(kIndex0)->GetDtype();
  auto dnnl_type_id = GetDnnlDataType(input_type_id);
  if (dnnl_type_id == dnnl::memory::data_type::undef) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', Resize failed, dnnl do not support data type:" << TypeIdToString(input_type_id);
    return KRET_RESIZE_FAILED;
  }
  if (src_shape_.empty()) {
    (void)src_shape_.insert(src_shape_.begin(), 1);
  }
  dnnl::memory::desc src_desc = GetExactMemDesc(src_shape_, dnnl_type_id);

  auto desc = GetForwardEltwiseDesc(src_desc);
  auto prim_desc = CreateDesc<dnnl::eltwise_forward::primitive_desc>(desc, engine_);
  primitive_ = CreatePrimitive<dnnl::eltwise_forward>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, src_desc);
  return KRET_OK;
}

std::vector<KernelAttr> EltWiseCpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR) << "For 'EltWise Op', the kernel name must be in " << kernel::Map2Str(kernel_attr_map_)
                  << ", but got " << kernel_name_;
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, EltWiseFunc> &pair) { return pair.first; });

  return support_list;
}

template <typename T>
bool EltWiseCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
  ExecutePrimitive();
  return true;
}

std::map<std::string, std::vector<std::pair<KernelAttr, EltWiseCpuKernelMod::EltWiseFunc>>>
  EltWiseCpuKernelMod::kernel_attr_map_ = {
    {kElu,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &EltWiseCpuKernelMod::LaunchKernel<float>}}},
    {kReLU,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &EltWiseCpuKernelMod::LaunchKernel<float>}}},
    {kReLU6,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &EltWiseCpuKernelMod::LaunchKernel<float>}}},
    {kExp,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &EltWiseCpuKernelMod::LaunchKernel<float>}}},
    {kLog,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &EltWiseCpuKernelMod::LaunchKernel<float>}}},
    {kSigmoid,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &EltWiseCpuKernelMod::LaunchKernel<float>}}},
    {kTanh,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &EltWiseCpuKernelMod::LaunchKernel<float>}}},
    {kSoftplus,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &EltWiseCpuKernelMod::LaunchKernel<float>}}},
    {prim::kPrimMish->name(),
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &EltWiseCpuKernelMod::LaunchKernel<float>}}},
    {prim::kPrimSqrt->name(),
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &EltWiseCpuKernelMod::LaunchKernel<float>}}}};

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Elu, []() { return std::make_shared<EltWiseCpuKernelMod>(kElu); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReLU,
                                 []() { return std::make_shared<EltWiseCpuKernelMod>(kReLU); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReLU6,
                                 []() { return std::make_shared<EltWiseCpuKernelMod>(kReLU6); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Exp, []() { return std::make_shared<EltWiseCpuKernelMod>(kExp); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Log, []() { return std::make_shared<EltWiseCpuKernelMod>(kLog); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Sigmoid,
                                 []() { return std::make_shared<EltWiseCpuKernelMod>(kSigmoid); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Tanh,
                                 []() { return std::make_shared<EltWiseCpuKernelMod>(kTanh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Softplus,
                                 []() { return std::make_shared<EltWiseCpuKernelMod>(kSoftplus); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Mish,
                                 []() { return std::make_shared<EltWiseCpuKernelMod>(prim::kPrimMish->name()); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Sqrt,
                                 []() { return std::make_shared<EltWiseCpuKernelMod>(prim::kPrimSqrt->name()); });
}  // namespace kernel
}  // namespace mindspore
