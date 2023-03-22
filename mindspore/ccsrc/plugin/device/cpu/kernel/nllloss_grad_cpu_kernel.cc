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

#include "plugin/device/cpu/kernel/nllloss_grad_cpu_kernel.h"
#include <map>
#include <string>
#include <utility>
#include <unordered_map>
#include "mindspore/core/ops/grad/nllloss_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kNLLLossGradInputsNum = 5;
constexpr size_t kNLLLossGradOutputsNum = 1;
const std::unordered_map<Reduction, ReductionType> kReductionMap = {
  {Reduction::MEAN, Reduction_Mean}, {Reduction::REDUCTION_SUM, Reduction_Sum}, {Reduction::NONE, Reduction_None}};
}  // namespace

bool NLLLossGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::NLLLossGrad>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast NLLLossGrad ops failed!";
    return false;
  }
  auto kernel_name = kernel_ptr->GetPrim()->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);

  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name << " does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;

  auto reduction = kernel_ptr->get_reduction();

  auto pair = kReductionMap.find(reduction);
  if (pair == kReductionMap.end()) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_
                      << ", the attr 'reduction' only support 'mean', 'sum' and 'none', but got " << reduction;
  }

  nllloss_param_.reduction_type_ = pair->second;
  return true;
}

int NLLLossGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != 0) {
    return ret;
  }

  auto logits_shape = inputs[0]->GetShapeVector();
  nllloss_param_.batch_ = LongToInt(logits_shape[0]);
  nllloss_param_.class_num_ = LongToInt(logits_shape[1]);
  return KRET_OK;
}

template <typename T>
bool NLLLossGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &workspace,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(kNLLLossGradInputsNum, inputs.size(), kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(kNLLLossGradOutputsNum, outputs.size(), kernel_name_);

  const auto *logits = reinterpret_cast<float *>(inputs[0]->addr);
  const auto *loss_grad = reinterpret_cast<float *>(inputs[1]->addr);
  const auto *labels = reinterpret_cast<T *>(inputs[2]->addr);
  const auto *weight = reinterpret_cast<float *>(inputs[3]->addr);
  const auto *total_weight = reinterpret_cast<float *>(inputs[4]->addr);
  auto *logits_grad = reinterpret_cast<float *>(outputs[0]->addr);

  if (logits == NULL || loss_grad == NULL || labels == NULL || weight == NULL || total_weight == NULL) {
    MS_LOG(ERROR) << "For NLLLossGrad, it does not support NULL input";
  }
  memset(logits_grad, 0, nllloss_param_.batch_ * nllloss_param_.class_num_ * sizeof(float));
  for (int i = 0; i < nllloss_param_.batch_; i++) {
    if (!(labels[i] < nllloss_param_.class_num_)) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', the labels should be smaller than the number of classes, but got " << labels[i];
    }
    int index = i * nllloss_param_.class_num_ + labels[i];
    float n_weight = weight[labels[i]];
    if (nllloss_param_.reduction_type_ == Reduction_Sum) {
      logits_grad[index] = -loss_grad[0] * n_weight;
    } else if (nllloss_param_.reduction_type_ == Reduction_Mean) {
      logits_grad[index] = -loss_grad[0] * n_weight / *total_weight;
    } else {
      logits_grad[index] = -loss_grad[i] * n_weight;
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, NLLLossGradCpuKernelMod::NLLLossGradFunc>> NLLLossGradCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &NLLLossGradCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &NLLLossGradCpuKernelMod::LaunchKernel<int64_t>}};

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NLLLossGrad, NLLLossGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
