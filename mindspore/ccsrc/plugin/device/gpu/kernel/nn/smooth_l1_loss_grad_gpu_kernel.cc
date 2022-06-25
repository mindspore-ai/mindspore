/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/nn/smooth_l1_loss_grad_gpu_kernel.h"
#include <string>
#include <map>
#include <functional>
#include <algorithm>
#include <utility>
#include "mindspore/core/ops/grad/smooth_l1_loss_grad.h"

namespace {
constexpr size_t kSmoothL1LossGradInputsNum = 3;
constexpr size_t kSmoothL1LossGradOutputsNum = 1;
}  // namespace
namespace mindspore {
namespace kernel {
bool SmoothL1LossGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SmoothL1LossGrad>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kSmoothL1LossGradInputsNum || outputs.size() != kSmoothL1LossGradOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kSmoothL1LossGradInputsNum
                  << " and " << kSmoothL1LossGradOutputsNum << ", but got " << inputs.size() << " and "
                  << outputs.size();
    return false;
  }

  beta_ = kernel_ptr->get_beta();
  if (beta_ == 0.0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << ", the 'beta' can not be 0.";
    return false;
  }

  std::string reduction = kernel_ptr->get_reduction();
  if (reduction == "none") {
    reduction_ = SmoothL1LossReductionMode::NONE;
  } else if (reduction == "mean") {
    reduction_ = SmoothL1LossReductionMode::MEAN;
  } else if (reduction == "sum") {
    reduction_ = SmoothL1LossReductionMode::SUM;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', reduction: " << reduction << " not support now.";
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[pair.second].second;
  return true;
}

int SmoothL1LossGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  auto predict_shape = inputs[kIndex0]->GetShapeVector();
  auto target_shape = inputs[kIndex1]->GetShapeVector();
  if (predict_shape != target_shape) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the predict_shape should be same as target_shape, but got predict_shape: " << predict_shape
                  << ", and target_shape" << target_shape;
    return KRET_RESIZE_FAILED;
  }
  tensor_size_ = std::accumulate(predict_shape.begin(), predict_shape.end(), int64_t(1), std::multiplies<int64_t>());
  return KRET_OK;
}

template <typename T>
bool SmoothL1LossGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &workspace,
                                                const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSmoothL1LossGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSmoothL1LossGradOutputsNum, kernel_name_);
  const auto *predict_addr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  const auto *target_addr = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  const auto *dloss_addr = reinterpret_cast<T *>(inputs[kIndex2]->addr);
  T *result_addr = reinterpret_cast<T *>(outputs[0]->addr);

  SmoothL1LossGrad(reduction_, tensor_size_, beta_, predict_addr, target_addr, dloss_addr, result_addr, device_id_,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

#define SMOOTH_L1_LOSS_GRAD_GPU_REG(MS_T, T)                                                 \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_T).AddInputAttr(MS_T).AddOutputAttr(MS_T), \
    &SmoothL1LossGradGpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, SmoothL1LossGradGpuKernelMod::SmoothL1LossGradFunc>>
  SmoothL1LossGradGpuKernelMod::func_list_ = {
    {SMOOTH_L1_LOSS_GRAD_GPU_REG(kNumberTypeFloat16, half)},
    {SMOOTH_L1_LOSS_GRAD_GPU_REG(kNumberTypeFloat32, float)},
    {SMOOTH_L1_LOSS_GRAD_GPU_REG(kNumberTypeFloat64, double)},
};

std::vector<KernelAttr> SmoothL1LossGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SmoothL1LossGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SmoothL1LossGrad, SmoothL1LossGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
