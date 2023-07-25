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

#include "plugin/device/gpu/kernel/nn/smooth_l1_loss_gpu_kernel.h"
#include <string>
#include <map>
#include <functional>
#include <algorithm>
#include <utility>
#include "abstract/utils.h"
#include "mindspore/core/ops/smooth_l1_loss.h"

namespace {
constexpr size_t kSmoothL1LossInputsNum = 2;
constexpr size_t kSmoothL1LossOutputsNum = 1;
}  // namespace
namespace mindspore {
namespace kernel {
bool SmoothL1LossGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SmoothL1Loss>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kSmoothL1LossInputsNum || outputs.size() != kSmoothL1LossOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kSmoothL1LossInputsNum << " and "
                  << kSmoothL1LossOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
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

int SmoothL1LossGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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

  // malloc double space for tmp_loss, prevents float overflow.
  if (reduction_ != SmoothL1LossReductionMode::NONE) {
    this->workspace_size_list_.clear();
    this->workspace_size_list_.push_back(sizeof(double));
  }
  return KRET_OK;
}

template <typename T>
bool SmoothL1LossGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSmoothL1LossInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSmoothL1LossOutputsNum, kernel_name_);
  const auto *predict_addr = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *target_addr = reinterpret_cast<T *>(inputs[1]->addr);
  T *result_addr = reinterpret_cast<T *>(outputs[0]->addr);
  if (this->reduction_ != SmoothL1LossReductionMode::NONE) {
    double *tmp_result_addr = reinterpret_cast<double *>(workspace[0]->addr);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemsetAsync(workspace[0]->addr, false, workspace[0]->size, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemsetAsync failed in SmoothL1LossGpuKernelMod::Launch.");
    auto status = SmoothL1Loss(reduction_, tensor_size_, beta_, predict_addr, target_addr, result_addr, tmp_result_addr,
                               device_id_, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
  } else {
    auto status = SmoothL1Loss(reduction_, tensor_size_, beta_, predict_addr, target_addr, result_addr, nullptr,
                               device_id_, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
  }

  return true;
}

#define SMOOTH_L1_LOSS_GPU_REG(MS_T, T) \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_T).AddOutputAttr(MS_T), &SmoothL1LossGpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, SmoothL1LossGpuKernelMod::SmoothL1LossFunc>> SmoothL1LossGpuKernelMod::func_list_ = {
  {SMOOTH_L1_LOSS_GPU_REG(kNumberTypeFloat16, half)},
  {SMOOTH_L1_LOSS_GPU_REG(kNumberTypeFloat32, float)},
  {SMOOTH_L1_LOSS_GPU_REG(kNumberTypeFloat64, double)},
};

std::vector<KernelAttr> SmoothL1LossGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SmoothL1LossFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SmoothL1Loss, SmoothL1LossGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
