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

#include "plugin/device/cpu/kernel/smooth_l1_loss_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <string>
#include <map>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/smooth_l1_loss.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSmoothL1LossInputsNum = 2;
constexpr size_t kSmoothL1LossOutputsNum = 1;
}  // namespace

template <typename T>
void SmoothL1LossCpuKernelMod::CalElements(const T *predict_addr, const T *target_addr, T *result_addr) {
  T zero = static_cast<T>(0.0);
  T half = static_cast<T>(0.5);
  T beta = static_cast<T>(beta_);
  auto task = [&](size_t start, size_t end) {
    for (uint64_t i = start; i < end; ++i) {
      T diff = predict_addr[i] - target_addr[i];
      if (diff < zero) {
        diff = -diff;
      }
      if (diff < beta) {
        result_addr[i] = half * diff * diff / beta;
      } else {
        result_addr[i] = diff - (half * beta);
      }
    }
  };
  ParallelLaunchAutoSearch(task, tensor_size_, this, &parallel_search_info_);
  return;
}

template <typename T>
bool SmoothL1LossCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSmoothL1LossInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSmoothL1LossOutputsNum, kernel_name_);
  const auto *predict_addr = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *target_addr = reinterpret_cast<T *>(inputs[1]->addr);
  T *result_addr = reinterpret_cast<T *>(outputs[0]->addr);
  if (reduction_ == ReductionType::NONE) {
    CalElements(predict_addr, target_addr, result_addr);
    return true;
  }

  T *workspace_addr = reinterpret_cast<T *>(workspace[0]->addr);
  CalElements(predict_addr, target_addr, workspace_addr);

  double tmp_sum{0};
  for (int64_t i = 0; i < tensor_size_; ++i) {
    tmp_sum += static_cast<double>(workspace_addr[i]);
  }
  result_addr[0] = static_cast<T>(tmp_sum);
  if (reduction_ == ReductionType::SUM) {
    return true;
  }

  result_addr[0] /= static_cast<T>(tensor_size_);
  return true;
}

bool SmoothL1LossCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
  if (beta_ < 0.0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << ", the 'beta' can not be less than 0.";
    return false;
  }

  std::string reduction = kernel_ptr->get_reduction();
  if (reduction == "none") {
    reduction_ = ReductionType::NONE;
  } else if (reduction == "mean") {
    reduction_ = ReductionType::MEAN;
  } else if (reduction == "sum") {
    reduction_ = ReductionType::SUM;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', reduction: " << reduction << " not support now.";
    return false;
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int SmoothL1LossCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  // when reduction is not set to none, we need extra space to record the result, with the same size with predict_size.
  if (reduction_ != ReductionType::NONE) {
    this->workspace_size_list_.push_back(input_size_list_[0]);
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

#define SMOOTH_L1_LOSS_CPU_REG(MS_T, T) \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_T).AddOutputAttr(MS_T), &SmoothL1LossCpuKernelMod::LaunchKernel<T>

const std::vector<std::pair<KernelAttr, SmoothL1LossCpuKernelMod::KernelRunFunc>>
  &SmoothL1LossCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SmoothL1LossCpuKernelMod::KernelRunFunc>> func_list = {
    {SMOOTH_L1_LOSS_CPU_REG(kNumberTypeFloat32, float)},
    {SMOOTH_L1_LOSS_CPU_REG(kNumberTypeFloat64, double)},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SmoothL1Loss, SmoothL1LossCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
