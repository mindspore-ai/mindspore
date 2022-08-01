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

#include "plugin/device/cpu/kernel/smooth_l1_loss_grad_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <string>
#include <map>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/grad/smooth_l1_loss_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSmoothL1LossGradInputsNum = 3;
constexpr size_t kSmoothL1LossGradOutputsNum = 1;
}  // namespace

bool SmoothL1LossGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
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
  if (std::equal_to<float>()(beta_, 0)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << ", the 'beta' can not be 0.";
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

int SmoothL1LossGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
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
                  << ", and target_shape: " << target_shape;
    return KRET_RESIZE_FAILED;
  }
  tensor_size_ = std::accumulate(predict_shape.begin(), predict_shape.end(), int64_t(1), std::multiplies<int64_t>());
  return KRET_OK;
}

template <typename T>
bool SmoothL1LossGradCpuKernelMod::CalNoReduce(const T *predict_addr, const T *target_addr, const T *dloss_addr,
                                               T *result_addr) {
  T beta = static_cast<T>(beta_);
  auto task = [&](size_t start, size_t end) {
    for (uint64_t i = start; i < end; ++i) {
      T diff = predict_addr[i] - target_addr[i];
      if (diff > beta) {
        result_addr[i] = dloss_addr[i];
      } else if (diff < -beta) {
        result_addr[i] = -dloss_addr[i];
      } else {
        result_addr[i] = (diff / beta) * dloss_addr[i];
      }
    }
  };
  ParallelLaunchAutoSearch(task, tensor_size_, this, &parallel_search_info_);
  return true;
}

template <typename T>
bool SmoothL1LossGradCpuKernelMod::CalMean(const T *predict_addr, const T *target_addr, const T *dloss_addr,
                                           T *result_addr) {
  T beta = static_cast<T>(beta_);
  T val = static_cast<T>(1) / static_cast<T>(tensor_size_);
  auto task = [&](size_t start, size_t end) {
    for (uint64_t i = start; i < end; ++i) {
      T diff = predict_addr[i] - target_addr[i];
      if (diff > beta) {
        result_addr[i] = dloss_addr[0] * val;
      } else if (diff < -beta) {
        result_addr[i] = -dloss_addr[0] * val;
      } else {
        result_addr[i] = (diff / beta) * dloss_addr[0] * val;
      }
    }
  };
  ParallelLaunchAutoSearch(task, tensor_size_, this, &parallel_search_info_);
  return true;
}

template <typename T>
bool SmoothL1LossGradCpuKernelMod::CalSum(const T *predict_addr, const T *target_addr, const T *dloss_addr,
                                          T *result_addr) {
  T beta = static_cast<T>(beta_);
  auto task = [&](size_t start, size_t end) {
    for (uint64_t i = start; i < end; ++i) {
      T diff = predict_addr[i] - target_addr[i];
      if (diff > beta) {
        result_addr[i] = dloss_addr[0];
      } else if (diff < -beta) {
        result_addr[i] = -dloss_addr[0];
      } else {
        result_addr[i] = (diff / beta) * dloss_addr[0];
      }
    }
  };
  ParallelLaunchAutoSearch(task, tensor_size_, this, &parallel_search_info_);
  return true;
}

template <typename T>
bool SmoothL1LossGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSmoothL1LossGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSmoothL1LossGradOutputsNum, kernel_name_);
  const auto *predict_addr = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *target_addr = reinterpret_cast<T *>(inputs[1]->addr);
  const auto *dloss_addr = reinterpret_cast<T *>(inputs[2]->addr);
  auto *result_addr = reinterpret_cast<T *>(outputs[0]->addr);
  switch (reduction_) {
    case ReductionType::NONE:
      return CalNoReduce(predict_addr, target_addr, dloss_addr, result_addr);
    case ReductionType::SUM:
      return CalSum(predict_addr, target_addr, dloss_addr, result_addr);
    case ReductionType::MEAN:
      return CalMean(predict_addr, target_addr, dloss_addr, result_addr);

    default:
      return false;
  }
}

#define SMOOTH_L1_LOSS_GRAD_CPU_REG(MS_T, T)                                                 \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_T).AddInputAttr(MS_T).AddOutputAttr(MS_T), \
    &SmoothL1LossGradCpuKernelMod::LaunchKernel<T>

const std::vector<std::pair<KernelAttr, SmoothL1LossGradCpuKernelMod::KernelRunFunc>>
  &SmoothL1LossGradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SmoothL1LossGradCpuKernelMod::KernelRunFunc>> func_list = {
    {SMOOTH_L1_LOSS_GRAD_CPU_REG(kNumberTypeFloat16, float16)},
    {SMOOTH_L1_LOSS_GRAD_CPU_REG(kNumberTypeFloat32, float)},
    {SMOOTH_L1_LOSS_GRAD_CPU_REG(kNumberTypeFloat64, double)},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SmoothL1LossGrad, SmoothL1LossGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
