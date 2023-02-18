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

#include <map>
#include <memory>
#include <utility>
#include <algorithm>
#include "include/common/thread_pool.h"
#include "mindspore/core/ops/grad/kl_div_loss_grad.h"
#include "mindspore/core/ops/op_name.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"
#include "plugin/device/cpu/kernel/kl_div_loss_grad_cpu_kernel.h"

namespace mindspore {
namespace kernel {
const size_t kInputsNum = 3;
const size_t kOutputsNum = 1;

bool KLDivLossGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::KLDivLossGrad>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(EXCEPTION) << "cast KLDivLoss ops failed!";
  }
  kernel_name_ = kernel_ptr->name();
  reductionMode_ = kernel_ptr->get_reduction();
  if (inputs[kIndex1]->GetShapeVector().size() >= 1) {
    batch_size_ = inputs[kIndex1]->GetShapeVector()[kIndex0];
  }

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  if (!KLDivLossGradCpuKernelMod::CheckParams()) {
    MS_LOG(EXCEPTION) << kernel_name_ << ": check param failed.";
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does not support this data type: " << kernel_attr;
  }

  kernel_func_ = func_list_[index].second;
  return true;
}

bool KLDivLossGradCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                       const std::vector<AddressPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(kernel_func_);
  return kernel_func_(this, inputs, workspace, outputs);
}

int KLDivLossGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &onHost) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, onHost)) != 0) {
    MS_LOG(WARNING) << kernel_name_ << " resize failed.";
    return ret;
  }

  input_grad_shape_size_ = 1;
  std::vector<int64_t> input_grad_shape = inputs[kIndex0]->GetShapeVector();
  if (input_grad_shape.size() >= 1) {
    for (size_t i = 0; i < input_grad_shape.size(); ++i) {
      input_grad_shape_size_ *= LongToSize(input_grad_shape[i]);
    }
  }

  input_x_shape_size_ = 1;
  std::vector<int64_t> input_x_shape = inputs[kIndex1]->GetShapeVector();
  if (input_x_shape.size() >= 1) {
    for (size_t i = 0; i < input_x_shape.size(); ++i) {
      input_x_shape_size_ *= LongToSize(input_x_shape[i]);
    }
  }

  input_target_shape_size_ = 1;
  std::vector<int64_t> input_target_shape = inputs[kIndex2]->GetShapeVector();
  if (input_target_shape.size() >= 1) {
    for (size_t i = 0; i < input_target_shape.size(); ++i) {
      input_target_shape_size_ *= LongToSize(input_target_shape[i]);
    }
  }

  return 0;
}

std::vector<KernelAttr> KLDivLossGradCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, KLDivLossGradCpuKernelMod::KLDivLossGradFunc> &pair) { return pair.first; });
  return support_list;
}

bool KLDivLossGradCpuKernelMod::CheckParams() const {
  // for kl div, shape size of input 1 and input 2 must be the same
  if (input_target_shape_size_ != input_x_shape_size_) {
    MS_LOG(ERROR) << kernel_name_ << ": input x shape size = " << input_x_shape_size_
                  << ", input target shape size = " << input_target_shape_size_ << ". They are not the same.";
    return false;
  }

  // for kl_div, shape size of input 0 is 1 if reductionmode is not None
  if (reductionMode_ != ops::kNone) {
    if (input_grad_shape_size_ != 1) {
      MS_LOG(ERROR) << kernel_name_ << ": input grad shape size = " << input_grad_shape_size_
                    << " is not 1 when reduction is None.";
      return false;
    }
  } else {
    if (input_grad_shape_size_ != input_x_shape_size_) {
      MS_LOG(ERROR) << kernel_name_ << ": input x shape size = " << input_x_shape_size_
                    << ", input grad shape size = " << input_grad_shape_size_ << ". They are not the same.";
      return false;
    }
  }
  return true;
}

template <typename T>
bool KLDivLossGradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                             const std::vector<AddressPtr> &outputs) {
  auto *input_grad = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto *input_target = reinterpret_cast<T *>(inputs[kIndex2]->addr);
  auto *y = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_grad(input_grad, input_grad_shape_size_, 1);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_target(input_target, input_target_shape_size_, 1);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_y(y, input_x_shape_size_, 1);

  double coefficient = -1.0;
  if (reductionMode_ == ops::kMean) {
    coefficient /= SizeToFloat(input_x_shape_size_);
  } else if (reductionMode_ == ops::kBatchMean) {
    coefficient /= LongToFloat(batch_size_);
  }

  double bcast = 1.0;
  if (reductionMode_ == ops::kNone) {
    array_y = array_target * array_grad;
  } else {
    array_y = array_target;
    bcast = static_cast<double>(input_grad[0]);
  }

  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      if (static_cast<double>(array_target[i]) <= 0.0) {
        array_y[i] = static_cast<T>(0);
      } else {
        array_y[i] *= (static_cast<T>(coefficient) * static_cast<T>(bcast));
      }
    }
  };
  ParallelLaunchAutoSearch(task, input_x_shape_size_, this, &parallel_search_info_);

  return true;
}

std::vector<std::pair<KernelAttr, KLDivLossGradCpuKernelMod::KLDivLossGradFunc>> KLDivLossGradCpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddOutputAttr(kNumberTypeFloat16),
    &KLDivLossGradCpuKernelMod::LaunchKernel<Eigen::half>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeFloat32),
    &KLDivLossGradCpuKernelMod::LaunchKernel<float>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeFloat64)
      .AddOutputAttr(kNumberTypeFloat64),
    &KLDivLossGradCpuKernelMod::LaunchKernel<double>}};

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, KLDivLossGrad, KLDivLossGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
