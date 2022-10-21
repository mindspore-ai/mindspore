/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/kl_div_loss_cpu_kernel.h"
#include <algorithm>
#include <memory>
#include <utility>
#include <map>
#include <functional>
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/kl_div_loss.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
const size_t kMyAddInputsNum = 2;
const size_t kMyAddOutputsNum = 1;

bool KLDivLossCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::KLDivLoss>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(EXCEPTION) << "cast KLDivLoss ops failed!";
  }
  kernel_name_ = kernel_ptr->name();
  reductionMode_ = kernel_ptr->get_reduction();

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMyAddInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMyAddOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does not support this data type: " << kernel_attr;
  }

  kernel_func_ = func_list_[index].second;
  return true;
}

bool KLDivLossCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(kernel_func_);
  return kernel_func_(this, inputs, workspace, outputs);
}

int KLDivLossCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &onHost) {
  int ret = 0;
  ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, onHost);
  if (ret != 0) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return ret;
  }

  input_x_shape_ = inputs[kIndex0]->GetShapeVector();
  input_target_shape_ = inputs[kIndex1]->GetShapeVector();
  output_before_reduction_shape_ = CPUKernelUtils::GetBroadcastShape(input_x_shape_, input_target_shape_);

  input_x_shape_size_ = std::accumulate(input_x_shape_.begin(), input_x_shape_.end(), 1, std::multiplies<int64_t>());
  input_target_shape_size_ =
    std::accumulate(input_target_shape_.begin(), input_target_shape_.end(), 1, std::multiplies<int64_t>());
  output_before_reduction_shape_size_ = std::accumulate(
    output_before_reduction_shape_.begin(), output_before_reduction_shape_.end(), 1, std::multiplies<int64_t>());

  size_t type_size = GetTypeByte(TypeIdToType(inputs[kIndex0]->GetDtype()));
  workspace_size_list_.push_back(input_target_shape_size_ * type_size);
  if (reductionMode_ != ops::kNone) {
    workspace_size_list_.push_back(output_before_reduction_shape_size_ * type_size);
  }

  if (reductionMode_ == ops::kBatchMean) {
    if (output_before_reduction_shape_.size() < 1) {
      MS_LOG(EXCEPTION) << kernel_name_
                        << ": for batchmean reduction, each input must be an array or matrix, but got a number";
    }
    batch_size_ = output_before_reduction_shape_[kIndex0];
  }
  return ret;
}

std::vector<KernelAttr> KLDivLossCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, KLDivLossCpuKernelMod::KLDivLossFunc> &pair) { return pair.first; });
  return support_list;
}

template <typename T>
bool KLDivLossCpuKernelMod::LaunchNoneReduction(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &workspace,
                                                const std::vector<AddressPtr> &outputs) {
  auto *input_x = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto *input_target = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  auto *input_target_log = reinterpret_cast<T *>(workspace[kIndex0]->addr);
  auto *y = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_x(input_x, input_x_shape_size_, 1);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_target(input_target, input_target_shape_size_, 1);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_log(input_target_log, input_target_shape_size_, 1);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_y(y, output_before_reduction_shape_size_, 1);

  array_log = Eigen::log(array_target);
  BroadcastIterator base_iter(input_x_shape_, input_target_shape_, output_before_reduction_shape_);
  auto task = [&](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; ++i) {
      if (std::isnan(static_cast<float>(array_log[iter.GetInputPosB()]))) {
        array_y[i] = static_cast<T>(0);
      } else {
        T diff = static_cast<T>(array_log[iter.GetInputPosB()] - array_x[iter.GetInputPosA()]);
        array_y[i] = static_cast<T>(diff * array_target[iter.GetInputPosB()]);
      }
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, output_before_reduction_shape_size_, this, &parallel_search_info_);
  return true;
}

template <typename T>
bool KLDivLossCpuKernelMod::LaunchOther(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                        const std::vector<AddressPtr> &outputs) {
  auto *input_x = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto *input_target = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  auto *input_target_log = reinterpret_cast<T *>(workspace[kIndex0]->addr);
  auto *tmp_result = reinterpret_cast<T *>(workspace[kIndex1]->addr);
  auto *y = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_x(input_x, input_x_shape_size_, 1);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_target(input_target, input_target_shape_size_, 1);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_log(input_target_log, input_target_shape_size_, 1);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_tmp(tmp_result, output_before_reduction_shape_size_, 1);

  array_log = Eigen::log(array_target);
  BroadcastIterator base_iter(input_x_shape_, input_target_shape_, output_before_reduction_shape_);
  auto task = [&](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; ++i) {
      if (std::isnan(static_cast<float>(array_log[iter.GetInputPosB()]))) {
        array_tmp[i] = static_cast<T>(0);
      } else {
        T diff = static_cast<T>(array_log[iter.GetInputPosB()] - array_x[iter.GetInputPosA()]);
        array_tmp[i] = static_cast<T>(diff * array_target[iter.GetInputPosB()]);
      }
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, output_before_reduction_shape_size_, this, &parallel_search_info_);

  if (reductionMode_ == ops::kSum) {
    y[kIndex0] = array_tmp.sum();
    return true;
  }

  if (reductionMode_ == ops::kMean) {
    y[kIndex0] = array_tmp.mean();
    return true;
  }

  if (reductionMode_ == ops::kBatchMean) {
    y[kIndex0] = array_tmp.sum() / static_cast<T>(batch_size_);
    return true;
  }
  return false;
}

template <typename T>
bool KLDivLossCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMyAddInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMyAddOutputsNum, kernel_name_);

  if (reductionMode_ == ops::kNone) {
    return LaunchNoneReduction<T>(inputs, workspace, outputs);
  }

  return LaunchOther<T>(inputs, workspace, outputs);
}

std::vector<std::pair<KernelAttr, KLDivLossCpuKernelMod::KLDivLossFunc>> KLDivLossCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &KLDivLossCpuKernelMod::LaunchKernel<Eigen::half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &KLDivLossCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &KLDivLossCpuKernelMod::LaunchKernel<double>}};

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, KLDivLoss, KLDivLossCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
