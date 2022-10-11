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

#include "plugin/device/cpu/kernel/bce_with_logits_loss_cpu_kernel.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <string>
#include <map>
#include <memory>
#include "plugin/device/cpu/kernel/nnacl/fp32/bce_with_logits_loss_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "mindspore/core/ops/bce_with_logits_loss.h"

namespace mindspore {
namespace kernel {
namespace {
using KernelRunFunc = BCEWithLogitsLossCpuKernelMod::KernelRunFunc;
}  // namespace
bool BCEWithLogitsLossCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (kernel_name_ != prim::kPrimBCEWithLogitsLoss->name()) {
    MS_LOG(ERROR) << "For 'BCEWithLogitsLoss', it's kernel name invalid, got " << kernel_name_;
    return false;
  }
  auto kernel_ptr = std::make_shared<ops::BCEWithLogitsLoss>(base_operator->GetPrim());
  const auto reduction = kernel_ptr->get_reduction();
  if (reduction == NONE) {
    reduction_ = kNone;
    is_reduction_ = false;
  } else if (reduction == MEAN) {
    reduction_ = kMean;
    is_reduction_ = true;
  } else if (reduction == SUM) {
    reduction_ = kSum;
    is_reduction_ = true;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'reduction' must be 'none', 'mean', or 'sum', but got "
                  << reduction;
    return false;
  }
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int BCEWithLogitsLossCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  input_logits_shape_ = inputs.at(kIndex0)->GetShapeVector();
  input_size_ = SizeOf(input_logits_shape_);
  input_label_shape_ = inputs.at(kIndex1)->GetShapeVector();
  input_weight_shape_ = inputs.at(kIndex2)->GetShapeVector();
  input_post_weight_shape_ = inputs.at(kIndex3)->GetShapeVector();

  thread_num_ = std::min(input_size_, pool_->GetKernelThreadNum());
  // The output_size_list_ should be clear and reset.
  output_size_list_.clear();
  workspace_size_list_.clear();
  size_t unit_byte_size = GetTypeByte(TypeIdToType(outputs.at(kIndex0)->GetDtype()));
  size_t input_byte_size = input_size_ * unit_byte_size;
  if (reduction_ == kNone) {
    // The output is a Tensor in ReductionType none.
    (void)output_size_list_.emplace_back(input_byte_size);
  } else {
    // The output is a scalar in ReductionType mean or sum.
    (void)output_size_list_.emplace_back(unit_byte_size);
    (void)workspace_size_list_.emplace_back(thread_num_ * unit_byte_size);
  }
  is_broadcast_ = input_post_weight_shape_ != input_label_shape_ || input_weight_shape_ != input_label_shape_;
  return KRET_OK;
}

void BCEWithLogitsLossCpuKernelMod::RunTask(int task_id) {
  auto stride_per_thread = SizeToInt(UP_DIV(input_size_, thread_num_));
  int start = stride_per_thread * task_id;
  int end = start + stride_per_thread;
  end = std::min(end, SizeToInt(input_size_));
  auto per_logits = reinterpret_cast<float *>(logits_);
  auto per_label = reinterpret_cast<float *>(label_);
  auto per_weight = reinterpret_cast<float *>(weight_);
  auto per_post_weight = reinterpret_cast<float *>(post_weight_);
  auto *per_output = reinterpret_cast<float *>(output_);

  float *per_reduction_sum = nullptr;
  if (is_reduction_) {
    per_reduction_sum = reinterpret_cast<float *>(reduction_output_) + task_id;
  }
  if (!is_broadcast_) {
    per_logits += start;
    per_label += start;
    per_weight += start;
    per_post_weight += start;
    per_output += start;
    BCEWithLogitLoss(per_logits, per_label, per_weight, per_post_weight, (end - start), is_reduction_, per_output,
                     per_reduction_sum);
    return;
  }
  MultipleBroadcastIterator multi_broadcast_iterator(
    {input_logits_shape_, input_label_shape_, input_weight_shape_, input_post_weight_shape_}, input_logits_shape_);
  constexpr float zero = 0.0f;
  constexpr float one = 1.0f;
  auto iter = multi_broadcast_iterator;
  float broadcast_reduction_sum = 0.0f;
  iter.SetPos(IntToSize(start));
  for (int i = start; i < end; i++) {
    auto logits_value = per_logits[iter.GetInputPos(kIndex0)];
    auto label_value = per_label[iter.GetInputPos(kIndex1)];
    auto weight_value = per_weight[iter.GetInputPos(kIndex2)];
    auto post_weight_value = per_post_weight[iter.GetInputPos(kIndex3)];
    float max_value = -logits_value;
    max_value = max_value > zero ? max_value : zero;
    const auto log_weight = (post_weight_value - one) * label_value + one;
    const auto log_exp_value = std::log(std::exp(-max_value) + std::exp(-logits_value - max_value));
    float loss = (one - label_value) * logits_value + log_weight * (log_exp_value + max_value);
    if (is_reduction_) {
      broadcast_reduction_sum += loss * weight_value;
    } else {
      per_output[i] = loss * weight_value;
    }
    iter.GenNextPos();
  }
  if (is_reduction_) {
    *per_reduction_sum = broadcast_reduction_sum;
  }
}

namespace {
int BCERun(void *c_data, int task_id, float, float) {
  if (c_data == nullptr) {
    MS_LOG(ERROR) << "bce_with_logits_loss kernel does Launch failed, for null data. Its task id is " << task_id;
    return -1;
  }
  auto bce_kernel = reinterpret_cast<BCEWithLogitsLossCpuKernelMod *>(c_data);
  bce_kernel->RunTask(task_id);
  return 0;
}
}  // namespace

bool BCEWithLogitsLossCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) {
  logits_ = inputs.at(kIndex0)->addr;
  label_ = inputs.at(kIndex1)->addr;
  weight_ = inputs.at(kIndex2)->addr;
  post_weight_ = inputs.at(kIndex3)->addr;
  if (is_reduction_) {
    reduction_output_ = workspace.at(kIndex0)->addr;
  }
  output_ = outputs.at(kIndex0)->addr;
  if (pool_->ParallelLaunch(BCERun, this, SizeToInt(thread_num_)) != THREAD_OK) {
    return false;
  }
  // Do Small Reduction.
  if (is_reduction_) {
    auto output = reinterpret_cast<float *>(output_);
    output[0] = 0.0f;
    auto reduction_output = reinterpret_cast<float *>(reduction_output_);
    for (size_t i = 0; i < thread_num_; ++i) {
      output[0] += reduction_output[i];
    }
    if (reduction_ == kMean) {
      output[0] = output[0] / static_cast<float>(input_size_);
    }
  }
  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &BCEWithLogitsLossCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &BCEWithLogitsLossCpuKernelMod::LaunchKernel},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BCEWithLogitsLoss, BCEWithLogitsLossCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
