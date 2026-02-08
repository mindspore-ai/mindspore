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

#include "kernel/cpu/native/in_top_k_cpu_kernel.h"

#include "include/utils/thread_pool.h"
#include "infer/in_top_k.h"

namespace mindspore {
namespace kernel {
namespace in_top_k_cpu {
namespace {
constexpr size_t kInTopKInputsNum = 2;
constexpr size_t kInTopKOutputsNum = 1;
constexpr size_t kInTopKShapeRank = 2;
constexpr size_t kInTopkTargetShapeSize = 1;
}  // namespace

bool InTopKCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  k_ = GetValue<int64_t>(primitive_->GetAttr("k"));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "InTopK does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int InTopKCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  auto prediction_shape = Convert2SizeT(inputs.at(0)->GetShapeVector());
  auto target_shape = Convert2SizeT(inputs.at(1)->GetShapeVector());
  if (prediction_shape.size() != kInTopKShapeRank) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the rank of the first input must be equal to "
                      << kInTopKShapeRank << ", but got " << prediction_shape.size();
  }
  if (target_shape.size() != kInTopkTargetShapeSize || target_shape[0] != prediction_shape[0]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the rank of the second input must be equal to "
                      << kInTopkTargetShapeSize << ", and the shape size must be euqal to " << prediction_shape[0]
                      << ", but got " << target_shape;
  }
  outer_size_ = 1;
  for (size_t i = 0; i < prediction_shape.size() - 1; ++i) {
    outer_size_ *= prediction_shape[i];
  }
  inner_size_ = prediction_shape[prediction_shape.size() - 1];
  return ret;
}

template <typename T, typename S>
bool InTopKCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInTopKInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kInTopKOutputsNum, kernel_name_);

  auto predictions = reinterpret_cast<T *>(inputs[0]->device_ptr());
  auto targets = reinterpret_cast<S *>(inputs[1]->device_ptr());
  auto output = reinterpret_cast<bool *>(outputs[0]->device_ptr());

  if (k_ < 1) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', the 'k' must be greater than 0, but got " << k_;
    auto ret = memset_s(output, outputs[0]->size(), 0, outputs[0]->size());
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output failed. Error no: " << ret;
    }
    return true;
  }

  auto task = [this, predictions, targets, output](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto base_input = i * inner_size_;
      auto target_pred = predictions[base_input + targets[i]];
      bool invalid = (static_cast<size_t>(targets[i]) >= inner_size_) || !std::isfinite(target_pred);
      int64_t pos_num = 0;
      if (!invalid) {
        for (size_t k = 0; k < inner_size_; k++) {
          auto pred = predictions[base_input + k];
          if (!std::isfinite(pred)) {
            invalid = true;
            break;
          } else if (pred > target_pred) {
            pos_num++;
            if (pos_num > k_) {
              break;
            }
          }
        }
      }
      output[i] = invalid ? false : (pos_num < k_);
    }
  };
  ParallelLaunchAutoSearch(task, outer_size_, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, InTopKCpuKernelMod::InTopKFunc>> InTopKCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
   &InTopKCpuKernelMod::LaunchKernel<float, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
   &InTopKCpuKernelMod::LaunchKernel<float, int64_t>}};

std::vector<KernelAttr> InTopKCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, InTopKFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, InTopK, InTopKCpuKernelMod);
}  // namespace in_top_k_cpu
}  // namespace kernel
}  // namespace mindspore
