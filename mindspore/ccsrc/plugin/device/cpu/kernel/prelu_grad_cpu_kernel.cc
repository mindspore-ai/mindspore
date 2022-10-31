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

#include "plugin/device/cpu/kernel/prelu_grad_cpu_kernel.h"
#include <utility>
#include <algorithm>
#include "ops/grad/prelu_grad.h"

namespace mindspore {
namespace kernel {
bool PReLUGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t input_num = 3;
  constexpr size_t output_num = 2;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  kernel_name_ = base_operator->GetPrim()->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int PReLUGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto x_shape = LongVecToSizeVec(inputs[kIndex1]->GetShapeVector());
  auto weight_shape = LongVecToSizeVec(inputs[kIndex2]->GetShapeVector());
  input_length_ = std::accumulate(x_shape.begin(), x_shape.end(), size_t(1), std::multiplies<>());
  size_t x_rank = x_shape.size();
  size_t channel_num;
  if (x_rank == 0) {
    channel_num = 1;
    per_channel_length_ = 1;
  } else if (x_rank == 1) {
    channel_num = 1;
    per_channel_length_ = x_shape[0];
  } else {
    channel_num = x_shape[1];
    const size_t beg_pos = 2;
    per_channel_length_ = std::accumulate(x_shape.begin() + beg_pos, x_shape.end(), size_t(1), std::multiplies<>());
  }

  if (weight_shape.size() != 1 || (weight_shape[0] != 1 && weight_shape[0] != channel_num)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of weight must be equal to 1 and "
                  << "weight.shape[0] must be equal to 1 or the channel number, but got the dimension of "
                  << "weight: " << weight_shape.size() << ", weight.shape[0]: " << weight_shape[0]
                  << ", the channel num: " << channel_num;
    return KRET_RESIZE_FAILED;
  }
  weight_length_ = weight_shape[0];
  return KRET_OK;
}

template <typename T>
bool PReLUGradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  auto *dy = reinterpret_cast<T *>(inputs[0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(dy, false);
  auto *x = reinterpret_cast<T *>(inputs[1]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(x, false);
  auto *w = reinterpret_cast<T *>(inputs[2]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(w, false);
  auto *dx = reinterpret_cast<T *>(outputs[0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(dx, false);
  auto *dw = reinterpret_cast<T *>(outputs[1]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(dw, false);
  auto ret = memset_s(dw, outputs[1]->size, 0, outputs[1]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed. Error no: " << ret;
  }

  size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  auto task = [this, dy, x, w, dx, dw](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      size_t channel_id = weight_length_ == 1 ? 0 : (i / per_channel_length_) % weight_length_;
      T threshold = static_cast<T>(0);
      dx[i] = x[i] <= threshold ? w[channel_id] * dy[i] : dy[i];
      if (x[i] < threshold) {
        dw[channel_id] = x[i] * dy[i] + dw[channel_id];
      }
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, PReLUGradCpuKernelMod::PReLUGradLaunchFunc>> PReLUGradCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &PReLUGradCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &PReLUGradCpuKernelMod::LaunchKernel<float>},
};

std::vector<KernelAttr> PReLUGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, PReLUGradCpuKernelMod::PReLUGradLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PReLUGrad, PReLUGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
