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

#include "plugin/device/cpu/kernel/fast_gelu_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "mindspore/core/ops/fast_gelu.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore::kernel {
constexpr auto kFastGeLU = "FastGeLU";
constexpr const size_t kFastGeluInputsNum = 1;
constexpr const size_t kFastGeluOutputsNum = 1;

template <typename T>
bool FastGeLUCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kFastGeluInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kFastGeluOutputsNum, kernel_name_);
  T *input = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);
  T *output = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(output, false);

  const size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  auto task = [&input, &output](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      T x = input[i];
      double double_x = static_cast<double>(x);
      T res_one = static_cast<T>(1.0) + static_cast<T>(std::exp(-1.702 * std::abs(double_x)));
      T res_two = static_cast<T>(std::exp(0.851 * (double_x - std::abs(double_x))));
      output[i] = x * res_two / res_one;
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, FastGeLUCpuKernelMod::KernelRunFunc>> &FastGeLUCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, FastGeLUCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &FastGeLUCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &FastGeLUCpuKernelMod::LaunchKernel<float>},
  };
  return func_list;
}

bool FastGeLUCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::FastGeLU>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kFastGeluInputsNum || outputs.size() != kFastGeluOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kFastGeluInputsNum << " and "
                  << kFastGeluOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int FastGeLUCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }
  std::vector<int64_t> input_shape = inputs[0]->GetShapeVector();
  std::vector<int64_t> output_shape = outputs[0]->GetShapeVector();
  auto in_shape_size = input_shape.size();
  if (in_shape_size > max_dims_) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of input should be less than or equal to max_dims 7, but got " << in_shape_size
                  << ".";
    return KRET_RESIZE_FAILED;
  }
  auto output_shape_size = output_shape.size();
  if (in_shape_size != output_shape_size) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input shape size should be the same as output shape size, but got"
                  << " input shape size " << in_shape_size << " output shape size" << output_shape_size;
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, FastGeLU,
                                 []() { return std::make_shared<FastGeLUCpuKernelMod>(kFastGeLU); });
}  // namespace mindspore::kernel
