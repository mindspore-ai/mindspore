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

#include "plugin/device/gpu/kernel/arrays/select_gpu_kernel.h"
#include "mindspore/core/ops/select.h"

namespace mindspore {
namespace kernel {
template <typename T>
bool SelectGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &workspace,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  auto *input_cond = reinterpret_cast<bool *>(inputs[0]->addr);
  auto *input_x = reinterpret_cast<T *>(inputs[1]->addr);
  auto *input_y = reinterpret_cast<T *>(inputs[2]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  CalSelect(output_size_, input_cond, input_x, input_y, output, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

using selectPair = std::pair<KernelAttr, SelectGpuKernelMod::KernelRunFunc>;
const std::vector<selectPair> &SelectGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SelectGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SelectGpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SelectGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &SelectGpuKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &SelectGpuKernelMod::LaunchKernel<int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SelectGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeBool),
     &SelectGpuKernelMod::LaunchKernel<bool>},
  };
  return func_list;
}

bool SelectGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Select>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
  kernel_name_ = kernel_ptr->name();

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int SelectGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }

  auto shape_signed = inputs[0]->GetShapeVector();
  auto shape = Convert2SizeTClipNeg(shape_signed);
  is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name_, "input");
  if (is_null_input_) {
    return true;
  }
  size_t input_num = inputs.size();
  if (input_num != kSelectInputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 3, but got " << input_num;
    return KRET_RESIZE_FAILED;
  }
  size_t output_num = outputs.size();
  if (output_num != kSelectOutputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
    return KRET_RESIZE_FAILED;
  }
  output_size_ = kSelectOutputSizeInit;
  for (size_t x : shape) {
    output_size_ = output_size_ * x;
  }
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Select, SelectGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
