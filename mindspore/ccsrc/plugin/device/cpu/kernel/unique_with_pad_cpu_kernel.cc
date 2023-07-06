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

#include "plugin/device/cpu/kernel/unique_with_pad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
int UniqueWithPadCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUniqueWithPadInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUniqueWithPadOutputsNum, kernel_name_);
  auto input_shape = inputs[0]->GetShapeVector();
  input_size_ = static_cast<size_t>(input_shape[0]);
  batch_size_ = 1;
  if (batch_rank_ > 0) {
    auto pad_shape = inputs[kPadNumIndex]->GetShapeVector();
    auto pad_nums = std::accumulate(pad_shape.begin(), pad_shape.end(), 1, std::multiplies<int64_t>());
    batch_size_ = LongToSize(
      std::accumulate(input_shape.begin(), input_shape.begin() + batch_rank_, 1, std::multiplies<int64_t>()));
    input_size_ = static_cast<size_t>(input_shape[input_shape.size() - 1]);
    if (pad_nums != static_cast<int64_t>(batch_size_)) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the elements num of input 'pad' must be equal to input 'x' batch size, "
                           "but got the elements num of input 'pad': "
                        << pad_shape << " and input 'x' batch size: " << batch_size_;
    }
  }
  (void)workspace_size_list_.emplace_back(input_size_ * sizeof(int64_t));
  (void)workspace_size_list_.emplace_back(input_size_ * sizeof(int64_t));
  (void)workspace_size_list_.emplace_back(input_size_ * sizeof(int64_t));
  return KRET_OK;
}

bool UniqueWithPadCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &workspace,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUniqueWithPadInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUniqueWithPadOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeInt32) {
    UniqueCpuKernelMod::LaunchKernel<int, int>(inputs, workspace, outputs);
    PadOutput<int>(inputs, outputs, output_sizes_);
  } else if (dtype_ == kNumberTypeInt64) {
    UniqueCpuKernelMod::LaunchKernel<int64_t, int64_t>(inputs, workspace, outputs);
    PadOutput<int64_t>(inputs, outputs, output_sizes_);
  } else if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat16) {
    UniqueCpuKernelMod::LaunchKernel<float, int>(inputs, workspace, outputs);
    PadOutput<float>(inputs, outputs, output_sizes_);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of input must be float16, float32, int32, or int64, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

template <typename T>
void UniqueWithPadCpuKernelMod::PadOutput(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs,
                                          const std::vector<size_t> &start) {
  if (inputs.size() < kUniqueWithPadInputsNum || outputs.size() < kUniqueWithPadOutputsNum) {
    return;
  }
  auto pad_num_p = static_cast<T *>(inputs[1]->addr);
  auto *out = static_cast<T *>(outputs[0]->addr);
  for (size_t batch_i = 0; batch_i < batch_size_; batch_i++) {
    T pad_num = *pad_num_p;
    for (size_t i = start[batch_i]; i < input_size_; ++i) {
      out[i] = pad_num;
    }
    pad_num_p++;
    out += input_size_;
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UniqueWithPad, UniqueWithPadCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
