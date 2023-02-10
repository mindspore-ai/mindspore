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

#include "plugin/device/cpu/kernel/channel_shuffle_cpu_kernel.h"
#include <functional>
#include <vector>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kChannelShuffleInputsNum = 1;
constexpr size_t kChannelShuffleOutputsNum = 1;
#define SHUFFLE_CHANNEL_COMPUTE_CASE(DTYPE, TYPE, INPUTS, OUTPUTS) \
  case (DTYPE): {                                                  \
    LaunchKernel<TYPE>(INPUTS, OUTPUTS);                           \
    break;                                                         \
  }
}  // namespace

bool ChannelShuffleCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  input_dtype_ = inputs[0]->GetDtype();
  group_ = GetValue<int64_t>(base_operator->GetAttr("group"));
  return true;
}

bool ChannelShuffleCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &workspace,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kChannelShuffleInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kChannelShuffleOutputsNum, kernel_name_);
  switch (input_dtype_) {
    SHUFFLE_CHANNEL_COMPUTE_CASE(kNumberTypeInt8, int8_t, inputs, outputs)
    SHUFFLE_CHANNEL_COMPUTE_CASE(kNumberTypeInt16, int16_t, inputs, outputs)
    SHUFFLE_CHANNEL_COMPUTE_CASE(kNumberTypeInt32, int32_t, inputs, outputs)
    SHUFFLE_CHANNEL_COMPUTE_CASE(kNumberTypeInt64, int64_t, inputs, outputs)
    SHUFFLE_CHANNEL_COMPUTE_CASE(kNumberTypeUInt8, uint8_t, inputs, outputs)
    SHUFFLE_CHANNEL_COMPUTE_CASE(kNumberTypeUInt16, uint16_t, inputs, outputs)
    SHUFFLE_CHANNEL_COMPUTE_CASE(kNumberTypeUInt32, uint32_t, inputs, outputs)
    SHUFFLE_CHANNEL_COMPUTE_CASE(kNumberTypeUInt64, uint64_t, inputs, outputs)
    SHUFFLE_CHANNEL_COMPUTE_CASE(kNumberTypeFloat16, float16, inputs, outputs)
    SHUFFLE_CHANNEL_COMPUTE_CASE(kNumberTypeFloat32, float, inputs, outputs)
    SHUFFLE_CHANNEL_COMPUTE_CASE(kNumberTypeFloat64, double, inputs, outputs)
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of input 'x' "
                        << TypeIdToType(input_dtype_)->ToString() << " not support.";
  }
  return true;
}

int ChannelShuffleCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[0]->GetShapeVector();
  outputs_ = outputs;
  return KRET_OK;
}

std::vector<KernelAttr> ChannelShuffleCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
    KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
    KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}

template <typename T>
bool ChannelShuffleCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &outputs) {
  int64_t dims = input_shape_.size();
  int64_t b = input_shape_[0];
  int64_t c = input_shape_[1];
  int64_t oc = c / group_;
  int64_t loc_out;
  int64_t loc_in;
  int64_t area = 1;
  for (int64_t i = 2; i < dims; i++) {
    area = area * input_shape_[i];
  }
  auto *in = reinterpret_cast<T *>(inputs[0]->addr);
  auto *out = reinterpret_cast<T *>(outputs[0]->addr);
  outputs_[0]->SetShapeVector(input_shape_);

  /*
    view the shape to n g c/g h*w,and transpose dim 1 and dim 2
  */
  for (int64_t l1 = 0; l1 < b; l1++) {
    for (int64_t l2 = 0; l2 < group_; l2++) {
      for (int64_t l3 = 0; l3 < oc; l3++) {
        for (int64_t l4 = 0; l4 < area; l4++) {
          loc_in = (l4 + l3 * area + l2 * oc * area + l1 * group_ * oc * area);
          loc_out = (l4 + l2 * area + l3 * group_ * area + l1 * group_ * oc * area);
          *(out + loc_out) = *(in + loc_in);
        }
      }
    }
  }
  return true;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ChannelShuffle, ChannelShuffleCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
