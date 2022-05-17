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

#include "plugin/device/cpu/kernel/bitwise_cpu_kernel.h"

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <algorithm>
#include <utility>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kBitwiseInputsNum = 2;
const size_t kBitwiseOutputsNum = 1;
}  // namespace

bool BitwiseCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  if (!base_operator) {
    MS_LOG(ERROR) << "For " << kernel_type_ << ", cast " << kernel_type_ << " ops failed!";
    return false;
  }
  kernel_name_ = base_operator->name();
  if (inputs.size() != kBitwiseInputsNum || outputs.size() != kBitwiseOutputsNum) {
    MS_LOG(ERROR) << "For" << kernel_name_ << ": input and output size should be " << kBitwiseInputsNum << " and "
                  << kBitwiseOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }
  input_type_1_ = inputs[0]->GetDtype();
  input_type_2_ = inputs[1]->GetDtype();
  if (input_type_1_ != input_type_2_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input1 and input2 must have the same type. But got input1 type "
                  << input_type_1_ << ", input2 type " << input_type_2_;
    return false;
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  switch (input_type_1_) {
    case kNumberTypeInt8:
      InitFunc<int8_t>();
      break;
    case kNumberTypeInt16:
      InitFunc<int16_t>();
      break;
    case kNumberTypeInt32:
      InitFunc<int32_t>();
      break;
    case kNumberTypeInt64:
      InitFunc<int64_t>();
      break;
    case kNumberTypeUInt8:
      InitFunc<uint8_t>();
      break;
    case kNumberTypeUInt16:
      InitFunc<uint16_t>();
      break;
    case kNumberTypeUInt32:
      InitFunc<uint32_t>();
      break;
    case kNumberTypeUInt64:
      InitFunc<uint64_t>();
      break;
    default:
      MS_LOG(ERROR) << "For '" << kernel_name_ << " kernel does not support " << TypeIdToString(input_type_1_);
      return false;
  }
  return true;
}

template <typename T>
void BitwiseAndFunc(const void *input1, const void *input2, void *output, size_t input1_index, size_t input2_index,
                    size_t output_index) {
  const auto *in1 = static_cast<const T *>(input1);
  const auto *in2 = static_cast<const T *>(input2);
  auto *out = static_cast<T *>(output);
  out[output_index] = in1[input1_index] & in2[input2_index];
}

template <typename T>
void BitwiseOrFunc(const void *input1, const void *input2, void *output, size_t input1_index, size_t input2_index,
                   size_t output_index) {
  const auto *in1 = static_cast<const T *>(input1);
  const auto *in2 = static_cast<const T *>(input2);
  auto *out = static_cast<T *>(output);
  out[output_index] = in1[input1_index] | in2[input2_index];
}

template <typename T>
void BitwiseXorFunc(const void *input1, const void *input2, void *output, size_t input1_index, size_t input2_index,
                    size_t output_index) {
  const auto *in1 = static_cast<const T *>(input1);
  const auto *in2 = static_cast<const T *>(input2);
  auto *out = static_cast<T *>(output);
  out[output_index] = in1[input1_index] ^ in2[input2_index];
}

template <typename T>
void BitwiseCpuKernelMod::InitFunc() {
  if (kernel_type_ == prim::kPrimBitwiseAnd->name()) {
    dist_func_ = BitwiseAndFunc<T>;
  } else if (kernel_type_ == prim::kPrimBitwiseOr->name()) {
    dist_func_ = BitwiseOrFunc<T>;
  } else if (kernel_type_ == prim::kPrimBitwiseXor->name()) {
    dist_func_ = BitwiseXorFunc<T>;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' kernel type should be " << kernel_name_ << " but got "
                      << kernel_type_;
  }
}

int BitwiseCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != 0) {
    return ret;
  }
  std::vector<int64_t> input_shape_1 = inputs[0]->GetShapeVector();
  std::vector<int64_t> input_shape_2 = inputs[1]->GetShapeVector();
  std::vector<int64_t> output_shape = outputs[0]->GetShapeVector();

  if (output_shape.size() > max_dims_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of output should be less than or equal to 7, but got " << output_shape.size()
                      << ".";
    return KRET_RESIZE_FAILED;
  }
  input_shape_1_.resize(input_shape_1.size(), 1);
  input_shape_2_.resize(input_shape_2.size(), 1);
  output_shape_.resize(output_shape.size(), 1);
  for (size_t i = 0; i < input_shape_1.size(); i++) {
    input_shape_1_[i] = static_cast<size_t>(input_shape_1[i]);
  }
  for (size_t i = 0; i < input_shape_2.size(); i++) {
    input_shape_2_[i] = static_cast<size_t>(input_shape_2[i]);
  }
  for (size_t i = 0; i < output_shape.size(); i++) {
    output_shape_[i] = static_cast<size_t>(output_shape[i]);
  }

  if (output_shape_.size() == 0) {
    (void)output_shape_.insert(output_shape_.begin(), 1);
  }
  for (size_t i = 0; i < output_shape_.size(); ++i) {
    output_size_ *= output_shape_[i];
  }

  return KRET_OK;
}

template <typename T>
bool BitwiseCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kBitwiseInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBitwiseOutputsNum, kernel_name_);
  const auto *input1 = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *input2 = reinterpret_cast<T *>(inputs[1]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  BroadcastIterator base_iter(input_shape_1_, input_shape_2_, output_shape_);
  auto task = [this, &input1, &input2, &output, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      this->dist_func_(input1, input2, output, iter.GetInputPosA(), iter.GetInputPosB(), i);
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, BitwiseCpuKernelMod::KernelRunFunc>> &BitwiseCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, BitwiseCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &BitwiseCpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &BitwiseCpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &BitwiseCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &BitwiseCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &BitwiseCpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &BitwiseCpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &BitwiseCpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &BitwiseCpuKernelMod::LaunchKernel<uint64_t>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, BitwiseAnd,
                                 []() { return std::make_shared<BitwiseCpuKernelMod>(prim::kPrimBitwiseAnd->name()); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, BitwiseOr,
                                 []() { return std::make_shared<BitwiseCpuKernelMod>(prim::kPrimBitwiseOr->name()); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, BitwiseXor,
                                 []() { return std::make_shared<BitwiseCpuKernelMod>(prim::kPrimBitwiseXor->name()); });
}  // namespace kernel
}  // namespace mindspore
