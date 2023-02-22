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
const size_t kMinThreadNum = 1;

template <class T>
struct BitwiseAndFunc {
  T operator()(T a, T b) const { return a & b; }
};

template <class T>
struct BitwiseOrFunc {
  T operator()(T a, T b) const { return a | b; }
};

template <class T>
struct BitwiseXorFunc {
  T operator()(T a, T b) const { return a ^ b; }
};
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
  return true;
}

int BitwiseCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != 0) {
    return ret;
  }
  input_shape_1_ = inputs[kIndex0]->GetShapeVector();
  input_shape_2_ = inputs[kIndex1]->GetShapeVector();
  output_shape_ = outputs[kIndex0]->GetShapeVector();
  if (output_shape_.size() > max_dims_) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of output should be less than or equal to max_dims 7, but got "
                  << output_shape_.size() << ".";
    return KRET_RESIZE_FAILED;
  }

  if (input_shape_1_ == input_shape_2_) {
    broadcast_ = false;
  } else {
    broadcast_ = true;
  }

  switch (input_type_1_) {
    case kNumberTypeBool:
      InitFunc<bool>();
      break;
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
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', does not support " << TypeIdToString(input_type_1_);
      return KRET_RESIZE_FAILED;
  }

  if (output_shape_.size() == 0) {
    (void)output_shape_.insert(output_shape_.begin(), 1);
  }
  output_size_ = 1;
  for (size_t i = 0; i < output_shape_.size(); ++i) {
    output_size_ *= static_cast<size_t>(output_shape_[i]);
  }

  if (output_size_ > kBitwiseBigShapeNum) {
    bitwise_parallel_func_ = &BitwiseCpuKernelMod::BitwiseParallelMaxThread;
  } else {
    bitwise_parallel_func_ = &BitwiseCpuKernelMod::BitwiseParallelSearch;
  }

  thread_num_ = std::min(static_cast<size_t>(output_size_), pool_->GetKernelThreadNum());
  if (thread_num_ == 0) {
    thread_num_ = kMinThreadNum;
  }
  block_size_ = static_cast<float>(output_size_) / thread_num_;

  return KRET_OK;
}

template <typename T>
void BitwiseCpuKernelMod::InitFunc() {
  if (broadcast_ == false) {
    if (kernel_name_ == prim::kPrimBitwiseAnd->name()) {
      bitwise_launch_func_ = &BitwiseCpuKernelMod::LaunchNoBroadcast<T, BitwiseAndFunc<T>>;
    } else if (kernel_name_ == prim::kPrimBitwiseOr->name()) {
      bitwise_launch_func_ = &BitwiseCpuKernelMod::LaunchNoBroadcast<T, BitwiseOrFunc<T>>;
    } else if (kernel_name_ == prim::kPrimBitwiseXor->name()) {
      bitwise_launch_func_ = &BitwiseCpuKernelMod::LaunchNoBroadcast<T, BitwiseXorFunc<T>>;
    } else {
      MS_LOG(ERROR) << "For Bitwise, kernel name should be BitwiseAnd, BitwiseOr or BitwiseXor, but got "
                    << kernel_name_;
    }
  } else {
    if (kernel_name_ == prim::kPrimBitwiseAnd->name()) {
      bitwise_launch_func_ = &BitwiseCpuKernelMod::LaunchBroadcast<T, BitwiseAndFunc<T>>;
    } else if (kernel_name_ == prim::kPrimBitwiseOr->name()) {
      bitwise_launch_func_ = &BitwiseCpuKernelMod::LaunchBroadcast<T, BitwiseOrFunc<T>>;
    } else if (kernel_name_ == prim::kPrimBitwiseXor->name()) {
      bitwise_launch_func_ = &BitwiseCpuKernelMod::LaunchBroadcast<T, BitwiseXorFunc<T>>;
    } else {
      MS_LOG(ERROR) << "For Bitwise, kernel name should be BitwiseAnd, BitwiseOr or BitwiseXor, but got "
                    << kernel_name_;
    }
  }
}

void BitwiseCpuKernelMod::BitwiseParallelSearch(const CTask &task) {
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

void BitwiseCpuKernelMod::BitwiseParallelMaxThread(const CTask &task) {
  ParallelLaunch(task, output_size_, block_size_, this, pool_);
}

template <typename T, typename BitwiseFunT>
bool BitwiseCpuKernelMod::LaunchBroadcast(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input1 = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  const auto *input2 = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  auto *output = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  BitwiseFunT bitwise_func;
  BroadcastIterator base_iter(input_shape_1_, input_shape_2_, output_shape_);
  auto task = [this, &input1, &input2, &output, &base_iter, bitwise_func](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      output[i] = bitwise_func(input1[iter.GetInputPosA()], input2[iter.GetInputPosB()]);
      iter.GenNextPos();
    }
  };
  bitwise_parallel_func_(this, task);
  return true;
}

template <typename T, typename BitwiseFunT>
bool BitwiseCpuKernelMod::LaunchNoBroadcast(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input1 = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  const auto *input2 = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  auto *output = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  BitwiseFunT bitwise_func;
  auto task = [&input1, &input2, &output, bitwise_func](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      output[i] = bitwise_func(input1[i], input2[i]);
    }
  };
  bitwise_parallel_func_(this, task);
  return true;
}

template <typename T>
bool BitwiseCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kBitwiseInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBitwiseOutputsNum, kernel_name_);
  return bitwise_launch_func_(this, inputs, outputs);
}

#define BITWISE_CPU_KERNEL_MATCH(MS_T, T) \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_T).AddOutputAttr(MS_T), &BitwiseCpuKernelMod::LaunchKernel<T>

const std::vector<std::pair<KernelAttr, BitwiseCpuKernelMod::KernelRunFunc>> &BitwiseCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, BitwiseCpuKernelMod::KernelRunFunc>> func_list = {
    {BITWISE_CPU_KERNEL_MATCH(kNumberTypeBool, bool)},       {BITWISE_CPU_KERNEL_MATCH(kNumberTypeInt8, int8_t)},
    {BITWISE_CPU_KERNEL_MATCH(kNumberTypeInt16, int16_t)},   {BITWISE_CPU_KERNEL_MATCH(kNumberTypeInt32, int32_t)},
    {BITWISE_CPU_KERNEL_MATCH(kNumberTypeInt64, int64_t)},   {BITWISE_CPU_KERNEL_MATCH(kNumberTypeUInt8, uint8_t)},
    {BITWISE_CPU_KERNEL_MATCH(kNumberTypeUInt16, uint16_t)}, {BITWISE_CPU_KERNEL_MATCH(kNumberTypeUInt32, uint32_t)},
    {BITWISE_CPU_KERNEL_MATCH(kNumberTypeUInt64, uint64_t)},
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
