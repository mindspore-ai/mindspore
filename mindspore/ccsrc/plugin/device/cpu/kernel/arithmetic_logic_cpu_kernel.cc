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

#include "plugin/device/cpu/kernel/arithmetic_logic_cpu_kernel.h"

#include <string>
#include <cmath>
#include <unordered_map>
#include <functional>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxLessSerialSize = 15000;
constexpr size_t kInputsNum = 2;
constexpr size_t kOutputsNum = 1;
}  // namespace

template <typename T>
template <typename Op>
void ArithmeticLogicCpuKernelMod<T>::BinaryOp(const T *input1, const T *input2, bool *out, Op op) {
  size_t input1_size = 1;
  size_t input2_size = 2;

  for (size_t i = 0; i < output_shape_.size(); i++) {
    input1_size *= input_shape1_[i];
    input2_size *= input_shape2_[i];
  }

  if (input_shape1_ == input_shape2_) {
    auto task = [this, input1, input2, out, op](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out[i] = op(input1[i], input2[i]);
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  } else if (input1_size == 1) {
    auto task = [this, input1, input2, out, op](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out[i] = op(input1[0], input2[i]);
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  } else if (input2_size == 1) {
    auto task = [this, input1, input2, out, op](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out[i] = op(input1[i], input2[0]);
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  } else {
    BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
    auto task = [this, input1, input2, out, op, &base_iter](size_t start, size_t end) {
      auto iter = base_iter;
      iter.SetPos(start);
      for (size_t i = start; i < end; i++) {
        auto x = input1[iter.GetInputPosA()];
        auto y = input2[iter.GetInputPosB()];
        out[i] = op(x, y);
        iter.GenNextPos();
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  }
}

template <typename T>
void ArithmeticLogicCpuKernelMod<T>::Less(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::less<T>());
}

template <typename T>
void ArithmeticLogicCpuKernelMod<T>::Equal(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::equal_to<T>());
}

template <typename T>
void ArithmeticLogicCpuKernelMod<T>::NotEqual(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::not_equal_to<T>());
}

template <typename T>
void ArithmeticLogicCpuKernelMod<T>::LogicalAnd(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::logical_and<T>());
}

template <typename T>
void ArithmeticLogicCpuKernelMod<T>::LogicalOr(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::logical_or<T>());
}

template <typename T>
void ArithmeticLogicCpuKernelMod<T>::LogicalXor(const T *input1, const T *input2, bool *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [input1, input2, out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      out[i] = input1[iter.GetInputPosA()] != input2[iter.GetInputPosB()];
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticLogicCpuKernelMod<T>::Greater(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::greater<T>());
}

template <typename T>
void ArithmeticLogicCpuKernelMod<T>::GreaterEqual(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::greater_equal<T>());
}

template <typename T>
void ArithmeticLogicCpuKernelMod<T>::LessEqual(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::less_equal<T>());
}

template <typename T>
void ArithmeticLogicCpuKernelMod<T>::InitComputeFunc() {
  static const std::unordered_map<std::string, TypeComputeFunc> arithmeticLogicFuncMap{
    {prim::kPrimGreater->name(), &ArithmeticLogicCpuKernelMod<T>::Greater},
    {prim::kPrimGreaterEqual->name(), &ArithmeticLogicCpuKernelMod<T>::GreaterEqual},
    {prim::kPrimLogicalAnd->name(), &ArithmeticLogicCpuKernelMod<T>::LogicalAnd},
    {prim::kPrimLessEqual->name(), &ArithmeticLogicCpuKernelMod<T>::LessEqual},
    {prim::kPrimLogicalOr->name(), &ArithmeticLogicCpuKernelMod<T>::LogicalOr},
    {prim::kPrimLogicalXor->name(), &ArithmeticLogicCpuKernelMod<T>::LogicalXor},
    {prim::kPrimLess->name(), &ArithmeticLogicCpuKernelMod<T>::Less},
    {prim::kPrimNotEqual->name(), &ArithmeticLogicCpuKernelMod<T>::NotEqual},
    {prim::kPrimEqual->name(), &ArithmeticLogicCpuKernelMod<T>::Equal}};
  if (arithmeticLogicFuncMap.find(kernel_name_) == arithmeticLogicFuncMap.end()) {
    MS_LOG(EXCEPTION) << "For 'ArithmeticLogic', only supports operators in "
                      << Unorderedmap2Str(arithmeticLogicFuncMap) << ", but got " << kernel_name_;
  }
  compute_func_ = arithmeticLogicFuncMap.at(kernel_name_);
}

template <typename T>
void ArithmeticLogicCpuKernelMod<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_shape1_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  input_shape2_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  output_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  if (output_shape_.empty()) {
    (void)output_shape_.insert(output_shape_.begin(), 1);
  }

  output_size_ = 1;
  for (size_t i = 0; i < output_shape_.size(); ++i) {
    output_size_ *= output_shape_[i];
  }

  size_t l = input_shape1_.size();
  for (size_t i = 0; i < output_shape_.size() - l; ++i) {
    (void)input_shape1_.insert(input_shape1_.begin(), 1);
  }
  l = input_shape2_.size();
  for (size_t i = 0; i < output_shape_.size() - l; ++i) {
    (void)input_shape2_.insert(input_shape2_.begin(), 1);
  }
  CPUKernelUtils::GetElementNumEveryDim(input_shape1_, &input_element_num1_);
  CPUKernelUtils::GetElementNumEveryDim(input_shape2_, &input_element_num2_);
  CPUKernelUtils::GetElementNumEveryDim(output_shape_, &output_element_num_);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  auto dtype_1 = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  if (dtype_ != dtype_1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'input1' and 'input2' should have the same data type, but got type of 'input1': "
                      << dtype_ << ", and the type of 'input2': " << dtype_1;
  }
  InitComputeFunc();
}

template <typename T>
bool ArithmeticLogicCpuKernelMod<T>::Launch(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> & /* workspace */,
                                            const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  const auto *input1 = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *input2 = reinterpret_cast<T *>(inputs[1]->addr);
  bool *output = reinterpret_cast<bool *>(outputs[0]->addr);
  compute_func_(this, input1, input2, output);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
