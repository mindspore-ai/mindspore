/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/arithmetic_logic_cpu_kernel.h"
#include <cmath>
#include <string>
#include <map>
#include <functional>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void ArithmeticLogicCPUKernel<T>::Less(const T *input1, const T *input2, bool *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  if (output_size_ > MAX_LESS_SERIAL_SIZE) {
    auto task = [&](size_t start, size_t end) {
      auto iter = base_iter;
      iter.SetPos(start);
      for (size_t i = start; i < end; i++) {
        auto x = input1[iter.GetInputPosA()];
        auto y = input2[iter.GetInputPosB()];
        out[i] = std::less<T>()(x, y);
        iter.GenNextPos();
      }
    };
    CPUKernelUtils::ParallelFor(task, output_size_);
  } else {
    base_iter.SetPos(0);
    for (size_t i = 0; i < output_size_; i++) {
      auto x = input1[base_iter.GetInputPosA()];
      auto y = input2[base_iter.GetInputPosB()];
      out[i] = std::less<T>()(x, y);
      base_iter.GenNextPos();
    }
  }
}

template <typename T>
void ArithmeticLogicCPUKernel<T>::Equal(const T *input1, const T *input2, bool *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      auto x = input1[iter.GetInputPosA()];
      auto y = input2[iter.GetInputPosB()];
      out[i] = std::equal_to<T>()(x, y);
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticLogicCPUKernel<T>::NotEqual(const T *input1, const T *input2, bool *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      auto x = input1[iter.GetInputPosA()];
      auto y = input2[iter.GetInputPosB()];
      out[i] = std::not_equal_to<T>()(x, y);
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticLogicCPUKernel<T>::LogicalAnd(const T *input1, const T *input2, bool *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      out[i] = input1[iter.GetInputPosA()] && input2[iter.GetInputPosB()];
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticLogicCPUKernel<T>::LogicalOr(const T *input1, const T *input2, bool *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      out[i] = input1[iter.GetInputPosA()] || input2[iter.GetInputPosB()];
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticLogicCPUKernel<T>::Greater(const T *input1, const T *input2, bool *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      auto x = input1[iter.GetInputPosA()];
      auto y = input2[iter.GetInputPosB()];
      out[i] = std::greater<T>()(x, y);
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticLogicCPUKernel<T>::GreaterEqual(const T *input1, const T *input2, bool *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      auto x = input1[iter.GetInputPosA()];
      auto y = input2[iter.GetInputPosB()];
      out[i] = std::greater_equal<T>()(x, y);
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticLogicCPUKernel<T>::LessEqual(const T *input1, const T *input2, bool *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      auto x = input1[iter.GetInputPosA()];
      auto y = input2[iter.GetInputPosB()];
      out[i] = std::less_equal<T>()(x, y);
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

static const std::map<std::string, OperateType> kArithmeticBinOpTypeMap = {
  {prim::kPrimGreater->name(), GREATER},       {prim::kPrimGreaterEqual->name(), GREATEREQUAL},
  {prim::kPrimLogicalAnd->name(), LOGICALAND}, {prim::kPrimLessEqual->name(), LESSEQUAL},
  {prim::kPrimLogicalOr->name(), LOGICALOR},   {prim::kPrimLess->name(), LESS},
  {prim::kPrimNotEqual->name(), NOTEQUAL},     {prim::kPrimEqual->name(), EQUAL}};

template <typename T>
void ArithmeticLogicCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
  if (kArithmeticBinOpTypeMap.find(kernel_name) != kArithmeticBinOpTypeMap.end()) {
    operate_type_ = kArithmeticBinOpTypeMap.at(kernel_name);
  } else {
    MS_LOG(EXCEPTION) << "Not support " << kernel_name;
  }

  input_shape1_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  input_shape2_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  if (output_shape_.size() == 0) {
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
  if (dtype_ != AnfAlgo::GetInputDeviceDataType(kernel_node, 1)) {
    MS_LOG(EXCEPTION) << "Input0 and input1 must has the same data type";
  }
  target_dtype_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
}

template <typename T>
bool ArithmeticLogicCPUKernel<T>::Launch(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> & /* workspace */,
                                         const std::vector<AddressPtr> &outputs) {
  T *input1 = reinterpret_cast<T *>(inputs[0]->addr);
  T *input2 = reinterpret_cast<T *>(inputs[1]->addr);
  bool *output = reinterpret_cast<bool *>(outputs[0]->addr);

  if (operate_type_ == LESS) {
    Less(input1, input2, output);
  } else if (operate_type_ == EQUAL) {
    Equal(input1, input2, output);
  } else if (operate_type_ == NOTEQUAL) {
    NotEqual(input1, input2, output);
  } else if (operate_type_ == GREATER) {
    Greater(input1, input2, output);
  } else if (operate_type_ == GREATEREQUAL) {
    GreaterEqual(input1, input2, output);
  } else if (operate_type_ == LESSEQUAL) {
    LessEqual(input1, input2, output);
  } else if (operate_type_ == LOGICALAND) {
    LogicalAnd(input1, input2, output);
  } else if (operate_type_ == LOGICALOR) {
    LogicalOr(input1, input2, output);
  } else {
    MS_LOG(EXCEPTION) << "Not support " << operate_type_;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
