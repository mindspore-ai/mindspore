/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/arithmetic_cpu_kernel.h"

#include <cmath>
#include <string>
#include <unordered_map>
#include <limits>

#include "backend/kernel_compiler/cpu/nnacl/fp32/power_fp32.h"
#include "backend/kernel_compiler/cpu/nnacl/fp32/sub_fp32.h"
#include "backend/kernel_compiler/cpu/nnacl/fp32/mul_fp32.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 2;
constexpr size_t kOutputsNum = 1;
constexpr float kMaxSubSerialSize = 10000.0;
constexpr float kMaxPowSerialSize = 700.0;

template <typename T>
void ElementRealDiv(const T *input1, const T *input2, T *out, size_t size, size_t delta_1, size_t delta_2) {
  size_t idx_1 = 0;
  size_t idx_2 = 0;
  auto zero = (T)0;
  for (size_t i = 0; i < size; ++i) {
    auto dividend = input1[idx_1];
    auto divisor = input2[idx_2];
    idx_1 += delta_1;
    idx_2 += delta_2;
    if (divisor == zero) {
      if (dividend == zero) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        out[i] = dividend > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        out[i] = dividend > zero ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    out[i] = dividend / divisor;
  }
}
}  // namespace

template <typename T>
void ArithmeticCPUKernel<T>::AssignAdd(T *input1, const T *input2, T *out) const {
  auto task = [&input1, &input2, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = input1[i] + input2[i];
      input1[i] = out[i];
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticCPUKernel<T>::Add(const T *input1, const T *input2, T *out) const {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      out[i] = input1[iter.GetInputPosA()] + input2[iter.GetInputPosB()];
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticCPUKernel<T>::Sub(const T *input1, const T *input2, T *out) {
  if constexpr (std::is_same_v<T, float>) {
    if (input_shape1_ == input_shape2_) {
      auto task = [this, input1, input2, out](size_t start, size_t end) {
        (void)ElementSub(input1 + start, input2 + start, out + start, end - start);
      };
      ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
      return;
    }
    if (op_para_.in_elements_num0_ == 1 || op_para_.in_elements_num1_ == 1) {
      auto task = [this, input1, input2, out](size_t start, size_t end) {
        if (op_para_.in_elements_num0_ == 1) {
          (void)ElementOptSub(input1, input2 + start, out + start, end - start, &op_para_);
        } else {
          (void)ElementOptSub(input1 + start, input2, out + start, end - start, &op_para_);
        }
      };
      ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
      return;
    }
  }

  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      out[i] = input1[iter.GetInputPosA()] - input2[iter.GetInputPosB()];
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_, kMaxSubSerialSize);
}

template <typename T>
void ArithmeticCPUKernel<T>::Mul(const T *input1, const T *input2, T *out) {
  if constexpr (std::is_same_v<T, float>) {
    if (input_shape1_ == input_shape2_) {
      auto task = [this, input1, input2, out](size_t start, size_t end) {
        (void)ElementMul(input1 + start, input2 + start, out + start, end - start);
      };
      ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
      return;
    }
    if (op_para_.in_elements_num0_ == 1 || op_para_.in_elements_num1_ == 1) {
      auto task = [this, input1, input2, out](size_t start, size_t end) {
        if (op_para_.in_elements_num0_ == 1) {
          (void)ElementOptMul(input1, input2 + start, out + start, end - start, &op_para_);
        } else {
          (void)ElementOptMul(input1 + start, input2, out + start, end - start, &op_para_);
        }
      };
      ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
      return;
    }
  }
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(input1[iter.GetInputPosA()] * input2[iter.GetInputPosB()]);
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticCPUKernel<T>::RealDiv(const T *input1, const T *input2, T *out) {
  if (input_shape1_ == input_shape2_) {
    auto task = [&](size_t start, size_t end) {
      ElementRealDiv<T>(input1 + start, input2 + start, out + start, end - start, 1, 1);
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
    return;
  }
  if (op_para_.in_elements_num0_ == 1) {
    auto task = [&](size_t start, size_t end) {
      ElementRealDiv<T>(input1, input2 + start, out + start, end - start, 0, 1);
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
    return;
  }
  if (op_para_.in_elements_num1_ == 1) {
    auto task = [&](size_t start, size_t end) {
      ElementRealDiv<T>(input1 + start, input2, out + start, end - start, 1, 0);
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
    return;
  }

  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      auto dividend = input1[iter.GetInputPosA()];
      auto divisor = input2[iter.GetInputPosB()];
      iter.GenNextPos();
      auto zero = (T)0;
      if (divisor == zero) {
        if (dividend == zero) {
          out[i] = std::numeric_limits<T>::quiet_NaN();
          continue;
        }
        if (std::numeric_limits<T>::has_infinity) {
          out[i] = dividend > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
        } else {
          out[i] = dividend > zero ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
        }
        continue;
      }
      out[i] = dividend / divisor;
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticCPUKernel<T>::Div(const T *input1, const T *input2, T *out) const {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      auto dividend = input1[iter.GetInputPosA()];
      auto divisor = input2[iter.GetInputPosB()];
      iter.GenNextPos();
      auto zero = (T)0;
      if (divisor == zero) {
        if (dividend == zero) {
          out[i] = std::numeric_limits<T>::quiet_NaN();
          continue;
        }
        if (std::numeric_limits<T>::has_infinity) {
          out[i] = dividend > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
        } else {
          out[i] = dividend > zero ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
        }
        continue;
      }
      out[i] = dividend / divisor;
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticCPUKernel<T>::FloorDiv(const T *input1, const T *input2, T *out) const {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      auto dividend = input1[iter.GetInputPosA()];
      auto divisor = input2[iter.GetInputPosB()];
      iter.GenNextPos();
      auto zero = static_cast<T>(0);
      if (divisor == zero) {
        if (dividend == zero) {
          out[i] = std::numeric_limits<T>::quiet_NaN();
          continue;
        }
        if (std::numeric_limits<T>::has_infinity) {
          out[i] = dividend > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
        } else {
          out[i] = dividend > zero ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
        }
        continue;
      }
      out[i] = static_cast<T>(floor(static_cast<double>(dividend) / static_cast<double>(divisor)));
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticCPUKernel<T>::Mod(const T *input1, const T *input2, T *out) const {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      auto x = static_cast<double>(input1[iter.GetInputPosA()]);
      auto y = static_cast<double>(input2[iter.GetInputPosB()]);
      iter.GenNextPos();
      auto data_div = x / y;
      auto data_div_min = data_div < 0.0 ? data_div : 0.0;
      auto data_div_max = data_div > 0.0 ? data_div : 0.0;
      auto data_div_max_floor = floor(data_div_max);
      auto data_div_min_ceil = ceil(data_div_min);
      auto data_div_res = data_div_max_floor + data_div_min_ceil;
      out[i] = static_cast<T>(x - data_div_res * y);
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticCPUKernel<T>::FloorMod(const T *input1, const T *input2, T *out) const {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      auto x = static_cast<double>(input1[iter.GetInputPosA()]);
      auto y = static_cast<double>(input2[iter.GetInputPosB()]);
      iter.GenNextPos();
      auto res = x - floor(x / y) * y;
      out[i] = static_cast<T>((std::abs(res) > 1e-9) && ((res < 0.0) != (y < 0.0)) ? res + y : res);
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticCPUKernel<T>::Pow(const T *input1, const T *input2, T *out) const {
  if constexpr (std::is_same_v<T, float>) {
    auto is_power_single = [this]() {
      bool is_power_single = false;
      if (input_shape1_.size() == input_shape2_.size()) {
        is_power_single = true;
        for (size_t i = 0; i < input_shape1_.size(); ++i) {
          if (input_shape1_[i] != input_shape2_[i]) {
            is_power_single = false;
            break;
          }
        }
      }
      return is_power_single;
    };

    if (op_para_.in_elements_num1_ == 1) {
      auto task = [&](size_t start, size_t end) {
        (void)Power(input1 + start, input2, out + start, end - start, 1, 0, true);
      };
      CPUKernelUtils::ParallelFor(task, output_size_);
      return;
    }
    if (is_power_single()) {
      auto task = [&](size_t start, size_t end) {
        (void)Power(input1 + start, input2 + start, out + start, end - start, 1, 0, false);
      };
      CPUKernelUtils::ParallelFor(task, output_size_);
      return;
    }
  }

  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  if (output_size_ > kMaxPowSerialSize) {
    auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
      auto iter = base_iter;
      iter.SetPos(start);
      for (size_t i = start; i < end; i++) {
        auto x = static_cast<double>(input1[iter.GetInputPosA()]);
        auto y = static_cast<double>(input2[iter.GetInputPosB()]);
        out[i] = static_cast<T>(std::pow(x, y));
        iter.GenNextPos();
      }
    };
    CPUKernelUtils::ParallelFor(task, output_size_);
  } else {
    base_iter.SetPos(0);
    for (size_t i = 0; i < output_size_; i++) {
      auto sx = static_cast<double>(input1[base_iter.GetInputPosA()]);
      auto sy = static_cast<double>(input2[base_iter.GetInputPosB()]);
      out[i] = static_cast<T>(std::pow(sx, sy));
      base_iter.GenNextPos();
    }
  }
}

template <typename T>
void ArithmeticCPUKernel<T>::SquaredDifference(const T *input1, const T *input2, T *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      T diff = input1[iter.GetInputPosA()] - input2[iter.GetInputPosB()];
      out[i] = static_cast<T>(diff * diff);
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T>
void ArithmeticCPUKernel<T>::Atan2(const T *input1, const T *input2, T *out) const {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(
        atan2(static_cast<double>(input1[iter.GetInputPosA()]), static_cast<double>(input2[iter.GetInputPosB()])));
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticCPUKernel<T>::InitComputeFunc() {
  if (kernel_name_ == prim::kPrimAssignAdd->name()) {
    return;
  }
  static const std::unordered_map<std::string, TypeComputeFunc> arithmeticMathFuncMap{
    {prim::kPrimAdd->name(), &ArithmeticCPUKernel<T>::Add},
    {prim::kPrimSub->name(), &ArithmeticCPUKernel<T>::Sub},
    {prim::kPrimMul->name(), &ArithmeticCPUKernel<T>::Mul},
    {prim::kPrimDiv->name(), &ArithmeticCPUKernel<T>::Div},
    {prim::kPrimMod->name(), &ArithmeticCPUKernel<T>::Mod},
    {prim::kPrimFloorMod->name(), &ArithmeticCPUKernel<T>::FloorMod},
    {prim::kPrimPow->name(), &ArithmeticCPUKernel<T>::Pow},
    {prim::kPrimFloorDiv->name(), &ArithmeticCPUKernel<T>::FloorDiv},
    {prim::kPrimAtan2->name(), &ArithmeticCPUKernel<T>::Atan2},
    {prim::kPrimRealDiv->name(), &ArithmeticCPUKernel<T>::RealDiv},
    {prim::kPrimSquaredDifference->name(), &ArithmeticCPUKernel<T>::SquaredDifference}};
  if (arithmeticMathFuncMap.find(kernel_name_) == arithmeticMathFuncMap.end()) {
    MS_LOG(EXCEPTION) << "ArithmeticCPUKernel does not support " << kernel_name_;
  }
  compute_func_ = arithmeticMathFuncMap.at(kernel_name_);
}

template <typename T>
void ArithmeticCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
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

  op_para_.in_elements_num0_ = 1;
  for (size_t i = 0; i < input_shape1_.size(); ++i) {
    op_para_.in_elements_num0_ *= input_shape1_[i];
  }

  op_para_.in_elements_num1_ = 1;
  for (size_t i = 0; i < input_shape2_.size(); ++i) {
    op_para_.in_elements_num1_ *= input_shape2_[i];
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
  InitComputeFunc();
}

template <typename T>
bool ArithmeticCPUKernel<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                    const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  auto *input1 = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *input2 = reinterpret_cast<T *>(inputs[1]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  if (kernel_name_ == prim::kPrimAssignAdd->name()) {
    AssignAdd(input1, input2, output);
  } else {
    compute_func_(this, input1, input2, output);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
