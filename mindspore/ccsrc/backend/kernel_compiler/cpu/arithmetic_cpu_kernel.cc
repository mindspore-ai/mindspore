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
#include <string>
#include <map>
#include "backend/kernel_compiler/cpu/arithmetic_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void ArithmeticCPUKernel<T>::AssignAdd(T *input1, const T *input2, T *out) {
  auto task = [&input1, &input2, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = input1[i] + input2[i];
      input1[i] = out[i];
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticCPUKernel<T>::Add(const T *input1, const T *input2, T *out) {
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
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  if (output_size_ > MAX_SUB_SERIAL_SIZE) {
    auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
      auto iter = base_iter;
      iter.SetPos(start);
      for (size_t i = start; i < end; i++) {
        out[i] = input1[iter.GetInputPosA()] - input2[iter.GetInputPosB()];
        iter.GenNextPos();
      }
    };
    CPUKernelUtils::ParallelFor(task, output_size_);
  } else {
    base_iter.SetPos(0);
    for (size_t i = 0; i < output_size_; i++) {
      out[i] = input1[base_iter.GetInputPosA()] - input2[base_iter.GetInputPosB()];
      base_iter.GenNextPos();
    }
  }
}

template <typename T>
void ArithmeticCPUKernel<T>::Mul(const T *input1, const T *input2, T *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      out[i] = input1[iter.GetInputPosA()] * input2[iter.GetInputPosB()];
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticCPUKernel<T>::RealDiv(const T *input1, const T *input2, T *out) {
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
void ArithmeticCPUKernel<T>::Div(const T *input1, const T *input2, T *out) {
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
void ArithmeticCPUKernel<T>::FloorDiv(const T *input1, const T *input2, T *out) {
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
      out[i] = (T)floor(static_cast<double>(dividend) / static_cast<double>(divisor));
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticCPUKernel<T>::Mod(const T *input1, const T *input2, T *out) {
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
void ArithmeticCPUKernel<T>::FloorMod(const T *input1, const T *input2, T *out) {
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
void ArithmeticCPUKernel<T>::Pow(const T *input1, const T *input2, T *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  if (output_size_ > MAX_POW_SERIAL_SIZE) {
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
      out[i] = diff * diff;
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

template <typename T>
void ArithmeticCPUKernel<T>::Atan2(const T *input1, const T *input2, T *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      out[i] =
        (T)atan2(static_cast<double>(input1[iter.GetInputPosA()]), static_cast<double>(input2[iter.GetInputPosB()]));
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
}

static const std::map<std::string, OperateType> kArithmeticBinOpTypeMap = {
  {prim::kPrimAdd->name(), ADD},
  {prim::kPrimSub->name(), SUB},
  {prim::kPrimMul->name(), MUL},
  {prim::kPrimDiv->name(), DIV},
  {prim::kPrimMod->name(), MOD},
  {prim::kPrimAssignAdd->name(), ASSIGNADD},
  {prim::kPrimPow->name(), POW},
  {prim::kPrimFloorDiv->name(), FLOORDIV},
  {prim::kPrimAtan2->name(), ATAN2},
  {prim::kPrimRealDiv->name(), REALDIV},
  {prim::kPrimSquaredDifference->name(), SQUAREDDIFFERENCE},
  {prim::kPrimFloorMod->name(), FLOORMOD}};

template <typename T>
void ArithmeticCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
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
bool ArithmeticCPUKernel<T>::Launch(const std::vector<AddressPtr> &inputs,
                                    const std::vector<AddressPtr> & /* workspace */,
                                    const std::vector<AddressPtr> &outputs) {
  T *input1 = reinterpret_cast<T *>(inputs[0]->addr);
  T *input2 = reinterpret_cast<T *>(inputs[1]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);

  if (operate_type_ == ADD) {
    Add(input1, input2, output);
  } else if (operate_type_ == SUB) {
    Sub(input1, input2, output);
  } else if (operate_type_ == MUL) {
    Mul(input1, input2, output);
  } else if (operate_type_ == REALDIV) {
    RealDiv(input1, input2, output);
  } else if (operate_type_ == DIV) {
    Div(input1, input2, output);
  } else if (operate_type_ == FLOORDIV) {
    FloorDiv(input1, input2, output);
  } else if (operate_type_ == MOD) {
    Mod(input1, input2, output);
  } else if (operate_type_ == FLOORMOD) {
    FloorMod(input1, input2, output);
  } else if (operate_type_ == POW) {
    Pow(input1, input2, output);
  } else if (operate_type_ == ASSIGNADD) {
    AssignAdd(input1, input2, output);
  } else if (operate_type_ == ATAN2) {
    Atan2(input1, input2, output);
  } else if (operate_type_ == SQUAREDDIFFERENCE) {
    SquaredDifference(input1, input2, output);
  } else {
    MS_LOG(EXCEPTION) << "Not support " << operate_type_;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
