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
#include <cmath>
#include <string>
#include <map>
#include "backend/kernel_compiler/cpu/arithmetic_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void ArithmeticCPUKernel::AssignAdd(T *input1, const T *input2, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = input1[i] + input2[i];
      input1[i] = out[i];
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::Add(const T *input1, const T *input2, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      out[i] = input1[idx[0]] + input2[idx[1]];
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::Sub(const T *input1, const T *input2, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      out[i] = input1[idx[0]] - input2[idx[1]];
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::Mul(const T *input1, const T *input2, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      out[i] = input1[idx[0]] * input2[idx[1]];
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::RealDiv(const T *input1, const T *input2, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      auto dividend = input1[idx[0]];
      auto divisor = input2[idx[1]];
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
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::Div(const T *input1, const T *input2, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      auto dividend = input1[idx[0]];
      auto divisor = input2[idx[1]];
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
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::FloorDiv(const T *input1, const T *input2, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      auto dividend = input1[idx[0]];
      auto divisor = input2[idx[1]];
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
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::Mod(const T *input1, const T *input2, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      auto x = static_cast<double>(input1[idx[0]]);
      auto y = static_cast<double>(input2[idx[1]]);
      auto data_div = x / y;
      auto data_div_min = data_div < 0.0 ? data_div : 0.0;
      auto data_div_max = data_div > 0.0 ? data_div : 0.0;
      auto data_div_max_floor = floor(data_div_max);
      auto data_div_min_ceil = ceil(data_div_min);
      auto data_div_res = data_div_max_floor + data_div_min_ceil;
      out[i] = static_cast<T>(x - data_div_res * y);
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::Pow(const T *input1, const T *input2, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      auto x = static_cast<double>(input1[idx[0]]);
      auto y = static_cast<double>(input2[idx[1]]);
      out[i] = static_cast<T>(std::pow(x, y));
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::Less(const T *input1, const T *input2, bool *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      out[i] = input1[idx[0]] < input2[idx[1]];
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::Equal(const T *input1, const T *input2, bool *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      out[i] = input1[idx[0]] == input2[idx[1]];
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::NotEqual(const T *input1, const T *input2, bool *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      out[i] = input1[idx[0]] != input2[idx[1]];
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::LogicalAnd(const T *input1, const T *input2, bool *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      out[i] = input1[idx[0]] && input2[idx[1]];
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::LogicalOr(const T *input1, const T *input2, bool *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      out[i] = input1[idx[0]] || input2[idx[1]];
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::SquaredDifference(const T *input1, const T *input2, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      T diff = input1[idx[0]] - input2[idx[1]];
      out[i] = diff * diff;
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::Greater(const T *input1, const T *input2, bool *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      out[i] = input1[idx[0]] > input2[idx[1]];
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::GreaterEqual(const T *input1, const T *input2, bool *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      out[i] = input1[idx[0]] >= input2[idx[1]];
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::LessEqual(const T *input1, const T *input2, bool *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      out[i] = input1[idx[0]] <= input2[idx[1]];
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ArithmeticCPUKernel::Atan2(const T *input1, const T *input2, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx;
      GenIndex(i, &idx);
      out[i] = (T)atan2(static_cast<double>(input1[idx[0]]), static_cast<double>(input2[idx[1]]));
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

static const std::map<std::string, OperateType> kArithmeticBinOpTypeMap = {
  {prim::kPrimGreater->name(), GREATER},
  {prim::kPrimAdd->name(), ADD},
  {prim::kPrimGreaterEqual->name(), GREATEREQUAL},
  {prim::kPrimSub->name(), SUB},
  {prim::kPrimLogicalAnd->name(), LOGICALAND},
  {prim::kPrimMul->name(), MUL},
  {prim::kPrimLessEqual->name(), LESSEQUAL},
  {prim::kPrimDiv->name(), DIV},
  {prim::kPrimLogicalOr->name(), LOGICALOR},
  {prim::kPrimMod->name(), MOD},
  {prim::kPrimAssignAdd->name(), ASSIGNADD},
  {prim::kPrimPow->name(), POW},
  {prim::kPrimFloorDiv->name(), FLOORDIV},
  {prim::kPrimLess->name(), LESS},
  {prim::kPrimNotEqual->name(), NOTEQUAL},
  {prim::kPrimAtan2->name(), ATAN2},
  {prim::kPrimRealDiv->name(), REALDIV},
  {prim::kPrimEqual->name(), EQUAL},
  {prim::kPrimSquaredDifference->name(), SQUAREDDIFFERENCE}};

void ArithmeticCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
  if (kArithmeticBinOpTypeMap.find(kernel_name) != kArithmeticBinOpTypeMap.end()) {
    operate_type_ = kArithmeticBinOpTypeMap.at(kernel_name);
  } else {
    MS_LOG(EXCEPTION) << "Not support " << kernel_name;
  }

  input_shape0_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  input_shape1_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  if (output_shape_.size() == 0) {
    output_shape_.insert(output_shape_.begin(), 1);
  }
  size_t l = input_shape0_.size();
  for (size_t i = 0; i < output_shape_.size() - l; ++i) {
    input_shape0_.insert(input_shape0_.begin(), 1);
  }
  l = input_shape1_.size();
  for (size_t i = 0; i < output_shape_.size() - l; ++i) {
    input_shape1_.insert(input_shape1_.begin(), 1);
  }
  CPUKernelUtils::GetElementNumEveryDim(input_shape0_, &input_element_num0_);
  CPUKernelUtils::GetElementNumEveryDim(input_shape1_, &input_element_num1_);
  CPUKernelUtils::GetElementNumEveryDim(output_shape_, &output_element_num_);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  if (dtype_ != AnfAlgo::GetInputDeviceDataType(kernel_node, 1)) {
    MS_LOG(EXCEPTION) << "Input0 and input1 must has the same data type";
  }
  target_dtype_ = AnfAlgo::GetOutputInferDataType(kernel_node, 0);
}

bool ArithmeticCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                 const std::vector<kernel::AddressPtr> & /*workspace*/,
                                 const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeBool) {
    LaunchKernelLogic<bool>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt8) {
    LaunchKernel<int8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt16) {
    LaunchKernel<int16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt8) {
    LaunchKernel<uint8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt32) {
    LaunchKernel<uint32_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Data type " << TypeIdLabel(dtype_) << "is not support.";
  }
  return true;
}

void ArithmeticCPUKernel::GenIndex(size_t num, std::vector<size_t> *idx) {
  std::vector<size_t> tmp;
  for (size_t i = 0; i < output_shape_.size() - 1; ++i) {
    if (output_element_num_[i] > num) {
      tmp.push_back(0);
    } else {
      tmp.push_back(num / output_element_num_[i]);
      num %= output_element_num_[i];
    }
  }
  tmp.push_back(num);
  size_t idx0 = 0;
  size_t idx1 = 0;
  for (size_t k = 0; k < tmp.size() - 1; ++k) {
    if (input_shape0_[k] > 1) {
      idx0 += tmp[k] * input_element_num0_[k];
    }
    if (input_shape1_[k] > 1) {
      idx1 += tmp[k] * input_element_num1_[k];
    }
  }
  if (input_shape0_[tmp.size() - 1] > 1) {
    idx0 += tmp[tmp.size() - 1];
  }
  if (input_shape1_[tmp.size() - 1] > 1) {
    idx1 += tmp[tmp.size() - 1];
  }
  idx->push_back(idx0);
  idx->push_back(idx1);
}

template <typename T>
void ArithmeticCPUKernel::LaunchKernelLogic(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &outputs) {
  T *input1 = reinterpret_cast<T *>(inputs[0]->addr);
  T *input2 = reinterpret_cast<T *>(inputs[1]->addr);
  bool *output = reinterpret_cast<bool *>(outputs[0]->addr);
  size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(bool)) : 1;
  if (operate_type_ == LESS) {
    Less<T>(input1, input2, output, lens);
  } else if (operate_type_ == EQUAL) {
    Equal<T>(input1, input2, output, lens);
  } else if (operate_type_ == NOTEQUAL) {
    NotEqual<T>(input1, input2, output, lens);
  } else if (operate_type_ == GREATER) {
    Greater<T>(input1, input2, output, lens);
  } else if (operate_type_ == GREATEREQUAL) {
    GreaterEqual<T>(input1, input2, output, lens);
  } else if (operate_type_ == LESSEQUAL) {
    LessEqual<T>(input1, input2, output, lens);
  } else if (operate_type_ == LOGICALAND) {
    LogicalAnd<T>(input1, input2, output, lens);
  } else if (operate_type_ == LOGICALOR) {
    LogicalOr<T>(input1, input2, output, lens);
  } else {
    MS_LOG(EXCEPTION) << "Not support " << operate_type_;
  }
}

template <typename T>
void ArithmeticCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  if (target_dtype_ == kNumberTypeBool) {
    LaunchKernelLogic<T>(inputs, outputs);
    return;
  }
  T *input1 = reinterpret_cast<T *>(inputs[0]->addr);
  T *input2 = reinterpret_cast<T *>(inputs[1]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);

  size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  if (operate_type_ == ADD) {
    Add<T>(input1, input2, output, lens);
  } else if (operate_type_ == SUB) {
    Sub<T>(input1, input2, output, lens);
  } else if (operate_type_ == MUL) {
    Mul<T>(input1, input2, output, lens);
  } else if (operate_type_ == REALDIV) {
    RealDiv<T>(input1, input2, output, lens);
  } else if (operate_type_ == DIV) {
    Div<T>(input1, input2, output, lens);
  } else if (operate_type_ == FLOORDIV) {
    FloorDiv<T>(input1, input2, output, lens);
  } else if (operate_type_ == MOD) {
    Mod<T>(input1, input2, output, lens);
  } else if (operate_type_ == POW) {
    Pow<T>(input1, input2, output, lens);
  } else if (operate_type_ == ASSIGNADD) {
    AssignAdd<T>(input1, input2, output, lens);
  } else if (operate_type_ == ATAN2) {
    Atan2<T>(input1, input2, output, lens);
  } else if (operate_type_ == SQUAREDDIFFERENCE) {
    SquaredDifference<T>(input1, input2, output, lens);
  } else {
    MS_LOG(EXCEPTION) << "Not support " << operate_type_;
  }
}
}  // namespace kernel
}  // namespace mindspore
