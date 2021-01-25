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
#include <thread>
#include "backend/kernel_compiler/cpu/arithmetic_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void ArithmeticCPUKernel::AssignAdd(T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    out[i] = input1[i] + input2[i];
    input1[i] = out[i];
  }
}

template <typename T>
void ArithmeticCPUKernel::Add(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    std::vector<size_t> idx;
    GenIndex(i, &idx);
    out[i] = input1[idx[0]] + input2[idx[1]];
  }
}

template <typename T>
void ArithmeticCPUKernel::Sub(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    std::vector<size_t> idx;
    GenIndex(i, &idx);
    out[i] = input1[idx[0]] - input2[idx[1]];
  }
}

template <typename T>
void ArithmeticCPUKernel::Mul(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    std::vector<size_t> idx;
    GenIndex(i, &idx);
    out[i] = input1[idx[0]] * input2[idx[1]];
  }
}

template <typename T>
void ArithmeticCPUKernel::RealDiv(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    std::vector<size_t> idx;
    GenIndex(i, &idx);
    auto dividend = input1[idx[0]];
    auto divisor = input2[idx[1]];
    if (divisor == 0) {
      if (dividend == 0) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        out[i] = dividend > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        out[i] = dividend > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    out[i] = dividend / divisor;
  }
}

template <typename T>
void ArithmeticCPUKernel::Div(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    std::vector<size_t> idx;
    GenIndex(i, &idx);
    auto dividend = input1[idx[0]];
    auto divisor = input2[idx[1]];
    if (divisor == 0) {
      if (dividend == 0) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        out[i] = dividend > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        out[i] = dividend > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    out[i] = dividend / divisor;
  }
}

template <typename T>
void ArithmeticCPUKernel::FloorDiv(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    std::vector<size_t> idx;
    GenIndex(i, &idx);
    auto dividend = input1[idx[0]];
    auto divisor = input2[idx[1]];
    if (divisor == 0) {
      if (dividend == 0) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        out[i] = dividend > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        out[i] = dividend > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    out[i] = floor(dividend / divisor);
  }
}

template <typename T>
void ArithmeticCPUKernel::Mod(const T *input1, const T *input2, T *out, size_t start, size_t end) {
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
}

template <typename T>
void ArithmeticCPUKernel::Pow(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    std::vector<size_t> idx;
    GenIndex(i, &idx);
    auto x = static_cast<double>(input1[idx[0]]);
    auto y = static_cast<double>(input2[idx[1]]);
    out[i] = static_cast<T>(std::pow(x, y));
  }
}

template <typename T>
void ArithmeticCPUKernel::Less(const T *input1, const T *input2, bool *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    std::vector<size_t> idx;
    GenIndex(i, &idx);
    out[i] = input1[idx[0]] < input2[idx[1]];
  }
}

template <typename T>
void ArithmeticCPUKernel::Equal(const T *input1, const T *input2, bool *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    std::vector<size_t> idx;
    GenIndex(i, &idx);
    out[i] = input1[idx[0]] == input2[idx[1]];
  }
}

template <typename T>
void ArithmeticCPUKernel::NotEqual(const T *input1, const T *input2, bool *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    std::vector<size_t> idx;
    GenIndex(i, &idx);
    out[i] = input1[idx[0]] != input2[idx[1]];
  }
}

template <typename T>
void ArithmeticCPUKernel::LogicalAnd(const T *input1, const T *input2, bool *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    std::vector<size_t> idx;
    GenIndex(i, &idx);
    out[i] = input1[idx[0]] && input2[idx[1]];
  }
}

template <typename T>
void ArithmeticCPUKernel::LogicalOr(const T *input1, const T *input2, bool *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    std::vector<size_t> idx;
    GenIndex(i, &idx);
    out[i] = input1[idx[0]] || input2[idx[1]];
  }
}

template <typename T>
void ArithmeticCPUKernel::SquaredDifference(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    std::vector<size_t> idx;
    GenIndex(i, &idx);
    T diff = input1[idx[0]] - input2[idx[1]];
    out[i] = diff * diff;
  }
}

template <typename T>
void ArithmeticCPUKernel::Greater(const T *input1, const T *input2, bool *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    std::vector<size_t> idx;
    GenIndex(i, &idx);
    out[i] = input1[idx[0]] > input2[idx[1]];
  }
}

template <typename T>
void ArithmeticCPUKernel::GreaterEqual(const T *input1, const T *input2, bool *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    std::vector<size_t> idx;
    GenIndex(i, &idx);
    out[i] = input1[idx[0]] >= input2[idx[1]];
  }
}

template <typename T>
void ArithmeticCPUKernel::LessEqual(const T *input1, const T *input2, bool *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    std::vector<size_t> idx;
    GenIndex(i, &idx);
    out[i] = input1[idx[0]] <= input2[idx[1]];
  }
}

void ArithmeticCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name == prim::kPrimTensorAdd->name()) {
    operate_type_ = ADD;
  } else if (kernel_name == prim::kPrimSub->name()) {
    operate_type_ = SUB;
  } else if (kernel_name == prim::kPrimMul->name()) {
    operate_type_ = MUL;
  } else if (kernel_name == prim::kPrimRealDiv->name()) {
    operate_type_ = REALDIV;
  } else if (kernel_name == prim::kPrimDiv->name()) {
    operate_type_ = DIV;
  } else if (kernel_name == prim::kPrimFloorDiv->name()) {
    operate_type_ = FLOORDIV;
  } else if (kernel_name == prim::kPrimMod->name()) {
    operate_type_ = MOD;
  } else if (kernel_name == prim::kPrimPow->name()) {
    operate_type_ = POW;
  } else if (kernel_name == prim::kPrimLess->name()) {
    operate_type_ = LESS;
  } else if (kernel_name == prim::kPrimEqual->name()) {
    operate_type_ = EQUAL;
  } else if (kernel_name == prim::kPrimNotEqual->name()) {
    operate_type_ = NOTEQUAL;
  } else if (kernel_name == prim::kPrimGreater->name()) {
    operate_type_ = GREATER;
  } else if (kernel_name == prim::kPrimGreaterEqual->name()) {
    operate_type_ = GREATEREQUAL;
  } else if (kernel_name == prim::kPrimLessEqual->name()) {
    operate_type_ = LESSEQUAL;
  } else if (kernel_name == prim::kPrimLogicalAnd->name()) {
    operate_type_ = LOGICALAND;
  } else if (kernel_name == prim::kPrimLogicalOr->name()) {
    operate_type_ = LOGICALOR;
  } else if (kernel_name == prim::kPrimAssignAdd->name()) {
    operate_type_ = ASSIGNADD;
  } else if (kernel_name == prim::kPrimSquaredDifference->name()) {
    operate_type_ = SQUAREDDIFFERENCE;
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
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  if (dtype_ != AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 1)) {
    MS_LOG(EXCEPTION) << "Input0 and input1 must has the same data type";
  }
  target_dtype_ = AnfAlgo::GetOutputInferDataType(kernel_node, 0);
}

bool ArithmeticCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                 const std::vector<kernel::AddressPtr> & /*workspace*/,
                                 const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeInt32 || dtype_ == kNumberTypeInt16 || dtype_ == kNumberTypeInt8) {
    LaunchKernel<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat16 || dtype_ == kNumberTypeFloat64) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeBool) {
    LaunchKernelLogic<bool>(inputs, outputs);
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
  auto max_thread_num = std::thread::hardware_concurrency();
  size_t thread_num = lens < 128 * max_thread_num ? std::ceil(lens / 128.0) : max_thread_num;
  MS_LOG(INFO) << "Lens=" << lens << "; use thread_num=" << thread_num << "; max_thread_num: " << max_thread_num;
  std::vector<std::thread> threads;
  if (thread_num < 1) {
    MS_LOG(ERROR) << "Invalid value: thread_num " << thread_num;
    return;
  }
  threads.reserve(thread_num);
  size_t start = 0;
  size_t once_compute_size = (lens + thread_num - 1) / thread_num;
  if (once_compute_size < 1) {
    MS_LOG(ERROR) << "Invalid value: once_compute_size " << once_compute_size;
    return;
  }
  while (start < lens) {
    size_t end = (start + once_compute_size) > lens ? lens : (start + once_compute_size);
    if (operate_type_ == LESS) {
      threads.emplace_back(std::thread(&ArithmeticCPUKernel::Less<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == EQUAL) {
      threads.emplace_back(std::thread(&ArithmeticCPUKernel::Equal<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == NOTEQUAL) {
      threads.emplace_back(std::thread(&ArithmeticCPUKernel::NotEqual<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == GREATER) {
      threads.emplace_back(std::thread(&ArithmeticCPUKernel::Greater<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == GREATEREQUAL) {
      threads.emplace_back(
        std::thread(&ArithmeticCPUKernel::GreaterEqual<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == LESSEQUAL) {
      threads.emplace_back(std::thread(&ArithmeticCPUKernel::LessEqual<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == LOGICALAND) {
      threads.emplace_back(std::thread(&ArithmeticCPUKernel::LogicalAnd<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == LOGICALOR) {
      threads.emplace_back(std::thread(&ArithmeticCPUKernel::LogicalOr<T>, this, input1, input2, output, start, end));
    } else {
      MS_LOG(EXCEPTION) << "Not support " << operate_type_;
    }
    start += once_compute_size;
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
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
  auto max_thread_num = std::thread::hardware_concurrency();
  size_t thread_num = lens < 128 * max_thread_num ? std::ceil(lens / 128.0) : max_thread_num;
  MS_LOG(INFO) << "Lens=" << lens << "; use thread_num=" << thread_num << "; max_thread_num: " << max_thread_num;
  std::vector<std::thread> threads;
  if (thread_num < 1) {
    MS_LOG(ERROR) << "Invalid value: thread_num " << thread_num;
    return;
  }
  threads.reserve(thread_num);
  size_t start = 0;
  size_t once_compute_size = (lens + thread_num - 1) / thread_num;
  if (once_compute_size < 1) {
    MS_LOG(ERROR) << "Invalid value: once_compute_size " << once_compute_size;
    return;
  }
  while (start < lens) {
    size_t end = (start + once_compute_size) > lens ? lens : (start + once_compute_size);
    if (operate_type_ == ADD) {
      threads.emplace_back(std::thread(&ArithmeticCPUKernel::Add<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == SUB) {
      threads.emplace_back(std::thread(&ArithmeticCPUKernel::Sub<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == MUL) {
      threads.emplace_back(std::thread(&ArithmeticCPUKernel::Mul<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == REALDIV) {
      threads.emplace_back(std::thread(&ArithmeticCPUKernel::RealDiv<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == DIV) {
      threads.emplace_back(std::thread(&ArithmeticCPUKernel::Div<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == FLOORDIV) {
      threads.emplace_back(std::thread(&ArithmeticCPUKernel::FloorDiv<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == MOD) {
      threads.emplace_back(std::thread(&ArithmeticCPUKernel::Mod<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == POW) {
      threads.emplace_back(std::thread(&ArithmeticCPUKernel::Pow<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == ASSIGNADD) {
      threads.emplace_back(std::thread(&ArithmeticCPUKernel::AssignAdd<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == SQUAREDDIFFERENCE) {
      threads.emplace_back(
        std::thread(&ArithmeticCPUKernel::SquaredDifference<T>, this, input1, input2, output, start, end));
    } else {
      MS_LOG(EXCEPTION) << "Not support " << operate_type_;
    }
    start += once_compute_size;
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}
}  // namespace kernel
}  // namespace mindspore
