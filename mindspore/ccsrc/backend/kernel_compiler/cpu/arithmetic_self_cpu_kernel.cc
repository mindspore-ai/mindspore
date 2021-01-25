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
#include "backend/kernel_compiler/cpu/arithmetic_self_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
template <typename T>
void Square(const T *in, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    out[i] = in[i] * in[i];
  }
}

template <typename T>
void Sign(const T *in, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    if (in[i] < 0) {
      out[i] = -1;
    } else if (in[i] > 0) {
      out[i] = 1;
    } else {
      out[i] = 0;
    }
  }
}

template <typename T>
void Neg(const T *in, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    out[i] = -in[i];
  }
}

template <typename T>
void LogicalNot(const T *in, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    out[i] = !in[i];
  }
}

template <typename T>
void OnesLike(const T *in, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    out[i] = static_cast<T>(1);
  }
}

template <typename T>
void ZerosLike(const T *in, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    out[i] = static_cast<T>(0);
  }
}

template <typename T>
void Floor(const T *in, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    out[i] = static_cast<T>(floor(in[i]));
  }
}

template <typename T>
void Reciprocal(const T *in, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    out[i] = static_cast<T>(1.0 / in[i]);
  }
}

template <typename T>
void Gelu(const T *in, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    T x = in[i];
    auto double_x = static_cast<T>(x);
    T tanh_res = (T)std::tanh(0.7978845608 * (double_x + 0.044715 * double_x * double_x * double_x));
    out[i] = x * ((T)1.0 + tanh_res) / (T)2.0;
  }
}
}  // namespace

void ArithmeticSelfCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name == prim::kPrimSquare->name()) {
    operate_type_ = SQUARE;
  } else if (kernel_name == prim::kPrimOnesLike->name()) {
    operate_type_ = ONESLIKE;
  } else if (kernel_name == prim::kPrimZerosLike->name()) {
    operate_type_ = ZEROSLIKE;
  } else if (kernel_name == prim::kPrimNeg->name()) {
    operate_type_ = NEG;
  } else if (kernel_name == prim::kPrimLogicalNot->name()) {
    operate_type_ = LOGICALNOT;
  } else if (kernel_name == prim::kPrimSign->name()) {
    operate_type_ = SIGN;
  } else if (kernel_name == prim::kPrimFloor->name()) {
    operate_type_ = FLOOR;
  } else if (kernel_name == prim::kPrimReciprocal->name()) {
    operate_type_ = RECIPROCAL;
  } else if (kernel_name == prim::kPrimGelu->name()) {
    operate_type_ = GELU;
  }
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  target_dtype_ = AnfAlgo::GetOutputInferDataType(kernel_node, 0);
}

bool ArithmeticSelfCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> & /*workspace*/,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat16 || dtype_ == kNumberTypeFloat64) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32 || dtype_ == kNumberTypeInt16 || dtype_ == kNumberTypeInt64) {
    LaunchKernel<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeBool) {
    LaunchKernelLogic<bool>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Data type is " << TypeIdLabel(dtype_) << "is not support.";
  }
  return true;
}

template <typename T>
void ArithmeticSelfCPUKernel::LaunchKernelLogic(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &outputs) {
  T *input = reinterpret_cast<T *>(inputs[0]->addr);
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
    if (operate_type_ == LOGICALNOT) {
      threads.emplace_back(std::thread(LogicalNot<T>, input, output, start, end));
    }
    start += once_compute_size;
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}

template <typename T>
void ArithmeticSelfCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &outputs) {
  if (target_dtype_ == kNumberTypeBool) {
    LaunchKernelLogic<T>(inputs, outputs);
    return;
  }
  T *input = reinterpret_cast<T *>(inputs[0]->addr);
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
    if (operate_type_ == SQUARE) {
      threads.emplace_back(std::thread(Square<T>, input, output, start, end));
    } else if (operate_type_ == NEG) {
      threads.emplace_back(std::thread(Neg<T>, input, output, start, end));
    } else if (operate_type_ == LOGICALNOT) {
      threads.emplace_back(std::thread(LogicalNot<T>, input, output, start, end));
    } else if (operate_type_ == ONESLIKE) {
      threads.emplace_back(std::thread(OnesLike<T>, input, output, start, end));
    } else if (operate_type_ == ZEROSLIKE) {
      threads.emplace_back(std::thread(ZerosLike<T>, input, output, start, end));
    } else if (operate_type_ == SIGN) {
      threads.emplace_back(std::thread(Sign<T>, input, output, start, end));
    } else if (operate_type_ == FLOOR) {
      threads.emplace_back(std::thread(Floor<T>, input, output, start, end));
    } else if (operate_type_ == RECIPROCAL) {
      threads.emplace_back(std::thread(Reciprocal<T>, input, output, start, end));
    } else if (operate_type_ == GELU) {
      threads.emplace_back(std::thread(Gelu<T>, input, output, start, end));
    }
    start += once_compute_size;
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}
}  // namespace kernel
}  // namespace mindspore
