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
#include "backend/kernel_compiler/cpu/arithmetic_cpu_kernel.h"
#include <thread>
#include <string>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
template <typename T>
void Add(const T *input1, const T *input2, T *out, size_t start, size_t end, bool is_number) {
  for (size_t i = start; i < end; i++) {
    out[i] = input1[i] + (is_number ? *input2 : input2[i]);
  }
}

template <typename T>
void Sub(const T *input1, const T *input2, T *out, size_t start, size_t end, bool is_number) {
  for (size_t i = start; i < end; i++) {
    out[i] = input1[i] - (is_number ? *input2 : input2[i]);
  }
}

template <typename T>
void Mul(const T *input1, const T *input2, T *out, size_t start, size_t end, bool is_number) {
  for (size_t i = start; i < end; i++) {
    out[i] = input1[i] * (is_number ? *input2 : input2[i]);
  }
}

template <typename T>
void Div(const T *input1, const T *input2, T *out, size_t start, size_t end, bool is_number) {
  for (size_t i = start; i < end; i++) {
    auto div_number = is_number ? *input2 : input2[i];
    if (div_number == 0) {
      MS_LOG(EXCEPTION) << "Cannot divided by 0!";
    }
    out[i] = input1[i] / div_number;
  }
}
}  // namespace

void ArithmeticCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name == prim::kPrimTensorAdd->name()) {
    operate_type_ = ADD;
  } else if (kernel_name == prim::kPrimSub->name()) {
    operate_type_ = SUB;
  } else if (kernel_name == prim::kPrimMul->name()) {
    operate_type_ = MUL;
  } else if (kernel_name == "Div") {
    operate_type_ = DIV;
  }

  auto shape0 = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto shape1 = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  if (shape1.size() == 0) {
    is_number_ = true;
  } else {
    is_number_ = false;
    if (shape0.size() != shape1.size()) {
      MS_LOG(EXCEPTION) << "Input0 and input1 must has the same shape";
    }
    for (size_t i = 0; i < shape0.size(); ++i) {
      if (shape0[i] != shape1[i]) {
        MS_LOG(EXCEPTION) << "Input0 and input1 must has the same shape";
      }
    }
  }
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  if (dtype_ != AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 1)) {
    MS_LOG(EXCEPTION) << "Input0 and input1 must has the same data type";
  }
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
  } else {
    MS_LOG(EXCEPTION) << "Only support int32, float32, but actual data type is " << TypeIdLabel(dtype_);
  }
  return true;
}

template <typename T>
void ArithmeticCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  T *input1 = reinterpret_cast<T *>(inputs[0]->addr);
  T *input2 = reinterpret_cast<T *>(inputs[1]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  auto lens = inputs[0]->size / sizeof(T);
  MS_LOG(INFO) << "lens=" << lens;

  const size_t thread_num = 24;
  std::vector<std::thread> threads;
  threads.reserve(thread_num);
  size_t start = 0;
  size_t once_compute_size = (lens + thread_num - 1) / thread_num;
  while (start < lens) {
    size_t end = (start + once_compute_size) > lens ? lens : (start + once_compute_size);
    if (operate_type_ == ADD) {
      threads.emplace_back(std::thread(Add<T>, input1, input2, output, start, end, is_number_));
    } else if (operate_type_ == SUB) {
      threads.emplace_back(std::thread(Sub<T>, input1, input2, output, start, end, is_number_));
    } else if (operate_type_ == MUL) {
      threads.emplace_back(std::thread(Mul<T>, input1, input2, output, start, end, is_number_));
    } else if (operate_type_ == DIV) {
      threads.emplace_back(std::thread(Div<T>, input1, input2, output, start, end, is_number_));
    }
    start += once_compute_size;
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}
}  // namespace kernel
}  // namespace mindspore
