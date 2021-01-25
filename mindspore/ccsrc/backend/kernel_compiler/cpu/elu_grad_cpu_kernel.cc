/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/cpu/elu_grad_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void EluGradCPUKernel::EluGrad(const T *input0, const T *input1, T *out, size_t start, size_t end) {
  const T alpha = static_cast<T>(1);
  for (size_t i = start; i < end; i++) {
    out[i] = (input1[i] < static_cast<T>(0)) ? input0[i] * (input1[i] + alpha) : input0[i];
  }
}

void EluGradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  if (dtype_ != AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 1)) {
    MS_LOG(EXCEPTION) << "Input0 and input1 must has the same data type";
  }
}

bool EluGradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                              const std::vector<kernel::AddressPtr> & /*workspace*/,
                              const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Data type is " << TypeIdLabel(dtype_) << "is not support.";
  }
  return true;
}

template <typename T>
void EluGradCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  T *input0 = reinterpret_cast<T *>(inputs[0]->addr);
  T *input1 = reinterpret_cast<T *>(inputs[1]->addr);
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
    threads.emplace_back(std::thread(&EluGradCPUKernel::EluGrad<T>, this, input0, input1, output, start, end));
    start += once_compute_size;
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}
}  // namespace kernel
}  // namespace mindspore
