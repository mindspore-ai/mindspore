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

#include "backend/kernel_compiler/cpu/apply_adagrad_cpu_kernel.h"

#include <thread>
#include <vector>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSizeFloat16 = 2;
constexpr size_t kSizeFloat32 = 4;
constexpr size_t kInputSize = 4;
constexpr size_t kOutputSize = 2;
}  // namespace
void ApplyAdagradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  update_slots_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "update_slots");
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
}

bool ApplyAdagradCPUKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> & /*workspace*/,
                                   const std::vector<AddressPtr> &outputs) {
  CheckParam(inputs, outputs);

  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  }

  return true;
}

void ApplyAdagradCPUKernel::CheckParam(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  // inputs: var, accum, lr, gradient
  if (inputs.size() != kInputSize) {
    MS_LOG(EXCEPTION) << "Input number is " << inputs.size() << ", but ApplyAdagrad needs 4 inputs.";
  }

  // outputs: var, accum
  if (outputs.size() != kOutputSize) {
    MS_LOG(EXCEPTION) << "Output number is " << outputs.size() << ", but ApplyAdagrad needs 2 outputs.";
  }

  if (inputs[0]->size != inputs[1]->size || inputs[0]->size != inputs[3]->size) {
    MS_LOG(EXCEPTION) << "Error input data size!";
  }

  if (inputs[2]->size != kSizeFloat16 && inputs[2]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "The attribute lr and grad must be float16 or float32!";
  }
}

template <typename T>
void ApplyAdagradCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  auto var = reinterpret_cast<T *>(inputs[0]->addr);
  auto accum = reinterpret_cast<T *>(inputs[1]->addr);
  auto lr = reinterpret_cast<T *>(inputs[2]->addr);
  auto gradient = reinterpret_cast<T *>(inputs[3]->addr);

  // multithreading
  size_t length = inputs[0]->size / sizeof(T);
  size_t max_thread_num = std::thread::hardware_concurrency();
  size_t use_thread_num = length < 128 * max_thread_num ? std::ceil(length / 128.0) : max_thread_num;
  std::vector<std::thread> threads;
  threads.reserve(use_thread_num);
  size_t start = 0;
  const size_t batch_size = (length + use_thread_num - 1) / use_thread_num;

  if (batch_size == 0) {
    MS_LOG(EXCEPTION) << "Error occur in launch kernel";
    return;
  }
  while (start < length) {
    size_t end = (start + batch_size) > length ? length : (start + batch_size);
    threads.emplace_back(
      std::thread(&ApplyAdagradCPUKernel::LaunchApplyAdagrad<T *>, this, var, accum, lr, gradient, start, end));
    start += batch_size;
  }

  for (auto &it : threads) {
    it.join();
  }

  // Copy result to output tensor
  auto output_var = reinterpret_cast<T *>(outputs[0]->addr);
  auto output_accum = reinterpret_cast<T *>(outputs[1]->addr);
  if (memcpy_s(output_var, outputs[0]->size, var, inputs[0]->size) != EOK) {
    MS_LOG(EXCEPTION) << "Launch kernel error: memcpy failed.";
  }

  if (memcpy_s(output_accum, outputs[1]->size, accum, inputs[1]->size) != EOK) {
    MS_LOG(EXCEPTION) << "Launch kernel error: memcpy failed.";
  }
}

template <typename T>
void ApplyAdagradCPUKernel::LaunchApplyAdagrad(T const var, T const accum, const T lr, const T gradient, size_t start,
                                               size_t end) {
  // DataType can only be float32 or float16, so eps will not be zero.
  using DataType = typename std::iterator_traits<T>::value_type;
  const DataType one = DataType(1);
  const DataType eps = DataType(1e-6);
  for (size_t i = start; i < end; ++i) {
    // update accum: accum += grad * grad
    if (update_slots_) {
      accum[i] += gradient[i] * gradient[i];
    }
    // update var: var -= lr * grad * \frac{1}{\sqrt{accum}}
    var[i] -= lr[0] * gradient[i] * (one / sqrt(accum[i] + eps));
  }
}
}  // namespace kernel
}  // namespace mindspore
