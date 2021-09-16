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

#include "backend/kernel_compiler/cpu/scatter_arithmetic_cpu_kernel.h"
#include <map>
#include <limits>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 3;
constexpr size_t kOutputNum = 1;
}  // namespace
template <typename T>
void ScatterArithmeticCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  CheckParam(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  input_size_ = 1;
  inner_size_ = 1;
  if (input_shape.empty()) {
    MS_LOG(EXCEPTION) << "Input shape is empty";
  }

  for (size_t i = 1; i < input_shape.size(); i++) {
    inner_size_ *= input_shape[i];
  }
  input_size_ = input_shape[0] * inner_size_;
  auto indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  indices_size_ = 1;
  for (size_t i = 0; i < indices_shape.size(); i++) {
    indices_size_ *= indices_shape[i];
  }
}

template <typename T>
void ScatterArithmeticCPUKernel<T>::CheckParam(const CNodePtr &kernel_node) const {
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != kInputNum) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but ScatterAdd needs 3 inputs.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != kOutputNum) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but ScatterAdd has 1 output.";
  }
}

template <typename T>
bool ScatterArithmeticCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &workspace,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  static const std::map<std::string, std::function<void(ScatterArithmeticCPUKernel *, T *, const int *, const T *)>>
    kScatterArithmeticBinOpFuncMap{{"ScatterAdd", &ScatterArithmeticCPUKernel<T>::ScatterAdd},
                                   {"ScatterSub", &ScatterArithmeticCPUKernel<T>::ScatterSub},
                                   {"ScatterMul", &ScatterArithmeticCPUKernel<T>::ScatterMul},
                                   {"ScatterDiv", &ScatterArithmeticCPUKernel<T>::ScatterDiv},
                                   {"ScatterMax", &ScatterArithmeticCPUKernel<T>::ScatterMax},
                                   {"ScatterMin", &ScatterArithmeticCPUKernel<T>::ScatterMin},
                                   {"ScatterUpdate", &ScatterArithmeticCPUKernel<T>::ScatterUpdate}};
  if (kScatterArithmeticBinOpFuncMap.find(kernel_name_) != kScatterArithmeticBinOpFuncMap.end()) {
    T *input = reinterpret_cast<T *>(inputs[INPUT]->addr);
    int *indices = reinterpret_cast<int *>(inputs[INDICES]->addr);
    T *updates = reinterpret_cast<T *>(inputs[UPDATES]->addr);
    T *output = reinterpret_cast<T *>(outputs[0]->addr);
    kScatterArithmeticBinOpFuncMap.at(kernel_name_)(this, input, indices, updates);
    auto bufferSize = outputs[0]->size;
    auto ret = memcpy_s(output, bufferSize, input, input_size_ * sizeof(T));
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "Memory copy failed!";
    }
  } else {
    MS_LOG(EXCEPTION) << "Not support operator:" << kernel_name_;
  }
  return true;
}

template <typename T>
void ScatterArithmeticCPUKernel<T>::ScatterAdd(T *input, const int *indices, const T *updates) {
  for (size_t i = 0; i < indices_size_; i++) {
    auto base_index_updates = i * inner_size_;
    auto base_index_input = indices[i] * inner_size_;
    for (size_t j = 0; j < inner_size_; j++) {
      input[base_index_input + j] += updates[base_index_updates + j];
    }
  }
}

template <typename T>
void ScatterArithmeticCPUKernel<T>::ScatterSub(T *input, const int *indices, const T *updates) {
  for (size_t i = 0; i < indices_size_; i++) {
    auto base_index_updates = i * inner_size_;
    auto base_index_input = indices[i] * inner_size_;
    for (size_t j = 0; j < inner_size_; j++) {
      input[base_index_input + j] -= updates[base_index_updates + j];
    }
  }
}

template <typename T>
void ScatterArithmeticCPUKernel<T>::ScatterMul(T *input, const int *indices, const T *updates) {
  for (size_t i = 0; i < indices_size_; i++) {
    auto base_index_updates = i * inner_size_;
    auto base_index_input = indices[i] * inner_size_;
    for (size_t j = 0; j < inner_size_; j++) {
      input[base_index_input + j] *= updates[base_index_updates + j];
    }
  }
}

template <typename T>
void ScatterArithmeticCPUKernel<T>::ScatterDiv(T *input, const int *indices, const T *updates) {
  for (size_t i = 0; i < indices_size_; i++) {
    for (size_t j = 0; j < inner_size_; j++) {
      auto dividend = input[indices[i] * inner_size_ + j];
      auto divisor = updates[i * inner_size_ + j];
      if (divisor == 0) {
        if (dividend == 0) {
          input[indices[i] * inner_size_ + j] = std::numeric_limits<T>::quiet_NaN();
          continue;
        }
        if (std::numeric_limits<T>::has_infinity) {
          input[indices[i] * inner_size_ + j] =
            dividend > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
        } else {
          input[indices[i] * inner_size_ + j] =
            dividend > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
        }
        continue;
      }
      input[indices[i] * inner_size_ + j] = dividend / divisor;
    }
  }
}

template <typename T>
void ScatterArithmeticCPUKernel<T>::ScatterMax(T *input, const int *indices, const T *updates) {
  for (size_t i = 0; i < indices_size_; i++) {
    auto base_index_updates = i * inner_size_;
    auto base_index_input = indices[i] * inner_size_;
    for (size_t j = 0; j < inner_size_; j++) {
      input[base_index_input + j] = input[base_index_input + j] > updates[base_index_updates + j]
                                      ? input[base_index_input + j]
                                      : updates[base_index_updates + j];
    }
  }
}

template <typename T>
void ScatterArithmeticCPUKernel<T>::ScatterMin(T *input, const int *indices, const T *updates) {
  for (size_t i = 0; i < indices_size_; i++) {
    auto base_index_updates = i * inner_size_;
    auto base_index_input = indices[i] * inner_size_;
    for (size_t j = 0; j < inner_size_; j++) {
      input[base_index_input + j] = input[base_index_input + j] < updates[base_index_updates + j]
                                      ? input[base_index_input + j]
                                      : updates[base_index_updates + j];
    }
  }
}

template <typename T>
void ScatterArithmeticCPUKernel<T>::ScatterUpdate(T *input, const int *indices, const T *updates) {
  for (size_t i = 0; i < indices_size_; i++) {
    auto base_index_updates = i * inner_size_;
    auto base_index_input = indices[i] * inner_size_;
    for (size_t j = 0; j < inner_size_; j++) {
      input[base_index_input + j] = updates[base_index_updates + j];
    }
  }
}
}  // namespace kernel
}  // namespace mindspore
