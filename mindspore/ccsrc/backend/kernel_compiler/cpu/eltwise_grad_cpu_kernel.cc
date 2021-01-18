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
#include "backend/kernel_compiler/cpu/eltwise_grad_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void EltWiseGradCPUKernel::ReluGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    if (input2[i] > 0) {
      out[i] = input1[i];
    } else {
      out[i] = 0;
    }
  }
}

template <typename T>
void EltWiseGradCPUKernel::ReLU6Grad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    if (input2[i] > 0 && input2[i] <= 6) {
      out[i] = input1[i];
    } else {
      out[i] = 0;
    }
  }
}

template <typename T>
void EltWiseGradCPUKernel::AbsGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    if (input1[i] > 0) {
      out[i] = input2[i];
    } else if (input1[i] < 0) {
      out[i] = -input2[i];
    } else {
      out[i] = 0;
    }
  }
}

template <typename T>
void EltWiseGradCPUKernel::SigmoidGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    out[i] = input2[i] * input1[i] * (1 - input1[i]);
  }
}

template <typename T>
void EltWiseGradCPUKernel::SqrtGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    out[i] = input2[i] / (input1[i] * 2);
  }
}

template <typename T>
void EltWiseGradCPUKernel::TanhGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    T tmp = input1[i] * input1[i];
    out[i] = input2[i] * (1 - tmp);
  }
}

template <typename T>
void EltWiseGradCPUKernel::GeluGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    T x = input2[i];
    auto double_x = static_cast<T>(x);
    T tanh_res = (T)std::tanh(0.7978845608 * (double_x + 0.044715 * double_x * double_x * double_x));
    T mul_right = (T)(0.7978845608 + 0.1070322244 * double_x * double_x);
    T y_res = (((T)1.0 + tanh_res) + x * ((T)1.0 - tanh_res * tanh_res) * mul_right) / (T)2.0;
    out[i] = input1[i] * y_res;
  }
}

void EltWiseGradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name == "ReluGrad") {
    operate_type_ = RELUGRAD;
  } else if (kernel_name == "ReLU6Grad") {
    operate_type_ = RELU6GRAD;
  } else if (kernel_name == "SigmoidGrad") {
    operate_type_ = SIGMOIDGRAD;
  } else if (kernel_name == "AbsGrad") {
    operate_type_ = ABSGRAD;
  } else if (kernel_name == "TanhGrad") {
    operate_type_ = TANHGRAD;
  } else if (kernel_name == "SqrtGrad") {
    operate_type_ = SQRTGRAD;
  } else if (kernel_name == "GeluGrad") {
    operate_type_ = GELUGRAD;
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
}

bool EltWiseGradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> & /*workspace*/,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeInt32 || dtype_ == kNumberTypeInt16) {
    LaunchKernel<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat16 || dtype_ == kNumberTypeFloat64) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Data type is " << TypeIdLabel(dtype_) << "is not support.";
  }
  return true;
}

template <typename T>
void EltWiseGradCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
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
    if (operate_type_ == RELUGRAD) {
      threads.emplace_back(std::thread(&EltWiseGradCPUKernel::ReluGrad<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == RELU6GRAD) {
      threads.emplace_back(std::thread(&EltWiseGradCPUKernel::ReLU6Grad<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == ABSGRAD) {
      threads.emplace_back(std::thread(&EltWiseGradCPUKernel::AbsGrad<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == SIGMOIDGRAD) {
      threads.emplace_back(
        std::thread(&EltWiseGradCPUKernel::SigmoidGrad<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == TANHGRAD) {
      threads.emplace_back(std::thread(&EltWiseGradCPUKernel::TanhGrad<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == SQRTGRAD) {
      threads.emplace_back(std::thread(&EltWiseGradCPUKernel::SqrtGrad<T>, this, input1, input2, output, start, end));
    } else if (operate_type_ == GELUGRAD) {
      threads.emplace_back(std::thread(&EltWiseGradCPUKernel::GeluGrad<T>, this, input1, input2, output, start, end));
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
