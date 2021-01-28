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
#include "backend/kernel_compiler/cpu/binary_cross_entropy_grad_kernel.h"

namespace mindspore {
namespace kernel {
template <typename T>
void BinaryCrossEntropyGradCpuKernel::Launchkernel(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &outputs) {
  T *input_x = reinterpret_cast<T *>(inputs[0]->addr);
  T *input_y = reinterpret_cast<T *>(inputs[1]->addr);
  T *dloss = reinterpret_cast<T *>(inputs[2]->addr);
  T *weight = reinterpret_cast<T *>(inputs[3]->addr);
  T *dx = reinterpret_cast<T *>(outputs[0]->addr);

  T epsilon = static_cast<T>(1e-12);
  T one = static_cast<T>(1);
  if (reduction_ == 0) {
    for (size_t i = 0; i < input_size_; i++) {
      T denominator = ((input_x[i] * (one - input_x[i])) > epsilon) ? (input_x[i] * (one - input_x[i])) : epsilon;
      T value = weight[i] * (input_x[i] - input_y[i]) / denominator;
      dx[i] = value * dloss[i];
    }
  } else {
    T dloss1 = dloss[0];
    if (reduction_ == 1) {
      dloss1 = dloss[0] / static_cast<T>(input_size_);
    }
    for (size_t i = 0; i < input_size_; i++) {
      T denominator = ((input_x[i] * (one - input_x[i])) > epsilon) ? (input_x[i] * (one - input_x[i])) : epsilon;
      T value = weight[i] * (input_x[i] - input_y[i]) / denominator;
      dx[i] = value * dloss1;
    }
  }
}

bool BinaryCrossEntropyGradCpuKernel::Launch(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs) {
  if (input_size_ > 0) {
    if (dtype_ == kNumberTypeFloat32) {
      Launchkernel<float>(inputs, outputs);
    } else if (dtype_ == kNumberTypeFloat16) {
      Launchkernel<float16>(inputs, outputs);
    }
  }
  return true;
}

void BinaryCrossEntropyGradCpuKernel::InitKernel(const CNodePtr &kernel_node) {
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  for (size_t i = 0; i < input_shape.size(); i++) {
    input_size_ *= input_shape[i];
  }
  string reduction = AnfAlgo::GetNodeAttr<string>(kernel_node, "reduction");
  if (reduction == "none") {
    reduction_ = 0;
  } else if (reduction == "sum") {
    reduction_ = 2;
  }

  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}
}  // namespace kernel
}  // namespace mindspore
