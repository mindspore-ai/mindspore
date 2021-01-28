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
#include "backend/kernel_compiler/cpu/binary_cross_entropy_cpu_kernel.h"

namespace mindspore {
namespace kernel {
template <typename T>
void BinaryCrossEntropyCpuKernel::LaunchToScalar(const int &input_size, const int &reduction, T *loss, T *tmp_loss) {
  if (input_size % 2 == 1) {
    tmp_loss[0] += tmp_loss[input_size - 1];
  }

  for (int stride = input_size / 2; stride > 0; stride >>= 1) {
    for (int i = 0; i < stride; i++) {
      tmp_loss[i] += tmp_loss[i + stride];
    }
    if (stride > 2 && stride % 2 == 1) {
      tmp_loss[0] += tmp_loss[stride - 1];
    }
  }

  loss[0] += tmp_loss[0];
  if (reduction == 1) {
    loss[0] /= static_cast<T>(input_size);
  }
}

template <typename T>
void BinaryCrossEntropyCpuKernel::Launchkernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &workspace,
                                               const std::vector<AddressPtr> &outputs) {
  T *input_x = reinterpret_cast<T *>(inputs[0]->addr);
  T *input_y = reinterpret_cast<T *>(inputs[1]->addr);
  T *weight = reinterpret_cast<T *>(inputs[2]->addr);
  T *loss = reinterpret_cast<T *>(outputs[0]->addr);
  std::vector<T> tmp_loss(input_size_);

  T epsilon = static_cast<T>(1e-12);
  T one = static_cast<T>(1);
  if (reduction_ == 0) {
    for (size_t i = 0; i < input_size_; i++) {
      T value =
        -weight[i] * (input_y[i] * log(input_x[i] + epsilon) + (one - input_y[i]) * log(one - input_x[i] + epsilon));
      loss[i] = value;
    }
  } else {
    for (size_t i = 0; i < input_size_; i++) {
      T value =
        -weight[i] * (input_y[i] * log(input_x[i] + epsilon) + (one - input_y[i]) * log(one - input_x[i] + epsilon));
      tmp_loss[i] = value;
    }
  }

  if (reduction_ != 0) {
    LaunchToScalar<T>(input_size_, reduction_, loss, tmp_loss.data());
  }
}

bool BinaryCrossEntropyCpuKernel::Launch(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  if (input_size_ > 0) {
    if (dtype_ == kNumberTypeFloat32) {
      Launchkernel<float>(inputs, workspace, outputs);
    } else if (dtype_ == kNumberTypeFloat16) {
      Launchkernel<float16>(inputs, workspace, outputs);
    }
  }
  return true;
}

void BinaryCrossEntropyCpuKernel::InitKernel(const CNodePtr &kernel_node) {
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
