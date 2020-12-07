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

#include "backend/kernel_compiler/cpu/rmsprop_cpu_kernel.h"

namespace mindspore {
namespace kernel {
void RMSPropCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  auto node_name = AnfAlgo::GetCNodeName(kernel_node);
  if (node_name == "ApplyCenteredRMSProp") {
    use_center_ = true;
  }

  if (node_name == "ApplyRMSProp") {
    decay_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "rho");
    momentum_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "momentum");
    epsilon_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "epsilon");
  }
  auto input_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  for (auto &dim : input_shape) {
    size_ *= dim;
  }
}

bool RMSPropCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                              const std::vector<kernel::AddressPtr> & /*workspace*/,
                              const std::vector<kernel::AddressPtr> &outputs) {
  if (!use_center_) {
    float *variable = reinterpret_cast<float *>(inputs[0]->addr);
    float *mean_square = reinterpret_cast<float *>(inputs[1]->addr);
    float *moment = reinterpret_cast<float *>(inputs[2]->addr);
    float *learning_rate = reinterpret_cast<float *>(inputs[3]->addr);
    float *gradients = reinterpret_cast<float *>(inputs[4]->addr);

    for (size_t i = 0; i < size_; i++) {
      mean_square[i] += (gradients[i] * gradients[i] - mean_square[i]) * (1.0 - decay_);
      moment[i] = moment[i] * momentum_ + (gradients[i] * learning_rate[0]) / sqrt(mean_square[i] + epsilon_);
      variable[i] -= moment[i];
    }
  } else {
    float *variable = reinterpret_cast<float *>(inputs[0]->addr);
    float *mean_gradients = reinterpret_cast<float *>(inputs[1]->addr);
    float *mean_square = reinterpret_cast<float *>(inputs[2]->addr);
    float *moment = reinterpret_cast<float *>(inputs[3]->addr);
    float *gradients = reinterpret_cast<float *>(inputs[4]->addr);
    float *learning_rate = reinterpret_cast<float *>(inputs[5]->addr);
    float *decay = reinterpret_cast<float *>(inputs[6]->addr);
    float *momentum = reinterpret_cast<float *>(inputs[7]->addr);
    float *epsilon = reinterpret_cast<float *>(inputs[8]->addr);

    for (size_t i = 0; i < size_; i++) {
      mean_square[i] += (gradients[i] * gradients[i] - mean_square[i]) * (1.0 - decay[0]);
      mean_gradients[i] += (gradients[i] - mean_gradients[i]) * (1.0 - decay[0]);
      auto denom = (mean_square[i] - mean_gradients[i] * mean_gradients[i]) + epsilon[0];
      if (denom > 0) {
        moment[i] = moment[i] * momentum[0] + (gradients[i] * learning_rate[0]) / sqrt(denom);
        variable[i] -= moment[i];
      }
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
