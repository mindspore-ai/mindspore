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

#include "backend/kernel_compiler/cpu/sgd_cpu_kernel.h"

#include <thread>
#include <vector>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputSize = 6;
constexpr size_t kOutputSize = 1;
}  // namespace
template <typename T>
void SGDCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  dampening_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "dampening");
  weight_decay_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "weight_decay");
  nesterov_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "nesterov");
}

template <typename T>
void SGDCPUKernel<T>::CheckParam(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  // inputs: params, grad, lr, accum, momentum, stat
  if (inputs.size() != kInputSize) {
    MS_LOG(EXCEPTION) << "Input number is " << inputs.size() << ", but SGD needs 6 inputs.";
  }

  // output: param
  if (outputs.size() != kOutputSize) {
    MS_LOG(EXCEPTION) << "Output number is " << outputs.size() << ", but SGD needs 1 outputs.";
  }
}

template <typename T>
bool SGDCPUKernel<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> & /*workspace*/,
                             const std::vector<AddressPtr> &outputs) {
  CheckParam(inputs, outputs);

  auto param = reinterpret_cast<T *>(inputs[0]->addr);
  auto grad = reinterpret_cast<T *>(inputs[1]->addr);
  auto lr = reinterpret_cast<T *>(inputs[2]->addr);
  auto accum = reinterpret_cast<T *>(inputs[3]->addr);
  auto momentum = reinterpret_cast<T *>(inputs[4]->addr);
  auto stat = reinterpret_cast<T *>(inputs[5]->addr);
  size_t elem_num = inputs[0]->size / sizeof(float);

  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      T grad_new = grad[i];
      if (weight_decay_ > 0) {
        grad_new += param[i] * static_cast<T>(weight_decay_);
      }
      if (momentum[0] > static_cast<T>(0)) {
        if (stat[i] > static_cast<T>(0)) {
          accum[i] = grad_new;
          stat[i] = static_cast<T>(0);
        } else {
          accum[i] = accum[i] * momentum[0] + static_cast<T>(1.0 - dampening_) * grad_new;
        }
        if (nesterov_) {
          grad_new += accum[i] * momentum[0];
        } else {
          grad_new = accum[i];
        }
      }
      param[i] -= lr[0] * grad_new;
    }
  };
  CPUKernelUtils::ParallelFor(task, elem_num);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
