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
constexpr size_t kSGDInputsNum = 6;
constexpr size_t kSGDOutputsNum = 1;
}  // namespace
template <typename T>
void SGDCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  dampening_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "dampening");
  weight_decay_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "weight_decay");
  nesterov_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "nesterov");
}

template <typename T>
bool SGDCPUKernel<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                             const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSGDInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSGDOutputsNum, kernel_name_);
  auto param = reinterpret_cast<T *>(inputs[PARAM]->addr);
  auto grad = reinterpret_cast<T *>(inputs[GRAD]->addr);
  auto lr = reinterpret_cast<T *>(inputs[LR]->addr);
  auto accum = reinterpret_cast<T *>(inputs[ACCUM]->addr);
  auto momentum = reinterpret_cast<T *>(inputs[MOMENTUM]->addr);
  auto stat = reinterpret_cast<T *>(inputs[STAT]->addr);
  auto output_param = reinterpret_cast<T *>(outputs[0]->addr);
  size_t elem_num = inputs[0]->size / sizeof(T);

  auto task = [this, &param, &grad, &lr, &accum, &momentum, &stat, &output_param](size_t start, size_t end) {
    T ZERO = static_cast<T>(0);
    T ONE = static_cast<T>(1);
    for (size_t i = start; i < end; i++) {
      T grad_new = grad[i];
      if (weight_decay_ > static_cast<float>(0.0)) {
        grad_new += param[i] * static_cast<T>(weight_decay_);
      }
      if (momentum[0] > ZERO) {
        if (stat[i] > ZERO) {
          accum[i] = grad_new;
          stat[i] = ZERO;
        } else {
          accum[i] = accum[i] * momentum[0] + (ONE - static_cast<T>(dampening_)) * grad_new;
        }
        if (nesterov_) {
          grad_new += accum[i] * momentum[0];
        } else {
          grad_new = accum[i];
        }
      }
      param[i] -= lr[0] * grad_new;
      output_param[i] = param[i];
    }
  };
  CPUKernelUtils::ParallelFor(task, elem_num);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
