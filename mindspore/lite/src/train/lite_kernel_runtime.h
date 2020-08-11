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

#ifndef MINDSPORE_LITE_SRC_TRAIN_LITE_KERNEL_RUNTIME_H_
#define MINDSPORE_LITE_SRC_TRAIN_LITE_KERNEL_RUNTIME_H_

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "src/runtime/allocator.h"
#include "src/executor.h"
// #include "runtime/device/kernel_runtime.h"
#include "runtime/device/device_address.h"
#include "src/lite_kernel.h"
#include "backend/session/kernel_graph.h"
namespace mindspore::lite {
class LiteInferKernelRuntime {
 public:
  LiteInferKernelRuntime() = default;
  ~LiteInferKernelRuntime() = default;

  bool Run(session::KernelGraph *graph, const std::vector<tensor::Tensor *> &inputs,
           std::vector<tensor::Tensor *> *outputs);

  void AssignKernelAddress(session::KernelGraph *graph) {}

 protected:
  void BindInputOutput(const session::KernelGraph *graph, const std::vector<tensor::Tensor *> &inputs,
                       std::vector<tensor::Tensor *> *outputs);

  std::vector<CNodePtr> GetGraphInputs(const std::vector<CNodePtr> &execution_order);
};

}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_TRAIN_LITE_KERNEL_RUNTIME_H_
