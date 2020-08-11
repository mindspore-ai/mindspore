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

#ifndef MINDSPORE_LITE_SRC_SCHEDULER_H_
#define MINDSPORE_LITE_SRC_SCHEDULER_H_

#include <vector>
#include "src/lite_kernel.h"
#include "include/context.h"
#include "include/model.h"

namespace mindspore::lite {
class Scheduler {
 public:
  explicit Scheduler(const Context *ctx) { context_ = const_cast<Context *>(ctx); }
  int Schedule(const lite::Model *model, std::vector<tensor::Tensor *> *tensors,
               std::vector<kernel::LiteKernel *> *kernels);

 protected:
  kernel::LiteKernel *ScheduleNode(const std::vector<tensor::Tensor *> &in_tensors,
                                   const std::vector<tensor::Tensor *> &out_tensors, const lite::Primitive *primitive);

 private:
  int InitOp2Kernel(const lite::Model *model, std::vector<tensor::Tensor *> *tensors,
                    std::vector<kernel::LiteKernel *> *kernels);

  // construct SubGraphKernel for each kernel-group in markedKernelGroup
  void ConstructSubgraphs(std::vector<kernel::LiteKernel *> *kernels);

  kernel::LiteKernel *CreateSubKernel(const std::vector<kernel::LiteKernel *> &kernels, kernel::KERNEL_ARCH arch);

 protected:
  Context *context_ = nullptr;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_SCHEDULER_H_
