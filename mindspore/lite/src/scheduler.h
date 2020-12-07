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
#include <map>
#include "src/sub_graph_kernel.h"
#include "src/inner_context.h"
#include "include/model.h"
#include "src/ops/primitive_c.h"

namespace mindspore::lite {
class Scheduler {
 public:
  explicit Scheduler(const InnerContext *ctx) { context_ = const_cast<InnerContext *>(ctx); }
  ~Scheduler() = default;

  int Schedule(const lite::Model *model, std::vector<Tensor *> *tensors, std::vector<kernel::LiteKernel *> *kernels);

  static int ReSizeKernels(const std::vector<kernel::LiteKernel *> &kernels);

 protected:
  kernel::LiteKernel *ScheduleNode(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                                   const mindspore::lite::PrimitiveC *primitive, const Model::Node *cnode);

  int BuildKernels(const lite::Model *model, const std::vector<Tensor *> *tensors,
                   std::vector<kernel::LiteKernel *> *kernels);

  static int InferShape(const lite::Model *model, std::vector<Tensor *> *tensors);

  int ConstructSubGraphs(std::vector<kernel::LiteKernel *> *kernels);

  kernel::SubGraphKernel *CreateSubGraphKernel(const std::vector<kernel::LiteKernel *> &kernels,
                                               kernel::SubGraphType type, int index);

  std::vector<kernel::LiteKernel *> FindAllSubGraphKernels(
    kernel::LiteKernel *head_kernel, std::map<const kernel::LiteKernel *, bool> *sinked_kernel_map);

  static TypeId GetFirstFp32Fp16OrInt8Type(const std::vector<Tensor *> &in_tensors);

  static void SetKernelTensorDataType(kernel::LiteKernel *kernel);

  static kernel::SubGraphType GetKernelSubGraphType(const kernel::LiteKernel *kernel);

 protected:
  InnerContext *context_ = nullptr;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_SCHEDULER_H_
