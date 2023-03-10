/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_SINGLE_GRAPH_SCHEDULER_H_
#define MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_SINGLE_GRAPH_SCHEDULER_H_
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "src/extendrt/graph_compiler/compile_result.h"
#include "src/extendrt/execution_flow.h"
#include "src/litert/inner_context.h"

namespace mindspore {
namespace infer {
enum KernelType {
  kKernelTypeUnknown = 0,
  kKernelTypeLite = 1,
  kKernelTypeCloud = 2,
};

class SingleGraphScheduler {
 public:
  explicit SingleGraphScheduler(lite::InnerContext *context) : context_(context) {}
  virtual ~SingleGraphScheduler() = default;
  ExecutionFlowPtr Schedule(const CompileResultPtr &node_list);

 private:
  int ScheduleToKernels(const CompileResultPtr &node_list);
  abstract::Kernel *CreateKernel(const CompileNode *compile_node);
  bool HandleWeightForKernels();
  bool AppendKernelToPlan(kernel::KernelExec *kernel);
  bool OptimizeTranspose(const std::vector<kernel::KernelExec *> &kernels);
  bool InferShape(const CompileResultPtr &node_list);

 private:
  lite::InnerContext *context_;
  ExecutionFlowPtr execution_plan_{nullptr};
  kernel::KERNEL_ARCH graph_arch_;
  TypeId graph_data_type_ = kTypeUnknown;

  std::map<std::string, OpParameter *> op_parameters_;
};
using SingleGraphSchedulerPtr = std::shared_ptr<SingleGraphScheduler>;
}  // namespace infer
}  // namespace mindspore
#endif
