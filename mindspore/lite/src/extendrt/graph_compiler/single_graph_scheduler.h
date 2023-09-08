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
#include "src/extendrt/graph_compiler/compile_option.h"
#include "src/infer/kernel.h"
#include "src/extendrt/kernel/kernel_selector/kernel_selector.h"
#include "src/infer/context.h"
#include "src/extendrt/delegate/type.h"
#include "src/executor/sub_graph_kernel.h"

namespace mindspore {
namespace lite {
class SingleGraphScheduler {
 public:
  explicit SingleGraphScheduler(InferContextPtr context, const std::shared_ptr<Context> &ctx,
                                std::shared_ptr<CompileOption> option)
      : context_(std::move(context)), ctx_(ctx), compile_option_(std::move(option)) {}
  virtual ~SingleGraphScheduler() = default;
  InferKernel *Schedule(const CompileResultPtr &node_list);

 private:
  int SelectKernel(const CompileResultPtr &node_list);
  bool HandleWeightForKernels();
  Status OptimizeTranspose(kernel::SubGraphKernel *kernels);

 private:
  InferContextPtr context_{nullptr};
  std::shared_ptr<Context> ctx_{nullptr};
  std::shared_ptr<CompileOption> compile_option_{nullptr};
  infer::ExecutionFlowPtr execution_flow_{nullptr};
  std::shared_ptr<kernel::KernelSelector> kernel_selector_{nullptr};
  void CreateDelegateKernel(const std::shared_ptr<CompileNode> &node, mindspore::ExtendDelegate *delegate,
                            std::vector<InferKernel *> *kernels);

  std::map<std::string, OpParameter *> op_parameters_;
};
using SingleGraphSchedulerPtr = std::shared_ptr<SingleGraphScheduler>;
}  // namespace lite
}  // namespace mindspore
#endif
