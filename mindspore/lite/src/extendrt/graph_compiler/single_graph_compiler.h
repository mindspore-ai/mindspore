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

#ifndef MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_SINGLE_GRAPH_COMPILER_H_
#define MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_SINGLE_GRAPH_COMPILER_H_
#include <string>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include "src/litert/inner_context.h"
#include "mindapi/ir/abstract.h"
#include "ir/anf.h"
#include "src/extendrt/graph_compiler/compile_result.h"
#include "src/extendrt/tensor.h"
#include "src/infer/execution_flow.h"
#include "src/infer/graph_compiler.h"
#include "src/extendrt/graph_compiler/single_graph_scheduler.h"

namespace mindspore {
namespace infer {
class SingleGraphCompiler : public abstract::GraphCompiler {
 public:
  explicit SingleGraphCompiler(std::shared_ptr<lite::InnerContext> context) : context_(std::move(context)) {}
  ~SingleGraphCompiler() override = default;
  abstract::ExecutionPlanPtr Compile(FuncGraphPtr graph) override { return nullptr; }
  abstract::ExecutionFlowPtr Compile(const GraphSegmentPtr &segment, const AnfNodePtrList &inputs,
                                     const AnfNodePtrList &outputs) override;

 private:
  CompileResultPtr Build(const GraphSegmentPtr &segment, const AnfNodePtrList &inputs, const AnfNodePtrList &outputs);
  abstract::ExecutionFlowPtr Schedule(const CompileResultPtr &node_list);

  SingleGraphSchedulerPtr scheduler_{nullptr};
  std::shared_ptr<lite::InnerContext> context_{nullptr};
};
}  // namespace infer
}  // namespace mindspore

#endif
