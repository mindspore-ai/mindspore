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
#ifndef MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_DEFAULT_GRAPH_COMPILER_H_
#define MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_DEFAULT_GRAPH_COMPILER_H_

#include <memory>
#include <vector>
#include <tuple>
#include "infer/graph_compiler.h"
#include "infer/context.h"
#include "src/extendrt/graph_compiler/compile_result.h"
#include "src/extendrt/graph_compiler/single_graph_scheduler.h"
#include "src/extendrt/graph_compiler/compile_option.h"

namespace mindspore::lite {
class DefaultGraphCompiler : public infer::abstract::GraphCompiler {
 public:
  explicit DefaultGraphCompiler(const std::shared_ptr<Context> &context) : context_(context) {
    inner_context_ = nullptr;
  }
  ~DefaultGraphCompiler() override = default;

  std::shared_ptr<infer::abstract::ExecutionPlan> Compile(FuncGraphPtr graph) override;

 protected:
  void InitCompileOption(const FuncGraphPtr &graph);
  std::shared_ptr<infer::abstract::ExecutionPlan> NonCFGCompile(const std::vector<GraphSegmentPtr> &graph_segments,
                                                                const FuncGraphPtr &func_graph);

  virtual std::vector<GraphSegmentPtr> Partition(const FuncGraphPtr &graph);

  CompileResultPtr Compile(const GraphSegmentPtr &segment, const std::vector<AnfNodePtr> &inputs,
                           const std::vector<AnfNodePtr> &outputs);

  std::vector<InferKernel *> Schedule(const CompileResultPtr &compile_result);

 private:
  Status CreateExecPlanKernels(const std::vector<GraphSegmentPtr> &graph_segments,
                               std::vector<AnfNodePtrList> *segments_outputs);
  Status UpdateSubGraphInoutMap(const kernel::KernelExec &subgraph, const AnfNodePtrList &inputs,
                                const AnfNodePtrList &outputs);
  std::tuple<AnfNodePtrList, AnfNodePtrList> GetSegmentInout(const GraphSegment &graph_segment);
  Status CreateExecPlanInputs(const FuncGraphPtr &func_graph);
  Status CreateExecPlanOutputs(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtrList> &segments_outputs);
  Status IsolateSubGraphs();
  static std::vector<InferTensor *> CreateTensors(const std::vector<AnfNodePtr> &nodes);
  std::vector<AnfNodePtr> SkipMakeTuple(const AnfNodePtr &origin_node);
  void ReplaceNodes(const std::shared_ptr<FuncGraph> &graph);

 private:
  std::shared_ptr<infer::ExecutionPlan> execution_plan_{nullptr};
  std::vector<InferTensor *> graph_input_tensors_;
  mindspore::HashMap<AnfNodePtr, InferTensor *> anf_tensor_map_;
  mindspore::HashMap<InferTensor *, AnfNodePtr> subgraph_input_map_;
  mindspore::HashMap<AnfNodePtr, InferTensor *> subgraph_output_map_;
  SingleGraphSchedulerPtr scheduler_{nullptr};
  const std::shared_ptr<Context> &context_;
  InferContextPtr inner_context_{nullptr};
  CompileOptionPtr option_{nullptr};
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_DEFAULT_GRAPH_COMPILER_H_
