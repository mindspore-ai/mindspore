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
  std::shared_ptr<infer::abstract::ExecutionPlan> NonCFGCompile(const std::vector<GraphSegmentPtr> &graph_segments,
                                                                const FuncGraphPtr &func_graph);

  virtual std::vector<GraphSegmentPtr> Partition(const FuncGraphPtr &graph);

  CompileResultPtr Compile(const GraphSegmentPtr &segment, const std::vector<AnfNodePtr> &inputs,
                           const std::vector<AnfNodePtr> &outputs);

  std::vector<InferKernel *> Schedule(const CompileResultPtr &compile_result);

 private:
  InferTensor *CreateTensor(const AnfNodePtr &node);
  std::vector<InferTensor *> CreateTensors(const std::vector<AnfNodePtr> &nodes);
  Status GetDTAndShapeFromParameter(const ParameterPtr &parameter, TypeId *data_type, ShapeVector *shape_vector);
  Status GetDTAndShapeFromAbTensor(const mindspore::abstract::AbstractTensorPtr &abstract, TypeId *data_type,
                                   ShapeVector *shape_vector);

 private:
  mindspore::HashMap<AnfNodePtr, InferTensor *> anf_tensor_map_;
  SingleGraphSchedulerPtr scheduler_{nullptr};
  const std::shared_ptr<Context> &context_;
  std::shared_ptr<mindspore::infer::abstract::Context> inner_context_;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_DEFAULT_GRAPH_COMPILER_H_
