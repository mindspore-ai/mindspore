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

namespace mindspore {
class DefaultGraphCompiler : public mindspore::infer::abstract::GraphCompiler {
 public:
  explicit DefaultGraphCompiler(const std::shared_ptr<Context> &context) : context_(context) {
    inner_context_ = nullptr;
  }
  virtual ~DefaultGraphCompiler() = default;

  std::shared_ptr<infer::abstract::ExecutionPlan> Compile(FuncGraphPtr graph) override;

  infer::abstract::ExecutionFlowPtr Compile(const GraphSegmentPtr &segment, const AnfNodePtrList &inputs,
                                            const AnfNodePtrList &outputs) override {
    return nullptr;
  }

 protected:
  virtual std::vector<GraphSegmentPtr> Partition(const FuncGraphPtr &graph);

  virtual std::shared_ptr<infer::abstract::ExecutionPlan> Schedule(const std::vector<GraphSegmentPtr> &graph_segments,
                                                                   FuncGraphPtr func_graph);

  virtual std::shared_ptr<infer::abstract::ExecutionFlow> Schedule(const GraphSegmentPtr &graph_segment,
                                                                   const std::vector<AnfNodePtr> &inputs,
                                                                   const std::vector<AnfNodePtr> &outputs);

 private:
  infer::abstract::Tensor *CreateTensor(AnfNodePtr node);
  std::vector<infer::abstract::Tensor *> CreateTensors(const std::vector<AnfNodePtr> &nodes);
  Status GetDTAndShapeFromParameter(ParameterPtr parameter, TypeId *data_type, ShapeVector *shape_vector);
  Status GetDTAndShapeFromAbTensor(const abstract::AbstractTensorPtr &abstract, TypeId *data_type,
                                   ShapeVector *shape_vector);

  std::tuple<FuncGraphPtr, AnfNodePtrList, AnfNodePtrList> TransformSegmentToAnfGraph(const AnfNodePtrList &lst);
  AnfNodePtrList GetOutput(const AnfNodePtrList &nodes, const NodeUsersMap &users,
                           const mindspore::HashSet<AnfNodePtr> &seen);
  AnfNodePtr RefSubGraphNode(const FuncGraphPtr &fg, const AnfNodePtr &node, AnfNodePtrList *inputs_ptr,
                             mindspore::HashMap<AnfNodePtr, AnfNodePtr> *eqv_ptr);

 private:
  mindspore::HashMap<AnfNodePtr, infer::abstract::Tensor *> anf_tensor_map_;
  const std::shared_ptr<Context> &context_;
  std::shared_ptr<mindspore::infer::abstract::Context> inner_context_;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_DEFAULT_GRAPH_COMPILER_H_
