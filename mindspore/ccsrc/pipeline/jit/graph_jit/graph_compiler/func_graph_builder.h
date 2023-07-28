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

#ifndef MINDSPORE_JIT_GRAPH_FUNC_GRAPH_BUILDER_H_
#define MINDSPORE_JIT_GRAPH_FUNC_GRAPH_BUILDER_H_

#include <map>
#include <memory>
#include <string>
#include "ir/func_graph.h"
#include "pipeline/jit/graph_jit/graph_compiler/pi_ir/ir_mutator.h"

namespace mindspore {
namespace jit {
namespace graph {
namespace py = pybind11;

class MindNode : public ir::Node {
 public:
  explicit MindNode(const AnfNodePtr &node) : node_(node) {}

  // Destructor.
  ~MindNode() override = default;
  JIT_DECLARE_PARENT(MindNode, Node);

  const AnfNodePtr &GetAnfNode() const { return node_; }

  /**
   * \brief Get the description of this Mind Node.
   * \return The description.
   */
  std::string ToString() const override { return node_->DebugString(); }

 private:
  AnfNodePtr node_;
};

using MindNodePtr = std::shared_ptr<MindNode>;

// FuncGraphBuilder to convert ir graph to function graph
class FuncGraphBuilder : public ir::IRMutator {
 public:
  explicit FuncGraphBuilder(const ir::FunctionNodePtr &func)
      : func_(func),
        args_({}),
        kwargs_(nullptr),
        func_graph_(std::make_shared<FuncGraph>()),
        func_graph_mgr_(nullptr) {}
  virtual ~FuncGraphBuilder() = default;
  static FuncGraphPtr BuildFuncGraph(const ir::FunctionNodePtr &func, const py::tuple &args, const py::dict &kwargs);
  FuncGraphPtr Build(const py::tuple &args, const py::dict &kwargs);
  FuncGraphPtr Build(const AnfNodePtrList &args, const AnfNodePtr &kwargs);

  FuncGraphManagerPtr GetFuncGraphManager() const { return func_graph_mgr_; }
  void SetFuncGraphManager(const FuncGraphManagerPtr &manager) { func_graph_mgr_ = manager; }
  void SpecifyLocalVarInitialValue(const std::string &name, const AnfNodePtr &value) { assigned_vars_[name] = value; }

  // overloadable Mutate function.
  ir::NodePtr Mutate_(const ir::RefNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::ParameterPtr &node) override;
  ir::NodePtr Mutate_(const ir::FunctionNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::ValuePtr &node) override;
  ir::NodePtr Mutate_(const ir::IfNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::BinaryOperationPtr &node) override;
  ir::NodePtr Mutate_(const ir::NegativeNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::NotNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::InvertNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::ReturnNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::CastNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::FormatNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::AddNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::SubNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::MulNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::DivNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::BitwiseNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::IsNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::ContainsNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::StoreNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::CompareNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::LoadNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::BuildNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::CallNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::UpdateNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::SubscrNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::AttrNodePtr &node) override;

 private:
  AnfNodePtr GetAnfNode(const ir::NodePtr &node);
  CNodePtr MergeList(const AnfNodePtrList &nodes);
  CNodePtr ConvertValueDictionaryToCNode(const AnfNodePtr &node);
  CNodePtr MergeDict(const AnfNodePtrList &nodes);

  const ir::FunctionNodePtr func_;
  AnfNodePtrList args_;
  AnfNodePtr kwargs_;
  FuncGraphPtr func_graph_;
  FuncGraphManagerPtr func_graph_mgr_;

  // Store variable's name, variable's node.
  std::map<std::string, AnfNodePtr> assigned_vars_;
};

using FuncGraphBuilderPtr = std::shared_ptr<FuncGraphBuilder>;
}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif  // MINDSPORE_JIT_GRAPH_FUNC_GRAPH_BUILDER_H_
