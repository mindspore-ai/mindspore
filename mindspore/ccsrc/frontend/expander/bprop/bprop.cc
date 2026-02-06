/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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
#include "frontend/expander/bprop/bprop.h"
#include <algorithm>
#include <queue>
#include <unordered_map>
#include "primitive/framework_ops.h"
#include "mindspore/ccsrc/include/utils/expander/infer.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "ir/graph_utils.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "frontend/expander/utils.h"
#include "include/utils/utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace expander {
namespace bprop {
using KernelGraph = session::KernelGraph;

void BpropExpander::FreeUselessValues(const PynativeCallback &cb) {
  auto handle = GetBpropIRBuilder(cb.opname());
  if (handle == nullptr) {
    MS_LOG(DEBUG) << "Bprop IRBuilder [" << cb.opname() << "] is not registered in bprop expander.";
    return;
  }
  if (!handle->unused_inputs.empty()) {
    cb.DeprecatedFreeDeviceAddress(handle->unused_inputs);
  }
  if (handle->free_useless_value_func != nullptr) {
    handle->free_useless_value_func(cb);
  }
}

bool BpropExpander::IsCloneInplaceInput(const PynativeCallback &cb) {
  auto handle = GetBpropIRBuilder(cb.opname());
  if (handle == nullptr) {
    MS_LOG(DEBUG) << "Bprop IRBuilder [" << cb.opname() << "] is not registered in bprop expander.";
    return false;
  }
  if (handle->clone_inplace_input_func != nullptr) {
    return handle->clone_inplace_input_func(cb);
  }
  return false;
}

class LazyInfer : public CppInfer {
 public:
  void Infer(const NodePtr &) override { return; }

  AbstractBasePtr GetAbstract(const NodePtr &node) override {
    auto anfnode = node->get();
    if (anfnode->abstract() == nullptr) {
      InferNow(anfnode);
    }
    return anfnode->abstract();
  }

 protected:
  void InferNow(const AnfNodePtr &node) {
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      for (size_t i = 1; i < cnode->size(); i++) {
        if (cnode->input(i)->abstract() == nullptr) {
          InferNow(cnode->input(i));
        }
      }
    }
    CppInfer::InferAnfnode(node);
  }
};

class GraphModeBuilder : public IrBuilder {
 public:
  GraphModeBuilder(const std::string &name, const FuncGraphPtr &func_graph, const ExpanderInferPtr &infer)
      : IrBuilder(name, func_graph, infer) {}

  NodePtrList Build(const NodePtrList &inputs, const mindspore::HashMap<std::string, ValuePtr> &attrs,
                    const BpropHandle &handle, const std::string &instance_name) {
    auto outputs = Run(inputs, attrs, handle, instance_name);
    InsertDepend(&outputs);
    auto mt = this->MakeTuple(outputs)->get();
    func_graph_->set_output(mt);
    if (has_ctrl_flow_) {
      // clear all abstract, to let the specializer re-infer the subgraph of controlflow graphs.
      auto todos = TopoSort(func_graph_->get_return(), SuccDeeperSimple, AlwaysInclude);
      for (auto &no : todos) {
        no->set_abstract(nullptr);
        if (IsValueNode<FuncGraph>(no)) {
          auto fg = GetValueNode<FuncGraphPtr>(no);
          for (auto &p : fg->parameters()) {
            p->set_abstract(nullptr);
          }
        }
      }
    }
    return outputs;
  }

  NodePtr Conditional(const NodePtr &cond, const BlockFunc &true_case, const BlockFunc &false_case) override {
    has_ctrl_flow_ = true;
    CtrlFlowBlock cfb(this, this->func_graph(),
                      [this](const FuncGraphPtr &fg, const ExpanderInferPtr &infer) -> EmitterPtr {
                        return std::make_shared<GraphModeBuilder>(this->name_ + "Conditional", fg, infer);
                      });
    this->func_graph()->set_flag(kFlagIsControlFlow, true);
    return cfb.IfThenElse(cond, true_case, false_case);
  }

  NodePtr While(const NodePtr &cond, const BlockFunc &body, const NodePtrList &init_list) override {
    has_ctrl_flow_ = true;
    CtrlFlowBlock cfb(this, this->func_graph(),
                      [this](const FuncGraphPtr &fg, const ExpanderInferPtr &infer) -> EmitterPtr {
                        return std::make_shared<GraphModeBuilder>(this->name_ + "While", fg, infer);
                      });
    this->func_graph()->set_flag(kFlagIsControlFlow, true);
    return cfb.While(cond, body, init_list);
  }

 protected:
  NodePtr EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) override {
    auto primpy = ConvertPrimToPrimPy(prim);
    AnfNodePtrList cnode_inputs = {NewValueNode(primpy ? primpy : prim)};
    cnode_inputs.reserve(inputs.size() + 1);
    (void)std::transform(inputs.cbegin(), inputs.cend(), std::back_inserter(cnode_inputs), [](const NodePtr &no) {
      MS_EXCEPTION_IF_NULL(no);
      return no->get();
    });
    // PyNative use kernel graph construct bprop graph
    auto cnode = func_graph_->isa<KernelGraph>() ? func_graph_->FuncGraph::NewCNode(cnode_inputs)
                                                 : func_graph_->NewCNode(cnode_inputs);
    if (scope_ != nullptr) {
      cnode->set_scope(scope_);
    }
    auto node = NewIrNode(cnode->cast<AnfNodePtr>());
    infer_->Infer(node);
    for (auto &inp : inputs) {
      (void)isolated_side_effect_nodes_.erase(inp);
    }
    if ((prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_MEM) && GetValue<bool>(prim->GetAttr(GRAPH_FLAG_SIDE_EFFECT_MEM))) ||
        (prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_IO) && GetValue<bool>(prim->GetAttr(GRAPH_FLAG_SIDE_EFFECT_IO)))) {
      isolated_side_effect_nodes_.add(node);
    }
    return node;
  }

  void InsertDepend(NodePtrList *outputs) {
    for (auto &out : *outputs) {
      isolated_side_effect_nodes_.erase(out);
    }
    if (isolated_side_effect_nodes_.empty() || outputs->empty()) {
      return;
    }
    if (isolated_side_effect_nodes_.size() == 1) {
      (*outputs)[0] = this->Depend((*outputs)[0], isolated_side_effect_nodes_.back());
    } else {
      NodePtrList nodes(isolated_side_effect_nodes_.begin(), isolated_side_effect_nodes_.end());
      auto mt = this->MakeTuple(nodes);
      (*outputs)[0] = this->Depend((*outputs)[0], mt);
    }
  }

  bool has_ctrl_flow_{false};
  // This variable is used to record isolated nodes in the graph. When the bprop graph construction is complete, all
  // isolated nodes are connected to outputs[0] using a Depend node.
  mindspore::OrderedSet<NodePtr> isolated_side_effect_nodes_;
};

bool ExpandBpropInGraphMode(const BpropHandle *handle, const PrimitivePtr &prim, const FuncGraphPtr &graph) {
  static const bool use_imm_infer = (common::GetEnv("MS_DEV_BPROP_IMM_INFER") == "on");
  static const bool dump_result = (common::GetEnv("MS_DEV_DUMP_BPROP") == "on");
  auto name = prim->name();
  if (handle == nullptr) {
    MS_LOG(DEBUG) << "Bprop IRBuilder [" << name << "] is not registered in bprop expander.";
    return false;
  }
  ExpanderInferPtr infer;
  if (use_imm_infer) {
    infer = std::make_shared<CppInfer>();
  } else {
    infer = std::make_shared<LazyInfer>();
  }
  GraphModeBuilder ir_builder(name, graph, infer);
  auto &parameters = graph->parameters();
  NodePtrList inputs;
  inputs.reserve(parameters.size());
  (void)std::transform(parameters.cbegin(), parameters.cend(), std::back_inserter(inputs),
                       [&ir_builder](const AnfNodePtr &no) { return std::make_shared<IrNode>(no, &ir_builder); });
  auto outputs = ir_builder.Build(inputs, prim->attrs(), *handle, prim->instance_name());
  if (outputs.empty()) {
    MS_LOG(DEBUG) << "The output nodes of bprop function [" << name << "] is empty.";
    return false;
  }
  if (dump_result) {
    DumpIR("bprop/bprop_expander_" + name + ".ir", graph, true);
  }
  return true;
}

#ifdef _MSC_VER
WinBpropRegister::WinBpropRegister() {}
#endif
}  // namespace bprop
}  // namespace expander
}  // namespace mindspore
