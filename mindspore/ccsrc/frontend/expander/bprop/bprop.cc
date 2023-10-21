/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "ops/sequence_ops.h"
#include "ops/array_ops.h"
#include "abstract/ops/primitive_infer_map.h"
#include "include/common/expander/core/infer.h"
#include "include/common/profiler.h"
#include "include/backend/kernel_graph.h"
#include "utils/anf_utils.h"
#include "include/common/debug/anf_ir_dump.h"
#include "frontend/expander/utils.h"

namespace mindspore {
namespace expander {
namespace bprop {
using BpropGraphCacheMap = std::unordered_map<abstract::AbstractBasePtrList, FuncGraphPtr,
                                              abstract::AbstractBasePtrListHasher, abstract::AbstractBasePtrListEqual>;
using KernelGraph = session::KernelGraph;

bool HasBpropExpander(const std::string &prim_name) {
  const BpropHandle *handle = BpropIRBuilderFactory::Instance().GetBuilder(prim_name);
  return (handle != nullptr);
}

class ShapeCalcException : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};

class PynativeIRBuilder : public BpropIRBuilder {
 public:
  PynativeIRBuilder(const std::string &name, const FuncGraphPtr &fg, const ExpanderInferPtr &infer, UserMap *users,
                    const AnfNodePtr &dout)
      : BpropIRBuilder(name, fg, infer), users_(users), dout_(dout) {
    MS_EXCEPTION_IF_NULL(users);
  }
  ~PynativeIRBuilder() = default;

  inline static std::unordered_map<PrimitivePtr, BpropGraphCacheMap, PrimitiveHasher, PrimitiveTotalEqual>
    bprop_op_graph_map;

  NodePtr OutZeros(const NodePtr &node) override {
    need_infer_ = false;
    auto ret = Emit(kZerosLikeOpName, {node});
    need_infer_ = true;
    return ret;
  }

  NodePtrList Build(const std::vector<NodePtr> &input_nodes, const HashMap<std::string, ValuePtr> &attrs,
                    const BpropHandle &handle, const std::string &instance_name) {
    auto output_nodes = Run(input_nodes, attrs, handle, instance_name);
    for (size_t i = 0; i < output_nodes.size(); i++) {
      auto &node = output_nodes[i];
      // A Value node gradient will loss the trace context in pynative, so emit a node. A example is Eye.
      if (node->isa<ValueNode>() || IsPrimitiveCNode(node->get(), prim::kPrimZerosLike)) {
        if (node->isa<ValueNode>()) {
          auto abs = node->abstract();
          MS_EXCEPTION_IF_NULL(abs);
          if (abs->isa<abstract::AbstractScalar>()) {
            node = OutZeros(Tensor(0, abs->BuildType()));
          } else {
            node = OutZeros(node);
          }
        }
        node->get()->set_abstract(input_nodes[i]->abstract()->Broaden());
      }
    }
    return output_nodes;
  }

  NodePtrList BuildWithCache(const NodePtrList &input_nodes, const HashMap<std::string, ValuePtr> &attrs,
                             const BpropHandle &handle, const std::string &instance_name, const PrimitivePtr &prim) {
    BpropGraphCacheMap &bprop_map = PynativeIRBuilder::bprop_op_graph_map[prim];
    AbstractBasePtrList abs_list;
    abs_list.reserve(input_nodes.size());
    (void)std::transform(input_nodes.cbegin(), input_nodes.cend(), std::back_insert_iterator(abs_list),
                         [](const NodePtr &no) { return no->abstract()->Clone(); });
    FuncGraphPtr graph;
    auto it = bprop_map.find(abs_list);
    if (it == bprop_map.end()) {
      bool skip_cache = false;
      graph = BuildBpropOpGraph(input_nodes, attrs, handle, instance_name, &skip_cache);
      bprop_map[abs_list] = skip_cache ? nullptr : graph;
    } else {
      graph = it->second;
    }
    if (graph == nullptr) {
      return Build(input_nodes, attrs, handle, instance_name);
    }

    need_infer_ = false;
    std::unordered_map<AnfNodePtr, NodePtr> node_map;
    auto parm = graph->parameters();
    auto output = graph->output();
    bool is_multi_outputs = graph->has_flag("multi");
    auto nodes = TopoSort(graph->get_return(), SuccIncoming,
                          [](const AnfNodePtr &node) { return node->isa<CNode>() ? FOLLOW : EXCLUDE; });
    nodes.pop_back();
    for (size_t i = 0; i < input_nodes.size(); i++) {
      node_map[parm[i]] = input_nodes[i];
    }
    for (auto &node : nodes) {
      auto cnode = node->cast<CNodePtr>();
      NodePtrList cnode_list;
      if (node == output && is_multi_outputs) {
        (void)std::transform(cnode->inputs().cbegin() + 1, cnode->inputs().cend(), std::back_inserter(cnode_list),
                             [&node_map](const AnfNodePtr &no) { return node_map.at(no); });
        return cnode_list;
      }
      PrimitivePtr primitive = nullptr;
      for (auto &no : cnode->inputs()) {
        if (no->isa<ValueNode>()) {
          auto value = no->cast<ValueNodePtr>()->value();
          if (value->isa<Primitive>()) {
            primitive = value->cast<PrimitivePtr>()->Clone();
          } else {
            auto value_node = NewValueNode(value);
            value_node->set_abstract(value->ToAbstract());
            cnode_list.emplace_back(NewNode(value_node));
          }
        } else {
          cnode_list.emplace_back(node_map.at(no));
        }
      }
      auto new_node = EmitOp(primitive, cnode_list);
      new_node->get()->set_abstract(node->abstract()->Clone());
      node_map[node] = new_node;
    }
    return NodePtrList{node_map.at(output)};
  }

 protected:
  FuncGraphPtr BuildBpropOpGraph(const NodePtrList &input_nodes, const HashMap<std::string, ValuePtr> &attrs,
                                 const BpropHandle &handle, const std::string &instance_name, bool *skip_cache) {
    // This section of code can have a very long runtime with certain operations, such as the MaskedSelect operator.
    auto graph = std::make_shared<FuncGraph>();
    NodePtrList inputs;
    inputs.reserve(input_nodes.size());
    std::vector<bool> value_index(input_nodes.size());
    for (size_t i = 0; i < input_nodes.size(); i++) {
      auto inp = input_nodes[i];
      auto p = inputs.emplace_back(NewNode(graph->add_parameter()));
      p->get()->set_abstract(inp->abstract()->Clone());
      if (!p->HasAbstractValue()) {
        p->SetValue(inp->Value());
        value_index[i] = true;
      }
    }

    std::swap(graph, func_graph_);
    need_record_users_ = false;
    auto output_nodes = Build(inputs, attrs, handle, instance_name);
    need_record_users_ = true;
    std::swap(graph, func_graph_);

    if (output_nodes.size() == 1) {
      graph->set_output(output_nodes[0]->get());
    } else {
      AnfNodePtrList new_outputs{NewValueNode(prim::kPrimMakeTuple)};
      (void)std::transform(output_nodes.cbegin(), output_nodes.cend(), std::back_inserter(new_outputs),
                           [](const NodePtr &node) { return node->get(); });
      auto mt = graph->NewCNode(new_outputs);
      graph->set_output(mt);
      graph->set_flag("multi", true);
    }
    for (size_t i = 0; i < inputs.size(); i++) {
      if (value_index[i] && inputs[i]->is_used_value()) {
        *skip_cache = true;
        break;
      }
    }
    return graph;
  }

  NodePtr EmitGetItemValue(const NodePtrList &inputs) {
    auto real_input = inputs[0]->get<ValueNodePtr>();
    if (real_input != nullptr) {
      auto real_input_value = real_input->value()->cast<ValueSequeuePtr>();
      if (real_input_value != nullptr) {
        auto item_idx = GetValue<int64_t>(inputs[1]->get<ValueNodePtr>()->value());
        auto valuenode = NewValueNode((*real_input_value)[item_idx]);
        valuenode->set_abstract(valuenode->value()->ToAbstract());
        return NewNode(valuenode);
      }
    }
    return nullptr;
  }

  NodePtr EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) override {
    if (prim->name() == prim::kPrimShapeCalc->name()) {
      // temporary solution, remove this after input parameter's value is set.
      throw ShapeCalcException("ShapeCalc is not supported in pynative mode.");
    }
    if (prim->name() == kTupleGetItemOpName) {
      // if the getitem's real input is a ValueSequence, just return the real Value of that.
      auto getitem_value = EmitGetItemValue(inputs);
      if (getitem_value != nullptr) {
        return getitem_value;
      }
    }
    AnfNodePtrList cnode_inputs{NewValueNode(prim)};
    cnode_inputs.reserve(inputs.size() + 1);
    (void)std::transform(inputs.cbegin(), inputs.cend(), std::back_inserter(cnode_inputs),
                         [](const NodePtr &inp) { return inp->get(); });
    // PyNative use kernel graph construct bprop graph, which indicate func_graph_ here is kernel graph;
    // And, use kernel graph create cnode will do PostNewCNode which is not necessary
    auto cnode = func_graph_->isa<KernelGraph>() ? func_graph_->FuncGraph::NewCNode(cnode_inputs)
                                                 : func_graph_->NewCNode(cnode_inputs);
    if (scope_ != nullptr) {
      cnode->set_scope(scope_);
    }

    auto node = NewNode(cnode->cast<AnfNodePtr>());
    if (need_infer_) {
      auto value_depend = abstract::GetValueDependArgIndices(cnode);
      if (!value_depend.empty()) {
        for (auto idx : value_depend) {
          size_t i = LongToSize(idx);
          if (i < inputs.size() && !inputs[i]->HasAbstractValue()) {
            auto v = inputs[i]->BuildValue();
            auto tensor = v->cast<tensor::TensorPtr>();
            if (tensor != nullptr) {
              tensor->data_sync();
            }
            inputs[i]->abstract()->set_value(v);
          }
        }
      }
      infer_->Infer(node);
    }
    if (!need_record_users_) {
      return node;
    }
    // record the users
    for (size_t i = 1; i < cnode_inputs.size(); i++) {
      auto &inp = cnode_inputs[i];
      if (inp == dout_ || inp->isa<Parameter>()) {
        (void)users_->dout_user_[inp].emplace_back(cnode, i);
      } else if (IsPrimitiveCNode(inp, prim::kPrimTupleGetItem)) {
        // record the dout's successor getitem's users
        auto getitem = inp->cast<CNodePtr>();
        auto real_input = getitem->input(kIndex1);
        if (real_input == dout_) {
          (void)users_->tuple_getitem_user_[inp].emplace_back(cnode, i);
        }
      }
    }
    return node;
  }

  UserMap *users_;
  AnfNodePtr dout_;
  bool need_infer_{true};
  bool need_record_users_{true};
};

void ClearBpropOpGraphMap() { PynativeIRBuilder::bprop_op_graph_map.clear(); }

bool BpropExpander::Run(const CNodePtr &cnode, const std::vector<ValuePtr> &input_values) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "Begin building bprop for " << cnode->fullname_with_scope();
  bool ret = true;
  if (outputs_ != nullptr) {
    outputs_->clear();
  }
  auto node_name = AnfUtils::GetCNodeName(cnode);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeGradExpander,
                                     node_name, true);
  if (OpEnvManager::UsePyBprop(node_name)) {
    MS_LOG(DEBUG) << "Python bprop will be used for op " << node_name;
    return false;
  }
  try {
    ret = RunBprop(cnode, input_values);
  } catch (const ShapeCalcException &e) {
    MS_LOG(INFO) << "Bprop \"" << node_name << "\" encounter a problem: [" << e.what()
                 << "]. python bprop will be used.";
    if (outputs_ != nullptr) {
      outputs_->clear();
    }
    ret = false;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Bprop \"" << node_name << "\" encounter a problem: [" << e.what() << "]";
    std::rethrow_exception(std::current_exception());
  }
  MS_LOG(DEBUG) << "Finish building bprop for " << cnode->fullname_with_scope();
  return ret;
}

const mindspore::HashSet<size_t> &BpropExpander::GetUnusedInputs(const string &op_name) {
  auto handle = BpropIRBuilderFactory::Instance().GetBuilder(op_name);
  if (handle == nullptr) {
    MS_LOG(DEBUG) << "Bprop IRBuilder [" << op_name << "] is not registered in bprop expander.";
    static const mindspore::HashSet<size_t> no_handle{INT_MAX};
    return no_handle;
  }
  return handle->unused_inputs;
}

bool BpropExpander::RunBprop(const CNodePtr &cnode, const std::vector<ValuePtr> &input_values) {
  const auto prim = GetCNodePrimitive(cnode);
  auto name = prim->name();
  PynativeIRBuilder ir_builder(name, cnode->func_graph(), std::make_shared<CppInfer>(), users_, cnode->inputs().back());
  input_nodes_.reserve(cnode->size());
  (void)std::transform(cnode->inputs().cbegin() + 1, cnode->inputs().cend(), std::back_inserter(input_nodes_),
                       [&ir_builder](const AnfNodePtr &no) { return std::make_shared<Node>(no, &ir_builder); });
  if (!input_values.empty()) {
    for (size_t i = 0; i < input_values.size(); ++i) {
      input_nodes_[i]->SetValue(input_values[i]);
    }
  }
  mindspore::HashMap<std::string, ValuePtr> attrs;
  {
    PrimitiveReadLock read_lock(prim->shared_mutex());
    attrs = prim->attrs();
  }
  auto handle = BpropIRBuilderFactory::Instance().GetBuilder(name);
  if (handle == nullptr) {
    MS_LOG(DEBUG) << "Bprop IRBuilder [" << name << "] is not registered in bprop expander.";
    return false;
  }
  static const bool cache_env = (common::GetEnv("MS_DEV_DISABLE_BPROP_CACHE") != "on");
  if (cache_env) {
    output_nodes_ = ir_builder.BuildWithCache(input_nodes_, attrs, *handle, prim->instance_name(), prim);
  } else {
    output_nodes_ = ir_builder.Build(input_nodes_, attrs, *handle, prim->instance_name());
  }
  if (output_nodes_.empty()) {
    MS_LOG(DEBUG) << "The output nodes of bprop function [" << name << "] is empty.";
    return false;
  }
  PostProcess(cnode);
  DumpResult(name);
  return true;
}

void BpropExpander::PostProcess(const CNodePtr &cnode) const {
  outputs_->reserve(output_nodes_.size());
  constexpr const size_t num_out_and_dout = 2;
  if (output_nodes_.size() + num_out_and_dout != input_nodes_.size()) {
    MS_LOG(EXCEPTION) << "For bprop [" << AnfUtils::GetCNodeName(cnode)
                      << "], the output size should be equal to input size (exclude out and dout), but got "
                      << output_nodes_.size() << " vs " << (input_nodes_.size() - num_out_and_dout);
  }
  for (size_t i = 0; i < output_nodes_.size(); i++) {
    (void)outputs_->emplace_back(output_nodes_[i]->get<CNodePtr>());
  }
}

void BpropExpander::DumpResult(const std::string &name) const {
  static const bool dump_result = (common::GetEnv("MS_DEV_DUMP_BPROP") == "on");
  if (!dump_result) {
    return;
  }
  auto fg = std::make_shared<FuncGraph>();
  std::map<AnfNodePtr, AnfNodePtr> node_map;
  CNodePtrList newcnodes;
  for (auto &inp : input_nodes_) {
    auto p = fg->add_parameter();
    p->set_abstract(inp->get()->abstract());
    node_map[inp->get()] = p;
  }
  std::queue<CNodePtr> que;
  (void)std::for_each(outputs_->cbegin(), outputs_->cend(), [&que](const CNodePtr &cnode) { que.push(cnode); });

  while (!que.empty()) {
    auto node = que.front();
    que.pop();
    if (node_map.count(node) != 0) {
      continue;
    }
    auto new_node = fg->NewCNode(node->inputs());
    new_node->CloneCNodeInfo(node);
    new_node->set_fullname_with_scope(node->fullname_with_scope());
    node_map[node] = new_node;
    newcnodes.push_back(new_node);
    for (size_t i = 1; i < node->size(); ++i) {
      const auto &inp = node->input(i);
      if (inp->isa<CNode>() && node_map.count(inp) == 0) {
        que.push(inp->cast<CNodePtr>());
      }
    }
  }

  for (auto &cnode : newcnodes) {
    for (size_t i = 1; i < cnode->size(); i++) {
      if (node_map.count(cnode->input(i)) != 0) {
        cnode->set_input(i, node_map[cnode->input(i)]);
      }
    }
  }

  if (outputs_->size() == 1) {
    fg->set_output(node_map[(*outputs_)[0]]);
  } else {
    AnfNodePtrList new_outputs{NewValueNode(prim::kPrimMakeTuple)};
    AbstractBasePtrList abs;
    (void)std::transform(outputs_->cbegin(), outputs_->cend(), std::back_inserter(new_outputs),
                         [&node_map, &abs](const CNodePtr &node) {
                           abs.push_back(node->abstract());
                           return node_map[node];
                         });
    auto mt = fg->NewCNode(new_outputs);
    mt->set_abstract(std::make_shared<abstract::AbstractTuple>(abs));
    fg->set_output(mt);
  }
  DumpIR("bprop/bprop_expander_" + name + ".ir", fg, true);

  if (users_ != nullptr) {
    for (auto &uiter : users_->dout_user_) {
      for (auto &iter : uiter.second) {
        auto user = iter.first.lock();
        if (user == nullptr) {
          continue;
        }
        MS_LOG(INFO) << "Node " << uiter.first->ToString() << " user: " << user->fullname_with_scope()
                     << "  index: " << iter.second;
      }
    }
  }
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

class GraphModeBuilder : public BpropIRBuilder {
 public:
  GraphModeBuilder(const std::string &name, const FuncGraphPtr &func_graph, const ExpanderInferPtr &infer)
      : BpropIRBuilder(name, func_graph, infer) {}

  NodePtrList Build(const NodePtrList &inputs, const mindspore::HashMap<std::string, ValuePtr> &attrs,
                    const BpropHandle &handle, const std::string &instance_name) {
    auto outputs = Run(inputs, attrs, handle, instance_name);
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

 protected:
  NodePtr EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) override {
    if (prim->name() == "Switch") {
      has_ctrl_flow_ = true;
    }
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
    auto node = NewNode(cnode->cast<AnfNodePtr>());
    infer_->Infer(node);
    return node;
  }

  bool has_ctrl_flow_{false};
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
                       [&ir_builder](const AnfNodePtr &no) { return std::make_shared<Node>(no, &ir_builder); });
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
void RegGradArrayOps();
void RegGradClipOps();
void RegGradCommOps();
void RegGradDebugOps();
void RegGradImageOps();
void RegGradImplementationsOps();
void RegGradInnerOps();
void RegGradLinalgOps();
void RegGradMathOps();
void RegGradNnOps();
void RegGradOtherOps();
void RegGradQuantOps();
void RegGradScipyOps();
void RegGradSparseOps();
void RegGradSequenceOps();
void RegGradScalarOps();

WinBpropRegister::WinBpropRegister() {
  RegGradArrayOps();
  RegGradClipOps();
  RegGradCommOps();
  RegGradDebugOps();
  RegGradImageOps();
  RegGradImplementationsOps();
  RegGradInnerOps();
  RegGradLinalgOps();
  RegGradMathOps();
  RegGradNnOps();
  RegGradOtherOps();
  RegGradQuantOps();
  RegGradScipyOps();
  RegGradSparseOps();
  RegGradSequenceOps();
  RegGradScalarOps();
}
#endif
}  // namespace bprop
}  // namespace expander
}  // namespace mindspore
