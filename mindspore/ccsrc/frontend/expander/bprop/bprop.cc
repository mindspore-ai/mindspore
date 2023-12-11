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
class SimpleNode {
 public:
  explicit SimpleNode(const AbstractBasePtr &abs) : abs_(abs->Clone()) {}
  SimpleNode(const ValuePtr &value, const AbstractBasePtr &abs) : abs_(abs->Clone()), value_(value) {}
  SimpleNode(const PrimitivePtr &prim, const AbstractBasePtr &abs, const std::vector<size_t> &input_indexs)
      : input_indexs(std::move(input_indexs)), abs_(abs->Clone()), prim_(prim->Clone()) {}
  ~SimpleNode() = default;
  bool is_valuenode() const { return value_ != nullptr; }
  PrimitivePtr get_primitive() const { return prim_->Clone(); }
  AbstractBasePtr get_abstract() const { return abs_->Clone(); }
  ValuePtr get_value() const { return value_; }

  std::vector<size_t> input_indexs;

 protected:
  AbstractBasePtr abs_;
  ValuePtr value_;
  PrimitivePtr prim_;
};
using SimpleNodePtr = std::shared_ptr<SimpleNode>;

struct SimpleGraph {
  std::vector<SimpleNodePtr> nodes;
  std::vector<size_t> output_indexs;
  std::vector<size_t> input_indexs;
};
using SimpleGraphPtr = std::shared_ptr<SimpleGraph>;
using BpropGraphCacheMap = std::unordered_map<abstract::AbstractBasePtrList, SimpleGraphPtr,
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
  PynativeIRBuilder(const PrimitivePtr &prim, const FuncGraphPtr &fg, const ExpanderInferPtr &infer, UserMap *users,
                    const AnfNodePtr &dout)
      : BpropIRBuilder(prim->name(), fg, infer), users_(users), dout_(dout), prim_(prim) {
    MS_EXCEPTION_IF_NULL(users);
  }
  ~PynativeIRBuilder() = default;

  NodePtr OutZeros(const NodePtr &node) override {
    need_infer_ = false;
    auto ret = Emit(kZerosLikeOpName, {node});
    need_infer_ = true;
    return ret;
  }

  virtual NodePtrList Build(const std::vector<NodePtr> &input_nodes, const std::vector<ValuePtr> &input_values,
                            const HashMap<std::string, ValuePtr> &attrs, const BpropHandle &handle) {
    if (!input_values.empty()) {
      for (size_t i = 0; i < input_values.size(); ++i) {
        input_nodes[i]->SetValue(input_values[i]);
      }
    }
    auto output_nodes = Run(input_nodes, attrs, handle, prim_->instance_name());
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

 protected:
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
  PrimitivePtr prim_;
};

class PynativeIRBuilderWithCache : public PynativeIRBuilder {
 public:
  using PynativeIRBuilder::PynativeIRBuilder;
  ~PynativeIRBuilderWithCache() = default;

  inline static std::unordered_map<PrimitivePtr, BpropGraphCacheMap, PrimitiveHasher, PrimitiveTotalEqual>
    bprop_op_graph_map;

  NodePtrList Build(const NodePtrList &input_nodes, const std::vector<ValuePtr> &input_values,
                    const HashMap<std::string, ValuePtr> &attrs, const BpropHandle &handle) override {
    AbstractBasePtrList abs_list;
    NodePtrList output_nodes;
    abs_list.reserve(input_nodes.size());
    (void)std::transform(input_nodes.cbegin(), input_nodes.cend(), std::back_insert_iterator(abs_list),
                         [](const NodePtr &no) { return no->abstract(); });
    std::vector<size_t> value_index(input_nodes.size());
    for (size_t i = 0; i < input_values.size(); ++i) {
      if (!input_nodes[i]->HasAbstractValue()) {
        input_nodes[i]->SetValue(input_values[i]);
        value_index[i] = true;
      }
    }
    BpropGraphCacheMap &bprop_map = PynativeIRBuilderWithCache::bprop_op_graph_map[prim_];
    auto it = bprop_map.find(abs_list);
    if (it == bprop_map.end()) {
      need_record_nodes_ = true;
      output_nodes = PynativeIRBuilder::Build(input_nodes, {}, attrs, handle);
      need_record_nodes_ = false;
      // need not grad if grad depend input_values.
      for (size_t i = 0; i < input_nodes.size(); i++) {
        if (value_index[i] && input_nodes[i]->is_used_value()) {
          return output_nodes;
        }
      }
      bprop_map[abs_list] = BuildBpropOpGraph(input_nodes, output_nodes);
    } else {
      need_infer_ = false;
      SimpleGraphPtr graph = it->second;
      std::vector<NodePtr> node_map(input_nodes);
      node_map.reserve(graph->nodes.size());
      auto SimpleNodeToMsNode = [&graph, &node_map, this](const SimpleNodePtr &node) -> NodePtr {
        if (node->is_valuenode()) {
          return EmitValue(node->get_value());
        }
        NodePtrList cnode_list;
        cnode_list.reserve(node->input_indexs.size());
        for (size_t i : node->input_indexs) {
          (void)cnode_list.emplace_back(node_map[i]);
        }
        NodePtr new_node = EmitOp(node->get_primitive(), cnode_list);
        AnfNodePtr ms_node = new_node->get();
        if (ms_node->abstract() == nullptr) {
          ms_node->set_abstract(node->get_abstract());
        }
        return new_node;
      };
      for (size_t i = graph->input_indexs.size(); i < graph->nodes.size(); i++) {
        (void)node_map.emplace_back(SimpleNodeToMsNode(graph->nodes[i]));
      }
      output_nodes.reserve(graph->output_indexs.size());
      for (size_t i : graph->output_indexs) {
        (void)output_nodes.emplace_back(node_map[i]);
      }
    }
    return output_nodes;
  }

 protected:
  NodePtr EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) override {
    auto node = PynativeIRBuilder::EmitOp(prim, inputs);
    if (need_record_nodes_) {
      (void)bprop_nodes_.emplace_back(std::make_pair(node, inputs));
    }
    return node;
  }

 private:
  SimpleGraphPtr BuildBpropOpGraph(const NodePtrList &input_nodes, const NodePtrList &output_nodes) {
    std::unordered_map<NodePtr, size_t> node_map;
    SimpleGraphPtr graph = std::make_shared<SimpleGraph>();
    for (auto &parm : input_nodes) {
      node_map[parm] = graph->nodes.size();
      (void)graph->input_indexs.emplace_back(graph->nodes.size());
      (void)graph->nodes.emplace_back(std::make_shared<SimpleNode>(parm->abstract()));
    }
    for (auto &[node, inputs] : bprop_nodes_) {
      std::vector<size_t> input_indexs;
      input_indexs.reserve(inputs.size());
      for (auto &no : inputs) {
        auto it = node_map.find(no);
        if (it == node_map.end()) {
          auto value = no->get<ValueNodePtr>()->value();
          node_map[node] = graph->nodes.size();
          (void)input_indexs.emplace_back(graph->nodes.size());
          (void)graph->nodes.emplace_back(std::make_shared<SimpleNode>(value, value->ToAbstract()));
        } else {
          (void)input_indexs.emplace_back(it->second);
        }
      }
      PrimitivePtr primitive = node->isa<ValueNode>() ? prim::kPrimTupleGetItem : GetCNodePrimitive(node->get());
      node_map[node] = graph->nodes.size();
      (void)graph->nodes.emplace_back(std::make_shared<SimpleNode>(primitive, node->abstract(), input_indexs));
    }
    graph->output_indexs.reserve(output_nodes.size());
    for (auto &node : output_nodes) {
      (void)graph->output_indexs.emplace_back(node_map[node]);
    }
    return graph;
  }

  bool need_record_nodes_{false};
  std::vector<std::pair<NodePtr, NodePtrList>> bprop_nodes_;
};

void ClearBpropOpGraphMap() { PynativeIRBuilderWithCache ::bprop_op_graph_map.clear(); }

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
  static const bool cache_env = (common::GetEnv("MS_DEV_DISABLE_BPROP_CACHE") != "on");
  const auto prim = GetCNodePrimitive(cnode);
  const auto name = prim->name();
  std::shared_ptr<PynativeIRBuilder> ir_builder;
  if (cache_env) {
    ir_builder = std::make_shared<PynativeIRBuilderWithCache>(prim, cnode->func_graph(), std::make_shared<CppInfer>(),
                                                              users_, cnode->inputs().back());
  } else {
    ir_builder = std::make_shared<PynativeIRBuilder>(prim, cnode->func_graph(), std::make_shared<CppInfer>(), users_,
                                                     cnode->inputs().back());
  }
  input_nodes_.reserve(cnode->size());
  (void)std::transform(cnode->inputs().cbegin() + 1, cnode->inputs().cend(), std::back_inserter(input_nodes_),
                       [&ir_builder](const AnfNodePtr &no) { return std::make_shared<Node>(no, ir_builder.get()); });
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
  output_nodes_ = ir_builder->Build(input_nodes_, input_values, attrs, *handle);
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
