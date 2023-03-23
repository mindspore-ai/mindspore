/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "backend/graph_compiler/transform.h"

#include <algorithm>
#include <map>
#include <queue>
#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/abstract_function.h"
#include "ir/graph_utils.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/ps/ps_context.h"
#endif

namespace mindspore {
namespace compile {
using mindspore::abstract::AbstractFunction;
using mindspore::abstract::AbstractFunctionPtr;
using PrimTypePair = std::pair<PrimitivePtr, AbstractFunctionPtr>;
using MapPrimTypeFuncGraph = std::map<PrimTypePair, FuncGraphPtr>;
using TypedPrimitiveAbstractClosurePtr = std::shared_ptr<abstract::TypedPrimitiveAbstractClosure>;

const std::vector<PrimitivePtr> &GetNonlinearOps() {
  static std::vector<PrimitivePtr> nonlinear_ops = {prim::kPrimReturn, prim::kPrimPartial, prim::kPrimSwitch,
                                                    prim::kPrimMakeTuple, prim::kPrimBpropCut};
  return nonlinear_ops;
}

const std::vector<PrimitivePtr> &GetControlOps() {
  static std::vector<PrimitivePtr> control_ops = {prim::kPrimReturn, prim::kPrimPartial, prim::kPrimSwitch,
                                                  prim::kPrimMakeTuple, prim::kPrimSwitchLayer};
  return control_ops;
}

const std::vector<PrimitivePtr> &GetMsNonlinearOps() {
  static const std::vector<PrimitivePtr> ms_nonlinear_ops = {prim::kPrimReturn,   prim::kPrimPartial,
                                                             prim::kPrimSwitch,   prim::kPrimMakeTuple,
                                                             prim::kPrimBpropCut, prim::kPrimSwitchLayer};
  return ms_nonlinear_ops;
}

CompileGraph::CompileGraph(const BackendPtr &backend, const std::vector<PrimitivePtr> &cut_list) : backend_(backend) {
  MS_EXCEPTION_IF_NULL(backend_);
  lin_convert_ = backend_->convert_fn();
  if (lin_convert_ == nullptr) {
    MS_LOG(EXCEPTION) << "Attribute 'lin_convert' is null.: " << backend->name();
  }
  graph_partition_ = std::make_shared<GraphPartition>(cut_list, backend->name());
}

// Push the value node on the stack.
void CompileGraph::Push(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (slots_.count(node) > 0) {
    MS_LOG(WARNING) << "Push failed node in slots:" << node->DebugString()
                    << " NodeInfo: " << trace::GetDebugInfo(node->debug_info());
    return;
  }
  MS_LOG(DEBUG) << "Push node: " << node->DebugString(true) << " height_: " << height_
                << " is parameter: " << node->isa<Parameter>();
  slots_[node] = height_;
  set_height(height_ + 1);
}

void CompileGraph::AddInst(const Instruction &inst, const int64_t &arg) {
  VectorRef args;
  args.push_back(arg);
  AddInst(inst, args);
}

void CompileGraph::AddInst(const Instruction &inst, const ValuePtr &arg) {
  VectorRef args;
  args.push_back(arg);
  AddInst(inst, args);
}

void CompileGraph::AddInst(const Instruction &inst, const VectorRef &args) {
  inst_.push_back(std::make_pair(inst, args));
}

// Gets the stack reference for the node value. If the node is a constant,
// it may actually cause the push in to not be mentioned before.
int64_t CompileGraph::Ref(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Start Ref node " << node->DebugString(true) << " height_: " << height_;
  if (slots_.count(node) == 0 && node->isa<ValueNode>()) {
    if (IsValueNode<FuncGraph>(node)) {
      MS_LOG(DEBUG) << "Push graph.";
      AddInst(Instruction::kGraph, GetValueNode(node));
    } else {
      MS_LOG(DEBUG) << "Push.";
      if (IsValueNode<Primitive>(node)) {
        MS_LOG(EXCEPTION) << "must not be primitive in here NodeInfo: " << trace::GetDebugInfo(node->debug_info());
      } else {
        AddInst(Instruction::kPush, GetValueNode(node));
      }
    }
    Push(node);
  }
  MS_LOG(DEBUG) << "End Ref node end height_: " << height_ << ", slots: " << slots_[node]
                << ", return: " << slots_[node] - height_;
  return slots_[node] - height_;
}

// Make sure the value of node is at the top of the stack.
void CompileGraph::AddInput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (slots_.count(node) == 0) {
    MS_LOG(DEBUG) << "Input node is null " << node->DebugString(true);
    (void)Ref(node);
    return;
  }
  AddInst(Instruction::kInput, Ref(node));
  set_height(height_ + 1);
}

// Call back effect in stack
void CompileGraph::Ret(int64_t nargs) { set_height(height_ - nargs); }

void CompileGraph::PushParameters(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> parameters = graph->parameters();
  for (size_t i = parameters.size(); i != 0; i--) {
    MS_EXCEPTION_IF_NULL(parameters[i - 1]);
    Push(parameters[i - 1]);
    MS_LOG(DEBUG) << "Push parameter " << (i - 1) << ": " << parameters[i - 1]->DebugString(true);
  }
}

int64_t CompileGraph::LinConvert(const FuncGraphPtr &graph, const GraphSegmentPtr &segment, const std::string &target) {
  MS_EXCEPTION_IF_NULL(segment);
  MS_LOG(DEBUG) << "LinConvert start";
  LinConvertResult result;

  result = lin_convert_(segment, target);

  if (result.run == nullptr) {
    MS_LOG(ERROR) << "LinConvert failed";
    return RET_FAILED;
  }

  if (!(*result.run)) {
    if (result.inputs.size() != result.outputs.size()) {
      MS_EXCEPTION_IF_NULL(graph);
      MS_LOG(EXCEPTION) << "must inputs equal outputs NodeInfo: " << trace::GetDebugInfo(graph->debug_info());
    } else {
      size_t size = result.inputs.size();
      for (size_t i = 0; i < size; i++) {
        Tie(result.inputs[i], result.outputs[i]);
      }
      return RET_CONTINUE;
    }
  }
  AddExternal(result);

  return RET_SUCCESS;
}

int64_t CompileGraph::InterpretNode(const FuncGraphPtr &graph, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Interpret node: " << node->DebugString(true);
  std::vector<AnfNodePtr> node_inputs = node->inputs();
  if (node_inputs.empty()) {
    MS_LOG(EXCEPTION) << "The node->inputs() is empty";
  }
  AnfNodePtr fn = node_inputs[0];
  if (IsValueNode<Primitive>(fn)) {
    PrimitivePtr value = GetValueNode<PrimitivePtr>(fn);
    MS_LOG(DEBUG) << "The fn is primitive " << (*value).name();
    for (size_t i = node_inputs.size() - 1; i > 0; i--) {
      AddInput(node->input(i));
    }
    if (IsPrimitive(fn, prim::kPrimReturn)) {
      AddReturn(node);
      return RET_BREAK;
    }
    if (IsPrimitive(fn, prim::kPrimPartial)) {
      AddPartial(node);
    } else if (IsPrimitive(fn, prim::kPrimSwitch)) {
      AddSwitch(node);
    } else if (IsPrimitive(fn, prim::kPrimSwitchLayer)) {
      AddSwitchLayer(node);
    } else if (IsPrimitive(fn, prim::kPrimMakeTuple)) {
      AddMakeTuple(node);
    } else {
      AddPrimitive(node, value);
    }
  } else {
    int64_t ret = AddCall(graph, node);
    if (ret == RET_BREAK) {
      return ret;
    }
  }
  Push(node);
  return RET_SUCCESS;
}

bool CompileGraph::Compile(const FuncGraphPtr &graph) {
  MS_LOG(DEBUG) << "Start split graph";
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(graph_partition_);
  auto segments = graph_partition_->Partition(graph);

  MS_LOG(DEBUG) << "Split nodes size:" << segments.size();
  for (auto &segment : segments) {
    MS_EXCEPTION_IF_NULL(segment);
    int64_t ret = RET_SUCCESS;
    if (!segment->is_cut_) {
      MS_LOG(DEBUG) << "Start a extern LinConvert";
      if (!segment->nodes_.empty()) {
        std::string cur_target = GetCNodeTarget(segment->nodes_[0]);
        ret = LinConvert(graph, segment, cur_target);
      } else {
        ret = LinConvert(graph, segment);
      }
      MS_LOG(DEBUG) << "End a extern LinConvert";
      if (ret == RET_FAILED) {
        return false;
      }
      if (ret == RET_CONTINUE) {
        continue;
      }
    } else if (!segment->nodes_.empty()) {
      MS_LOG(DEBUG) << "Start a cut node";
      auto &cut_node = segment->nodes_[0];
      MS_EXCEPTION_IF_NULL(cut_node);
      if (!cut_node->isa<CNode>()) {
        MS_LOG(EXCEPTION) << "must be anfnode here NodeInfo: " << trace::GetDebugInfo(graph->debug_info());
      }
      auto node = cut_node->cast<CNodePtr>();
      ret = InterpretNode(graph, node);
      MS_LOG(DEBUG) << "End a cut node";
      if (ret == RET_BREAK) {
        break;
      }
    }
  }
  MS_LOG(DEBUG) << "End split graph";
  return true;
}

InstSet CompileGraph::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);

  Reset();
  PushParameters(graph);

  int64_t param_height = height_;
  MS_EXCEPTION_IF_NULL(graph->get_return());
  MS_LOG(DEBUG) << "'param_height': " << height_ << " to split graph: " << graph->get_return()->DebugString(true);

  if (!Compile(graph)) {
    return inst_;
  }

  AddPadStack(param_height);
  auto ret = inst_;
  Reset();
  return ret;
}

void CompileGraph::AddPadStack(int64_t param_height) {
  int64_t stack_sizes = max_height_ - param_height;
  MS_LOG(DEBUG) << "Pad stack max_height_:" << max_height_ << " param:" << param_height
                << " need_stack:" << stack_sizes;
  if (stack_sizes > 0) {
    VectorRef need_stacks({stack_sizes});
    (void)inst_.insert(inst_.cbegin(), std::make_pair(Instruction::kPadStack, need_stacks));
  }
}

void CompileGraph::AddTailCall(const AnfNodePtr &fn, size_t size) {
  VectorRef args;
  args.emplace_back(Ref(fn));
  args.emplace_back(height_);
  args.emplace_back(static_cast<int64_t>(size - 1));
  MS_LOG(DEBUG) << "Tail call:" << Ref(fn) << ", " << height_ << ", " << (size - 1);
  AddInst(Instruction::kTailCall, args);
}

void CompileGraph::AddPartial(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto inputs = node->inputs();
  VectorRef args;
  if (inputs.size() <= 1) {
    MS_LOG(EXCEPTION) << "The node:" << node->DebugString() << "do not have two input.";
  }
  auto fn = inputs[1];
  if (!IsValueNode<FuncGraph>(fn)) {
    MS_LOG(EXCEPTION) << "The type of 1st input of node must be FuncGraph, but got:" << fn->ToString();
  }
  for (size_t i = 1; i < inputs.size(); i++) {
    args.emplace_back(Ref(inputs[i]));
  }
  AddInst(Instruction::kPartial, args);
}

void CompileGraph::AddMakeTuple(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto inputs = node->inputs();
  VectorRef args;
  for (size_t i = 1; i < inputs.size(); i++) {
    args.emplace_back(Ref(inputs[i]));
  }
  AddInst(Instruction::kTuple, args);
}

void CompileGraph::AddSwitch(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto inputs = node->inputs();
  if (inputs.size() < kSwitchInputSize) {
    MS_LOG(EXCEPTION) << "Length of inputs of primitive " << prim::kPrimSwitch->name() << " is less than 4";
  }
  VectorRef args;
  args.emplace_back(Ref(inputs[kPartialGraphIndex]));
  args.emplace_back(Ref(inputs[kSwitchTrueBranchIndex]));
  args.emplace_back(Ref(inputs[kSwitchFalseBranchIndex]));
  AddInst(Instruction::kSwitch, args);
}

void CompileGraph::AddSwitchLayer(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto inputs = node->inputs();
  if (inputs.size() != kSwitchLayerInputSize) {
    MS_LOG(EXCEPTION) << "Switch layer must have index and branches.";
  }
  VectorRef args;
  const size_t cond_index = 1;
  const size_t tuple_index = 2;
  args.emplace_back(Ref(inputs[cond_index]));
  args.emplace_back(Ref(inputs[tuple_index]));
  AddInst(Instruction::kSwitchLayer, args);
}

void CompileGraph::AddReturn(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  VectorRef args;
  if (node->inputs().size() <= 1) {
    MS_LOG(EXCEPTION) << "The node:" << node->DebugString() << "do not have two input.";
  }
  args.emplace_back(Ref(node->input(1)));
  args.emplace_back(height_);
  AddInst(Instruction::kReturn, args);
}

void CompileGraph::AddPrimitive(const CNodePtr &node, const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(node);
  auto inputs = node->inputs();
  VectorRef args;
  args.push_back(prim);
  for (size_t i = 1; i < inputs.size(); i++) {
    args.emplace_back(Ref(inputs[i]));
  }
  AddInst(Instruction::kPrim, args);
}

int64_t CompileGraph::AddCall(const FuncGraphPtr &graph, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto inputs = node->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "The node->inputs() is empty.";
  }
  AnfNodePtr fn = inputs[0];
  (void)Ref(fn);
  size_t size = inputs.size();
  for (size_t i = size - 1; i > 0; i--) {
    AddInput(inputs[i]);
  }
  if (node == graph->output()) {
    AddTailCall(fn, size);
    return RET_BREAK;
  }
  MS_LOG(DEBUG) << "Call:" << Ref(fn) << ", " << height_ << ", " << (size - 1);
  AddInst(Instruction::kCall, Ref(fn));
  Ret(static_cast<int64_t>(size - 1));

  for (size_t i = size - 1; i > 0; i--) {
    const auto iter = slots_.find(inputs[i]);
    if (iter != slots_.end() && iter->second >= height_) {
      (void)slots_.erase(inputs[i]);
    }
  }
  return RET_SUCCESS;
}

void CompileGraph::AddExternal(const LinConvertResult &result) {
  VectorRef args;
  args.push_back(result.run);
  args.push_back(result.simu_run);
  size_t size = result.inputs.size();
  for (size_t i = 0; i < size; i++) {
    args.emplace_back(Ref(result.inputs[i]));
  }
  AddInst(Instruction::kExternal, args);
  for (auto &out : result.outputs) {
    Push(out);
  }
}

void TraverseGraphMap(
  const FuncGraphManagerPtr &manager_ptr, FuncGraphTransaction *tr, const FuncGraphSet &fgs,
  const std::function<std::shared_ptr<FuncGraph>(const PrimitivePtr, const AbstractFunctionPtr)> &get_prim_graph) {
  MS_EXCEPTION_IF_NULL(manager_ptr);
  MS_EXCEPTION_IF_NULL(tr);
  for (const auto &fg : fgs) {
    MS_EXCEPTION_IF_NULL(fg);
    for (const auto &ct_any : fg->value_nodes()) {
      AnfNodePtr const_primitive_node = ct_any.first;
      if (const_primitive_node != nullptr && IsValueNode<Primitive>(const_primitive_node)) {
        auto users = manager_ptr->node_users()[const_primitive_node];
        for (auto &use : users) {
          CNodePtr node = use.first->cast<CNodePtr>();
          MS_EXCEPTION_IF_NULL(node);
          if (node->func_graph() != fg) {
            continue;
          }
          int64_t key = use.second;
          if (key != 0) {
            MS_EXCEPTION_IF_NULL(node->input(0));
            bool key_is_const = node->input(0)->isa<ValueNode>();
            PrimitivePtr value = GetValueNode<PrimitivePtr>(node->input(0));
            if (value != nullptr) {
              bool is_prim_array_map = !(prim::kPrimArrayMap->name().compare(value->name()));
              bool is_prim_array_reduce = !(prim::kPrimArrayReduce->name().compare(value->name()));
              if (key == 1 && key_is_const && (is_prim_array_map || is_prim_array_reduce)) {
                continue;
              }
            }
            FuncGraphPtr g = get_prim_graph(GetValueNode<PrimitivePtr>(const_primitive_node),
                                            dyn_cast<AbstractFunction>(const_primitive_node->abstract()));
            tr->SetEdge(node, key, NewValueNode(g));
          }
        }
      }
    }
  }
}

FuncGraphPtr WrapPrimitives(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  FuncGraphManagerPtr manager_ptr = graph->manager();
  MS_EXCEPTION_IF_NULL(manager_ptr);
  MapPrimTypeFuncGraph prim_graphs;
  const auto &get_prim_graph = [&prim_graphs](const PrimitivePtr &prim, const AbstractFunctionPtr &type) {
    PrimTypePair prim_type = std::make_pair(prim, type);
    if (prim_graphs.end() == prim_graphs.find(prim_type)) {
      FuncGraphPtr g = std::make_shared<FuncGraph>();
      std::vector<AnfNodePtr> args;
      ValueNodePtr prim_ct = NewValueNode(prim);
      MS_EXCEPTION_IF_NULL(prim_ct);
      prim_ct->set_abstract(type);
      args.push_back(prim_ct);
      MS_EXCEPTION_IF_NULL(type);
      TypedPrimitiveAbstractClosurePtr tp = dyn_cast<abstract::TypedPrimitiveAbstractClosure>(type->GetUnique());
      MS_EXCEPTION_IF_NULL(tp);
      MS_EXCEPTION_IF_NULL(g);
      for (auto t : tp->args_spec_list()) {
        ParameterPtr p = g->add_parameter();
        p->set_abstract(t);
        args.push_back(p);
      }
      AnfNodePtr out = g->NewCNode(args);
      out->set_abstract(tp->output());
      g->set_output(out);
      prim_graphs[prim_type] = g;
    }

    return prim_graphs[prim_type];
  };

  FuncGraphTransaction tr = manager_ptr->Transact();
  auto &fgs = manager_ptr->func_graphs();
  TraverseGraphMap(manager_ptr, &tr, fgs, get_prim_graph);
  tr.Commit();

  return graph;
}

CompileGraphs::CompileGraphs(const BackendPtr &backend, const std::vector<PrimitivePtr> &cut_list) : backend_(backend) {
  MS_EXCEPTION_IF_NULL(backend);
  MS_LOG(DEBUG) << "Start vm: " << backend->name();
  transform_ = std::make_shared<CompileGraph>(backend, cut_list);
  Reset();
}

// Convert graphs to unlinked instructions.
void CompileGraphs::Compile(const FuncGraphPtr &graph) {
  MS_LOG(DEBUG) << "Start";
  mapping_[graph] = static_cast<int64_t>(insts_.size());
  if (transform_ != nullptr) {
    InstSet insts = transform_->Run(graph);
    if (!insts.empty()) {
      (void)insts_.insert(insts_.cend(), insts.cbegin(), insts.cend());
    }
  }
  MS_LOG(DEBUG) << "End";
}

// Link instructions from multiple function graphs together.
FinalVMPtr CompileGraphs::Link() {
  MS_LOG(DEBUG) << "Start";
  for (std::size_t i = 0; i < insts_.size(); i++) {
    InstType inst = insts_[i];
    MS_LOG(DEBUG) << "Link point:" << inst_str[inst.first];
    if (Instruction::kGraph == inst.first) {
      if (inst.second.empty()) {
        MS_LOG(EXCEPTION) << "The second element of inst is empty";
      }
      FuncGraphPtr func_graph = utils::cast<ValuePtr>(inst.second[0])->cast<FuncGraphPtr>();
      MS_LOG(DEBUG) << "Link graph:" << func_graph->ToString();
      insts_[i] = std::make_pair(Instruction::kPush, VectorRef(std::vector<BaseRef>{mapping_[func_graph]}));
    }
  }

  FinalVMPtr rt = std::make_shared<FinalVM>(insts_, backend_);
  MS_LOG(DEBUG) << "End";
  return rt;
}

// Convert all graphs to unlinked instructions and link them.
FinalVMPtr CompileGraphs::CompileAndLink(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(DEBUG) << "Start";
  Reset();
  MS_LOG(DEBUG) << "Begin parameter:" << graph->parameters().size();

  FuncGraphPtr prim_graph = WrapPrimitives(graph);
  Compile(prim_graph);
  MS_EXCEPTION_IF_NULL(prim_graph);
  MS_EXCEPTION_IF_NULL(prim_graph->manager());
  FuncGraphSet graphs = prim_graph->manager()->func_graphs();
  for (const auto &g : graphs) {
    if (g != graph && g != nullptr) {
      Compile(g);
    }
  }

  FinalVMPtr rt = Link();
  Reset();
  MS_LOG(DEBUG) << "End";
  return rt;
}

BackendPtr CreateBackend() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string name = context_ptr->backend_policy();
  MS_LOG(INFO) << "CreateBackend is: " << name;
  context_ptr->Refresh();

  if (name == kMsConvert || name == kGeVm) {
    std::string target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    BackendPtr backend = nullptr;
    // Create MindRTBackend or MsBackend according to whether mindrt is used.
    if (context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
      backend = std::make_shared<MindRTBackend>(name, target, device_id);
    } else {
      backend = std::make_shared<MsBackend>(name, target, device_id);
    }
    if (target == kAscendDevice && context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
      backend->set_is_multi_graph_sink(false);
    }
    return backend;
  }

  return std::make_shared<Backend>(name);
}

void SetMindRTEnable() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);

#if defined(__linux__) && defined(WITH_BACKEND)
  if (ps::PSContext::instance()->is_ps_mode() && !ps::PSContext::instance()->enable_distributed_mindrt()) {
    context_ptr->set_param<bool>(MS_CTX_ENABLE_MINDRT, false);
    return;
  }
#endif

  std::string target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (common::GetEnv("DISABLE_ASCEND_MINDRT") == "1" && target == kAscendDevice) {
    context_ptr->set_param<bool>(MS_CTX_ENABLE_MINDRT, false);
    return;
  }

  MS_LOG(DEBUG) << "Enable mindRT.";
  context_ptr->set_param<bool>(MS_CTX_ENABLE_MINDRT, true);
}
}  // namespace compile
}  // namespace mindspore
