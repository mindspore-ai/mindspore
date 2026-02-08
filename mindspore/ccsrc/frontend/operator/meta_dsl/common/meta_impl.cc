/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/common/meta_impl.h"
#include <algorithm>
#include <utility>
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ops/op_def.h"
#include "abstract/abstract_value.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "include/utils/python_adapter.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/cc_implementations.h"
#include "frontend/jit/ps/parse/resolve.h"
#include "frontend/jit/ps/parse/parse_base.h"
#include "utils/compile_config.h"
#include "primitive/framework_ops.h"
#include "primitive/structure_ops.h"
#include "primitive/sequence_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "ir/func_graph_flag.h"

namespace mindspore::prim {
namespace {
ValuePtr GetPyValueWithCache(const std::string &module_name, const std::string &op_name) {
  static std::map<std::string, ValuePtr> py_value_map;
  auto full_name = module_name + "." + op_name;
  auto iter = py_value_map.find(full_name);
  if (iter != py_value_map.end()) {
    return iter->second;
  }
  py::gil_scoped_acquire gil;
  auto value = prim::GetPythonOps(op_name, module_name);
  MS_EXCEPTION_IF_NULL(value);
  py_value_map[full_name] = value;
  return value;
}

ValueNodePtr GetMultitypeOps(const std::string &op_name) {
  return NewValueNode(GetPyValueWithCache("mindspore.ops.composite.multitype_ops", op_name));
}

ValuePtr CreateCalssType(const std::string &module_name, const std::string &op_name) {
  py::object obj = python_adapter::GetPyFn(module_name, op_name);
  return std::make_shared<parse::ClassType>(obj, op_name);
}

ValuePtr GetClassTypeValue(const TypeId &type) {
  py::gil_scoped_acquire gil;
  switch (type) {
    case TypeId::kObjectTypeTensorType:
      return CreateCalssType("mindspore.common.tensor", "Tensor");
    case TypeId::kNumberTypeInt:
      return CreateCalssType("builtins", "int");
    case TypeId::kNumberTypeFloat:
      return CreateCalssType("builtins", "float");
    case TypeId::kNumberTypeBool:
      return CreateCalssType("builtins", "bool");
    case TypeId::kObjectTypeTuple:
      return CreateCalssType("builtins", "tuple");
    case TypeId::kObjectTypeList:
      return CreateCalssType("builtins", "list");
    case TypeId::kObjectTypeString:
      return CreateCalssType("builtins", "str");
    case TypeId::kNumberTypeComplex:
      return CreateCalssType("builtins", "complex");
    case TypeId::kObjectTypeNumber:
      return CreateCalssType("numbers", "Number");
    default:
      MS_LOG(INTERNAL_EXCEPTION) << "Unsupported TypeId '" << TypeIdToString(type) << "'.";
  }
}
}  // namespace

FuncGraphPtr MetaImpl::GenerateFuncGraph(const AbstractBasePtrList &input_args) {
  input_args_ = input_args;
  CheckInputs(input_args);
  BeginFunc("total");
  GenerateFunction();
  return EndFunc();
}

void MetaImpl::set_prim(const PrimitivePtr &prim) { prim_ = prim; }

PrimitivePtr MetaImpl::prim() const { return prim_; }

void MetaImpl::set_manager(const FuncGraphManagerPtr &manager) { manager_ = manager; }

void MetaImpl::CheckInputs(const AbstractBasePtrList &input_args) const {
  if (prim_ == nullptr) {
    return;
  }
  // Check inputs' number.
  const auto &prim_name = prim_->name();
  const auto &op_def = ops::GetOpDef(prim_name);
  if (op_def == nullptr) {
    return;
  }
  auto args_size = op_def->args_.size();
  if (input_args.size() != args_size) {
    MS_LOG(EXCEPTION) << "Operator[" << prim_name << "] requires " << args_size << " arguments, but got "
                      << input_args.size() << ".";
  }
  // Check inputs' abstract.
  const auto &check_func = RegMetaImplFactory::GetInstance().GetCheckFunc(prim_name);
  if (check_func != nullptr) {
    check_func(prim_, input_args);
  }
}

void MetaImpl::BeginFunc(const std::string &func_name) {
  auto builder = std::make_shared<MetaFuncBuilder>(name_ + "_" + func_name);
  builder->BeginFunc();
  func_builder_stack_.push(builder);
}

FuncGraphPtr MetaImpl::EndFunc() {
  if (func_builder_stack_.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "func_builder_stack_ is empty when trying to pop MetaFuncBuilder.";
  }
  auto graph = func_builder_stack_.top()->EndFunc();
  func_builder_stack_.pop();
  // Add graph to manager.
  if (manager_ != nullptr) {
    manager_->AddFuncGraph(graph);
  }
  // Process top graph.
  if (func_builder_stack_.empty()) {
    // Define custom bprop for current op.
    DefineCustomBprop(graph);
    // Dump IR for top graph.
    DumpIRForMetaDsl(graph);
  }
  return graph;
}

void MetaImpl::DefineCustomBprop(const FuncGraphPtr &graph) {
  bprop_graph_ = RegMetaImplFactory::GetInstance().GetBprop(prim_);
  if (bprop_graph_ != nullptr) {
    // Associate bprop to graph.
    MS_LOG(DEBUG) << "Define custom bprop for " << name_ << ": " << bprop_graph_->ToString();
    (void)graph->transforms().emplace(parse::CUSTOM_BPROP_NAME, FuncGraphTransform(bprop_graph_));
    (void)bprop_graph_->transforms().emplace("primal", FuncGraphTransform(graph));
    graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
    graph->set_flag(FUNC_GRAPH_FLAG_PRIMAL_OF_BPROP, true);
    // Add bprop_graph to manager.
    if (manager_ != nullptr) {
      manager_->AddFuncGraph(bprop_graph_);
    }
  }
}

void MetaImpl::DumpIRForMetaDsl(const FuncGraphPtr &graph) const {
  std::string str(common::GetCompileConfig("DUMP_IR_META_DSL"));
  if (!str.empty() && str == name_) {
    std::string file_name = "MetaDSL_" + name_ + ".ir";
    DumpIR(file_name, graph);
  }
}

void MetaImpl::Return(const NodePtr &output) { func_builder_stack_.top()->SetOutput(output); }

NodePtr MetaImpl::NewParam(const std::string &name, int index) {
  auto param = func_builder_stack_.top()->AddParameter(name);
  // Set abstract for parameters of top graph.
  if (index > 0) {
    param->set_abstract(input_args_[index - 1]);
  }
  return param;
}

void MetaImpl::ConvertTypeIdToType(NodePtrList *nodes) {
  constexpr size_t index_prim = 0;
  auto do_trans = GetValueNode<prim::DoTransPrimitiveFunctionPtr>(nodes->at(index_prim));
  if (do_trans == nullptr) {
    return;
  }
  const auto &op_name = do_trans->function()->name();
  const auto &op_def = mindspore::ops::GetOpDef(op_name);
  MS_EXCEPTION_IF_NULL(op_def);
  const auto &op_args = op_def->args_;
  if (op_args.size() != nodes->size() - 1) {
    MS_LOG(INTERNAL_EXCEPTION) << "Operator[" << op_name << "] expects " << op_args.size() << " arguments, but got "
                               << nodes->size() - 1;
  }
  for (size_t i = 0; i < op_args.size(); ++i) {
    const auto &op_arg = op_args[i];
    if (!op_arg.arg_handler_.empty() && op_arg.arg_handler_ == "dtype_to_type_id") {
      NodePtrList convert_node_list = {NewValueNode(prim::kPrimEnumToDtype), (*nodes)[i + 1]};
      (*nodes)[i + 1] = func_builder_stack_.top()->CreateNode(convert_node_list);
    }
  }
}

NodePtr MetaImpl::NewNode(const NodePtrList &nodes) {
  NodePtrList node_list = nodes;
  ConvertTypeIdToType(&node_list);
  return func_builder_stack_.top()->CreateNode(node_list);
}

FuncGraphPtr MetaImpl::BuildSubFunction(const std::string &func_name, const BlockFunc &sub_func) {
  BeginFunc(func_name);
  sub_func();
  return EndFunc();
}

NodePtr MetaImpl::IfCond(const NodePtr &condition, const BlockFunc &true_branch, const BlockFunc &false_branch,
                         const NodePtrList &args) {
  auto true_branch_graph = BuildSubFunction("ifexp_true", true_branch);
  auto false_branch_graph = BuildSubFunction("ifexp_false", false_branch);
  auto bool_condition = NewNode({{NewValueNode(prim::kPrimCond), condition, NewValueNode(MakeValue(false))}});
  auto switch_node = NewNode({NewValueNode(prim::kPrimSwitch), bool_condition, NewValueNode(true_branch_graph),
                              NewValueNode(false_branch_graph)});
  NodePtrList node_list{switch_node};
  (void)std::copy(args.begin(), args.end(), std::back_inserter(node_list));
  return NewNode(node_list);
}

NodePtr MetaImpl::If(const NodePtr &condition, const BlockFunc &true_branch, const BlockFunc &false_branch) {
  return IfCond(condition, true_branch, false_branch, {});
}

NodePtr MetaImpl::If(const std::vector<std::pair<NodePtr, BlockFunc>> &if_branches, const BlockFunc &else_branch) {
  MS_EXCEPTION_IF_CHECK_FAIL(!if_branches.empty(), "if_branches should not be empty.");
  return IfBranchesInner(if_branches, else_branch, 0);
}

NodePtr MetaImpl::IfBranchesInner(const std::vector<std::pair<NodePtr, BlockFunc>> &if_branches,
                                  const BlockFunc &else_branch, size_t index) {
  constexpr auto first_index = 0;
  const auto &[condition, true_branch] = if_branches[first_index];
  if (index == if_branches.size() - 1) {
    return IfCond(condition, true_branch, else_branch, {});
  }
  auto false_branch = [&]() { Return(IfBranchesInner(if_branches, else_branch, index + 1)); };
  return IfCond(condition, true_branch, false_branch, {});
}

namespace {
FuncGraphPtr BuildForBodyGraph(const FuncGraphPtr &loop_func_graph, const FuncGraphPtr &for_iter_graph) {
  /* def body_func(sequence, result, index, len):
   *   result = loop_func(index, sequence[index], result)
   *   index = index + 1
   *   return for_impl(sequence, result, index, len) */
  auto body_graph = std::make_shared<FuncGraph>();
  auto sequence_param = body_graph->add_parameter();
  auto result_param = body_graph->add_parameter();
  auto index_param = body_graph->add_parameter();
  auto len_param = body_graph->add_parameter();
  auto item = body_graph->NewCNodeInOrder({GetMultitypeOps("getitem"), sequence_param, index_param});
  auto new_result = body_graph->NewCNodeInOrder({NewValueNode(loop_func_graph), index_param, item, result_param});
  auto new_index = body_graph->NewCNodeInOrder(
    {NewValueNode(prim::kPrimScalarAdd), index_param, NewValueNode(static_cast<int64_t>(1))});
  auto output =
    body_graph->NewCNodeInOrder({NewValueNode(for_iter_graph), sequence_param, new_result, new_index, len_param});
  body_graph->set_output(output);
  return body_graph;
}

FuncGraphPtr BuildForReturnGraph() {
  /* def return_func(sequence, result, index, len):
   *   return result */
  auto return_graph = std::make_shared<FuncGraph>();
  return_graph->debug_info()->set_name("for_return");
  (void)return_graph->add_parameter();
  auto result_param = return_graph->add_parameter();
  (void)return_graph->add_parameter();
  (void)return_graph->add_parameter();
  return_graph->set_output(result_param);
  return return_graph;
}

FuncGraphPtr BuildForIterGraph(const FuncGraphPtr &graph, const FuncGraphPtr &body_func_graph,
                               const FuncGraphPtr &return_func_graph) {
  /* def for_iter(sequence, result, index, len):
   *   return index < len ? body_func(...) : return_func(...) */
  graph->debug_info()->set_name("for_iter");
  auto sequence_param = graph->add_parameter();
  auto result_param = graph->add_parameter();
  auto index_param = graph->add_parameter();
  auto len_param = graph->add_parameter();
  auto compare_node = graph->NewCNodeInOrder({NewValueNode(prim::kPrimScalarLt), index_param, len_param});
  auto cond_node = graph->NewCNodeInOrder({NewValueNode(prim::kPrimCond), compare_node, NewValueNode(MakeValue(true))});
  auto switch_node = graph->NewCNodeInOrder(
    {NewValueNode(prim::kPrimSwitch), cond_node, NewValueNode(body_func_graph), NewValueNode(return_func_graph)});
  auto output_node = graph->NewCNodeInOrder({switch_node, sequence_param, result_param, index_param, len_param});
  graph->set_output(output_node);
  return graph;
}
}  // namespace

NodePtr MetaImpl::For(const std::function<void(const NodePtr &, const NodePtr &, const NodePtr &)> &loop_func,
                      const NodePtr &sequence, const NodePtr &result, const NodePtr &lower, const NodePtr &upper) {
  // Define for_iter(sequence, result, index, len)
  auto for_iter_graph = std::make_shared<FuncGraph>();
  for_iter_graph->set_manager(manager_);
  // Define loop_func(index, item, result)
  BeginFunc("loop_func");
  auto param_loop_index = NewParam("index");
  auto param_loop_item = NewParam("item");
  auto param_loop_result = NewParam("result");
  loop_func(param_loop_index, param_loop_item, param_loop_result);
  auto loop_func_graph = EndFunc();
  // Build body graph.
  auto body_func_graph = BuildForBodyGraph(loop_func_graph, for_iter_graph);
  body_func_graph->set_manager(manager_);
  // Build return graph.
  auto return_func_graph = BuildForReturnGraph();
  return_func_graph->set_manager(manager_);
  // Build for_iter graph.
  BuildForIterGraph(for_iter_graph, body_func_graph, return_func_graph);
  // Call for_iter(sequence, result, lower=0, upper=len(sequence))
  auto new_lower = lower != nullptr ? lower : NewValueNode(static_cast<int64_t>(0));
  auto new_upper = upper != nullptr ? upper : Len(sequence);
  NodePtrList node_list{NewValueNode(for_iter_graph), sequence, result, new_lower, new_upper};
  return NewNode(node_list);
}

NodePtr MetaImpl::ForiLoop(const NodePtr &lower, const NodePtr &upper,
                           const std::function<void(const NodePtr &, const NodePtr &)> &loop_func,
                           const NodePtr &init_val) {
  // Build graph for loop body.
  BeginFunc("ForiLoop");
  auto param_index = NewParam("index");
  auto param_value = NewParam("value");
  loop_func(param_index, param_value);
  auto loop_graph = EndFunc();
  // Create node with ForiLoop.
  NodePtrList node_list{NewValueNode(prim::kPrimForiLoop), lower, upper, NewValueNode(loop_graph), init_val};
  return NewNode(node_list);
}

NodePtr MetaImpl::While(const std::function<void(const NodePtr &)> &cond_func,
                        const std::function<void(const NodePtr &)> &loop_func, const NodePtr &init_val) {
  // Build graph for condition.
  BeginFunc("while_cond");
  auto param_cond = NewParam("cond");
  cond_func(param_cond);
  auto cond_graph = EndFunc();
  // Build graph for loop body.
  BeginFunc("while_loop");
  auto param_loop = NewParam("loop");
  loop_func(param_loop);
  auto loop_graph = EndFunc();
  // Create node with WhileLoop.
  NodePtrList node_list{NewValueNode(prim::kPrimWhileLoop), NewValueNode(cond_graph), NewValueNode(loop_graph),
                        init_val};
  return NewNode(node_list);
}

NodePtr MetaImpl::Scan(const std::function<void(const NodePtr &, const NodePtr &)> &loop_func, const NodePtr &init,
                       const NodePtr &xs, const NodePtr &length) {
  // Build graph for loop body.
  BeginFunc("scan_loop");
  auto param_result = NewParam("result");
  auto param_elem = NewParam("elem");
  loop_func(param_result, param_elem);
  auto loop_graph = EndFunc();
  // Create node with Scan.
  NodePtrList node_list{NewValueNode(prim::kPrimScan), NewValueNode(loop_graph), init, xs, length};
  return NewNode(node_list);
}

NodePtr MetaImpl::MakeTuple(const std::vector<NodePtr> &nodes) {
  auto cnode_inputs(nodes);
  cnode_inputs.insert(cnode_inputs.begin(), NewValueNode(prim::kPrimMakeTuple));
  return NewNode(cnode_inputs);
}

NodePtr MetaImpl::ListToTuple(const NodePtr &node) { return NewNode({NewValueNode(prim::kPrimListToTuple), node}); }

NodePtr MetaImpl::SequenceLen(const NodePtr &node) { return NewNode({NewValueNode(prim::kPrimSequenceLen), node}); }

NodePtr MetaImpl::ZerosLike(const NodePtr &x) { return NewNode({GetMultitypeOps("zeros_like"), x}); }

NodePtr MetaImpl::OnesLike(const NodePtr &x) { return NewNode({GetMultitypeOps("ones_like"), x}); }

NodePtr MetaImpl::Equal(const NodePtr &x, const NodePtr &y) { return NewNode({GetMultitypeOps("equal"), x, y}); }

NodePtr MetaImpl::NotEqual(const NodePtr &x, const NodePtr &y) { return NewNode({GetMultitypeOps("not_equal"), x, y}); }

NodePtr MetaImpl::Greater(const NodePtr &x, const NodePtr &y) { return NewNode({GetMultitypeOps("greater"), x, y}); }

NodePtr MetaImpl::Less(const NodePtr &x, const NodePtr &y) { return NewNode({GetMultitypeOps("less"), x, y}); }

NodePtr MetaImpl::GreaterEqual(const NodePtr &x, const NodePtr &y) {
  return NewNode({GetMultitypeOps("greater_equal"), x, y});
}

NodePtr MetaImpl::LessEqual(const NodePtr &x, const NodePtr &y) {
  return NewNode({GetMultitypeOps("less_equal"), x, y});
}

NodePtr MetaImpl::GetItem(const NodePtr &x, const NodePtr &y) { return NewNode({GetMultitypeOps("getitem"), x, y}); }

NodePtr MetaImpl::SetItem(const NodePtr &x, const NodePtr &y, const NodePtr &z) {
  return NewNode({GetMultitypeOps("setitem"), x, y, z});
}

NodePtr MetaImpl::IsNone(const NodePtr &node) {
  return NewNode({NewValueNode(prim::kPrimIs_), node, NewValueNode(kNone)});
}

NodePtr MetaImpl::IsNotNone(const NodePtr &node) {
  return NewNode({NewValueNode(prim::kPrimIsNot), node, NewValueNode(kNone)});
}

NodePtr MetaImpl::And(const NodePtr &x, const NodePtr &y) {
  auto true_branch = [&]() {
    (void)NewParam("x");
    auto param_y = NewParam("y");
    Return(param_y);
  };
  auto false_branch = [&]() {
    auto param_x = NewParam("x");
    (void)NewParam("y");
    Return(param_x);
  };
  return IfCond(x, true_branch, false_branch, {x, y});
}

NodePtr MetaImpl::Or(const NodePtr &x, const NodePtr &y) {
  auto true_branch = [&]() {
    auto param_x = NewParam("x");
    (void)NewParam("y");
    Return(param_x);
  };
  auto false_branch = [&]() {
    (void)NewParam("x");
    auto param_y = NewParam("y");
    Return(param_y);
  };
  return IfCond(x, true_branch, false_branch, {x, y});
}

NodePtr MetaImpl::Len(const NodePtr &x) {
  auto len_func = NewNode({NewValueNode(prim::kPrimGetAttr), x, NewValueNode(MakeValue("__len__"))});
  return NewNode({len_func});
}

NodePtr MetaImpl::ImplAllAny(const NodePtr &input, bool is_all) {
  constexpr int idx_zero = 0;
  constexpr int idx_first = 1;
  constexpr int idx_second = 2;
  auto cond_func = [&](const NodePtr &val) {
    auto index = GetItem(val, Value(idx_zero));
    auto result = GetItem(val, Value(idx_first));
    auto iterable = GetItem(val, Value(idx_second));
    auto index_valid = NewNode({NewValueNode(prim::kPrimScalarLt), index, Len(iterable)});
    // All: index < len(iterable) and result. Any: index < len(iterable) and not result.
    auto check = is_all ? result : Not(result);
    Return(And(index_valid, check));
  };
  auto loop_func = [&](const NodePtr &val) {
    auto index = GetItem(val, Value(idx_zero));
    auto result = GetItem(val, Value(idx_first));
    auto iterable = GetItem(val, Value(idx_second));
    // Example code: bool(iterable[index])
    auto new_res = NewNode({NewValueNode(prim::kPrimCond), GetItem(iterable, index), NewValueNode(MakeValue(false))});
    // Example code: (index + 1, new_res, iterable)
    auto new_index = NewNode({NewValueNode(prim::kPrimScalarAdd), index, Value(idx_first)});
    Return(Tuple(new_index, new_res, iterable));
  };
  auto default_value = is_all ? Value(true) : Value(false);
  auto init_val = Tuple(Value(idx_zero), default_value, input);
  auto res = While(cond_func, loop_func, init_val);
  return GetItem(res, Value(idx_first));
}

NodePtr MetaImpl::All(const NodePtr &iterable) { return ImplAllAny(iterable, true); }

NodePtr MetaImpl::Any(const NodePtr &iterable) { return ImplAllAny(iterable, false); }

NodePtr MetaImpl::ScalarAdd(const NodePtr &x, const NodePtr &y) {
  return NewNode({NewValueNode(prim::kPrimScalarAdd), x, y});
}

NodePtr MetaImpl::ScalarSub(const NodePtr &x, const NodePtr &y) {
  return NewNode({NewValueNode(prim::kPrimScalarSub), x, y});
}

NodePtr MetaImpl::ScalarMul(const NodePtr &x, const NodePtr &y) {
  return NewNode({NewValueNode(prim::kPrimScalarMul), x, y});
}

NodePtr MetaImpl::ScalarDiv(const NodePtr &x, const NodePtr &y) {
  return NewNode({NewValueNode(prim::kPrimScalarDiv), x, y});
}

NodePtr MetaImpl::ScalarFloorDiv(const NodePtr &x, const NodePtr &y) {
  return NewNode({NewValueNode(prim::kPrimScalarFloorDiv), x, y});
}

NodePtr MetaImpl::ScalarMod(const NodePtr &x, const NodePtr &y) {
  return NewNode({NewValueNode(prim::kPrimScalarMod), x, y});
}

NodePtr MetaImpl::ScalarPow(const NodePtr &x, const NodePtr &y) {
  return NewNode({NewValueNode(prim::kPrimScalarPow), x, y});
}

NodePtr MetaImpl::Shape(const NodePtr &x) { return NewNode({NewValueNode(prim::kPrimShape), x}); }

NodePtr MetaImpl::Rank(const NodePtr &x) { return NewNode({NewValueNode(prim::kPrimRank), x}); }

NodePtr MetaImpl::DTypeId(const NodePtr &x) { return NewNode({NewValueNode(prim::kPrimDTypeId), x}); }

NodePtr MetaImpl::Reshape(const NodePtr &x, const NodePtr &shape) {
  return NewNode({NewValueNode(prim::kPrimReshape), x, shape});
}

NodePtr MetaImpl::Not(const NodePtr &x) { return NewNode({GetMultitypeOps("logical_not"), x}); }

NodePtr MetaImpl::Raise(const std::string &exception_type, const std::string &exception_msg) {
  auto node = NewNode(
    {NewValueNode(prim::kPrimRaise), NewValueNode(exception_type), NewValueNode(exception_msg), NewValueNode("None")});
  node->func_graph()->set_is_tensor_condition_branch(true);
  return node;
}

NodePtr MetaImpl::IsInstance(const NodePtr &x, const TypeId &type) {
  auto class_type = NewValueNode(GetClassTypeValue(type));
  return NewNode({NewValueNode(prim::kPrimIsInstance), x, class_type});
}

NodePtr MetaImpl::IsInstance(const NodePtr &x, const std::vector<TypeId> &types) {
  NodePtrList class_type_list{NewValueNode(prim::kPrimMakeTuple)};
  (void)std::transform(types.begin(), types.end(), std::back_inserter(class_type_list),
                       [](const auto &type) { return NewValueNode(GetClassTypeValue(type)); });
  auto class_type_list_node = NewNode(class_type_list);
  return NewNode({NewValueNode(prim::kPrimIsInstance), x, class_type_list_node});
}

RegMetaImplFactory &RegMetaImplFactory::GetInstance() {
  static RegMetaImplFactory instance{};
  return instance;
}

bool RegMetaImplFactory::IsMetaImpl(const std::string &name) { return registry_.find(name) != registry_.end(); }

void RegMetaImplFactory::AddMetaImpl(const std::string &name, const CreateFunc &creator) {
  (void)registry_.emplace(name, creator);
}

MetaImplPtr RegMetaImplFactory::CreateMetaImpl(const std::string &name) {
  const auto &it = registry_.find(name);
  if (it == registry_.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Failed to create MetaImpl: " << name;
  }
  return it->second();
}

void RegMetaImplFactory::RegBprop(const PrimitivePtr &prim, const CreateFunc &creator) {
  (void)bprop_map_.emplace(prim->name(), creator);
}

FuncGraphPtr RegMetaImplFactory::GetBprop(const PrimitivePtr &prim) {
  if (prim == nullptr) {
    return nullptr;
  }
  const auto &prim_name = prim->name();
  const auto &it = bprop_map_.find(prim_name);
  if (it == bprop_map_.end()) {
    return nullptr;
  }
  auto bprop_meta_impl = it->second();
  MS_LOG(DEBUG) << "Get bprop " << bprop_meta_impl->ToString() << " for Operator[" << prim_name << "].";
  // Implement bprop graph.
  auto bprop_graph = std::make_shared<FuncGraph>();
  bprop_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  MS_EXCEPTION_IF_NULL(bprop_graph->debug_info());
  bprop_graph->debug_info()->set_name(prim_name + "_" + parse::CUSTOM_BPROP_NAME);
  constexpr auto extend_size = 2;
  const auto &op_def = ops::GetOpDef(prim_name);
  MS_EXCEPTION_IF_NULL(op_def);
  auto args_size = op_def->args_.size();
  auto params_size = args_size + extend_size;
  AnfNodePtrList inputs{NewValueNode(bprop_meta_impl)};
  for (size_t i = 0; i < params_size; ++i) {
    (void)inputs.emplace_back(bprop_graph->add_parameter());
  }
  CNodePtr cnode = bprop_graph->NewCNodeInOrder(inputs);
  bprop_graph->set_output(cnode);
  if (GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_BACKPROP)) {
    bprop_graph->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
  }
  return bprop_graph;
}

void RegMetaImplFactory::RegCheckFunc(const std::string &name, const CheckFunc &check_func) {
  (void)check_func_map_.emplace(name, check_func);
}

CheckFunc RegMetaImplFactory::GetCheckFunc(const std::string &prim_name) {
  const auto &it = check_func_map_.find(prim_name);
  return it != check_func_map_.end() ? it->second : nullptr;
}
}  // namespace mindspore::prim
