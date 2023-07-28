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

#include "pipeline/jit/graph_jit/graph_compiler/func_graph_builder.h"
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "frontend/operator/ops.h"
#include "ir/anf.h"
#include "ir/value.h"
#include "ops/framework_ops.h"
#include "ops/math_ops.h"
#include "ops/sequence_ops.h"
#include "ops/structure_ops.h"
#include "pipeline/jit/graph_jit/graph_compiler/func_wrapper.h"
#include "pipeline/jit/graph_jit/graph_compiler/pi_ir/ctrl_flow.h"
#include "pipeline/jit/graph_jit/graph_compiler/pi_ir/custom_nodes.h"
#include "pipeline/jit/graph_jit/graph_compiler/utils.h"
#include "frontend/operator/composite/unpack_call.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/jit/ps/parse/parse.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace jit {
namespace graph {
namespace ir {
STATIC_IR_FUNCTOR(IRMutator, vtable).set_dispatch<MindNode>([](const NodePtr &node, IRMutator *m) { return node; });
}  // namespace ir

FuncGraphPtr FuncGraphBuilder::BuildFuncGraph(const ir::FunctionNodePtr &func, const py::tuple &args,
                                              const py::dict &kwargs) {
  FuncGraphBuilderPtr builder = std::make_shared<FuncGraphBuilder>(func);
  return builder->Build(args, kwargs);
}

FuncGraphPtr FuncGraphBuilder::Build(const py::tuple &args, const py::dict &kwargs) {
  AnfNodePtrList anf_args;
  bool broaden = func_->GetAttr("enable_tuple_broaden");
  std::transform(args.begin(), args.end(), std::back_inserter(anf_args), [broaden](const py::handle &arg) {
    auto node = GraphUtils::ConvertPythonObjectToAnfNode(py::cast<py::object>(arg));
    node->set_abstract(GraphUtils::ArgsToAbstract(py::cast<py::object>(arg), GetValueNode<ValuePtr>(node), broaden));
    return node;
  });
  AnfNodePtr anf_kwargs = GraphUtils::ConvertPythonObjectToAnfNode(kwargs);
  anf_kwargs->set_abstract(GraphUtils::ArgsToAbstract(kwargs, GetValueNode<ValuePtr>(anf_kwargs), broaden));
  return Build(anf_args, anf_kwargs);
}

FuncGraphPtr FuncGraphBuilder::Build(const AnfNodePtrList &args, const AnfNodePtr &kwargs) {
  func_graph_->debug_info()->set_name(func_->GetName());
  if (func_graph_->manager() == nullptr) {
    if (func_graph_mgr_ == nullptr) {
      func_graph_mgr_ = Manage(func_graph_, true);
    } else {
      func_graph_mgr_->AddFuncGraph(func_graph_);
    }
  }
  args_ = args;
  kwargs_ = kwargs;
  // used for create sub function
  func_->Sort();
  (void)Mutate(func_);
  parse::Parser::UpdateTopFuncGraph(func_graph_);
  return func_graph_;
}

#define DEFINE_UN_NODE_MUTATE_(OP)                                           \
  ir::NodePtr FuncGraphBuilder::Mutate_(const OP &node) {                    \
    auto op = GraphUtils::GetPrimOrMetaFuncGraph(node->GetOpCode());         \
    auto n = func_graph_->NewCNodeInOrder({op, GetAnfNode(node->GetArg())}); \
    return std::make_shared<MindNode>(n);                                    \
  }

DEFINE_UN_NODE_MUTATE_(ir::NegativeNodePtr)
DEFINE_UN_NODE_MUTATE_(ir::NotNodePtr)

#define DEFINE_BIN_NODE_MUTATE_(OP)                                  \
  ir::NodePtr FuncGraphBuilder::Mutate_(const OP &node) {            \
    auto op = GraphUtils::GetPrimOrMetaFuncGraph(node->GetOpCode()); \
    auto left = GetAnfNode(node->GetLeftArg());                      \
    auto right = GetAnfNode(node->GetRightArg());                    \
    CNodePtr n = func_graph_->NewCNodeInOrder({op, left, right});    \
    return std::make_shared<MindNode>(n);                            \
  }

DEFINE_BIN_NODE_MUTATE_(ir::AddNodePtr)
DEFINE_BIN_NODE_MUTATE_(ir::SubNodePtr)
DEFINE_BIN_NODE_MUTATE_(ir::MulNodePtr)
DEFINE_BIN_NODE_MUTATE_(ir::DivNodePtr)
DEFINE_BIN_NODE_MUTATE_(ir::BitwiseNodePtr)
DEFINE_BIN_NODE_MUTATE_(ir::BinaryOperationPtr)

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::RefNodePtr &node) {
  node->SetRealNode(Mutate(node->GetRealNode()));
  return node;
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::ParameterPtr &node) {
  auto param = std::make_shared<Parameter>(func_graph_);
  MS_EXCEPTION_IF_NULL(param);
  std::string name = node->GetName();
  param->set_name(name);
  MS_EXCEPTION_IF_NULL(param->debug_info());
  param->debug_info()->set_name(name);
  auto category = node->GetCategory();
  // kwargs
  py::object object;
  if (category == ir::Parameter::KEYWORD) {
    param->set_abstract(kwargs_->abstract());
  } else {
    MS_EXCEPTION_IF_CHECK_FAIL(node->GetIndex() < args_.size(), "Parameter " + name + " has no arguments");
    param->set_abstract(args_[node->GetIndex()]->abstract());
  }
  auto defalut_value = node->GetDefaultValue();
  if (defalut_value != nullptr) {
    AnfNodePtr value_node = GetAnfNode(defalut_value);
    func_graph_->set_param_default_value(name, value_node);
  }
  return std::make_shared<MindNode>(param);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::FunctionNodePtr &node) {
  func_graph_->set_has_vararg(node->HasVarArg());
  func_graph_->set_kwonlyargs_count(node->GetKwOnlyArgsCnt());
  func_graph_->set_has_kwarg(node->HasKwArg());
  bool drop_param = node->IsMethod();
  for (auto &parameter : node->GetParameters()) {
    auto name = parameter->cast<ir::ParameterPtr>()->GetName();
    if (assigned_vars_.find(name) != assigned_vars_.end()) {
      continue;
    }
    parameter = Mutate(parameter);
    MS_EXCEPTION_IF_CHECK_FAIL(parameter->isa<MindNode>(), "Not Excepted value of parameter.");
    auto param = GetAnfNode(parameter)->cast<ParameterPtr>();
    if (!drop_param) {
      func_graph_->add_parameter(param);
    }
    drop_param = false;
    assigned_vars_.emplace(param->name(), param);
  }
  auto nodes = node->GetNodes();
  for (auto iter = nodes.begin(); iter != nodes.end();) {
    auto new_node = Mutate(*iter);
    if (new_node == nullptr) {
      iter = nodes.erase(iter);
    } else {
      *iter = new_node;
      iter++;
    }
  }
  return node;
}

bool IsGraphCell(const py::object &node) {
  if (!py::hasattr(node, "__class__")) {
    return false;
  }
  auto cls = py::getattr(node, "__class__");
  if (!py::hasattr(cls, "__name__")) {
    return false;
  }
  return py::cast<std::string>(py::getattr(cls, "__name__")) == "GraphCell";
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::ValuePtr &node) {
  auto obj = node->GetValue();
  auto dict = python_adapter::GetPyObjAttr(python_adapter::GetPyModule("mindspore._extends.parse.resources"),
                                           "convert_object_map");
  auto special_obj = PyDict_GetItem(dict.ptr(), obj.ptr());
  if (special_obj != nullptr) {
    obj = py::cast<py::object>(special_obj);
  }
  auto value = GraphUtils::ConvertPythonObjectToAnfNode(obj);
  if (IsValueNode<FuncGraph>(value) && IsGraphCell(obj)) {
    GetValueNode<FuncGraphPtr>(value)->set_flag("is_load", true);
  }
  return std::make_shared<MindNode>(value);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::IfNodePtr &node) {
  auto cond = node->GetCondition();
  MS_EXCEPTION_IF_CHECK_FAIL(cond->isa<ir::JumpNode>(), cond->ToString() + " can't be a condition.");
  auto jump = cond->cast<ir::JumpNodePtr>();
  auto condition = Mutate(jump->GetCondition())->cast<MindNodePtr>()->GetAnfNode();
  if (jump->GetOpCode() == POP_JUMP_IF_TRUE) {
    condition = func_graph_->NewCNodeInOrder({GraphUtils::GetMetaFuncGraph(UNARY_NOT), condition});
  }
  auto _then = node->GetThen();
  if (!_then.empty() && _then.back()->isa<ir::JumpNode>()) {
    _then.pop_back();
  }
  const std::string prefix = func_->GetName() + "_sub_func_" + std::to_string(cond->GetOffset());
  FuncWrapperPtr wrapper_then = std::make_shared<FuncWrapper>(prefix + "_true", _then);
  auto outputs_then = wrapper_then->GetOutputs();
  auto _else = node->GetElse();
  if (!_else.empty() && _else.back()->isa<ir::JumpNode>()) {
    _else.pop_back();
  }
  FuncWrapperPtr wrapper_else = std::make_shared<FuncWrapper>(prefix + "_false", _else);
  auto outputs_else = wrapper_else->GetOutputs();
  std::set<std::string> var_names;
  std::vector<ir::ValuePtr> outputs;
  std::for_each(outputs_then.begin(), outputs_then.end(), [&outputs, &var_names](const ir::ValuePtr &var) {
    auto var_name = var->GetValue().cast<std::string>();
    if (var_names.find(var_name) == var_names.end()) {
      outputs.push_back(var);
    }
    var_names.insert(var_name);
  });
  std::for_each(outputs_else.begin(), outputs_else.end(), [&outputs, &var_names](const ir::ValuePtr &var) {
    auto var_name = var->GetValue().cast<std::string>();
    if (var_names.find(var_name) == var_names.end()) {
      outputs.push_back(var);
    }
    var_names.insert(var_name);
  });
  wrapper_then->SpecifyOutputs(outputs);
  wrapper_else->SpecifyOutputs(outputs);

  ir::FunctionNodePtr func = wrapper_then->Wrapper();
  FuncGraphBuilderPtr builder = std::make_shared<FuncGraphBuilder>(func);
  AnfNodePtrList args;
  std::transform(func->GetParameters().begin(), func->GetParameters().end(), std::back_inserter(args),
                 [&builder, this](const ir::NodePtr &param) {
                   auto name = param->cast<ir::ParameterPtr>()->GetName();
                   MS_EXCEPTION_IF_CHECK_FAIL(assigned_vars_.find(name) != assigned_vars_.end(),
                                              "Local var " + name + " is not defined.");
                   builder->SpecifyLocalVarInitialValue(name, assigned_vars_.at(name));
                   return assigned_vars_.at(name);
                 });
  auto graph_true = builder->Build(args, NewValueNode(kNone));

  func = wrapper_else->Wrapper();
  builder = std::make_shared<FuncGraphBuilder>(func);
  args.clear();
  std::transform(func->GetParameters().begin(), func->GetParameters().end(), std::back_inserter(args),
                 [&builder, this](const ir::NodePtr &param) {
                   auto name = param->cast<ir::ParameterPtr>()->GetName();
                   MS_EXCEPTION_IF_CHECK_FAIL(assigned_vars_.find(name) != assigned_vars_.end(),
                                              "Local var " + name + " is not defined.");
                   builder->SpecifyLocalVarInitialValue(name, assigned_vars_.at(name));
                   return assigned_vars_.at(name);
                 });
  auto graph_false = builder->Build(args, NewValueNode(kNone));
  CNodePtr switch_node =
    func_graph_->NewCNodeInOrder(prim::kPrimSwitch, {condition, NewValueNode(graph_true), NewValueNode(graph_false)});
  CNodePtr call_switch = func_graph_->NewCNodeInOrder({switch_node});
  size_t index = 0;
  std::for_each(outputs.begin(), outputs.end(), [call_switch, &index, this](const ir::ValuePtr &value) {
    const auto name = py::cast<std::string>(value->GetValue());
    MS_EXCEPTION_IF_CHECK_FAIL(assigned_vars_.find(name) != assigned_vars_.end(),
                               "Local var " + name + " is not defined.");
    auto subscr = NewValueNode(std::make_shared<Int64Imm>(index));
    assigned_vars_[name] =
      func_graph_->NewCNodeInOrder({GraphUtils::GetMetaFuncGraph(BINARY_SUBSCR), call_switch, subscr});
  });
  return std::make_shared<MindNode>(call_switch);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::InvertNodePtr &node) {
  auto arg = GetAnfNode(node->GetArg());
  auto op = IsValueNode<Scalar>(arg) ? GraphUtils::GetPrimOrMetaFuncGraph(node->GetOpCode())
                                     : NewValueNode(prim::GetPythonOps("logical_not", "mindspore.ops.functional"));
  return std::make_shared<MindNode>(func_graph_->NewCNodeInOrder({op, arg}));
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::ReturnNodePtr &node) {
  auto op = GraphUtils::GetPrimOrMetaFuncGraph(node->GetOpCode());
  auto arg = GetAnfNode(node->GetArg());
  auto ret = func_graph_->NewCNodeInOrder({op, arg});
  func_graph_->set_return(ret);
  return std::make_shared<MindNode>(ret);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::CastNodePtr &node) {
  auto op = GraphUtils::GetPrimOrMetaFuncGraph(node->GetOpCode());
  auto arg = GetAnfNode(node->GetArg());
  if (!IsValueNode<ValueList>(arg)) {
    MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(arg, prim::kPrimMakeList),
                               arg->DebugString() + " is invalid for list_to_tuple.");
    arg->cast<CNodePtr>()->set_input(0, op);
  } else {
    auto value_list = GetValueNode<ValueListPtr>(arg);
    AnfNodePtrList values = {op};
    std::transform(value_list->value().begin(), value_list->value().end(), std::back_inserter(values),
                   [](const ValuePtr &arg) { return NewValueNode(arg); });
    arg = func_graph_->NewCNodeInOrder(values);
  }
  return std::make_shared<MindNode>(arg);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::FormatNodePtr &node) {
  auto arg = node->GetArg(0);
  MS_EXCEPTION_IF_CHECK_FAIL(arg->isa<ir::Value>(), "The arg of format must be object.");
  py::object top = arg->cast<ir::ValuePtr>()->GetValue();
  py::object format;
  auto fmt_flag = node->GetFormatType();
  switch (fmt_flag & 0x03) {
    case 0x00: {
      break;
    }
    case 0x01: {
      top = py::reinterpret_steal<py::object>(PyObject_Str(top.ptr()));
      break;
    }
    case 0x02: {
      top = py::reinterpret_steal<py::object>(PyObject_Repr(top.ptr()));
      break;
    }
    case 0x03: {
      top = py::reinterpret_steal<py::object>(PyObject_ASCII(top.ptr()));
      break;
    }
    default: {
      if ((fmt_flag & 0x04) == 0x04) {
        arg = node->GetArg(1);
        MS_EXCEPTION_IF_CHECK_FAIL(arg->isa<ir::Value>(), "The fmt must be object.");
        format = arg->cast<ir::ValuePtr>()->GetValue();
      }
      break;
    }
  }
  py::str obj = py::cast<py::str>(PyObject_Format(top.ptr(), format.ptr()));
  AnfNodePtr value = GraphUtils::ConvertPythonObjectToAnfNode(obj);
  return std::make_shared<MindNode>(value);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::IsNodePtr &node) {
  auto left = GetAnfNode(node->GetLeftArg());
  auto right = GetAnfNode(node->GetRightArg());
  PrimitivePtr prim = node->IsInvert() ? prim::kPrimIsNot : prim::kPrimIs_;
  AnfNodePtr n = func_graph_->NewCNodeInOrder(prim, {left, right});
  return std::make_shared<MindNode>(n);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::ContainsNodePtr &node) {
  auto left = GetAnfNode(node->GetLeftArg());
  auto right = GetAnfNode(node->GetRightArg());
  auto name = node->IsInvert() ? "not_in_" : "in_";
  CNodePtr n = func_graph_->NewCNodeInOrder({GraphUtils::GetMetaFuncGraph(name), left, right});
  return std::make_shared<MindNode>(n);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::StoreNodePtr &node) {
  auto left = GetAnfNode(node->GetLeftArg());
  auto right = GetAnfNode(node->GetRightArg());
  MS_EXCEPTION_IF_CHECK_FAIL(IsValueNode<StringImm>(right), "Excepted var name.");
  assigned_vars_[GetValue<std::string>(GetValueNode(right))] = left;
  return nullptr;
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::CompareNodePtr &node) {
  std::vector<std::string> ops = {
    "less",
    "less_equal",
    "equal",
    "not_equal",
    "greater",
    "greater_equal"
#if (PY_MAJOR_VERSION == 3 && (PY_MINOR_VERSION == 7 || PY_MINOR_VERSION == 8))
    ,
    "in_",
    "not_in_",
    "is",
    "is_not"
#endif  // #if (PY_MAJOR_VERSION == 3 && (PY_MINOR_VERSION == 7 || PY_MINOR_VERSION == 8))
  };
  auto left = GetAnfNode(node->GetLeftArg());
  auto right = GetAnfNode(node->GetRightArg());
  const std::string &op = ops[node->GetInstrArg()];
#if (PY_MAJOR_VERSION == 3 && (PY_MINOR_VERSION == 7 || PY_MINOR_VERSION == 8))
  if (op == "is" || op == "is_not") {
    auto prim = (op == "is") ? prim::kPrimIs_ : prim::kPrimIsNot;
    return std::make_shared<MindNode>(func_graph_->NewCNodeInOrder(prim, {left, right}));
  }
#endif  // #if (PY_MAJOR_VERSION == 3 && (PY_MINOR_VERSION == 7 || PY_MINOR_VERSION == 8))
  CNodePtr n = func_graph_->NewCNodeInOrder({GraphUtils::GetMetaFuncGraph(op), left, right});
  return std::make_shared<MindNode>(n);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::UpdateNodePtr &node) {
  auto left = GetAnfNode(node->GetLeftArg());
  auto right = GetAnfNode(node->GetRightArg());
  AnfNodePtrList array = {left, right};
#if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 7)
  if (node->GetOpCode() == MAP_ADD) {
    std::reverse(array.begin(), array.end());
  }
#endif  // #if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 7)
  CNodePtr n = nullptr;
  if (node->GetOpCode() == LIST_EXTEND) {
    n = MergeList(array);
  } else {
    n = MergeDict(array);
  }
  return std::make_shared<MindNode>(n);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::LoadNodePtr &node) {
  AnfNodePtr n = GetAnfNode(node->GetArg(0));
  ir::OpCode op_code = node->GetOpCode();
  if (op_code == LOAD_CONST) {
    // no need to do anything
  } else if (op_code == LOAD_FAST) {
    MS_EXCEPTION_IF_CHECK_FAIL(IsValueNode<StringImm>(n), "Invalid var name.");
    std::string key = GetValue<std::string>(GetValueNode(n));
    auto found = assigned_vars_.find(key);
    MS_EXCEPTION_IF_CHECK_FAIL(found != assigned_vars_.end(), "Found not define var " + key + ".");
    n = assigned_vars_[key];
  } else if (op_code == LOAD_GLOBAL || op_code == LOAD_CLOSURE) {
    n = GetAnfNode(node->GetArg(1));
  } else if (op_code == LOAD_DEREF || op_code == LOAD_CLASSDEREF) {
    auto arg = node->GetArg(1);
    MS_EXCEPTION_IF_CHECK_FAIL(arg->isa<ir::Value>(), "Excepted a python object as arg of load_closure.");
    auto cell = arg->cast<ir::ValuePtr>()->GetValue();
    n = GraphUtils::ConvertPythonObjectToAnfNode(py::cast<py::object>(PyCell_Get(cell.ptr())));
  } else {
    auto base = GetAnfNode(node->GetArg(1));
    auto attr = GetAnfNode(node->GetArg(0));
    MS_EXCEPTION_IF_CHECK_FAIL(IsValueNode<StringImm>(attr), "Excepted attr/name.");
    n = func_graph_->NewCNodeInOrder(prim::kPrimGetAttr, {base, attr});
  }
  return std::make_shared<MindNode>(n);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::BuildNodePtr &node) {
  AnfNodePtrList array;
  std::transform(node->GetArgs().begin(), node->GetArgs().end(), std::back_inserter(array),
                 [&](const ir::NodePtr &arg) { return GetAnfNode(arg); });
  auto prim = GraphUtils::GetPrimitive(node->GetOpCode());
  if (prim == prim::kPrimStringConcat) {
    while (array.size() > 2) {
      size_t array_new_size = array.size() - 2;
      CNodePtr string_concat =
        func_graph_->NewCNodeInOrder(prim::kPrimStringConcat, {array.back(), array[array_new_size]});
      array.resize(array_new_size);
      array.push_back(string_concat);
    }
  }
  if (prim == prim::kPrimMakeSlice) {
    while (array.size() < 3) {
      array.push_back(NewValueNode(kNone));
    }
  }
  if (prim == prim::kPrimMakeDict) {
    AnfNodePtrList keys;
    AnfNodePtrList values;
    if (node->GetOpCode() == BUILD_MAP) {
      for (size_t index = 0; index < array.size(); index += 2) {
        values.push_back(array[index]);
        keys.push_back(array[index + 1]);
      }
    } else {
      AnfNodePtr const_key = array.back();
      array.pop_back();
      auto tuple = GetValueNode<ValueTuplePtr>(const_key);
      MS_EXCEPTION_IF_CHECK_FAIL((tuple->size() == array.size()), "The count of keys not match the size of Dict.");
      std::transform(tuple->value().begin(), tuple->value().end(), std::back_inserter(keys),
                     [](const ValuePtr &value) { return NewValueNode(value); });
      values.insert(values.begin(), array.begin(), array.end());
    }
    CNodePtr cnode_keys = func_graph_->NewCNodeInOrder(prim::kPrimMakeTuple, keys);
    CNodePtr cnode_values = func_graph_->NewCNodeInOrder(prim::kPrimMakeTuple, values);
    array.clear();
    array.push_back(cnode_keys);
    array.push_back(cnode_values);
  }
  CNodePtr n = func_graph_->NewCNodeInOrder(prim, array);
  return std::make_shared<MindNode>(n);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::CallNodePtr &node) {
  AnfNodePtrList nodes;
  size_t index = 0;
  if (node->GetOpCode() == CALL_FUNCTION_KW || node->GetOpCode() == CALL_FUNCTION_EX) {
    nodes.push_back(NewValueNode(std::make_shared<prim::UnpackCall>("unpack_call")));
    index++;
  }
  std::transform(node->GetArgs().begin(), node->GetArgs().end(), std::back_inserter(nodes),
                 [&](const ir::NodePtr &arg) { return GetAnfNode(arg); });
  if (IsValueNode<FuncGraph>(nodes[index])) {
    auto func_graph = GetValueNode<FuncGraphPtr>(nodes[index]);
    if (func_graph->has_flag("is_load")) {
      func_graph_mgr_->AddFuncGraph(func_graph);
      GraphUtils::ResolveFuncGraph(func_graph);
    }
  }
  if (node->GetOpCode() == CALL_FUNCTION_EX && !IsPrimitiveCNode(nodes.back(), prim::kPrimMakeDict)) {
    auto args = nodes.back();
    nodes.erase(nodes.begin());
    nodes.resize(1);
    if (IsValueNode<ValueSequence>(args)) {
      auto inputs = GetValueNode<ValueSequencePtr>(args)->value();
      std::transform(inputs.begin(), inputs.end(), std::back_inserter(nodes),
                     [](const ValuePtr &input) { return NewValueNode(input); });
    } else {
      MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(args, prim::kPrimMakeTuple), "Expected tuple node.");
      auto inputs = args->cast<CNodePtr>()->inputs();
      std::transform(inputs.begin() + 1, inputs.end(), std::back_inserter(nodes),
                     [](const AnfNodePtr &input) { return input; });
    }
  }
  if (node->GetOpCode() == CALL_FUNCTION_KW) {
    MS_EXCEPTION_IF_CHECK_FAIL(IsValueNode<ValueTuple>(nodes.back()), "Expected tuple node.");
    auto keys_tuple = GetValueNode<ValueTuplePtr>(nodes.back());
    nodes.pop_back();
    AnfNodePtrList keys;
    std::transform(keys_tuple->value().begin(), keys_tuple->value().end(), std::back_inserter(keys),
                   [](const ValuePtr &key) { return NewValueNode(key); });
    CNodePtr keys_cnode = func_graph_->NewCNodeInOrder(prim::kPrimMakeTuple, std::move(keys));
    size_t args_cnt = nodes.size() - keys_tuple->size();
    AnfNodePtrList values(nodes.begin() + args_cnt, nodes.end());
    nodes.resize(args_cnt);
    if (args_cnt > 2) {
      AnfNodePtrList pos_args(nodes.begin() + 2, nodes.end());
      nodes.resize(2);
      CNodePtr pos_args_cnode = func_graph_->NewCNodeInOrder(prim::kPrimMakeTuple, std::move(pos_args));
      nodes.push_back(pos_args_cnode);
    }
    CNodePtr values_cnode = func_graph_->NewCNodeInOrder(prim::kPrimMakeTuple, std::move(values));
    CNodePtr kwargs_node = func_graph_->NewCNodeInOrder(prim::kPrimMakeDict, {keys_cnode, values_cnode});
    nodes.push_back(kwargs_node);
  }
  return std::make_shared<MindNode>(func_graph_->NewCNodeInOrder(std::move(nodes)));
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::SubscrNodePtr &node) {
  auto object = GetAnfNode(node->GetObject());
  auto subscr = GetAnfNode(node->GetSubscr());
  CNodePtr n = func_graph_->NewCNodeInOrder({GraphUtils::GetMetaFuncGraph(BINARY_SUBSCR), object, subscr});
  return std::make_shared<MindNode>(n);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::AttrNodePtr &node) {
  auto object = GetAnfNode(node->GetObject());
  auto attr = GetAnfNode(node->GetAttr());
  CNodePtr n = func_graph_->NewCNodeInOrder(prim::kPrimGetAttr, {object, attr});
  return std::make_shared<MindNode>(n);
}

AnfNodePtr FuncGraphBuilder::GetAnfNode(const ir::NodePtr &node) {
  if (node->isa<MindNode>()) {
    return node->cast<MindNodePtr>()->GetAnfNode();
  }
  if (node->isa<ir::RefNode>()) {
    return GetAnfNode(node->cast<ir::RefNodePtr>()->GetRealNode());
  }
  return GetAnfNode(Mutate(node));
}

CNodePtr FuncGraphBuilder::MergeList(const AnfNodePtrList &nodes) {
  MS_EXCEPTION_IF_CHECK_FAIL(IsValueNode<ValueTuple>(nodes.back()), "Invalid args of list extend.");
  AnfNodePtrList list;
  for (size_t index = 0; index < nodes.size() - 1; index++) {
    MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(nodes[index], prim::kPrimMakeList),
                               "Invalid args of list extend target.");
    auto inputs = nodes[index]->cast<CNodePtr>()->inputs();
    list.insert(list.end(), inputs.begin() + 1, inputs.end());
  }
  auto valueTuple = GetValuePtr<ValueTuple>(nodes.back());
  std::transform(valueTuple->value().begin(), valueTuple->value().end(), std::back_inserter(list),
                 [](const ValuePtr &value) { return NewValueNode(value); });
  return func_graph_->NewCNodeInOrder(prim::kPrimMakeList, list);
}

CNodePtr FuncGraphBuilder::ConvertValueDictionaryToCNode(const AnfNodePtr &node) {
  bool is_make_dict_node = IsPrimitiveCNode(node, prim::kPrimMakeDict);
  bool is_param = node->isa<Parameter>();
  MS_EXCEPTION_IF_CHECK_FAIL((is_make_dict_node || is_param || IsValueNode<ValueDictionary>(node)),
                             "Can't convert non-dictionary to dict node.");
  if (is_make_dict_node) {
    return node->cast<CNodePtr>();
  }
  if (is_param) {
    auto value = node->abstract()->BuildValue();
    return ConvertValueDictionaryToCNode(NewValueNode(value));
  }
  auto dict = GetValueNode<ValueDictionaryPtr>(node);
  AnfNodePtrList keys;
  keys.push_back(NewValueNode(prim::kPrimMakeTuple));
  AnfNodePtrList values;
  values.push_back(NewValueNode(prim::kPrimMakeTuple));
  for (auto &[key, value] : dict->value()) {
    keys.push_back(NewValueNode(key));
    values.push_back(NewValueNode(value));
  }
  CNodePtr keys_cnode = func_graph_->NewCNodeInOrder(std::move(keys));
  CNodePtr values_cnode = func_graph_->NewCNodeInOrder(std::move(values));
  return func_graph_->NewCNodeInOrder({NewValueNode(prim::kPrimMakeDict), keys_cnode, values_cnode});
}

CNodePtr FuncGraphBuilder::MergeDict(const AnfNodePtrList &nodes) {
  AnfNodePtrList keys;
  AnfNodePtrList values;
  std::for_each(nodes.begin(), nodes.end(), [&](const AnfNodePtr &node) {
    auto cnode = ConvertValueDictionaryToCNode(node);
    auto inputs = cnode->input(1)->cast<CNodePtr>()->inputs();
    keys.insert(keys.end(), inputs.begin() + 1, inputs.end());
    inputs = cnode->input(2)->cast<CNodePtr>()->inputs();
    values.insert(values.end(), inputs.begin() + 1, inputs.end());
  });
  CNodePtr keys_cnode = func_graph_->NewCNodeInOrder(prim::kPrimMakeTuple, keys);
  CNodePtr values_cnode = func_graph_->NewCNodeInOrder(prim::kPrimMakeTuple, values);
  return func_graph_->NewCNodeInOrder({NewValueNode(prim::kPrimMakeDict), keys_cnode, values_cnode});
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
