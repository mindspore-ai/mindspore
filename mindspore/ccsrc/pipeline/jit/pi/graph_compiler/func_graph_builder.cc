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

#include "pipeline/jit/pi/graph_compiler/func_graph_builder.h"
#include <algorithm>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "frontend/operator/composite/unpack_call.h"
#include "frontend/operator/ops.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/python_adapter.h"
#include "ops/framework_ops.h"
#include "ops/math_ops.h"
#include "ops/sequence_ops.h"
#include "ops/structure_ops.h"
#include "pipeline/jit/pi/graph_compiler/func_wrapper.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/ir_mutator.h"
#include "pipeline/jit/pi/graph_compiler/utils.h"
#include "pipeline/jit/ps/parse/parse.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace pijit {
namespace ir {
STATIC_IR_FUNCTOR(IRMutator, vtable).set_dispatch<MindNode>([](const NodePtr &node, IRMutator *m) { return node; });
}  // namespace ir

FuncGraphPtr FuncGraphBuilder::BuildFuncGraph(const ir::FunctionNodePtr &func, const py::tuple &args,
                                              const py::dict &kwargs) {
  AnfNodePtrList anf_args;
  bool broaden = func->GetAttr("enable_tuple_broaden");
  std::transform(args.begin(), args.end(), std::back_inserter(anf_args), [broaden](const py::handle &arg) {
    auto node = GraphUtils::ConvertPythonObjectToAnfNode(py::cast<py::object>(arg));
    node->set_abstract(GraphUtils::ArgsToAbstract(py::cast<py::object>(arg), GetValueNode<ValuePtr>(node), broaden));
    return node;
  });
  AnfNodePtr anf_kwargs = GraphUtils::ConvertPythonObjectToAnfNode(kwargs);
  anf_kwargs->set_abstract(GraphUtils::ArgsToAbstract(kwargs, GetValueNode<ValuePtr>(anf_kwargs), broaden));
  return BuildFuncGraph(func, anf_args, anf_kwargs);
}

FuncGraphPtr FuncGraphBuilder::BuildFuncGraph(const ir::FunctionNodePtr &func, const AnfNodePtrList &args,
                                              const AnfNodePtr &kwargs) {
  auto builder = std::make_shared<FuncGraphBuilder>(func, args, kwargs);
  parse::Parser::UpdateTopFuncGraph(builder->func_graph_);
  auto func_graph = GetValueNode<FuncGraphPtr>(builder->Mutate(func)->cast<MindNodePtr>()->GetAnfNode());
  return func_graph;
}

#define DEFINE_UN_NODE_MUTATE_(OP)                                           \
  ir::NodePtr FuncGraphBuilder::Mutate_(const OP &node) {                    \
    auto op = GraphUtils::GetPrimOrMetaFuncGraph(node->GetOpCode());         \
    auto n = func_graph_->NewCNodeInOrder({op, GetAnfNode(node->GetArg())}); \
    UpdateLocation(n, node);                                                 \
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
    UpdateLocation(n, node);                                         \
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
  auto name = node->GetName();
  auto index = node->GetIndex();
  param_name_to_index_[name] = index;
  if (!func_->NeedGenParameters()) {
    MS_EXCEPTION_IF_CHECK_FAIL(static_cast<size_t>(index) < args_.size(), "Invalid paramete[" + name + "].");
    assigned_vars_[name] = args_[index];
    return node;
  }
  auto param = std::make_shared<Parameter>(func_graph_);
  MS_EXCEPTION_IF_NULL(param);
  param->set_name(name);
  MS_EXCEPTION_IF_NULL(param->debug_info());
  param->debug_info()->set_name(name);
  UpdateLocation(param, node);
  auto category = node->GetCategory();
  // kwargs
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
  assigned_vars_[name] = param;
  func_graph_->add_parameter(param);
  return std::make_shared<MindNode>(param);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::FunctionNodePtr &node) {
  func_graph_->set_has_vararg(node->HasVarArg());
  func_graph_->set_kwonlyargs_count(node->GetKwOnlyArgsCnt());
  func_graph_->set_has_kwarg(node->HasKwArg());
  func_graph_->debug_info()->set_name(node->GetName());
  // used for create sub function
  node->Sort();
  auto first_if_node = std::find_if(node->GetNodes().begin(), node->GetNodes().end(),
                                    [](const ir::NodePtr &n) { return n->isa<ir::IfNode>(); });
  if (first_if_node != node->GetNodes().end()) {
    size_t index = std::distance(node->GetNodes().begin(), first_if_node);
    std::vector<ir::NodePtr> nodes(node->GetNodes().begin() + index + 1, node->GetNodes().end());
    node->GetNodes().resize(index + 1);
    auto if_node = node->GetNodes().back()->cast<ir::IfNodePtr>();
    if (if_node->GetThen().empty() || !if_node->GetThen().back()->isa<ir::ReturnNode>()) {
      if (!if_node->GetThen().empty() && if_node->GetThen().back()->isa<ir::JumpNode>()) {
        if_node->GetThen().pop_back();
      }
      std::for_each(nodes.begin(), nodes.end(), [&if_node](const ir::NodePtr &n) { if_node->AddThen(n); });
    }
    if (if_node->GetElse().empty() || !if_node->GetElse().back()->isa<ir::ReturnNode>()) {
      if (!if_node->GetElse().empty() && if_node->GetElse().back()->isa<ir::JumpNode>()) {
        if_node->GetElse().pop_back();
      }
      std::for_each(nodes.begin(), nodes.end(), [&if_node](const ir::NodePtr &n) { if_node->AddElse(n); });
    }
    node->AddNode(std::make_shared<ir::ReturnNode>(if_node));
  }
  MUTATE_NODE_LIST(node->GetParameters())
  MUTATE_NODE_LIST(node->GetNodes())
  return std::make_shared<MindNode>(NewValueNode(func_graph_));
}

void FuncGraphBuilder::UpdateLocation(const AnfNodePtr &anf_node, const ir::NodePtr &node) {
  if (!enable_debug_info_) {
    return;
  }
  // Refer to Location::Location() for each node: line, column, line_end, column_end, expr_src.
  auto debug_info = node->GetDebugInfo();
  auto line_no = debug_info->GetLineNo();
  line_no = (line_no == 0) ? last_line_no_ : line_no;
  last_line_no_ = line_no;
  auto loc = anf_node->debug_info()->location();
  if (loc == nullptr) {
    std::vector<std::string> comments;
    anf_node->debug_info()->set_location(std::make_shared<Location>(debug_info->GetFileName(), line_no, 0, line_no, 0,
                                                                    debug_info->GetDesc(), std::move(comments)));
  } else {
    loc->set_file_name(debug_info->GetFileName());
    loc->set_line(line_no);
    loc->set_line_end(line_no);
    loc->set_expr_src(debug_info->GetDesc());
  }
}

AnfNodePtr FuncGraphBuilder::ConvertListOrTupleToCNode(const py::object &obj) {
  MS_EXCEPTION_IF_CHECK_FAIL((py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)),
                             "Should be a list or tuple.");
  auto tuple = py::cast<py::tuple>(obj);
  auto parameter = python_adapter::GetPyObjAttr(python_adapter::GetPyModule("mindspore"), "Parameter");
  auto prim = py::isinstance<py::list>(obj) ? prim::kPrimMakeList : prim::kPrimMakeTuple;
  CNodePtr cnode = func_graph_->NewCNodeInOrder(prim, {});
  for (size_t idx = 0; idx < tuple.size(); idx++) {
    if (py::isinstance(tuple[idx], parameter)) {
      cnode->add_input(parse::ResolveParameterObj(func_graph_, tuple[idx]));
    } else if (py::isinstance<py::list>(tuple[idx]) || py::isinstance<py::tuple>(tuple[idx])) {
      cnode->add_input(ConvertListOrTupleToCNode(tuple[idx]));
    } else {
      cnode->add_input(GraphUtils::ConvertPythonObjectToAnfNode(tuple[idx]));
    }
  }
  return cnode;
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::ValuePtr &node) {
  auto obj = node->GetValue();
  auto dict = python_adapter::GetPyObjAttr(python_adapter::GetPyModule("mindspore._extends.parse.resources"),
                                           "convert_object_map");
  auto special_obj = PyDict_GetItem(dict.ptr(), obj.ptr());
  if (special_obj != nullptr) {
    obj = py::cast<py::object>(special_obj);
  }
  bool is_list_or_tuple = (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj));
  auto value = is_list_or_tuple ? ConvertListOrTupleToCNode(obj) : GraphUtils::ConvertPythonObjectToAnfNode(obj);
  UpdateLocation(value, node);
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
  func->MarkNoNeedGenParameters();
  AnfNodePtrList args;
  std::transform(func->GetParameters().begin(), func->GetParameters().end(), std::back_inserter(args),
                 [this](const ir::NodePtr &param) {
                   auto name = param->cast<ir::ParameterPtr>()->GetName();
                   MS_EXCEPTION_IF_CHECK_FAIL(assigned_vars_.find(name) != assigned_vars_.end(),
                                              "Local var " + name + " is not defined.");
                   return assigned_vars_.at(name);
                 });
  FuncGraphBuilderPtr builder = std::make_shared<FuncGraphBuilder>(func, args, NewValueNode(kNone));
  auto graph_true = GetValueNode<FuncGraphPtr>(builder->Mutate(func)->cast<MindNodePtr>()->GetAnfNode());

  func = wrapper_else->Wrapper();
  func->MarkNoNeedGenParameters();
  args.clear();
  std::transform(func->GetParameters().begin(), func->GetParameters().end(), std::back_inserter(args),
                 [this](const ir::NodePtr &param) {
                   auto name = param->cast<ir::ParameterPtr>()->GetName();
                   MS_EXCEPTION_IF_CHECK_FAIL(assigned_vars_.find(name) != assigned_vars_.end(),
                                              "Local var " + name + " is not defined.");
                   return assigned_vars_.at(name);
                 });
  builder = std::make_shared<FuncGraphBuilder>(func, args, NewValueNode(kNone));
  auto graph_false = GetValueNode<FuncGraphPtr>(builder->Mutate(func)->cast<MindNodePtr>()->GetAnfNode());
  CNodePtr switch_node =
    func_graph_->NewCNodeInOrder(prim::kPrimSwitch, {condition, NewValueNode(graph_true), NewValueNode(graph_false)});
  CNodePtr call_switch = func_graph_->NewCNodeInOrder({switch_node});
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
  UpdateLocation(arg, node);
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
  UpdateLocation(value, node);
  return std::make_shared<MindNode>(value);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::IsNodePtr &node) {
  auto left = GetAnfNode(node->GetLeftArg());
  auto right = GetAnfNode(node->GetRightArg());
  PrimitivePtr prim = node->IsInvert() ? prim::kPrimIsNot : prim::kPrimIs_;
  AnfNodePtr n = func_graph_->NewCNodeInOrder(prim, {left, right});
  UpdateLocation(n, node);
  return std::make_shared<MindNode>(n);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::ContainsNodePtr &node) {
  auto left = GetAnfNode(node->GetLeftArg());
  auto right = GetAnfNode(node->GetRightArg());
  auto name = node->IsInvert() ? "not_in_" : "in_";
  CNodePtr n = func_graph_->NewCNodeInOrder({GraphUtils::GetMetaFuncGraph(name), left, right});
  UpdateLocation(n, node);
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
  UpdateLocation(n, node);
  return std::make_shared<MindNode>(n);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::UpdateNodePtr &node) {
  auto left = GetAnfNode(node->GetLeftArg());
  auto right = GetAnfNode(node->GetRightArg());
#if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 7)
  if (node->GetOpCode() == MAP_ADD) {
    left.swap(right);
  }
#endif  // #if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 7)
  AnfNodePtr n = nullptr;
  if (node->GetOpCode() == LIST_EXTEND) {
    n = MergeList(left, right);
  } else {
    n = MergeDict(left, right);
  }
  UpdateLocation(n, node);
  return std::make_shared<MindNode>(n);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::LoadValueNodePtr &node) {
  AnfNodePtr n = GetAnfNode(node->GetArg(0));
  ir::OpCode op_code = node->GetOpCode();
  if (op_code == LOAD_FAST) {
    MS_EXCEPTION_IF_CHECK_FAIL(IsValueNode<StringImm>(n), "Invalid var name.");
    std::string key = GetValue<std::string>(GetValueNode(n));
    auto found = assigned_vars_.find(key);
    MS_EXCEPTION_IF_CHECK_FAIL(found != assigned_vars_.end(), "Found not define var " + key + ".");
    n = assigned_vars_[key];
  } else if (op_code == LOAD_DEREF || op_code == LOAD_CLASSDEREF) {
    auto arg = node->GetArg();
    MS_EXCEPTION_IF_CHECK_FAIL(arg->isa<ir::Value>(), "Excepted a python object as arg of load_closure.");
    auto cell = arg->cast<ir::ValuePtr>()->GetValue();
    n = GraphUtils::ConvertPythonObjectToAnfNode(py::cast<py::object>(PyCell_Get(cell.ptr())));
  } else {
    // no need to do anything
    MS_EXCEPTION_IF_CHECK_FAIL((op_code == LOAD_CONST || op_code == LOAD_GLOBAL || op_code == LOAD_CLOSURE),
                               "Not Expected bytecode.");
  }
  UpdateLocation(n, node);
  return std::make_shared<MindNode>(n);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::LoadFieldNodePtr &node) {
  auto instance = GetAnfNode(node->GetArg(0));
  auto field = GetAnfNode(node->GetArg(1));
  MS_EXCEPTION_IF_CHECK_FAIL(IsValueNode<StringImm>(field), "Excepted attr/name.");
  return std::make_shared<MindNode>(func_graph_->NewCNodeInOrder(prim::kPrimGetAttr, {instance, field}));
}

AnfNodePtrList GetConstKeys(const AnfNodePtr &node) {
  AnfNodePtrList keys;
  if (node->isa<CNode>()) {
    auto inputs = node->cast<CNodePtr>()->inputs();
    keys.insert(keys.begin(), inputs.begin() + 1, inputs.end());
  } else {
    MS_EXCEPTION_IF_CHECK_FAIL(IsValueNode<ValueTuple>(node), "The keys must be a ValueTuple.");
    auto tuple = GetValueNode<ValueTuplePtr>(node);
    std::transform(tuple->value().begin(), tuple->value().end(), std::back_inserter(keys),
                   [](const ValuePtr &value) { return NewValueNode(value); });
  }
  return keys;
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
      auto key_list = GetConstKeys(array.back());
      keys.insert(keys.begin(), key_list.begin(), key_list.end());
      values.insert(values.begin(), array.begin(), array.end() - 1);
      MS_EXCEPTION_IF_CHECK_FAIL((keys.size() == values.size()), "The keys and values of Dict are not match.");
    }
    CNodePtr cnode_keys = func_graph_->NewCNodeInOrder(prim::kPrimMakeTuple, keys);
    CNodePtr cnode_values = func_graph_->NewCNodeInOrder(prim::kPrimMakeTuple, values);
    array.clear();
    array.push_back(cnode_keys);
    array.push_back(cnode_values);
  }
  CNodePtr n = func_graph_->NewCNodeInOrder(prim, array);
  UpdateLocation(n, node);
  return std::make_shared<MindNode>(n);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::CallNodePtr &node) {
  AnfNodePtrList nodes;
  if (node->GetOpCode() == CALL_FUNCTION_KW || node->GetOpCode() == CALL_FUNCTION_EX) {
    nodes.push_back(NewValueNode(std::make_shared<prim::UnpackCall>("unpack_call")));
  }
  std::transform(node->GetArgs().begin(), node->GetArgs().end(), std::back_inserter(nodes),
                 [&](const ir::NodePtr &arg) { return GetAnfNode(arg); });

  if (node->GetOpCode() == CALL_FUNCTION_KW) {
    MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(nodes.back(), prim::kPrimMakeTuple), "Expected tuple node.");
    CNodePtr keys_cnode = nodes.back()->cast<CNodePtr>();
    size_t args_cnt = nodes.size() - keys_cnode->size();
    nodes.pop_back();
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
  auto n = func_graph_->NewCNodeInOrder(std::move(nodes));
  UpdateLocation(n, node);
  return std::make_shared<MindNode>(n);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::SubscrNodePtr &node) {
  auto object = GetAnfNode(node->GetObject());
  auto subscr = GetAnfNode(node->GetSubscr());
  CNodePtr n = func_graph_->NewCNodeInOrder({GraphUtils::GetMetaFuncGraph(BINARY_SUBSCR), object, subscr});
  UpdateLocation(n, node);
  return std::make_shared<MindNode>(n);
}

ir::NodePtr FuncGraphBuilder::Mutate_(const ir::AttrNodePtr &node) {
  auto object = GetAnfNode(node->GetObject());
  auto attr = GetAnfNode(node->GetAttr());
  CNodePtr n = func_graph_->NewCNodeInOrder(prim::kPrimGetAttr, {object, attr});
  UpdateLocation(n, node);
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

AnfNodePtr FuncGraphBuilder::MergeList(const AnfNodePtr &left, const AnfNodePtr &right) {
  MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(left, prim::kPrimMakeList), "Invalid args of list extend target.");
  MS_EXCEPTION_IF_CHECK_FAIL(IsValueNode<ValueTuple>(right), "Invalid args of list extend.");
  auto inputs = left->cast<CNodePtr>()->inputs();
  AnfNodePtrList values(inputs.begin() + 1, inputs.end());
  auto valueTuple = GetValuePtr<ValueTuple>(right);
  std::for_each(valueTuple->value().begin(), valueTuple->value().end(),
                [&](const ValuePtr &value) { return values.push_back(NewValueNode(value)); });
  return func_graph_->NewCNodeInOrder(prim::kPrimMakeTuple, values);
}

std::pair<AnfNodePtrList, AnfNodePtrList> FuncGraphBuilder::GetKeysAndValueOfDict(const AnfNodePtr &node) {
  AnfNodePtrList keys;
  AnfNodePtrList values;
  if (node->isa<Parameter>()) {
    auto param = node->cast<ParameterPtr>();
    return GetKeysAndValueOfDict(assigned_vars_.at(param->name()));
  } else if (IsPrimitiveCNode(node, prim::kPrimMakeDict)) {
    auto key_tuple = node->cast<CNodePtr>()->input(1)->cast<CNodePtr>()->inputs();
    keys.assign(key_tuple.begin() + 1, key_tuple.end());
    auto value_tuple = node->cast<CNodePtr>()->input(2)->cast<CNodePtr>()->inputs();
    values.assign(value_tuple.begin() + 1, value_tuple.end());
  } else {
    MS_EXCEPTION_IF_CHECK_FAIL(IsValueNode<ValueDictionary>(node), "Can't convert non-dictionary to dict node.");
    auto dict = GetValueNode<ValueDictionaryPtr>(node);
    std::for_each(dict->value().begin(), dict->value().end(), [&](const auto &kv) {
      keys.push_back(NewValueNode(kv.first));
      values.push_back(NewValueNode(kv.second));
    });
  }
  return std::make_pair(keys, values);
}

bool IsEmptyTuple(const AnfNodePtr &node) {
  return (IsValueNode<ValueTuple>(node) && GetValueNode<ValueTuplePtr>(node)->size() == 0) ||
         (IsPrimitiveCNode(node, prim::kPrimMakeTuple) && node->cast<CNodePtr>()->size() == 1);
}

bool IsEmptyDict(const AnfNodePtr &node) {
  return (IsValueNode<ValueDictionary>(node) && GetValueNode<ValueDictionaryPtr>(node)->size() == 0) ||
         (IsPrimitiveCNode(node, prim::kPrimMakeDict) &&
          (node->cast<CNodePtr>()->size() == 1 || IsEmptyTuple(node->cast<CNodePtr>()->input(1))));
}

AnfNodePtr FuncGraphBuilder::MergeDict(const AnfNodePtr &left, const AnfNodePtr &right) {
  MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(left, prim::kPrimMakeDict), "Invalid args of dict merge target.");
  if (IsEmptyDict(left)) {
    return right;
  }
  if (IsEmptyDict(right)) {
    return left;
  }
  auto kv = GetKeysAndValueOfDict(left);
  AnfNodePtrList keys(kv.first.begin(), kv.first.end());
  AnfNodePtrList values(kv.second.begin(), kv.second.end());
  kv = GetKeysAndValueOfDict(right);
  keys.insert(keys.end(), kv.first.begin(), kv.first.end());
  values.insert(values.end(), kv.second.begin(), kv.second.end());
  CNodePtr keys_cnode = func_graph_->NewCNodeInOrder(prim::kPrimMakeTuple, keys);
  CNodePtr values_cnode = func_graph_->NewCNodeInOrder(prim::kPrimMakeTuple, values);
  return func_graph_->NewCNodeInOrder({NewValueNode(prim::kPrimMakeDict), keys_cnode, values_cnode});
}
}  // namespace pijit
}  // namespace mindspore
