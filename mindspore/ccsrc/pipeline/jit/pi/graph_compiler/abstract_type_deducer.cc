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
#include "pipeline/jit/pi/graph_compiler/abstract_type_deducer.h"
#include <algorithm>
#include <exception>
#include <iterator>
#include <memory>
#include <string>
#include "abstract/abstract_value.h"
#include "include/common/utils/python_adapter.h"
#include "ir/anf.h"
#include "ops/framework_ops.h"
#include "ops/structure_ops.h"
#include "pipeline/jit/pi/graph_compiler/abstract_type.h"
#include "pipeline/jit/pi/graph_compiler/utils.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "pipeline/jit/ps/pass.h"
#include "pipeline/jit/ps/resource.h"
#include "pipeline/jit/ps/static_analysis/static_analysis.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace jit {
namespace graph {
abstract::AbstractBasePtr GetAbstract(const ir::TypePtr &type) {
  auto abs_type = std::dynamic_pointer_cast<AbstractType>(type);
  if (abs_type == nullptr) {
    return nullptr;
  }
  return abs_type->GetAbstract();
}

void ExpandFuncGraphVarargs(const FuncGraphPtr &func_graph, AbstractBasePtrList *args_spec) {
  if (!func_graph->has_vararg()) {
    return;
  }
  auto index = func_graph->GetPositionalArgsCount();
  MS_EXCEPTION_IF_CHECK_FAIL((*args_spec)[index]->isa<abstract::AbstractTuple>(), "Varargs should be a tuple.");
  auto elems = (*args_spec)[index]->cast<abstract::AbstractTuplePtr>()->elements();
  args_spec->insert(args_spec->begin() + index, elems.begin(), elems.end());
  args_spec->erase(args_spec->begin() + index);
}

void ExpandFuncGraphKwargs(const FuncGraphPtr &func_graph, AbstractBasePtrList *args_spec) {
  if (!func_graph->has_kwarg()) {
    return;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(args_spec->back()->isa<abstract::AbstractDictionary>(), "Kwargs should be a dict.");
  auto elems = args_spec->back()->cast<abstract::AbstractDictionaryPtr>()->elements();
  args_spec->pop_back();
  for (const auto &elem : elems) {
    auto key = GetValue<std::string>(elem.first->BuildValue());
    auto kwarg_abs = std::make_shared<abstract::AbstractKeywordArg>(key, elem.second);
    args_spec->push_back(kwarg_abs);
  }
}

void PrepareEvalFuncGraph(const AnfNodePtr &func, AbstractBasePtrList *args_spec) {
  if (!IsValueNode<FuncGraph>(func)) {
    return;
  }
  auto func_graph = GetValueNode<FuncGraphPtr>(func);
  pipeline::ResourcePtr res = std::make_shared<pipeline::Resource>();
  res->set_func_graph(func_graph);
  res->set_args_abs(*args_spec);
  parse::ResolveFuncGraph(func_graph, res);
  pipeline::MetaUnpackPreparePass(res);
  ExpandFuncGraphVarargs(func_graph, args_spec);
  ExpandFuncGraphKwargs(func_graph, args_spec);
}

abstract::AbstractBasePtr EvalFunctionValue(const AnfNodePtr &func, AbstractBasePtrList *args_spec) {
  if (func->isa<Primitive>()) {
    auto prim = GetValueNode<PrimitivePtr>(func);
    auto res = abstract::EvalOnePrim(prim, *args_spec);
    if (res != nullptr) {
      return res->abstract();
    }
  }
  PrepareEvalFuncGraph(func, args_spec);
  return pipeline::AbstractAnalyze(GetValueNode(func), *args_spec).eval_result->abstract();
}

abstract::AbstractBasePtr DeduceNodeAbstract(const ir::NodePtr &node) {
  MS_EXCEPTION_IF_CHECK_FAIL(node->isa<ir::Operation>(), "Invalid Node[" + node->GetNodeName() + "] to deduce.");
  auto op = node->cast<ir::OperationPtr>();
  // prim::pPrimMakeDict and no elements
  if (op->GetOpCode() == BUILD_MAP && op->GetArgsCnt() == 0) {
    return abstract::ToAbstract(parse::data_converter::PyDataToValue(py::dict()));
  }
  auto func = GraphUtils::GetPrimOrMetaFuncGraph(op->GetOpCode());
  ir::NodePtrList args(op->GetArgs().begin(), op->GetArgs().end());
  if (node->isa<ir::CallNode>()) {
    auto type = std::dynamic_pointer_cast<AbstractType>(node->cast<ir::CallNodePtr>()->GetArg()->GetType());
    MS_EXCEPTION_IF_NULL(type);
    auto callable = type->GetPythonObject();
    MS_EXCEPTION_IF_CHECK_FAIL(!py::isinstance<py::none>(callable), "callable can't be none.");
    func = GraphUtils::ConvertPythonObjectToAnfNode(callable);
    // first opnd is callable object
    args.erase(args.begin());
  }
  if (func == nullptr || func.get() == nullptr) {
    return nullptr;
  }
  abstract::AbstractBasePtrList args_spec;
  std::for_each(args.begin(), args.end(), [&args_spec](const ir::NodePtr &arg) {
    auto abs = GetAbstract(arg->GetType());
    if (abs != nullptr) {
      args_spec.push_back(abs);
    }
  });
  if (args.size() != args_spec.size()) {
    return nullptr;
  }
  try {
    return EvalFunctionValue(func, &args_spec);
  } catch (const std::exception &e) {
    MS_LOG_WARNING << e.what();
  }
  return nullptr;
}

void AbstractTypeDeducer::Deduce(const ir::FunctionNodePtr &func, const ir::py::tuple &args,
                                 const ir::py::dict &kwargs) {
  auto deducer = std::make_shared<AbstractTypeDeducer>();
  std::for_each(func->GetParameters().begin(), func->GetParameters().end(), [&](const ir::NodePtr &node) {
    auto param = node->cast<ir::ParameterPtr>();
    auto var = (param->GetCategory() == ir::Parameter::KEYWORD) ? std::make_shared<ir::Value>(kwargs)
                                                                : std::make_shared<ir::Value>(args[param->GetIndex()]);
    deducer->Visit(var);
    param->SetType(var->GetType());
    deducer->assigned_local_vars_[param->GetName()] = var;
  });
  deducer->Visit_(func);
}

#define DEFINE_NODE_VISIT_FUNC_(OP)                                          \
  void AbstractTypeDeducer::Visit_(const OP &node) {                         \
    VISIT_NODE_LIST(node->GetArgs())                                         \
    node->SetType(std::make_shared<AbstractType>(DeduceNodeAbstract(node))); \
  }

DEFINE_NODE_VISIT_FUNC_(ir::CastNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::DeleteNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::GetNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::InvertNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::NegativeNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::NotNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::ReturnNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::UnaryOperationPtr)

DEFINE_NODE_VISIT_FUNC_(ir::AddNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::SubNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::MulNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::DivNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::BitwiseNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::CompareNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::ContainsNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::IsNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::JumpNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::BinaryOperationPtr)

DEFINE_NODE_VISIT_FUNC_(ir::BuildNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::CallNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::NaryWithFlagNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::FormatNodePtr)
DEFINE_NODE_VISIT_FUNC_(ir::NaryOperationPtr)

void AbstractTypeDeducer::Visit_(const ir::RefNodePtr &node) {
  Visit(node->GetRealNode());
  node->SetType(node->GetRealNode()->GetType());
}

void AbstractTypeDeducer::Visit_(const ir::ValuePtr &node) {
  auto value = GraphUtils::ConvertPythonObjectToAnfNode(node->GetValue());
  node->SetType(std::make_shared<AbstractType>(abstract::ToAbstract(GetValueNode(value)), node->GetValue()));
}

void AbstractTypeDeducer::Visit_(const ir::FunctionNodePtr &node) {
  VISIT_NODE_LIST(node->GetParameters())
  VISIT_NODE_LIST(node->GetNodes())
  node->SetType(node->GetNodes().back()->GetType());
}

void AbstractTypeDeducer::Visit_(const ir::LoadValueNodePtr &node) {
  VISIT_NODE_LIST(node->GetArgs())
  node->SetType(node->GetArg()->GetType());
  auto op = node->GetOpCode();
  if (op != LOAD_FAST && op != LOAD_GLOBAL) {
    return;
  }
  auto arg = node->GetArg();
  MS_EXCEPTION_IF_CHECK_FAIL(arg->isa<ir::Value>(), "Invalid arg[" + arg->GetNodeName() + "] for LoadValueNode.");
  auto name = arg->cast<ir::ValuePtr>()->GetName();
  if (op == LOAD_FAST) {
    MS_EXCEPTION_IF_CHECK_FAIL(assigned_local_vars_.find(name) != assigned_local_vars_.end(),
                               "Not defined var[" + name + "].");
    node->SetType(assigned_local_vars_.at(name)->GetType());
  } else {
    if (assigned_global_vars_.find(name) != assigned_global_vars_.end()) {
      node->SetType(assigned_global_vars_.at(name)->GetType());
    }
  }
}

void AbstractTypeDeducer::Visit_(const ir::LoadFieldNodePtr &node) {
  VISIT_NODE_LIST(node->GetArgs())
  auto abs = DeduceNodeAbstract(node);
  py::object cls = std::dynamic_pointer_cast<AbstractType>(node->GetArg()->GetType())->GetPythonObject();
  auto arg = node->GetArg(1);
  MS_EXCEPTION_IF_CHECK_FAIL(arg->isa<ir::Value>(), "Invalid arg[" + arg->GetNodeName() + "] for LoadFieldNode.");
  auto attr = arg->cast<ir::ValuePtr>()->GetName();
  py::object obj = python_adapter::GetPyObjAttr(cls, attr);
  if (abs == nullptr && !py::isinstance<py::none>(obj)) {
    abs = abstract::ToAbstract(parse::data_converter::PyDataToValue(obj));
  }
  node->SetType(std::make_shared<AbstractType>(abs, obj));
}

void AbstractTypeDeducer::Visit_(const ir::StoreNodePtr &node) {
  VISIT_NODE_LIST(node->GetArgs())
  node->SetType(node->GetArg()->GetType());
  auto op = node->GetOpCode();
  if (op != STORE_FAST && op != STORE_GLOBAL) {
    return;
  }
  auto arg = node->GetArg(1);
  MS_EXCEPTION_IF_CHECK_FAIL(arg->isa<ir::Value>(), "Invalid arg[" + arg->GetNodeName() + "] for StoreNode.");
  auto name = arg->cast<ir::ValuePtr>()->GetName();
  if (op == STORE_FAST) {
    assigned_local_vars_[name] = node->GetArg();
  } else {
    assigned_global_vars_[name] = node->GetArg();
  }
}

void AbstractTypeDeducer::Visit_(const ir::UpdateNodePtr &node) {
  VISIT_NODE_LIST(node->GetArgs())
  auto type = std::dynamic_pointer_cast<AbstractType>(node->GetArg(1)->GetType());
  if (node->GetOpCode() == LIST_EXTEND) {
    abstract::AbstractBasePtr abs = type->GetAbstract();
    MS_EXCEPTION_IF_NULL(abs);
    auto new_abs = std::make_shared<abstract::AbstractList>(abs->cast<abstract::AbstractSequencePtr>()->elements());
    node->GetArg(1)->SetType(std::make_shared<AbstractType>(new_abs));
  }
  node->SetType(std::make_shared<AbstractType>(DeduceNodeAbstract(node)));
  if (node->GetOpCode() == LIST_EXTEND) {
    node->GetArg(1)->SetType(type);
  }
}

void AbstractTypeDeducer::Visit_(const ir::IfNodePtr &node) {
  Visit(node->GetCondition());
  VISIT_NODE_LIST(node->GetThen())
  VISIT_NODE_LIST(node->GetElse())
}

void AbstractTypeDeducer::Visit_(const ir::WhileNodePtr &node) {
  Visit(node->GetCondition());
  VISIT_NODE_LIST(node->GetBody())
}

void AbstractTypeDeducer::Visit_(const ir::SubscrNodePtr &node) {
  Visit(node->GetObject());
  Visit(node->GetSubscr());
  auto func = GraphUtils::GetPrimOrMetaFuncGraph(BINARY_SUBSCR);
  abstract::AbstractBasePtrList args_spec = {GetAbstract(node->GetObject()->GetType()),
                                             GetAbstract(node->GetSubscr()->GetType())};
  node->SetType(std::make_shared<AbstractType>(EvalFunctionValue(func, &args_spec)));
}

void AbstractTypeDeducer::Visit_(const ir::AttrNodePtr &node) {
  Visit(node->GetObject());
  Visit(node->GetAttr());
  abstract::AbstractBasePtrList args_spec = {GetAbstract(node->GetObject()->GetType()),
                                             GetAbstract(node->GetAttr()->GetType())};
  node->SetType(std::make_shared<AbstractType>(EvalFunctionValue(NewValueNode(prim::kPrimGetAttr), &args_spec)));
}

void AbstractTypeDeducer::Visit_(const ir::PairNodePtr &node) {
  Visit(node->GetFirst());
  Visit(node->GetSecond());
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
