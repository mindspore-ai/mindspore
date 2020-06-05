
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

#include "operator/composite/composite.h"
#include <algorithm>
#include <utility>
#include <sstream>

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "pipeline/static_analysis/abstract_value.h"
#include "pipeline/static_analysis/abstract_function.h"
#include "pipeline/static_analysis/dshape.h"
#include "pipeline/static_analysis/param_validator.h"
#include "operator/cc_implementations.h"
#include "optimizer/opt.h"
#include "utils/symbolic.h"
#include "pybind_api/api_register.h"
#include "./common.h"
#include "ir/signature.h"
#include "debug/trace.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
using AbstractTensor = mindspore::abstract::AbstractTensor;
using FuncGraphAbstractClosure = mindspore::abstract::FuncGraphAbstractClosure;

using mindspore::abstract::AbstractAttribute;
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractClass;
using mindspore::abstract::AbstractDictionary;
using mindspore::abstract::AbstractDictionaryPtr;
using mindspore::abstract::AbstractEllipsis;
using mindspore::abstract::AbstractEllipsisPtr;
using mindspore::abstract::AbstractFunction;
using mindspore::abstract::AbstractFunctionPtr;
using mindspore::abstract::AbstractList;
using mindspore::abstract::AbstractNone;
using mindspore::abstract::AbstractScalar;
using mindspore::abstract::AbstractSlice;
using mindspore::abstract::AbstractTuple;

ElemwiseMap kElemwiseMap = {{"__add__", kPrimScalarAdd}, {"__sub__", kPrimScalarSub}, {"__mul__", kPrimScalarMul},
                            {"__truediv__", nullptr},    {"__floordiv__", nullptr},   {"__mod__", kPrimScalarMod},
                            {"__pow__", kPrimScalarPow}, {"__eq__", kPrimScalarEq},   {"__lt__", kPrimScalarLt},
                            {"__gt__", kPrimScalarGt},   {"__ne__", kPrimScalarNe},   {"__le__", kPrimScalarLe},
                            {"__ge__", kPrimScalarGe}};

const MetaFuncGraphPtr kTail = std::make_shared<Tail>("tail");

// copy from python API: reduce.
// Apply a function of two arguments cumulatively to the items of a sequence,
// from left to right, so as to reduce the sequence to a single value.For example,
// reduce(lambda x, y: x + y, [ 1, 2, 3, 4, 5 ]) calculates ((((1 + 2) + 3) + 4) + 5).
AnyPtr Reduce(const OpsFunction &func, const AnyPtrList &list) {
  std::shared_ptr<Any> ret;
  size_t size = list.size();
  if (size < 2) {
    MS_LOG(EXCEPTION) << "length of inputs of Reduce is less than 2";
  }

  AnyPtrList input;
  input.push_back(list[0]);
  input.push_back(list[1]);
  ret = std::make_shared<Any>(func(input));

  for (size_t i = 2; i < size; ++i) {
    input.clear();
    input.push_back(ret);
    input.push_back(list[i]);
    ret = std::make_shared<Any>(func(input));
  }

  return ret;
}

AnfNodePtr Reduce(const AnfNodeOpsFunction &func, const std::vector<AnfNodePtr> &list) {
  size_t size = list.size();
  if (size < 2) {
    MS_LOG(EXCEPTION) << "length of inputs of Reduce is less than 2";
  }

  std::vector<AnfNodePtr> input;
  input.push_back(list[0]);
  input.push_back(list[1]);
  AnfNodePtr ret = func(input);

  for (size_t i = 2; i < size; ++i) {
    input.clear();
    input.push_back(ret);
    input.push_back(list[i]);
    ret = func(input);
  }

  return ret;
}

ValuePtr kCompositeHyperMap = std::make_shared<HyperMap>();

void HyperMap::Init() {
  if (fn_leaf_) {
    name_ = "hyper_map[" + fn_leaf_->name() + "]";
  }
  signatures_ =
    // def hypermap(func:read, *args:ref):
    std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                            {"args", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindVarPositional}});
}

HyperMap::HyperMap(const std::shared_ptr<MultitypeFuncGraph> &fn_leaf)
    : MetaFuncGraph("hyper_map"),
      fn_leaf_(fn_leaf),
      broadcast_(false),
      nonleaf_({kObjectTypeList, kObjectTypeTuple, kObjectTypeClass}) {
  Init();
}

HyperMap::HyperMap(const HyperMap &h)
    : MetaFuncGraph("hyper_map"), fn_leaf_(h.fn_leaf_), broadcast_(h.broadcast_), nonleaf_(h.nonleaf_) {
  Init();
}

AnfNodePtr HyperMap::FullMake(TypePtr, const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg,
                              const ArgsPairList &arg_map) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> inputs;
  if (fn_arg != nullptr) {
    inputs.push_back(fn_arg);
  } else {
    inputs.push_back(NewValueNode(fn_leaf_));
  }

  (void)std::transform(arg_map.begin(), arg_map.end(), std::back_inserter(inputs),
                       [](const std::pair<AnfNodePtr, Any> &item) { return item.first; });
  return func_graph->NewCNode(inputs);
}

AnfNodePtr HyperMap::FullMake(const std::shared_ptr<List> &type, const FuncGraphPtr &func_graph,
                              const AnfNodePtr &fn_arg, const ArgsPairList &arg_map) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(type);

  std::size_t size = type->elements().size();
  bool is_not_same = std::any_of(arg_map.begin(), arg_map.end(), [size](const std::pair<AnfNodePtr, TypePtr> &item) {
    auto lhs = std::static_pointer_cast<List>(item.second);
    MS_EXCEPTION_IF_NULL(lhs);
    return lhs->elements().size() != size;
  });
  if (is_not_same) {
    MS_LOG(EXCEPTION) << "List in HyperMap should have same length";
  }

  // cannot use shared_from_base() also known as this, as it will make a reference cycle on
  // hypermap and graph generated, it will cause memory leak.
  auto fn_rec = NewValueNode(std::make_shared<HyperMap>(*this));
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimMakeList));

  for (int i = 0; i < SizeToInt(size); ++i) {
    std::vector<AnfNodePtr> inputs2;
    inputs2.push_back(fn_rec);
    if (fn_arg != nullptr) {
      inputs2.push_back(fn_arg);
    }

    (void)std::transform(
      arg_map.begin(), arg_map.end(), std::back_inserter(inputs2),
      [&func_graph, i](const std::pair<AnfNodePtr, Any> &item) {
        return func_graph->NewCNode({NewValueNode(prim::kPrimListGetItem), item.first, NewValueNode(i)});
      });

    inputs.push_back(func_graph->NewCNode(inputs2));
  }
  return func_graph->NewCNode(inputs);
}

AnfNodePtr HyperMap::FullMake(const std::shared_ptr<Tuple> &type, const FuncGraphPtr &func_graph,
                              const AnfNodePtr &fn_arg, const ArgsPairList &arg_map) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(type);

  std::size_t size = type->elements().size();
  bool is_not_same = std::any_of(arg_map.begin(), arg_map.end(), [size](const std::pair<AnfNodePtr, TypePtr> &item) {
    auto lhs = std::static_pointer_cast<Tuple>(item.second);
    MS_EXCEPTION_IF_NULL(lhs);
    return lhs->elements().size() != size;
  });
  if (is_not_same) {
    MS_LOG(EXCEPTION) << "tuple in HyperMap should have same length";
  }

  // cannot use shared_from_base() also known as this, as it will make a reference cycle on
  // hypermap and graph generated, it will cause memory leak.
  auto fn_rec = NewValueNode(std::make_shared<HyperMap>(*this));
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimMakeTuple));

  for (int i = 0; i < SizeToInt(size); ++i) {
    std::vector<AnfNodePtr> inputs2;
    inputs2.push_back(fn_rec);
    if (fn_arg != nullptr) {
      inputs2.push_back(fn_arg);
    }

    (void)std::transform(
      arg_map.begin(), arg_map.end(), std::back_inserter(inputs2), [&func_graph, &i](std::pair<AnfNodePtr, Any> item) {
        return func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), item.first, NewValueNode(i)});
      });

    inputs.push_back(func_graph->NewCNode(inputs2));
  }
  return func_graph->NewCNode(inputs);
}

AnfNodePtr HyperMap::FullMake(const std::shared_ptr<Class> &type, const FuncGraphPtr &func_graph,
                              const AnfNodePtr &fn_arg, const ArgsPairList &arg_map) {
  MS_EXCEPTION_IF_NULL(type);
  MS_EXCEPTION_IF_NULL(func_graph);

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimMakeRecord));
  inputs.push_back(NewValueNode(type));

  // cannot use shared_from_base() also known as this, as it will make a reference cycle on
  // hypermap and graph generated, it will cause memory leak.
  auto fn_rec = NewValueNode(std::make_shared<HyperMap>(*this));
  std::size_t attrSize = type->GetAttributes().size();
  for (std::size_t i = 0; i < attrSize; ++i) {
    std::vector<AnfNodePtr> inputs2;
    inputs2.push_back(fn_rec);
    if (fn_arg) {
      inputs2.push_back(fn_arg);
    }

    int j = 0;
    for (auto item : arg_map) {
      inputs2.push_back(func_graph->NewCNode({NewValueNode(prim::kPrimGetAttr), item.first, NewValueNode(j)}));
      j++;
    }

    inputs.push_back(func_graph->NewCNode(inputs2));
  }
  return func_graph->NewCNode(inputs);
}

AnfNodePtr HyperMap::Make(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg, const ArgsPairList &arg_map) {
  bool found = false;
  TypeId id = kObjectTypeEnd;
  std::pair<AnfNodePtr, TypePtr> pair;
  for (auto &item : arg_map) {
    pair = item;
    id = item.second->type_id();
    if (nonleaf_.count(id)) {
      found = true;
      break;
    }
  }

  if (found) {
    // In a nonleaf situation, all arguments must have the same generic.
    bool is_not_same = std::any_of(arg_map.begin(), arg_map.end(), [pair](const std::pair<AnfNodePtr, TypePtr> &item) {
      if (item.first != pair.first) {
        return item.second->type_id() != pair.second->type_id();
      }
      return false;
    });
    if (is_not_same) {
      std::ostringstream oss;
      oss << "There are " << arg_map.size() << " inputs of `" << name_ << "`, corresponding type info:\n"
          << trace::GetDebugInfo(func_graph->debug_info()) << "\n";
      int idx = 0;
      for (auto &item : arg_map) {
        oss << ++idx << ": " << item.second->ToString() << "\n";
      }
      MS_LOG(EXCEPTION) << "HyperMap cannot match up all input types of arguments.\n" << oss.str();
    }
  }

  switch (id) {
    case kObjectTypeList: {
      auto type = std::static_pointer_cast<List>(pair.second);
      return FullMake(type, func_graph, fn_arg, arg_map);
    }
    case kObjectTypeTuple: {
      auto type = std::static_pointer_cast<Tuple>(pair.second);
      return FullMake(type, func_graph, fn_arg, arg_map);
    }
    case kObjectTypeClass: {
      auto type = std::static_pointer_cast<Class>(pair.second);
      return FullMake(type, func_graph, fn_arg, arg_map);
    }
    default:
      return FullMake(pair.second, func_graph, fn_arg, arg_map);
  }
}

ArgsPairList HyperMap::Harmonize(const FuncGraphPtr &func_graph, const ArgsPairList &args_spec_list) {
  TypePtr type_tensor = std::make_shared<TensorType>();
  bool flag = std::any_of(
    args_spec_list.begin(), args_spec_list.end(),
    [type_tensor](const std::pair<AnfNodePtr, TypePtr> &item) { return IsSubType(item.second, type_tensor); });
  if (flag && broadcast_) {
    ArgsPairList ret;
    for (auto &item : args_spec_list) {
      if (!IsSubType(item.second, type_tensor)) {
        TypePtr type_tensor_ele = std::make_shared<TensorType>(item.second);
        ret.push_back(
          std::make_pair(func_graph->NewCNode({NewValueNode(prim::kPrimScalarToArray), item.first}), type_tensor_ele));
      } else {
        ret.push_back(std::make_pair(item.first, item.second));
      }
    }
    return ret;
  }
  return args_spec_list;
}

FuncGraphPtr HyperMap::GenerateFromTypes(const TypePtrList &args_spec_list) {
  FuncGraphPtr ptrGraph = std::make_shared<FuncGraph>();
  ptrGraph->set_flags(FUNC_GRAPH_FLAG_CORE, true);
  ptrGraph->set_flags(FUNC_GRAPH_FLAG_SPECIALIZE_PARAMETER, true);
  ptrGraph->debug_info()->set_name("hyper_map");

  AnfNodePtr ptrFnArg = nullptr;
  std::size_t i = 0;
  ArgsPairList argmap;
  ArgsPairList argmap2;
  if (fn_leaf_ == nullptr) {
    ptrFnArg = ptrGraph->add_parameter();
    i = 1;
  }

  std::size_t size = args_spec_list.size();
  for (; i < size; ++i) {
    argmap.push_back(std::make_pair(ptrGraph->add_parameter(), args_spec_list[i]));
  }

  argmap2 = Harmonize(ptrGraph, argmap);
  ptrGraph->set_output(Make(ptrGraph, ptrFnArg, argmap2));
  return ptrGraph;
}

abstract::AbstractBasePtrList HyperMap::NormalizeArgs(const AbstractBasePtrList &args_spec_list) const {
  if (fn_leaf_ == nullptr) {
    MS_EXCEPTION_IF_NULL(args_spec_list[0]);
    // Assert that hypermap's function param does not contain free variables
    if (args_spec_list[0]->isa<FuncGraphAbstractClosure>()) {
      auto graph_func = dyn_cast<FuncGraphAbstractClosure>(args_spec_list[0]);
      auto func_graph = graph_func->func_graph();
      if (func_graph->parent() != nullptr) {
        MS_LOG(EXCEPTION) << "HyperMap don't support Closure with free variable yet.";
      }
    }
  }

  AbstractBasePtrList broadened;
  (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(broadened),
                       [](const AbstractBasePtr &arg) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(arg);
                         return arg->Broaden();
                       });
  return broadened;
}

REGISTER_PYBIND_DEFINE(HyperMap_, ([](const py::module *m) {
                         (void)py::class_<HyperMapPy, MetaFuncGraph, std::shared_ptr<HyperMapPy>>(*m, "HyperMap_")
                           .def(py::init<std::shared_ptr<MultitypeFuncGraph>>(), py::arg("leaf"))
                           .def(py::init<>());
                       }));

FuncGraphPtr Tail::GenerateTupleFuncGraph(const abstract::AbstractTuplePtr &a_tuple) {
  MS_EXCEPTION_IF_NULL(a_tuple);

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flags(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("tail");
  AnfNodePtr ptrTup = ret->add_parameter();

  std::vector<AnfNodePtr> elems;
  elems.push_back(NewValueNode(prim::kPrimMakeTuple));

  int tuple_size = SizeToInt(a_tuple->size());
  for (int i = 1; i < tuple_size; ++i) {
    elems.push_back(ret->NewCNode({NewValueNode(prim::kPrimTupleGetItem), ptrTup, NewValueNode(i)}));
  }

  ret->set_output(ret->NewCNode(elems));
  return ret;
}

FuncGraphPtr Tail::GenerateListFuncGraph(const abstract::AbstractListPtr &a_list) {
  MS_EXCEPTION_IF_NULL(a_list);

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flags(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("tail");
  AnfNodePtr ptrList = ret->add_parameter();

  std::vector<AnfNodePtr> elems;
  elems.push_back(NewValueNode(prim::kPrimMakeList));

  int list_size = SizeToInt(a_list->size());
  for (int i = 1; i < list_size; ++i) {
    elems.push_back(ret->NewCNode({NewValueNode(prim::kPrimListGetItem), ptrList, NewValueNode(i)}));
  }

  ret->set_output(ret->NewCNode(elems));
  return ret;
}

FuncGraphPtr Tail::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  if (args_spec_list.size() != 1) {
    MS_LOG(EXCEPTION) << "tail requires a non-empty tuple.";
  }

  AbstractBasePtr a = args_spec_list[0];
  abstract::AbstractTuplePtr a_tuple = dyn_cast<AbstractTuple>(a);
  if (a_tuple != nullptr) {
    return GenerateTupleFuncGraph(a_tuple);
  }

  abstract::AbstractListPtr a_list = dyn_cast<AbstractList>(a);
  if (a_list != nullptr) {
    return GenerateListFuncGraph(a_list);
  }

  MS_LOG(EXCEPTION) << "arg0 must be AbstractTuple or AbstractList, but: " << a->ToString();
}

REGISTER_PYBIND_DEFINE(
  Tail_, ([](const py::module *m) {
    (void)py::class_<Tail, MetaFuncGraph, std::shared_ptr<Tail>>(*m, "Tail_").def(py::init<std::string &>());
  }));

FuncGraphPtr MakeTupleGradient::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  int tuple_size = SizeToInt(args_spec_list.size());

  std::ostringstream ss;
  ss << "▶make_tuple_" << tuple_size;
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  fg->debug_info()->set_name(ss.str());

  std::vector<AnfNodePtr> params;
  params.push_back(NewValueNode(prim::kPrimMakeTuple));
  for (int i = 0; i < tuple_size; ++i) {
    params.push_back(fg->add_parameter());
  }

  // make fprob first result, maketuple's forward result.
  AnfNodePtr out = fg->NewCNode(params);

  // make fprob second result, maketuple's backward function.
  FuncGraphPtr b = std::make_shared<FuncGraph>();

  ss.clear();
  ss << "◀make_tuple_" << tuple_size;
  b->debug_info()->set_name(ss.str());
  AnfNodePtr dout = b->add_parameter();

  std::vector<AnfNodePtr> grads;
  grads.push_back(NewValueNode(prim::kPrimMakeTuple));
  grads.push_back(NewValueNode(newenv));
  for (int i = 0; i < tuple_size; ++i) {
    grads.push_back(b->NewCNode({NewValueNode(prim::kPrimTupleGetItem), dout, NewValueNode(i)}));
  }

  b->set_flags(FUNC_GRAPH_FLAG_CORE, true);
  b->set_output(b->NewCNode(grads));

  fg->set_flags(FUNC_GRAPH_FLAG_CORE, true);
  fg->set_output(fg->NewCNode({NewValueNode(prim::kPrimMakeTuple), out, NewValueNode(b)}));
  (void)fg->transforms().emplace("primal", FuncGraphTransform(prim::kPrimMakeTuple));
  return fg;
}

GradOperation::GradOperation(const std::string &name, bool get_all, bool get_by_list, bool sens_param)
    : MetaFuncGraph(name), get_all_(get_all), get_by_list_(get_by_list), sens_param_(sens_param) {
  if (get_by_list) {
    signatures_ =
      // def grad(func:read, weight_list:ref):
      std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                              {"weight_list", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindDefault}});
  }
}

FuncGraphPtr GradOperation::GetGrad(AnfNodePtr node, const AnfNodePtr &weights,
                                    const std::vector<AnfNodePtr> &params_list, const std::vector<AnfNodePtr> &args,
                                    bool applyJ) {
  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flags(FUNC_GRAPH_FLAG_CORE, true);

  auto weights_node = weights;
  if (weights == nullptr && !args.empty()) {
    weights_node = ret->NewCNode(args);
  }

  ValueNodePtr opsJ = NewValueNode(prim::kPrimJ);
  ValueNodePtr opsTupleItem = NewValueNode(prim::kPrimTupleGetItem);

  std::vector<AnfNodePtr> inputs;
  if (applyJ) {
    inputs.push_back(opsJ);
    inputs.push_back(node);
    node = ret->NewCNode(inputs);
  }

  std::vector<AnfNodePtr> params;
  for (size_t i = 0; i < params_list.size(); ++i) {
    params.push_back(ret->add_parameter());
  }

  inputs.clear();
  inputs.push_back(node);
  (void)std::copy(params.begin(), params.end(), std::back_inserter(inputs));
  AnfNodePtr cnode = ret->NewCNode(inputs);

  inputs.clear();
  inputs.push_back(opsTupleItem);
  inputs.push_back(cnode);
  inputs.push_back(NewValueNode(0));
  auto out = ret->NewCNode(inputs);

  inputs.clear();
  inputs.push_back(opsTupleItem);
  inputs.push_back(cnode);
  inputs.push_back(NewValueNode(1));
  AnfNodePtr ptrBprop = ret->NewCNode(inputs);

  doGetGrad(ret, out, ptrBprop, weights_node, opsTupleItem);
  return ret;
}

void GradOperation::doGetGrad(const FuncGraphPtr &func_graph, AnfNodePtr out, AnfNodePtr ptrBprop, AnfNodePtr weights,
                              ValueNodePtr opsTupleItem) {
  MS_EXCEPTION_IF_NULL(func_graph);

  AnfNodePtr ptrBPropArg = nullptr;
  if (sens_param_) {
    ptrBPropArg = func_graph->add_parameter();
  } else {
    auto ones_like = prim::GetPythonOps("ones_like");
    ptrBPropArg = func_graph->NewCNode({NewValueNode(ones_like), out});
  }

  AnfNodePtr ptrBApp = func_graph->NewCNode({ptrBprop, ptrBPropArg});

  CNodePtr fv_bprop = nullptr;
  if (get_by_list_) {
    // python code: grads = hyper_map(F.partial(env_get, env), weights)
    AnfNodePtr env = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), ptrBApp, NewValueNode(0)});
    AnfNodePtr partial_env_get =
      func_graph->NewCNode({NewValueNode(prim::kPrimPartial), NewValueNode(prim::GetPythonOps("env_get")), env});
    MetaFuncGraphPtr hyper_map = std::make_shared<HyperMap>();
    fv_bprop = func_graph->NewCNode({NewValueNode(hyper_map), partial_env_get, weights});
  }

  CNodePtr inputs_bprop = nullptr;
  if (get_all_) {
    inputs_bprop = func_graph->NewCNode({NewValueNode(kTail), ptrBApp});
  }

  // Gradients wrt inputs and parameters
  if (fv_bprop != nullptr && inputs_bprop != nullptr) {
    func_graph->set_output(func_graph->NewCNode({NewValueNode(kPrimMakeTuple), inputs_bprop, fv_bprop}));
    return;
  }

  // Gradients wrt parameters
  if (fv_bprop != nullptr) {
    func_graph->set_output(fv_bprop);
    return;
  }

  // Gradients wrt inputs
  if (inputs_bprop != nullptr) {
    func_graph->set_output(inputs_bprop);
    return;
  }

  // Gradients wrt first input.
  // ptrBApp returns (EnvInstance(grads wrt params), grads wrt input0, grads wrt input1, ...), so 1 is for first input
  func_graph->set_output(func_graph->NewCNode({opsTupleItem, ptrBApp, NewValueNode(1)}));
}

// Generate the graph.
FuncGraphPtr GradOperation::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  if (args_spec_list.size() < 1) {
    MS_LOG(EXCEPTION) << "GenerateGraph requires at least 1 parameters, while the input size is "
                      << args_spec_list.size() << ".";
  }

  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  AbstractFunctionPtr fn = dyn_cast<AbstractFunction>(args_spec_list[0]);
  if (fn == nullptr) {
    MS_LOG(EXCEPTION) << "GradOperation arg0 must be AbstractFunction, but " << args_spec_list[0]->ToString();
  }

  // Waiting for implementation.
  auto real_fn = dyn_cast<FuncGraphAbstractClosure>(fn);
  MS_EXCEPTION_IF_NULL(real_fn);

  FuncGraphPtr ptrGraph = real_fn->func_graph();
  MS_EXCEPTION_IF_NULL(ptrGraph);
  TraceManager::DebugTrace(std::make_shared<TraceGradOperation>(ptrGraph->debug_info()));
  FuncGraphPtr dfBuilder = std::make_shared<FuncGraph>();
  TraceManager::EndTrace();
  auto nparam = ptrGraph->parameters().size();

  std::ostringstream ss;
  ss << "grad{" << nparam << "}";
  dfBuilder->set_flags(FUNC_GRAPH_FLAG_CORE, true);
  dfBuilder->debug_info()->set_name(ss.str());
  ParameterPtr param_graph = dfBuilder->add_parameter();

  AnfNodePtr weights = nullptr;
  if (get_by_list_) {
    weights = dfBuilder->add_parameter();
  }

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimJ));
  inputs.push_back(param_graph);
  auto jf = dfBuilder->NewCNode(inputs);
  // df is checked in GetGrad
  TraceManager::DebugTrace(std::make_shared<TraceGradOperation>(ptrGraph->debug_info()));
  auto df = GetGrad(jf, weights, ptrGraph->parameters());
  TraceManager::EndTrace();
  dfBuilder->set_output(NewValueNode(df));

  return dfBuilder;
}

REGISTER_PYBIND_DEFINE(GradOperation_, ([](const py::module *m) {
                         (void)py::class_<GradOperation, MetaFuncGraph, std::shared_ptr<GradOperation>>(
                           *m, "GradOperation_")
                           .def(py::init<std::string &>(), py::arg("fn"))
                           .def(py::init<std::string &, bool, bool, bool>(), py::arg("fn"), py::arg("get_all"),
                                py::arg("get_by_list"), py::arg("sens_param"));
                       }));

// Generate the ListMap func graph.
FuncGraphPtr ListMap::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  size_t args_num = args_spec_list.size();
  // args: fn, list1, list2, ...
  if (args_num < 2) {
    MS_LOG(EXCEPTION) << "list_map takes at least two arguments";
  }

  for (size_t i = 1; i < args_num; ++i) {
    if (typeid(args_spec_list[i]) != typeid(AbstractBase)) {
      // The function currently not be use
      MS_LOG(EXCEPTION) << "list_map requires lists, not {t}'";
    }
  }

  FuncGraphPtr fg_ptr = std::make_shared<FuncGraph>();
  fg_ptr->set_flags(FUNC_GRAPH_FLAG_CORE, true);
  fg_ptr->debug_info()->set_name("list_map");
  AnfNodePtr fn = fg_ptr->add_parameter();

  std::vector<AnfNodePtr> lists;
  for (size_t i = 1; i < args_num; ++i) {
    lists.push_back(fg_ptr->add_parameter());
  }

  std::vector<AnfNodePtr> iters;
  (void)std::transform(lists.begin(), lists.end(), std::back_inserter(iters), [fg_ptr](AnfNodePtr item) {
    return fg_ptr->NewCNode({NewValueNode(std::string("list_iter")), item});
  });

  std::vector<AnfNodePtr> nexts;
  (void)std::transform(iters.begin(), iters.end(), std::back_inserter(nexts), [fg_ptr](AnfNodePtr item) {
    return fg_ptr->NewCNode({NewValueNode(std::string("next")), item});
  });

  std::vector<AnfNodePtr> values;
  (void)std::transform(nexts.begin(), nexts.end(), std::back_inserter(values), [fg_ptr](AnfNodePtr item) {
    return fg_ptr->NewCNode({NewValueNode(prim::kPrimTupleGetItem), item});
  });

  (void)std::transform(nexts.begin(), nexts.end(), std::back_inserter(iters), [fg_ptr](AnfNodePtr item) {
    return fg_ptr->NewCNode({NewValueNode(prim::kPrimTupleGetItem), item, NewValueNode(1)});
  });

  (void)values.insert(values.begin(), fn);
  AnfNodePtr cnode_graph = fg_ptr->NewCNode(values);
  AnfNodePtr resl = fg_ptr->NewCNode({NewValueNode(prim::kPrimMakeList), cnode_graph});

  FuncGraphPtr fgnext_ptr = std::make_shared<FuncGraph>();
  fgnext_ptr->debug_info()->set_name("body");

  FuncGraphPtr fgcond_ptr = std::make_shared<FuncGraph>();
  fgcond_ptr->debug_info()->set_name("cond");

  MakeCond(lists, fgnext_ptr, fgcond_ptr);
  MakeNext(lists, fgcond_ptr, fgnext_ptr);

  CNodePtr output_cnode = fg_ptr->NewCNode({NewValueNode(fgcond_ptr), fn, resl});

  auto inputs = output_cnode->inputs();
  (void)inputs.insert(inputs.end(), iters.begin(), iters.end());
  output_cnode->set_inputs(inputs);

  fg_ptr->set_output(output_cnode);
  return fg_ptr;
}

void ListMap::MakeCond(const std::vector<AnfNodePtr> &lists, const FuncGraphPtr &fgnext_ptr,
                       const FuncGraphPtr &fg_ptr) {
  MS_EXCEPTION_IF_NULL(fg_ptr);

  AnfNodePtr fn = fg_ptr->add_parameter();
  AnfNodePtr resl = fg_ptr->add_parameter();

  std::vector<AnfNodePtr> iters;
  (void)std::transform(lists.begin(), lists.end(), std::back_inserter(iters),
                       [fg_ptr](AnfNodePtr) { return fg_ptr->add_parameter(); });

  std::vector<AnfNodePtr> hasnexts;
  (void)std::transform(iters.begin(), iters.end(), std::back_inserter(hasnexts), [fg_ptr](AnfNodePtr item) {
    return fg_ptr->NewCNode({NewValueNode(std::string("hasnext")), item});
  });

  // cond = reduce(lambda a, b: g.apply(P.bool_and, a, b), hasnexts)
  FuncGraphPtr fgtrue_ptr = std::make_shared<FuncGraph>();
  fgtrue_ptr->debug_info()->set_name("ftrue");
  fgtrue_ptr->set_flags(FUNC_GRAPH_FLAG_CORE, true);

  CNodePtr fgtrue_output_cnode = fgtrue_ptr->NewCNode({NewValueNode(fgnext_ptr), fn, resl});
  auto inputs = fgtrue_output_cnode->inputs();
  (void)inputs.insert(inputs.end(), iters.begin(), iters.end());
  fgtrue_output_cnode->set_inputs(inputs);
  fgtrue_ptr->set_output(fgtrue_output_cnode);

  FuncGraphPtr fgfalse_ptr = std::make_shared<FuncGraph>();
  fgfalse_ptr->debug_info()->set_name("ffalse");
  fgfalse_ptr->set_flags(FUNC_GRAPH_FLAG_CORE, true);
  fgfalse_ptr->set_output(resl);

  AnfNodePtr output_cnode = fg_ptr->NewCNode({NewValueNode(prim::kPrimSwitch), NewValueNode(std::string("cond")),
                                              NewValueNode(fgtrue_ptr), NewValueNode(fgfalse_ptr)});
  fgtrue_ptr->set_output(output_cnode);
}

void ListMap::MakeNext(const std::vector<AnfNodePtr> &lists, const FuncGraphPtr &fgcond_ptr,
                       const FuncGraphPtr &fg_ptr) {
  MS_EXCEPTION_IF_NULL(fg_ptr);
  AnfNodePtr fn = fg_ptr->add_parameter();

  std::vector<AnfNodePtr> iters;
  (void)std::transform(lists.begin(), lists.end(), std::back_inserter(iters),
                       [fg_ptr](AnfNodePtr) { return fg_ptr->add_parameter(); });

  std::vector<AnfNodePtr> nexts;
  (void)std::transform(iters.begin(), iters.end(), std::back_inserter(nexts), [fg_ptr](AnfNodePtr item) {
    return fg_ptr->NewCNode({NewValueNode(std::string("next")), item});
  });

  std::vector<AnfNodePtr> values;
  (void)std::transform(nexts.begin(), nexts.end(), std::back_inserter(values), [fg_ptr](AnfNodePtr item) {
    return fg_ptr->NewCNode({NewValueNode(prim::kPrimTupleGetItem), item, nullptr});
  });

  iters.clear();
  (void)std::transform(nexts.begin(), nexts.end(), std::back_inserter(iters), [fg_ptr](AnfNodePtr item) {
    return fg_ptr->NewCNode({NewValueNode(prim::kPrimTupleGetItem), item, NewValueNode(1)});
  });

  (void)values.insert(values.begin(), fn);
  AnfNodePtr cnode_graph = fg_ptr->NewCNode(values);
  AnfNodePtr resl = fg_ptr->NewCNode({NewValueNode(prim::kPrimListAppend), cnode_graph});
  CNodePtr output_cnode = fg_ptr->NewCNode({NewValueNode(fgcond_ptr), fn, resl});

  auto inputs = output_cnode->inputs();
  (void)inputs.insert(inputs.end(), iters.begin(), iters.end());
  output_cnode->set_inputs(inputs);
  fg_ptr->set_output(output_cnode);
}

FuncGraphPtr TupleAdd::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  // args: tuple1, tuple2
  abstract::CheckArgsSize("TupleAdd", args_spec_list, 2);
  AbstractBasePtr abs_a = args_spec_list[0];
  AbstractBasePtr abs_b = args_spec_list[1];

  abstract::AbstractTuplePtr a_tuple = dyn_cast<AbstractTuple>(abs_a);
  abstract::AbstractTuplePtr b_tuple = dyn_cast<AbstractTuple>(abs_b);
  if (a_tuple == nullptr || b_tuple == nullptr) {
    MS_LOG(EXCEPTION) << "TupleAdd argument should be tuple,but " << args_spec_list[0]->ToString() << ", "
                      << args_spec_list[1]->ToString();
  }

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flags(FUNC_GRAPH_FLAG_CORE, true);
  AnfNodePtr p_tup_a = ret->add_parameter();
  AnfNodePtr p_tup_b = ret->add_parameter();

  std::vector<AnfNodePtr> elems;
  elems.push_back(NewValueNode(prim::kPrimMakeTuple));

  int tuple_size = SizeToInt(a_tuple->size());
  for (int i = 0; i < tuple_size; ++i) {
    elems.push_back(ret->NewCNode({NewValueNode(prim::kPrimTupleGetItem), p_tup_a, NewValueNode(i)}));
  }

  tuple_size = SizeToInt(b_tuple->size());
  for (int i = 0; i < tuple_size; ++i) {
    elems.push_back(ret->NewCNode({NewValueNode(prim::kPrimTupleGetItem), p_tup_b, NewValueNode(i)}));
  }

  ret->set_output(ret->NewCNode(elems));
  return ret;
}

int GetArgScalarValue(const abstract::AbstractScalarPtr &scalar, const std::string &) {
  MS_EXCEPTION_IF_NULL(scalar);
  return GetValue<int>(scalar->BuildValue());
}

bool CheckIndexInRange(int index, int min, int max) { return (index >= min && index <= max); }

int GetPositiveIndex(int index, int length) {
  if (index < 0) {
    index += length;
  }
  return index;
}

int CheckSliceMember(const AbstractBasePtr &member, int default_value, const std::string &member_name) {
  MS_EXCEPTION_IF_NULL(member);

  if (member->isa<AbstractScalar>()) {
    return GetArgScalarValue(dyn_cast<AbstractScalar>(member), member_name);
  }

  if (member->isa<AbstractNone>()) {
    return default_value;
  }

  MS_LOG(EXCEPTION) << member_name << " should be a AbstractScalar or AbstractNone, but got " << member->ToString();
}

void GenerateTupleSliceParameter(const AbstractTuplePtr &tuple, const AbstractSlicePtr &slice, int *start_index,
                                 int *stop_index, int *step_value) {
  MS_EXCEPTION_IF_NULL(tuple);
  MS_EXCEPTION_IF_NULL(slice);
  MS_EXCEPTION_IF_NULL(start_index);
  MS_EXCEPTION_IF_NULL(stop_index);
  MS_EXCEPTION_IF_NULL(step_value);

  const std::string start_name("Slice start index");
  const std::string stop_name("Slice stop index");
  const std::string step_name("Slice step value");

  int tuple_size = SizeToInt(tuple->size());
  int start_default = 0;
  int stop_default = tuple_size;
  int step_default = 1;

  *step_value = CheckSliceMember(slice->step(), step_default, step_name);
  if (*step_value == 0) {
    MS_LOG(EXCEPTION) << "TupleSlice require the step value could not be 0, but got 0.";
  }

  if (*step_value < 0) {
    start_default = tuple_size - 1;
    stop_default = -1;
  }

  *start_index = CheckSliceMember(slice->start(), start_default, start_name);
  *stop_index = CheckSliceMember(slice->stop(), stop_default, stop_name);
  if (!CheckIndexInRange(*start_index, -tuple_size, tuple_size - 1) ||
      !CheckIndexInRange(*stop_index, -tuple_size - 1, tuple_size)) {
    MS_LOG(EXCEPTION) << "TupleSlice the start index " << *start_index << " or end end index " << *stop_index
                      << " out of range, tuple size " << tuple_size << ".";
  }

  *start_index = GetPositiveIndex(*start_index, tuple_size);
  if (!slice->stop()->isa<AbstractNone>()) {
    *stop_index = GetPositiveIndex(*stop_index, tuple_size);
  }
}

FuncGraphPtr TupleSlice::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  // slice a tuple
  // args: tuple, start index, end index, step
  const std::string op_name("TupleSlice");
  abstract::CheckArgsSize(op_name, args_spec_list, 2);
  AbstractTuplePtr tuple = abstract::CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  AbstractSlicePtr slice = abstract::CheckArg<AbstractSlice>(op_name, args_spec_list, 1);

  int start_index;
  int stop_index;
  int step_value;
  GenerateTupleSliceParameter(tuple, slice, &start_index, &stop_index, &step_value);

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flags(FUNC_GRAPH_FLAG_CORE, true);
  AnfNodePtr p_tuple = ret->add_parameter();
  (void)ret->add_parameter();

  std::vector<AnfNodePtr> elems;
  elems.push_back(NewValueNode(prim::kPrimMakeTuple));
  if (step_value > 0) {
    for (int index = start_index; index < stop_index; index = index + step_value) {
      elems.push_back(ret->NewCNode({NewValueNode(prim::kPrimTupleGetItem), p_tuple, NewValueNode(index)}));
    }
  } else {
    for (int index = start_index; index > stop_index; index = index + step_value) {
      elems.push_back(ret->NewCNode({NewValueNode(prim::kPrimTupleGetItem), p_tuple, NewValueNode(index)}));
    }
  }

  ret->set_output(ret->NewCNode(elems));
  return ret;
}

FuncGraphPtr TupleGetItemTensor::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  // select indexed item
  // args: tuple of items, index
  const std::string op_name = std::string("TupleGetItemTensor");
  abstract::CheckArgsSize(op_name, args_spec_list, 2);
  AbstractTuplePtr branches_abs = abstract::CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  AbstractBasePtrList branches = branches_abs->elements();
  if (branches.size() > 0 && branches[0] != nullptr && branches[0]->isa<AbstractFunction>()) {
    FuncGraphPtr ret_graph = std::make_shared<FuncGraph>();
    ret_graph->set_flags(FUNC_GRAPH_FLAG_CORE, true);
    AnfNodePtr functions = ret_graph->add_parameter();
    auto index = ret_graph->add_parameter();

    ret_graph->set_output(ret_graph->NewCNode({NewValueNode(prim::kPrimSwitchLayer), index, functions}));
    return ret_graph;
  }

  MS_LOG(EXCEPTION) << "TupleGetItemTensor does not support to index " << branches_abs->ToString() << ".";
}

REGISTER_PYBIND_DEFINE(TupleAdd_, ([](const py::module *m) {
                         (void)py::class_<TupleAdd, MetaFuncGraph, std::shared_ptr<TupleAdd>>(*m, "TupleAdd_")
                           .def(py::init<std::string &>());
                       }));

REGISTER_PYBIND_DEFINE(TupleSlice_, ([](const py::module *m) {
                         (void)py::class_<TupleSlice, MetaFuncGraph, std::shared_ptr<TupleSlice>>(*m, "TupleSlice_")
                           .def(py::init<std::string &>());
                       }));

REGISTER_PYBIND_DEFINE(TupleGetItemTensor_, ([](const py::module *m) {
                         (void)py::class_<TupleGetItemTensor, MetaFuncGraph, std::shared_ptr<TupleGetItemTensor>>(
                           *m, "TupleGetItemTensor_")
                           .def(py::init<std::string &>());
                       }));
}  // namespace prim
}  // namespace mindspore
