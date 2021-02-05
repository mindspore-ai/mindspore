/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "frontend/operator/composite/map.h"
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "abstract/abstract_value.h"
#include "abstract/abstract_function.h"
#include "abstract/dshape.h"
#include "pybind_api/api_register.h"
#include "debug/trace.h"
#include "frontend/operator/ops.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
using FuncGraphAbstractClosure = mindspore::abstract::FuncGraphAbstractClosure;

AnfNodePtr Map::FullMakeLeaf(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg, const AnfNodePtrList &args) {
  MS_LOG(DEBUG) << "Map FullMakeLeaf non recursive.\n";
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> inputs;
  if (fn_arg != nullptr) {
    inputs.emplace_back(fn_arg);
  } else {
    inputs.emplace_back(NewValueNode(fn_leaf_));
  }
  inputs.insert(inputs.end(), args.begin(), args.end());
  return func_graph->NewCNodeInOrder(inputs);
}

FuncGraphPtr Map::GenerateLeafFunc(const size_t &args_size) {
  // Generate func for leaf nodes
  FuncGraphPtr ptrGraph = std::make_shared<FuncGraph>();
  ptrGraph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ptrGraph->set_flag(FUNC_GRAPH_FLAG_SPECIALIZE_PARAMETER, true);
  ptrGraph->debug_info()->set_name("map");
  AnfNodePtr ptrFnArg = nullptr;
  if (fn_leaf_ == nullptr) {
    ptrFnArg = ptrGraph->add_parameter();
  }
  AnfNodePtrList args;
  for (size_t i = 0; i < args_size; ++i) {
    args.emplace_back(ptrGraph->add_parameter());
  }
  ptrGraph->set_output(FullMakeLeaf(ptrGraph, ptrFnArg, args));
  return ptrGraph;
}

AnfNodePtr Map::FullMakeList(const std::shared_ptr<List> &type, const FuncGraphPtr &func_graph,
                             const AnfNodePtr &fn_arg, const ArgsPairList &arg_pairs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(type);

  std::size_t size = type->elements().size();
  bool is_not_same =
    std::any_of(arg_pairs.begin(), arg_pairs.end(), [size](const std::pair<AnfNodePtr, TypePtr> &item) {
      auto lhs = std::dynamic_pointer_cast<List>(item.second);
      MS_EXCEPTION_IF_NULL(lhs);
      return lhs->elements().size() != size;
    });
  if (is_not_same) {
    MS_LOG(EXCEPTION) << "List in Map should have same length";
  }

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimMakeList));

  for (int64_t i = 0; i < SizeToLong(size); ++i) {
    MS_LOG(DEBUG) << "GenerateLeafFunc for the " << i << "th arg of the target";
    auto ptrGraph = GenerateLeafFunc(arg_pairs.size());
    auto fn = NewValueNode(ptrGraph);

    std::vector<AnfNodePtr> inputs2;
    inputs2.push_back(fn);
    if (fn_arg != nullptr) {
      inputs2.push_back(fn_arg);
    }

    (void)std::transform(
      arg_pairs.begin(), arg_pairs.end(), std::back_inserter(inputs2),
      [&func_graph, i](const std::pair<AnfNodePtr, Any> &item) {
        return func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimListGetItem), item.first, NewValueNode(i)});
      });

    inputs.push_back(func_graph->NewCNodeInOrder(inputs2));
  }
  return func_graph->NewCNodeInOrder(inputs);
}

AnfNodePtr Map::FullMakeTuple(const std::shared_ptr<Tuple> &type, const FuncGraphPtr &func_graph,
                              const AnfNodePtr &fn_arg, const ArgsPairList &arg_pairs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(type);

  std::size_t size = type->elements().size();
  bool is_not_same =
    std::any_of(arg_pairs.begin(), arg_pairs.end(), [size](const std::pair<AnfNodePtr, TypePtr> &item) {
      auto lhs = std::dynamic_pointer_cast<Tuple>(item.second);
      MS_EXCEPTION_IF_NULL(lhs);
      return lhs->elements().size() != size;
    });
  if (is_not_same) {
    MS_LOG(EXCEPTION) << "tuple in Map should have same length";
  }

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimMakeTuple));

  for (int64_t i = 0; i < SizeToLong(size); ++i) {
    MS_LOG(DEBUG) << "GenerateLeafFunc for the " << i << "th arg of the tuple inputs";
    auto ptrGraph = GenerateLeafFunc(arg_pairs.size());
    auto fn = NewValueNode(ptrGraph);

    std::vector<AnfNodePtr> inputs2;
    inputs2.push_back(fn);
    if (fn_arg != nullptr) {
      inputs2.push_back(fn_arg);
    }

    (void)std::transform(
      arg_pairs.begin(), arg_pairs.end(), std::back_inserter(inputs2),
      [&func_graph, &i](std::pair<AnfNodePtr, Any> item) {
        return func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), item.first, NewValueNode(i)});
      });

    inputs.push_back(func_graph->NewCNodeInOrder(inputs2));
  }
  return func_graph->NewCNodeInOrder(inputs);
}

AnfNodePtr Map::FullMakeClass(const std::shared_ptr<Class> &type, const FuncGraphPtr &func_graph,
                              const AnfNodePtr &fn_arg, const ArgsPairList &arg_pairs) {
  MS_EXCEPTION_IF_NULL(type);
  MS_EXCEPTION_IF_NULL(func_graph);

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimMakeRecord));
  inputs.push_back(NewValueNode(type));

  std::size_t attrSize = type->GetAttributes().size();
  for (std::size_t i = 0; i < attrSize; ++i) {
    MS_LOG(DEBUG) << "GenerateLeafFunc for the " << i << "th element of the inputs";
    auto ptrGraph = GenerateLeafFunc(arg_pairs.size());
    auto fn = NewValueNode(ptrGraph);

    std::vector<AnfNodePtr> inputs2;
    inputs2.push_back(fn);
    if (fn_arg != nullptr) {
      inputs2.push_back(fn_arg);
    }

    int64_t j = 0;
    for (auto item : arg_pairs) {
      inputs2.push_back(func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimGetAttr), item.first, NewValueNode(j)}));
      j++;
    }

    inputs.push_back(func_graph->NewCNodeInOrder(inputs2));
  }
  return func_graph->NewCNodeInOrder(inputs);
}

AnfNodePtr Map::Make(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg, const ArgsPairList &arg_pairs) {
  if (arg_pairs.empty()) {
    MS_EXCEPTION(TypeError) << "map() must have at least two arguments";
  }
  bool found = false;
  TypeId id = kObjectTypeEnd;
  std::pair<AnfNodePtr, TypePtr> pair;
  for (auto &item : arg_pairs) {
    pair = item;
    MS_LOG(DEBUG) << "Map " << pair.second->ToString();
    id = item.second->type_id();
    if (nonleaf_.count(id)) {
      found = true;
      break;
    }
  }

  if (found) {
    // In a nonleaf situation, all arguments must have the same generic.
    bool is_not_same =
      std::any_of(arg_pairs.begin(), arg_pairs.end(), [pair](const std::pair<AnfNodePtr, TypePtr> &item) {
        if (item.first != pair.first) {
          return item.second->type_id() != pair.second->type_id();
        }
        return false;
      });
    if (is_not_same) {
      std::ostringstream oss;
      oss << "There are " << arg_pairs.size() << " inputs of `" << name_ << "`, corresponding type info:\n"
          << trace::GetDebugInfo(func_graph->debug_info()) << "\n";
      int64_t idx = 0;
      for (auto &item : arg_pairs) {
        oss << ++idx << ": " << item.second->ToString() << "\n";
      }
      MS_LOG(EXCEPTION) << "Map cannot match up all input types of arguments.\n"
                        << oss.str() << pair.second->ToString() << "\n";
    }
  }

  switch (id) {
    case kObjectTypeList: {
      auto type = std::static_pointer_cast<List>(pair.second);
      return FullMakeList(type, func_graph, fn_arg, arg_pairs);
    }
    case kObjectTypeTuple: {
      auto type = std::static_pointer_cast<Tuple>(pair.second);
      return FullMakeTuple(type, func_graph, fn_arg, arg_pairs);
    }
    case kObjectTypeClass: {
      auto type = std::static_pointer_cast<Class>(pair.second);
      return FullMakeClass(type, func_graph, fn_arg, arg_pairs);
    }
    default:
      MS_LOG(EXCEPTION) << "Map can only be applied to list, tuple and class "
                        << ", but got " << pair.second->ToString();
  }
}

FuncGraphPtr Map::GenerateFromTypes(const TypePtrList &args_spec_list) {
  FuncGraphPtr ptrGraph = std::make_shared<FuncGraph>();
  ptrGraph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ptrGraph->set_flag(FUNC_GRAPH_FLAG_SPECIALIZE_PARAMETER, true);
  ptrGraph->debug_info()->set_name("map");

  AnfNodePtr ptrFnArg = nullptr;
  std::size_t i = 0;
  if (fn_leaf_ == nullptr) {
    ptrFnArg = ptrGraph->add_parameter();
    i = 1;
  }
  ArgsPairList arg_pairs;
  std::size_t size = args_spec_list.size();
  for (; i < size; ++i) {
    MS_LOG(DEBUG) << "GenerateFromTypes for elements from " << args_spec_list[i]->ToString();
    arg_pairs.push_back(std::make_pair(ptrGraph->add_parameter(), args_spec_list[i]));
  }

  ptrGraph->set_output(Make(ptrGraph, ptrFnArg, arg_pairs));
  return ptrGraph;
}

abstract::AbstractBasePtrList Map::NormalizeArgs(const AbstractBasePtrList &args_spec_list) const {
  if (fn_leaf_ == nullptr) {
    MS_EXCEPTION_IF_NULL(args_spec_list[0]);
    // Assert that map's function param does not contain free variables
    if (args_spec_list[0]->isa<FuncGraphAbstractClosure>()) {
      auto graph_func = dyn_cast<FuncGraphAbstractClosure>(args_spec_list[0]);
      auto func_graph = graph_func->func_graph();
      if (func_graph->parent() != nullptr) {
        MS_LOG(EXCEPTION) << "Map don't support Closure with free variable yet.";
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

REGISTER_PYBIND_DEFINE(Map_, ([](const py::module *m) {
                         (void)py::class_<MapPy, MetaFuncGraph, std::shared_ptr<MapPy>>(*m, "Map_")
                           .def(py::init<std::shared_ptr<MultitypeFuncGraph>>(), py::arg("leaf"))
                           .def(py::init<>());
                       }));
}  // namespace prim
}  // namespace mindspore
