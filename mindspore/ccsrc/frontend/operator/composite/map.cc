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
#include "include/common/pybind_api/api_register.h"
#include "pipeline/jit/debug/trace.h"
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
  (void)inputs.insert(inputs.cend(), args.cbegin(), args.cend());
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

std::pair<std::string, std::string> Map::GetMapInputIndex(size_t num) const {
  std::string error_index;
  std::string next_index;
  const size_t first_index = 1;
  const size_t second_index = 2;
  if (num == first_index) {
    // The first element in Map is func_graph
    error_index = "first";
    next_index = "second";
  } else if (num == second_index) {
    error_index = "second";
    next_index = "third";
  } else {
    error_index = std::to_string(num) + "th";
    next_index = std::to_string(num + 1) + "th";
  }
  return std::pair<std::string, std::string>(error_index, next_index);
}

AnfNodePtr Map::FullMakeList(const std::shared_ptr<List> &type, const FuncGraphPtr &func_graph,
                             const AnfNodePtr &fn_arg, const ArgsPairList &arg_pairs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(type);

  std::size_t size = type->elements().size();
  size_t num = 0;
  std::ostringstream oss;
  bool is_not_same = false;
  for (auto &item : arg_pairs) {
    num++;
    auto lhs = std::dynamic_pointer_cast<List>(item.second);
    if (lhs->dynamic_len()) {
      MS_LOG(EXCEPTION) << "For 'map', the dynamic length input is unsupported in graph mode";
    }
    auto [error_index, next_index] = GetMapInputIndex(num);
    if (lhs == nullptr) {
      MS_LOG(EXCEPTION) << "The " << error_index << " element in Map has wrong type, expected a List, but got "
                        << item.second->ToString() << ".";
    }
    if (lhs->elements().size() != size) {
      oss << "\nThe length of the " << error_index << " element in Map is " << size << ", but the length of the "
          << next_index << " element in Map is " << lhs->elements().size() << ".\n";
      is_not_same = true;
      break;
    }
  }
  if (is_not_same) {
    MS_LOG(EXCEPTION) << "For 'Map', the length of lists must be the same. " << oss.str();
  }

  constexpr size_t kPrimHoldLen = 1;
  std::vector<AnfNodePtr> inputs;
  inputs.reserve(size + kPrimHoldLen);
  inputs.push_back(NewValueNode(prim::kPrimMakeList));

  for (size_t i = 0; i < size; i++) {
    MS_LOG(DEBUG) << "FullMakeList for the " << i << "th arg of the target, reverse_: " << reverse_ << ".";
    auto ptrGraph = GenerateLeafFunc(arg_pairs.size());
    auto fn = NewValueNode(ptrGraph);

    std::vector<AnfNodePtr> inputs2;
    inputs2.push_back(fn);
    if (fn_arg != nullptr) {
      inputs2.push_back(fn_arg);
    }

    size_t pos = (reverse_ ? (size - 1 - i) : i);
    (void)std::transform(arg_pairs.begin(), arg_pairs.end(), std::back_inserter(inputs2),
                         [&func_graph, pos](const std::pair<AnfNodePtr, Any> &item) {
                           return func_graph->NewCNodeInOrder(
                             {NewValueNode(prim::kPrimListGetItem), item.first, NewValueNode(SizeToLong(pos))});
                         });

    auto call_node = func_graph->NewCNodeInOrder(inputs2);
    if (reverse_) {
      (void)inputs.insert(inputs.cbegin() + 1, call_node);
    } else {
      inputs.emplace_back(call_node);
    }
  }
  return func_graph->NewCNodeInOrder(inputs);
}

AnfNodePtr Map::FullMakeTuple(const std::shared_ptr<Tuple> &type, const FuncGraphPtr &func_graph,
                              const AnfNodePtr &fn_arg, const ArgsPairList &arg_pairs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(type);

  size_t size = type->elements().size();
  size_t num = 0;
  std::ostringstream oss;
  bool is_not_same = false;
  for (auto &item : arg_pairs) {
    num++;
    auto lhs = std::dynamic_pointer_cast<Tuple>(item.second);
    if (lhs->dynamic_len()) {
      MS_LOG(EXCEPTION) << "For 'map', the dynamic length input is unsupported in graph mode";
    }
    auto [error_index, next_index] = GetMapInputIndex(num);
    if (lhs == nullptr) {
      MS_LOG(EXCEPTION) << "The " << error_index << " element in Map has wrong type, expected a Tuple, but got "
                        << item.second->ToString() << ".";
    }
    if (lhs->elements().size() != size) {
      oss << "\nThe length of the " << error_index << " element in Map is " << size << ", but the length of the "
          << next_index << " element in Map is " << lhs->elements().size() << ".\n";
      is_not_same = true;
      break;
    }
  }
  if (is_not_same) {
    MS_LOG(EXCEPTION) << "For 'Map', the length of tuples must be the same. " << oss.str();
  }

  constexpr size_t kPrimHoldLen = 1;
  std::vector<AnfNodePtr> inputs;
  inputs.reserve(size + kPrimHoldLen);
  inputs.push_back(NewValueNode(prim::kPrimMakeTuple));

  for (size_t i = 0; i < size; i++) {
    MS_LOG(DEBUG) << "FullMakeTuple for the " << i << "th arg of the tuple inputs, reverse_: " << reverse_ << ".";
    auto ptrGraph = GenerateLeafFunc(arg_pairs.size());
    auto fn = NewValueNode(ptrGraph);

    std::vector<AnfNodePtr> inputs2;
    inputs2.push_back(fn);
    if (fn_arg != nullptr) {
      inputs2.push_back(fn_arg);
    }

    size_t pos = (reverse_ ? (size - 1 - i) : i);
    (void)std::transform(arg_pairs.begin(), arg_pairs.end(), std::back_inserter(inputs2),
                         [&func_graph, &pos](const std::pair<AnfNodePtr, Any> &item) {
                           return func_graph->NewCNodeInOrder(
                             {NewValueNode(prim::kPrimTupleGetItem), item.first, NewValueNode(SizeToLong(pos))});
                         });

    auto call_node = func_graph->NewCNodeInOrder(inputs2);
    if (reverse_) {
      (void)inputs.insert(inputs.cbegin() + 1, call_node);
    } else {
      inputs.emplace_back(call_node);
    }
  }
  return func_graph->NewCNodeInOrder(inputs);
}

AnfNodePtr Map::Make(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg, const ArgsPairList &arg_pairs) {
  if (arg_pairs.empty()) {
    MS_EXCEPTION(TypeError) << "The Map operator must have at least two arguments. But the size of arguments is "
                            << (arg_pairs.size() + 1) << ".";
  }
  bool found = false;
  TypeId id = kObjectTypeEnd;
  std::pair<AnfNodePtr, TypePtr> pair;
  for (auto &arg_pair : arg_pairs) {
    pair = arg_pair;
    MS_LOG(DEBUG) << "Map " << pair.second->ToString();
    id = arg_pair.second->type_id();
    if (nonleaf_.count(id) != 0) {
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
      oss << "There are " << (arg_pairs.size() + 1) << " inputs of `" << name_ << "`, corresponding type info:\n"
          << trace::GetDebugInfo(func_graph->debug_info()) << ".\n";
      int64_t idx = 0;
      std::string str_index = "first";
      for (auto &item : arg_pairs) {
        if (idx == 0) {
          // The first element in HyperMap is func_graph
          str_index = "second";
        } else if (idx == 1) {
          str_index = "third";
        } else {
          constexpr auto arg_start_idx = 2;
          str_index = std::to_string(idx + arg_start_idx) + "th";
        }
        ++idx;
        oss << "The type of the " << str_index << " argument in Map is: " << item.second->ToString() << ".\n";
      }
      MS_LOG(EXCEPTION) << "The types of arguments in Map must be consistent, "
                        << "but the types of arguments are inconsistent.\n"
                        << oss.str();
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
    default:
      MS_LOG(EXCEPTION) << "Map can only be applied to list, tuple, but got " << pair.second->ToString() << ".";
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
    MS_LOG(DEBUG) << "GenerateFromTypes for elements from " << args_spec_list[i]->ToString() << ".";
    arg_pairs.push_back(std::make_pair(ptrGraph->add_parameter(), args_spec_list[i]));
  }

  ptrGraph->set_output(Make(ptrGraph, ptrFnArg, arg_pairs));
  return ptrGraph;
}

abstract::AbstractBasePtrList Map::NormalizeArgs(const AbstractBasePtrList &args_spec_list) const {
  if (fn_leaf_ == nullptr) {
    if (args_spec_list.empty()) {
      MS_LOG(EXCEPTION) << "The arguments of Map operator should not be empty.";
    }
    MS_EXCEPTION_IF_NULL(args_spec_list[0]);
    // Assert that map's function param does not contain free variables
    if (args_spec_list[0]->isa<FuncGraphAbstractClosure>()) {
      auto graph_func = dyn_cast<FuncGraphAbstractClosure>(args_spec_list[0]);
      auto func_graph = graph_func->func_graph();
      if (func_graph->parent() != nullptr) {
        MS_LOG(EXCEPTION) << "The Map operator don't support Closure with free variable yet.";
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
}  // namespace prim
}  // namespace mindspore
