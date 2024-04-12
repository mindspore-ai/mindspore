/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "frontend/operator/composite/unpack_call.h"
#include <algorithm>
#include <utility>

#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "frontend/operator/cc_implementations.h"
#include "ir/anf.h"
#include "frontend/optimizer/opt.h"
#include "include/common/pybind_api/api_register.h"
#include "pipeline/jit/ps/fallback.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
using mindspore::abstract::AbstractAny;
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractDictionary;
using mindspore::abstract::AbstractDictionaryPtr;
using mindspore::abstract::AbstractElementPair;
using mindspore::abstract::AbstractFunction;
using mindspore::abstract::AbstractKeywordArg;
using mindspore::abstract::AbstractList;
using mindspore::abstract::AbstractListPtr;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;

FuncGraphPtr ConvertUnpackToPyInterpretFuncGraph(const AbstractBasePtrList &args_abs_list) {
  // No need to check, check will be done in infer.
  auto res_graph = std::make_shared<FuncGraph>();
  res_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  res_graph->debug_info()->set_name("UnpackCallToPyInterpret");

  // Generate pyinterpret node's inputs
  AnfNodePtrList local_key_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  AnfNodePtrList local_value_inputs = {NewValueNode(prim::kPrimMakeTuple)};

  // Get function
  std::stringstream script_buffer;
  const std::string call_func_str = "__call_func_str__";
  script_buffer << call_func_str << "(";
  (void)local_key_inputs.emplace_back(NewValueNode(call_func_str));
  (void)local_value_inputs.emplace_back(res_graph->add_parameter());

  // Get input parameters:
  // UnpackCall(__call_func_str__, (a, b), args(AbstractAny), {kwargs})
  // -> PyInterpret(__call_func_str__, a, b, args, kwargs)
  // -> eval(__call_func_str__(a, b, *args, **kwargs))
  // 1. Process stable parameters, must be a tuple
  size_t index = 1;
  if (args_abs_list[index]->isa<AbstractTuple>()) {
    auto arg_tuple = args_abs_list[index++]->cast<AbstractTuplePtr>();
    AnfNodePtr para_tuple = res_graph->add_parameter();
    for (size_t i = 0; i < arg_tuple->size(); i++) {
      const auto param_str = "__input__" + std::to_string(i) + "__";
      script_buffer << param_str << ",";
      (void)local_key_inputs.emplace_back(NewValueNode(param_str));
      (void)local_value_inputs.emplace_back(
        res_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), para_tuple, NewValueNode(SizeToLong(i))}));
    }
  }

  // 2. Process *args(AbstractAny)
  if (index < args_abs_list.size() && args_abs_list[index]->isa<AbstractAny>()) {
    const auto param_str = "args";
    script_buffer << "*" << param_str << ",";
    AnfNodePtrList abstract_any_inputs = {NewValueNode(prim::kPrimMakeTuple)};
    while (index < args_abs_list.size() && args_abs_list[index]->isa<AbstractAny>()) {
      (void)abstract_any_inputs.emplace_back(res_graph->add_parameter());
      index++;
    }
    (void)local_key_inputs.emplace_back(NewValueNode(param_str));
    (void)local_value_inputs.emplace_back(res_graph->NewCNode(abstract_any_inputs));
  }

  // 3. Process **kwargs, must be a dictionary
  if (index < args_abs_list.size() && args_abs_list[index]->isa<AbstractDictionary>()) {
    const auto param_str = "kwargs";
    script_buffer << "**" << param_str;
    (void)local_key_inputs.emplace_back(NewValueNode(param_str));
    (void)local_value_inputs.emplace_back(res_graph->add_parameter());
  }
  script_buffer << ")";

  // Set func_graph output as generated pyinterpret node
  const auto &script = script_buffer.str();
  const auto key_tuple = res_graph->NewCNode(local_key_inputs);
  const auto value_tuple = res_graph->NewCNode(local_value_inputs);
  auto local_dict_node = res_graph->NewCNode({NewValueNode(prim::kPrimMakeDict), key_tuple, value_tuple});
  auto res = fallback::CreatePyInterpretCNode(res_graph, script, py::dict(), local_dict_node);
  res_graph->set_output(res);

  MS_LOG(DEBUG) << "Convert UnpackCall funcgraph as PyInterpret: " << res->DebugString();
  return res_graph;
}

FuncGraphPtr UnpackCall::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  size_t arg_length = args_abs_list.size();
  const size_t min_args_size = 2;
  if (arg_length < min_args_size) {
    MS_LOG(INTERNAL_EXCEPTION) << "The UnpackCall operator requires arguments >=2, but got " << arg_length << ".";
  }

  bool existAny = false;
  std::for_each(args_abs_list.begin() + 1, args_abs_list.end(), [&existAny](const AbstractBasePtr &abs) {
    MS_EXCEPTION_IF_NULL(abs);
    if (abs->isa<AbstractAny>()) {
      existAny = true;
      return;
    }
    if (!abs->isa<AbstractTuple>() && !abs->isa<AbstractList>() && !abs->isa<AbstractDictionary>()) {
      MS_LOG(INTERNAL_EXCEPTION) << "The arguments of UnpackCall operator should be tuple, list or dict, but got "
                                 << abs->ToString();
    }
  });
  if (existAny) {
    MS_LOG(DEBUG) << "The arguments of UnpackCall operator should not be AbstractAny, convert to PyInterpret";
    return ConvertUnpackToPyInterpretFuncGraph(args_abs_list);
  }

  // No need to check, check will be done in infer.
  auto res_graph = std::make_shared<FuncGraph>();
  res_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  res_graph->debug_info()->set_name("UnpackCall");

  AnfNodePtr fn_node = res_graph->add_parameter();
  std::vector<AnfNodePtr> elems;
  elems.push_back(fn_node);
  for (size_t index = 1; index < arg_length; index++) {
    MS_EXCEPTION_IF_NULL(args_abs_list[index]);
    if (args_abs_list[index]->isa<AbstractTuple>()) {
      auto arg_tuple = args_abs_list[index]->cast<AbstractTuplePtr>();
      AnfNodePtr para_tuple = res_graph->add_parameter();
      for (size_t i = 0; i < arg_tuple->size(); ++i) {
        elems.push_back(
          res_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), para_tuple, NewValueNode(SizeToLong(i))}));
      }
    } else if (args_abs_list[index]->isa<AbstractList>()) {
      auto arg_list = args_abs_list[index]->cast<AbstractListPtr>();
      AnfNodePtr para_list = res_graph->add_parameter();
      for (size_t i = 0; i < arg_list->size(); ++i) {
        elems.push_back(
          res_graph->NewCNode({NewValueNode(prim::kPrimListGetItem), para_list, NewValueNode(SizeToLong(i))}));
      }
    } else {
      AbstractDictionaryPtr arg_dict = args_abs_list[index]->cast<AbstractDictionaryPtr>();
      AnfNodePtr para_dict = res_graph->add_parameter();
      auto dict_elems = arg_dict->elements();
      (void)std::transform(
        dict_elems.cbegin(), dict_elems.cend(), std::back_inserter(elems),
        [res_graph, para_dict](const AbstractElementPair &item) {
          // Dict_elems's first element represents parameter names, which should be string type.
          auto key_value = GetValue<std::string>(item.first->BuildValue());
          auto dict_get_item =
            res_graph->NewCNode({NewValueNode(prim::kPrimDictGetItem), para_dict, NewValueNode(key_value)});
          return res_graph->NewCNode({NewValueNode(prim::kPrimMakeKeywordArg), NewValueNode(key_value), dict_get_item});
        });
    }
  }
  // Add to order list to trace if fn_node had side effect.
  res_graph->set_output(res_graph->NewCNodeInOrder(elems));
  return res_graph;
}
}  // namespace prim
}  // namespace mindspore
