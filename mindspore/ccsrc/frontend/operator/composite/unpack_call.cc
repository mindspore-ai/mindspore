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

#include "frontend/operator/composite/unpack_call.h"
#include <algorithm>
#include <utility>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "frontend/operator/cc_implementations.h"
#include "ir/anf.h"
#include "frontend/optimizer/opt.h"
#include "include/common/pybind_api/api_register.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
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

FuncGraphPtr UnpackCall::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  // slice a tensor
  // args: tensor, slice or slice tuple
  size_t arg_length = args_abs_list.size();
  const size_t min_args_size = 2;
  if (arg_length < min_args_size) {
    MS_LOG(EXCEPTION) << "The UnpackCall operator requires at least two arguments, but got " << arg_length << ".";
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
    } else if (args_abs_list[index]->isa<AbstractDictionary>()) {
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
    } else {
      MS_LOG(EXCEPTION) << "The arguments of UnpackCall operator should be tuple, list or dict, but got "
                        << args_abs_list[index]->ToString();
    }
  }
  // Add to order list to trace if fn_node had side effect.
  res_graph->set_output(res_graph->NewCNodeInOrder(elems));
  return res_graph;
}
}  // namespace prim
}  // namespace mindspore
