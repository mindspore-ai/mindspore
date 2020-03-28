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

#include "operator/composite/unpack_call.h"
#include <algorithm>
#include <utility>

#include "./common.h"
#include "pipeline/static_analysis/abstract_value.h"
#include "pipeline/static_analysis/dshape.h"
#include "pipeline/static_analysis/param_validator.h"
#include "operator/cc_implementations.h"
#include "ir/anf.h"
#include "optimizer/opt.h"
#include "utils/symbolic.h"
#include "pybind_api/api_register.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
using mindspore::abstract::AbstractAttribute;
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractDictionary;
using mindspore::abstract::AbstractDictionaryPtr;
using mindspore::abstract::AbstractFunction;
using mindspore::abstract::AbstractKeywordArg;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;

FuncGraphPtr UnpackCall::GenerateFuncGraph(const AbstractBasePtrList& args_spec_list) {
  // slice a tensor
  // args: tensor, slice or slice tuple
  const std::string op_name = std::string("UnpackCall");
  size_t arg_length = args_spec_list.size();
  if (arg_length < 2) {
    MS_LOG(EXCEPTION) << op_name << " requires at least two args, but got " << arg_length << ".";
  }

  (void)abstract::CheckArg<AbstractFunction>(op_name, args_spec_list, 0);
  auto ret_graph = std::make_shared<FuncGraph>();
  ret_graph->set_flags(FUNC_GRAPH_FLAG_CORE, true);

  AnfNodePtr fnNode = ret_graph->add_parameter();
  std::vector<AnfNodePtr> elems;
  elems.push_back(fnNode);
  for (size_t index = 1; index < arg_length; index++) {
    MS_EXCEPTION_IF_NULL(args_spec_list[index]);
    if (args_spec_list[index]->isa<AbstractTuple>()) {
      auto arg_tuple = args_spec_list[index]->cast<AbstractTuplePtr>();
      AnfNodePtr para_tuple = ret_graph->add_parameter();
      for (size_t i = 0; i < arg_tuple->size(); ++i) {
        elems.push_back(
          ret_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), para_tuple, NewValueNode(SizeToInt(i))}));
      }
    } else if (args_spec_list[index]->isa<AbstractDictionary>()) {
      AbstractDictionaryPtr arg_dict = args_spec_list[index]->cast<AbstractDictionaryPtr>();
      AnfNodePtr para_dict = ret_graph->add_parameter();
      auto dict_elems = arg_dict->elements();
      (void)std::transform(dict_elems.begin(), dict_elems.end(), std::back_inserter(elems),
                           [ret_graph, para_dict](const AbstractAttribute& item) {
                             auto dict_get_item = ret_graph->NewCNode(
                               {NewValueNode(prim::kPrimDictGetItem), para_dict, NewValueNode(item.first)});
                             return ret_graph->NewCNode(
                               {NewValueNode(prim::kPrimMakeKeywordArg), NewValueNode(item.first), dict_get_item});
                           });
    } else {
      MS_LOG(EXCEPTION) << op_name << " require args should be tuple or dict, but got "
                        << args_spec_list[index]->ToString();
    }
  }
  ret_graph->set_output(ret_graph->NewCNode(elems));
  return ret_graph;
}

REGISTER_PYBIND_DEFINE(UnpackCall_, ([](const py::module* m) {
                         (void)py::class_<UnpackCall, MetaFuncGraph, std::shared_ptr<UnpackCall>>(*m, "UnpackCall_")
                           .def(py::init<std::string&>());
                       }));

}  // namespace prim
}  // namespace mindspore
