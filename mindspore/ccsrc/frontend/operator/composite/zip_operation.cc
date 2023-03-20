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

#include "frontend/operator/composite/zip_operation.h"
#include <algorithm>

#include "abstract/abstract_value.h"
#include "ir/anf.h"
#include "abstract/dshape.h"
#include "frontend/operator/cc_implementations.h"
#include "frontend/optimizer/opt.h"
#include "include/common/pybind_api/api_register.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractList;
using mindspore::abstract::AbstractSequence;
using mindspore::abstract::AbstractSequencePtr;
using mindspore::abstract::AbstractTuple;

FuncGraphPtr ZipOperation::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  // zip operation:
  // input: tuple arguments
  // output: tuple of items of input iterated on every input
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "The zip operator must have at least 1 argument, but the size of arguments is 0.";
  }

  for (size_t idx = 0; idx < args_spec_list.size(); idx++) {
    auto abs = args_spec_list[idx];
    if (!abs->isa<AbstractSequence>()) {
      std::string error_index;
      if (idx == 0) {
        error_index = "first";
      } else if (idx == 1) {
        error_index = "second";
      } else if (idx == 2) {
        error_index = "third";
      } else {
        error_index = std::to_string(idx) + "th";
      }
      MS_LOG(EXCEPTION) << "For 'zip', the all inputs must be list or tuple, but the " << error_index
                        << " argument is not list or tuple.\nThe " << error_index
                        << " argument detail: " << args_spec_list[idx]->ToString() << ".";
    }
    if (abs->cast<AbstractSequencePtr>()->dynamic_len()) {
      MS_LOG(EXCEPTION) << "For 'zip', the dynamic length input is unsupported in graph mode";
    }
  }

  auto min_abs = std::min_element(
    args_spec_list.begin(), args_spec_list.end(), [](const AbstractBasePtr &x, const AbstractBasePtr &y) {
      return (x->cast<AbstractSequencePtr>()->size() < y->cast<AbstractSequencePtr>()->size());
    });
  FuncGraphPtr ret_graph = std::make_shared<FuncGraph>();
  ret_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  for (size_t idx = 0; idx < args_spec_list.size(); idx++) {
    (void)ret_graph->add_parameter();
  }

  // generate tuple output of zipped arguments input
  std::vector<AnfNodePtr> make_tuple_nodes;
  make_tuple_nodes.push_back(NewValueNode(prim::kPrimMakeTuple));
  for (size_t idx = 0; idx < (*min_abs)->cast<AbstractSequencePtr>()->size(); idx++) {
    std::vector<AnfNodePtr> make_tuple_zip_nodes;
    make_tuple_zip_nodes.push_back(NewValueNode(prim::kPrimMakeTuple));
    std::string module_name = "mindspore.ops.composite.multitype_ops.getitem_impl";
    ValuePtr op = prim::GetPythonOps("getitem", module_name);
    for (size_t arg_idx = 0; arg_idx < args_spec_list.size(); arg_idx++) {
      std::vector<AnfNodePtr> tuple_get_item_nodes{NewValueNode(op), ret_graph->parameters()[arg_idx],
                                                   NewValueNode(SizeToLong(idx))};
      auto tuple_get_item_op = ret_graph->NewCNode(tuple_get_item_nodes);
      make_tuple_zip_nodes.push_back(tuple_get_item_op);
    }
    auto make_tuple_zip_op = ret_graph->NewCNode(make_tuple_zip_nodes);
    make_tuple_nodes.push_back(make_tuple_zip_op);
  }
  ret_graph->set_output(ret_graph->NewCNode(make_tuple_nodes));
  return ret_graph;
}
}  // namespace prim
}  // namespace mindspore
