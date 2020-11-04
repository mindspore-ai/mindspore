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
#include "pybind_api/api_register.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractList;
using mindspore::abstract::AbstractSequeue;
using mindspore::abstract::AbstractSequeuePtr;
using mindspore::abstract::AbstractTuple;

FuncGraphPtr ZipOperation::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  // zip operation:
  // input: tuple arguments
  // output: tuple of items of input iterated on every input
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "For 'zip', there is at least one input.";
  }

  auto is_all_sequeue =
    std::all_of(args_spec_list.begin(), args_spec_list.end(), [](const AbstractBasePtr &abs) -> bool {
      MS_EXCEPTION_IF_NULL(abs);
      return abs->isa<AbstractSequeue>();
    });
  if (!is_all_sequeue) {
    MS_LOG(EXCEPTION) << "For 'zip', all inputs must be sequence.";
  }

  auto min_abs = std::min_element(
    args_spec_list.begin(), args_spec_list.end(), [](const AbstractBasePtr &x, const AbstractBasePtr &y) {
      return (x->cast<AbstractSequeuePtr>()->size() < y->cast<AbstractSequeuePtr>()->size());
    });
  FuncGraphPtr ret_graph = std::make_shared<FuncGraph>();
  ret_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  for (size_t idx = 0; idx < args_spec_list.size(); idx++) {
    (void)ret_graph->add_parameter();
  }

  // generate tuple output of ziped arguments input
  std::vector<AnfNodePtr> make_tuple_nodes;
  make_tuple_nodes.push_back(NewValueNode(prim::kPrimMakeTuple));
  for (size_t idx = 0; idx < (*min_abs)->cast<AbstractSequeuePtr>()->size(); idx++) {
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

REGISTER_PYBIND_DEFINE(ZipOperation_, ([](const py::module *m) {
                         (void)py::class_<ZipOperation, MetaFuncGraph, std::shared_ptr<ZipOperation>>(*m,
                                                                                                      "ZipOperation_")
                           .def(py::init<std::string &>());
                       }));
}  // namespace prim
}  // namespace mindspore
