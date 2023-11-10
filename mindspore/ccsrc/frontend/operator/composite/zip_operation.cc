/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include <vector>

#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "abstract/abstract_value.h"
#include "ir/anf.h"
#include "abstract/dshape.h"
#include "frontend/operator/cc_implementations.h"
#include "frontend/optimizer/opt.h"
#include "include/common/pybind_api/api_register.h"
#include "include/common/fallback.h"
#include "pipeline/jit/ps/fallback.h"
#include "mindspore/core/utils/ms_context.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractDictionary;
using mindspore::abstract::AbstractList;
using mindspore::abstract::AbstractScalar;
using mindspore::abstract::AbstractSequence;
using mindspore::abstract::AbstractSequencePtr;
using mindspore::abstract::AbstractTuple;

FuncGraphPtr ZipOperation::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  // zip operation:
  // input: tuple arguments
  // output: tuple of items of input iterated on every input
  if (args_abs_list.empty()) {
    MS_LOG(EXCEPTION) << "The zip operator must have at least 1 argument, but the size of arguments is 0.";
  }

  FuncGraphPtr ret_graph = std::make_shared<FuncGraph>();
  ret_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  for (auto arg : args_abs_list) {
    if (arg->isa<abstract::AbstractTensor>()) {
      const auto &build_shape = arg->BuildShape();
      if (build_shape->IsDimZero()) {
        MS_EXCEPTION(TypeError) << "Cannot iterate over a scalar tensor.";
      }
    }
  }

  bool convert_to_interpret = std::any_of(args_abs_list.begin(), args_abs_list.end(), [](const AbstractBasePtr &abs) {
    MS_EXCEPTION_IF_NULL(abs);
    return !abs->isa<abstract::AbstractSequence>();
  });

  if (convert_to_interpret) {
    AnfNodePtrList node_inputs{NewValueNode(prim::kPrimMakeTuple)};
    AnfNodePtrList keys_inputs{NewValueNode(prim::kPrimMakeTuple)};
    std::stringstream script_buffer;
    script_buffer << "tuple(zip(";
    for (size_t i = 0; i < args_abs_list.size(); ++i) {
      const std::string cur_name = "__zip_input_" + std::to_string(i) + "__";
      script_buffer << cur_name << ",";
      (void)keys_inputs.emplace_back(NewValueNode(cur_name));
      (void)node_inputs.emplace_back(ret_graph->add_parameter());
    }
    script_buffer << "))";
    auto script_text = script_buffer.str();
    auto script = std::make_shared<parse::Script>(script_text);
    auto script_node = NewValueNode(script);
    auto global_dict_node = NewValueNode(std::make_shared<parse::InterpretedObject>(py::dict()));
    auto keys_tuple = ret_graph->NewCNode(keys_inputs);
    auto values_tuple = ret_graph->NewCNode(node_inputs);
    auto local_dict_node = ret_graph->NewCNode({NewValueNode(prim::kPrimMakeDict), keys_tuple, values_tuple});
    auto ret_node =
      ret_graph->NewCNode({NewValueNode(prim::kPrimPyInterpret), script_node, global_dict_node, local_dict_node});
    ret_graph->set_output(ret_node);
    return ret_graph;
  }

  bool has_dynamic_length = std::any_of(args_abs_list.begin(), args_abs_list.end(), [](const AbstractBasePtr &abs) {
    // Abs must be sequence since it is checked before.
    auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
    return seq_abs->dynamic_len();
  });
  if (has_dynamic_length) {
    MS_LOG(EXCEPTION) << "For 'zip', the dynamic length input is unsupported in graph mode.";
  }

  auto min_abs = std::min_element(
    args_abs_list.begin(), args_abs_list.end(), [](const AbstractBasePtr &x, const AbstractBasePtr &y) {
      return (x->cast<AbstractSequencePtr>()->size() < y->cast<AbstractSequencePtr>()->size());
    });

  for (size_t idx = 0; idx < args_abs_list.size(); idx++) {
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
    for (size_t arg_idx = 0; arg_idx < args_abs_list.size(); arg_idx++) {
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
