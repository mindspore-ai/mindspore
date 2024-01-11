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

void CheckValidityOfZipInput(const AbstractBasePtrList &args_abs_list) {
  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
  for (size_t idx = 0; idx < args_abs_list.size(); idx++) {
    auto abs = args_abs_list[idx];
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
      if (allow_fallback_runtime) {
        MS_EXCEPTION(TypeError) << "For 'zip', the all inputs must be iterable objects. For example:"
                                << "list, tuple, dict, string or multi dimensional Tensor, but the " << error_index
                                << " argument is:" << args_abs_list[idx]->ToString() << ".";
      } else {
        MS_EXCEPTION(TypeError) << "In JIT strict mode, for 'zip', the all inputs must be list or tuple, but the "
                                << error_index << " argument is:" << args_abs_list[idx]->ToString() << ".";
      }
    }
    if (abs->cast<AbstractSequencePtr>()->dynamic_len()) {
      MS_LOG(EXCEPTION) << "For 'zip', the dynamic length input is unsupported in graph mode.";
    }
  }
}

FuncGraphPtr ZipOperation::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  // zip operation:
  // input: tuple arguments
  // output: tuple of items of input iterated on every input
  if (args_abs_list.empty()) {
    MS_LOG(EXCEPTION) << "The zip operator must have at least 1 argument, but the size of arguments is 0.";
  }

  FuncGraphPtr ret_graph = std::make_shared<FuncGraph>();
  ret_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  bool has_tensor = false;
  for (auto arg : args_abs_list) {
    if (arg->isa<abstract::AbstractTensor>()) {
      const auto &build_shape = arg->BuildShape();
      if (build_shape->IsDimZero()) {
        MS_EXCEPTION(TypeError) << "Cannot iterate over a scalar tensor.";
      }
      has_tensor = true;
    }
  }
  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
  bool has_any = std::any_of(args_abs_list.begin(), args_abs_list.end(),
                             [](const AbstractBasePtr &abs) { return fallback::ContainsSequenceAnyType(abs); });
  bool has_scalar_string = std::any_of(args_abs_list.begin(), args_abs_list.end(), [](const AbstractBasePtr &abs) {
    MS_EXCEPTION_IF_NULL(abs);
    if (abs->isa<AbstractScalar>()) {
      auto type = abs->BuildType();
      MS_EXCEPTION_IF_NULL(type);
      if (type->type_id() == kObjectTypeString) {
        return true;
      }
    }
    return false;
  });
  bool has_dict = std::any_of(args_abs_list.begin(), args_abs_list.end(),
                              [](const AbstractBasePtr &abs) { return abs->isa<AbstractDictionary>(); });
  if (allow_fallback_runtime && (has_any || has_tensor || has_scalar_string || has_dict)) {
    std::vector<AnfNodePtr> make_tuple_inputs;
    make_tuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
    for (size_t idx = 0; idx < args_abs_list.size(); idx++) {
      if (args_abs_list[idx]->isa<AbstractDictionary>()) {
        auto key_node = ret_graph->NewCNodeInOrder({NewValueNode(prim::kPrimDictGetKeys), ret_graph->add_parameter()});
        (void)make_tuple_inputs.emplace_back(key_node);
      } else {
        (void)make_tuple_inputs.emplace_back(ret_graph->add_parameter());
      }
    }
    auto make_tuple = ret_graph->NewCNode(make_tuple_inputs);
    auto pyexecute_node = fallback::ConvertCNodeToPyExecuteForPrim(make_tuple, "zip");
    std::vector<AnfNodePtr> keys_tuple_node_inputs{NewValueNode(prim::kPrimMakeTuple)};
    std::vector<AnfNodePtr> values_tuple_node_inputs{NewValueNode(prim::kPrimMakeTuple)};
    (void)keys_tuple_node_inputs.emplace_back(NewValueNode(std::make_shared<StringImm>("key")));
    (void)values_tuple_node_inputs.emplace_back(pyexecute_node);
    auto script_node = NewValueNode(std::make_shared<StringImm>("tuple(key)"));
    auto keys_tuple_node = ret_graph->NewCNodeInOrder(keys_tuple_node_inputs);
    auto values_tuple_node = ret_graph->NewCNodeInOrder(values_tuple_node_inputs);
    auto tuple_pyexecute_node = fallback::CreatePyExecuteCNodeInOrder(ret_graph, script_node, keys_tuple_node,
                                                                      values_tuple_node, pyexecute_node->debug_info());
    MS_LOG(DEBUG) << "Convert: " << make_tuple->DebugString() << " -> " << tuple_pyexecute_node->DebugString();
    ret_graph->set_output(tuple_pyexecute_node);
    return ret_graph;
  }

  CheckValidityOfZipInput(args_abs_list);

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
