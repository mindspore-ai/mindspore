/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/converter/adapter/acl/mapper/spatial_node_adapter.h"
#include <vector>
#include <set>
#include <memory>
#include <string>
#include "tools/converter/adapter/acl/common/utils.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/common/tensor_util.h"
#include "include/errorcode.h"
#include "base/base.h"
#include "mindspore/core/ops/core_ops.h"
#include "ops/concat.h"
#include "ops/batch_norm.h"
#include "ops/fused_batch_norm.h"
#include "ops/instance_norm.h"
#include "ops/stack.h"
#include "ops/tuple_get_item.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kCnodeInputMinNum = 2;
constexpr auto kAnfPrimitiveIndex = 0;
constexpr auto kNamewiEltwise = "Eltwise";
const std::set<std::string> kCNodeWithMultiOutputs = {ops::kNameBatchNorm, ops::kNameFusedBatchNorm,
                                                      ops::kNameInstanceNorm};
const std::set<std::string> kCNodeWithDynamicInput = {kNamewiEltwise, ops::kNameConcat, ops::kNameStack,
                                                      acl::kNameConcatV2};
}  // namespace

CNodePtr CreateTupleGetItemNode(const FuncGraphPtr &func_graph, const CNodePtr &input_cnode) {
  CNodePtr get_item_cnode = nullptr;
  auto tuple_get_item_prim_ptr = std::make_shared<ops::TupleGetItem>();
  MS_CHECK_TRUE_MSG(tuple_get_item_prim_ptr != nullptr, nullptr, "New TupleGetItem failed");
  auto tuple_get_item_prim_c = tuple_get_item_prim_ptr->GetPrim();
  auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_c);
  MS_CHECK_TRUE_MSG(tuple_get_item_prim != nullptr, nullptr, "tuple_prim is nullptr.");
  auto get_item_value = NewValueNode(MakeValue<int64_t>(0));
  MS_CHECK_TRUE_MSG(get_item_value != nullptr, nullptr, "item_value is nullptr.");
  AnfNodePtrList inputs{tuple_get_item_prim, input_cnode, get_item_value};
  get_item_cnode = func_graph->NewCNode(inputs);
  MS_CHECK_TRUE_MSG(get_item_cnode != nullptr, nullptr, "New get item cnode failed.");

  std::vector<int64_t> shape;
  if (acl::GetShapeVectorFromCNode(input_cnode, &shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get shape failed.";
    return nullptr;
  }
  TypeId type = acl::GetTypeFromNode(input_cnode);
  auto tensor_abstract = CreateTensorAbstract(shape, type);
  MS_CHECK_TRUE_MSG(tensor_abstract != nullptr, nullptr, "Create tensor abstract failed.");
  get_item_cnode->set_abstract(tensor_abstract);
  get_item_cnode->set_fullname_with_scope(input_cnode->fullname_with_scope() + "_getitem");
  AbstractBasePtrList abstract_list = {tensor_abstract};
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  MS_CHECK_TRUE_MSG(abstract_tuple != nullptr, nullptr, "Create abstract Tuple failed.");
  input_cnode->set_abstract(abstract_tuple);
  return get_item_cnode;
}

static STATUS AdapteNodeWithMultiOutputs(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                         const FuncGraphManagerPtr &manager) {
  std::string cnode_func_name = GetCNodeFuncName(cnode);
  if (cnode_func_name == prim::kTupleGetItem) {
    return lite::RET_OK;
  }

  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    auto input = cnode->input(i);
    MS_CHECK_TRUE_MSG(input != nullptr, lite::RET_ERROR, "input is nullptr.");
    if (!utils::isa<CNode>(input)) {
      continue;
    }
    auto input_cnode = input->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(input_cnode != nullptr, lite::RET_ERROR, "input_cnode is nullptr.");
    std::string input_func_name = GetCNodeFuncName(input_cnode);
    if (kCNodeWithMultiOutputs.find(input_func_name) != kCNodeWithMultiOutputs.end()) {
      MS_LOG(DEBUG) << "Input " << input_func_name << " of cnode " << cnode_func_name << " has multioutputs";
      CNodePtr get_item_cnode = CreateTupleGetItemNode(func_graph, input_cnode);
      if (get_item_cnode == nullptr) {
        MS_LOG(ERROR) << "Create tuple item for " << input_func_name << " of " << cnode_func_name << " failed.";
        return lite::RET_ERROR;
      }
      if (!manager->Replace(input_cnode, get_item_cnode)) {
        MS_LOG(ERROR) << "Replace " << input_func_name << " of " << cnode_func_name << " failed.";
        return lite::RET_ERROR;
      }
    }
  }
  return lite::RET_OK;
}

static STATUS AdapteNodeWithDynamicInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  std::string cnode_func_name = GetCNodeFuncName(cnode);
  if (kCNodeWithDynamicInput.find(cnode_func_name) == kCNodeWithDynamicInput.end()) {
    return lite::RET_OK;
  }
  MS_LOG(DEBUG) << "Adapter cnode with dynamic input: " << cnode_func_name;
  auto make_tuple_val_node = NewValueNode(prim::kPrimMakeTuple);
  MS_CHECK_TRUE_MSG(make_tuple_val_node != nullptr, lite::RET_ERROR, "New make tuple val node failed.");
  AnfNodePtrList new_inputs = {make_tuple_val_node};
  auto cnode_inputs = cnode->inputs();
  if (cnode_inputs.size() < kCnodeInputMinNum) {
    MS_LOG(ERROR) << "Input size " << cnode_inputs.size() << " is less than " << kCnodeInputMinNum;
    return lite::RET_ERROR;
  }
  if (cnode_func_name == acl::kNameConcatV2) {
    new_inputs.insert(new_inputs.end(), cnode_inputs.begin() + 1, cnode_inputs.end() - 1);
  } else {
    new_inputs.insert(new_inputs.end(), cnode_inputs.begin() + 1, cnode_inputs.end());
  }
  auto make_tuple_cnode = func_graph->NewCNode(new_inputs);
  MS_CHECK_TRUE_MSG(make_tuple_cnode != nullptr, lite::RET_ERROR, "New make tuple cnode failed.");
  AbstractBasePtrList elem;
  std::transform(new_inputs.begin() + 1, new_inputs.end(), std::back_inserter(elem),
                 [](const AnfNodePtr &node) { return node->abstract(); });
  make_tuple_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(elem));

  std::vector<AnfNodePtr> replace_node;
  if (cnode_func_name == acl::kNameConcatV2) {
    replace_node = std::vector<AnfNodePtr>({cnode_inputs[0], make_tuple_cnode, cnode_inputs[cnode_inputs.size() - 1]});
  } else {
    replace_node = std::vector<AnfNodePtr>({cnode_inputs[0], make_tuple_cnode});
  }
  cnode->set_inputs(replace_node);
  return lite::RET_OK;
}

STATUS AdapteSpatialNode(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  auto cnodes = func_graph->GetOrderedCnodes();
  for (const auto &cnode : cnodes) {
    MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_ERROR, "Cnode is nullptr.");
    if (AdapteNodeWithMultiOutputs(func_graph, cnode, manager) != lite::RET_OK) {
      MS_LOG(ERROR) << "Adapter node with multioutput failed.";
      return lite::RET_ERROR;
    }
    if (AdapteNodeWithDynamicInput(func_graph, cnode) != lite::RET_OK) {
      MS_LOG(ERROR) << "Adapter node with dynamic input failed.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}
}  // namespace lite
}  // namespace mindspore
