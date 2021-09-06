/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/converter/acl/mapper/spatial_node_adapter.h"
#include <vector>
#include <set>
#include <memory>
#include <string>
#include "tools/converter/acl/common/utils.h"
#include "tools/converter/ops/ops_def.h"
#include "tools/common/tensor_util.h"
#include "include/errorcode.h"
#include "base/base.h"
#include "base/core_ops.h"
#include "ops/concat.h"
#include "ops/batch_norm.h"
#include "ops/fused_batch_norm.h"
#include "ops/stack.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kCnodeInputMinNum = 2;
constexpr auto kAnfPrimitiveIndex = 0;
constexpr auto kNamewiEltwise = "Eltwise";
const std::set<std::string> kCNodeWithMultiOutputs = {ops::kNameBatchNorm, ops::kNameFusedBatchNorm};
const std::set<std::string> kCNodeWithDynamicInput = {kNamewiEltwise, ops::kNameConcat, ops::kNameStack};
}  // namespace

CNodePtr CreateTupleGetItemNode(const FuncGraphPtr &func_graph, const CNodePtr &input_cnode) {
  CNodePtr get_item_cnode = nullptr;
  auto tuple_get_item_prim_ptr = std::make_shared<lite::TupleGetItem>();
  if (tuple_get_item_prim_ptr == nullptr) {
    MS_LOG(ERROR) << "New TupleGetItem failed";
    return nullptr;
  }
  auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_ptr);
  auto get_item_value = NewValueNode(MakeValue<int64_t>(0));
  AnfNodePtrList inputs{tuple_get_item_prim, input_cnode, get_item_value};
  get_item_cnode = func_graph->NewCNode(inputs);
  if (get_item_cnode == nullptr) {
    MS_LOG(ERROR) << "New get item cnode failed.";
    return nullptr;
  }

  std::vector<int64_t> shape;
  if (acl::GetShapeVectorFromCNode(input_cnode, &shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get shape failed.";
    return nullptr;
  }
  TypeId type = acl::GetTypeFromNode(input_cnode);
  auto get_item_abstract = CreateTensorAbstract(shape, type);
  if (get_item_abstract == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstract failed.";
    return nullptr;
  }
  get_item_cnode->set_abstract(get_item_abstract);
  get_item_cnode->set_fullname_with_scope(input_cnode->fullname_with_scope() + "_getitem");
  return get_item_cnode;
}

static STATUS AdapteNodeWithMultiOutputs(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                         const FuncGraphManagerPtr &manager) {
  std::string cnode_func_name = GetCNodeFuncName(cnode);
  if (cnode_func_name == prim::kTupleGetItem || cnode_func_name == kNameReturn) {
    return lite::RET_OK;
  }

  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    auto input = cnode->input(i);
    if (!utils::isa<CNode>(input)) {
      continue;
    }
    auto input_cnode = input->cast<CNodePtr>();
    std::string input_func_name = GetCNodeFuncName(input_cnode);
    if (kCNodeWithMultiOutputs.find(input_func_name) != kCNodeWithMultiOutputs.end()) {
      MS_LOG(INFO) << "Adapter cnode with multioutputs: " << cnode_func_name;
      CNodePtr get_item_cnode = CreateTupleGetItemNode(func_graph, input_cnode);
      if (get_item_cnode == nullptr) {
        MS_LOG(ERROR) << "Create tuple item for " << cnode_func_name << " failed.";
        return lite::RET_ERROR;
      }
      if (!manager->Replace(input_cnode, get_item_cnode)) {
        MS_LOG(ERROR) << "Replace " << cnode_func_name << " failed.";
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
  MS_LOG(INFO) << "Adapter cnode with dynamic input: " << cnode_func_name;
  auto make_tuple_val_node = NewValueNode(prim::kPrimMakeTuple);
  if (make_tuple_val_node == nullptr) {
    MS_LOG(ERROR) << "New make tuple val node failed.";
    return lite::RET_ERROR;
  }
  AnfNodePtrList new_inputs = {make_tuple_val_node};
  auto cnode_inputs = cnode->inputs();
  if (cnode_inputs.size() >= kCnodeInputMinNum) {
    new_inputs.insert(new_inputs.end(), cnode_inputs.begin() + 1, cnode_inputs.end());
  }
  auto make_tuple_cnode = func_graph->NewCNode(new_inputs);
  if (make_tuple_cnode == nullptr) {
    MS_LOG(ERROR) << "New make tuple cnode failed.";
    return lite::RET_ERROR;
  }

  const std::vector<AnfNodePtr> replace_node = {cnode_inputs[0], make_tuple_cnode};
  cnode->set_inputs(replace_node);
  return lite::RET_OK;
}

STATUS AdapteSpatialNode(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  auto cnodes = func_graph->GetOrderedCnodes();
  for (const auto &cnode : cnodes) {
    if (cnode == nullptr) {
      MS_LOG(ERROR) << "Cnode is nullptr.";
      return lite::RET_ERROR;
    }
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
