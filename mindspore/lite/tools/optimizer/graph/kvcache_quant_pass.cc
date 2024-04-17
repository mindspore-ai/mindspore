/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <memory>
#include <vector>
#include <string>
#include "ops/array_ops.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "ops/op_name.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "mindspore/core/utils/anf_utils.h"
#include "tools/optimizer/graph/kvcache_quant_pass.h"

/* This pass changes the following pattern(s).

  1. Do quant op pass
    ###############
    Pattern:
    Mul -> Add -> Round -> Cast

    Replace:
    Mul -> Add -> Quant
    ###############

    The quantization algorithm is composed of four nodes: Mul -> Add -> Round -> Cast, which is converted into
    Mul -> Add -> Quant nodes for the Ascend backend, so that the internal operator fusion strategy of Ascend can
    improve the inference performance

  2. Do dequant op pass
    ###############
    Pattern:
    Cast -> Add -> Mul

    Replace:
    AntiQuant -> Add -> Mul
    ###############

    The dequantization algorithm is composed of three nodes: cast->add->mul, which is converted into
    AntiQuant -> Add -> Mul nodes for the Ascend backend, so that the internal operator fusion strategy of Ascend can
    improve the inference performance

*/
namespace mindspore::opt {
CNodePtr KVCacheQuantPass::NewQuantNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, TypeId dst_type) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input_node);
  auto dst_prim = std::make_shared<lite::acl::Quant>();
  if (dst_prim == nullptr) {
    return nullptr;
  }
  dst_prim->AddAttr("scale", MakeValue(1.0f));
  dst_prim->AddAttr("offset", MakeValue(0.0f));
  TypePtr type_ptr = TypeIdToType(dst_type);
  dst_prim->AddAttr(ops::kDstType, type_ptr);
  std::vector<AnfNodePtr> quant_op_inputs = {NewValueNode(dst_prim), input_node};
  auto quant_cnode = func_graph->NewCNode(quant_op_inputs);
  quant_cnode->set_fullname_with_scope(input_node->fullname_with_scope() + "-Quant");
  quant_cnode->set_abstract(input_node->abstract()->Clone());
  quant_cnode->abstract()->set_type(type_ptr);

  auto ret = lite::quant::UpdateDataType(quant_cnode, dst_type);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << quant_cnode->fullname_with_scope() << " update datatype failed.";
    return nullptr;
  }
  return quant_cnode;
}

CNodePtr KVCacheQuantPass::NewAscendAntiQuantNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node,
                                                  TypeId dst_type) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input_node);
  auto dst_prim = std::make_shared<lite::acl::AscendAntiQuant>();
  if (dst_prim == nullptr) {
    return nullptr;
  }
  dst_prim->AddAttr("scale", MakeValue(1.0f));
  dst_prim->AddAttr("offset", MakeValue(0.0f));
  TypePtr type_ptr = TypeIdToType(dst_type);
  dst_prim->AddAttr(ops::kOutputDType, type_ptr);
  std::vector<AnfNodePtr> ascendantiquant_op_inputs = {NewValueNode(dst_prim), input_node};
  auto anti_cnode = func_graph->NewCNode(ascendantiquant_op_inputs);
  anti_cnode->set_fullname_with_scope(input_node->fullname_with_scope() + "-AntiQuant");
  anti_cnode->set_abstract(input_node->abstract()->Clone());
  anti_cnode->abstract()->set_type(type_ptr);
  auto ret = lite::quant::UpdateDataType(anti_cnode, dst_type);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << anti_cnode->fullname_with_scope() << " update datatype failed.";
    return nullptr;
  }
  return anti_cnode;
}

STATUS KVCacheQuantPass::ReplaceCastOpToQuantOp(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                                                const FuncGraphManagerPtr &manager) {
  CHECK_NULL_RETURN(func_graph);
  CHECK_NULL_RETURN(anf_node);
  CHECK_NULL_RETURN(manager);
  auto cast_cnode = anf_node->cast<CNodePtr>();
  auto input_node = cast_cnode->input(kIndexOne);

  TypeId dst_type_id;
  if (opt::GetDataTypeFromAnfNode(anf_node, &dst_type_id) != RET_OK) {
    MS_LOG(ERROR) << anf_node->fullname_with_scope() << " Get data type failed.";
    return RET_ERROR;
  }

  auto quant_node = NewQuantNode(func_graph, input_node, dst_type_id);
  CHECK_NULL_RETURN(quant_node);
  auto node_users = manager->node_users()[cast_cnode];
  for (auto &node_user : node_users) {
    auto post_cnode = node_user.first->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(post_cnode != nullptr, lite::RET_ERROR);
    manager->SetEdge(post_cnode, node_user.second, quant_node);
  }
  return RET_OK;
}

STATUS KVCacheQuantPass::ReplaceCastOpToAntiQuantOp(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                                                    const FuncGraphManagerPtr &manager) {
  CHECK_NULL_RETURN(func_graph);
  CHECK_NULL_RETURN(anf_node);
  CHECK_NULL_RETURN(manager);
  auto cast_cnode = anf_node->cast<CNodePtr>();
  auto input_node = cast_cnode->input(kIndexOne);

  TypeId dst_type_id;
  if (opt::GetDataTypeFromAnfNode(anf_node, &dst_type_id) != RET_OK) {
    MS_LOG(ERROR) << anf_node->fullname_with_scope() << " Get data type failed.";
    return RET_ERROR;
  }

  auto antiquant_node = NewAscendAntiQuantNode(func_graph, input_node, dst_type_id);
  CHECK_NULL_RETURN(antiquant_node);
  auto node_users = manager->node_users()[cast_cnode];
  for (auto &node_user : node_users) {
    auto post_cnode = node_user.first->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(post_cnode != nullptr, lite::RET_ERROR);
    manager->SetEdge(post_cnode, node_user.second, antiquant_node);
  }
  return RET_OK;
}

STATUS KVCacheQuantPass::RemoveOp(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                                  const FuncGraphManagerPtr &manager) {
  CHECK_NULL_RETURN(func_graph);
  CHECK_NULL_RETURN(anf_node);
  CHECK_NULL_RETURN(manager);
  if (!utils::isa<CNodePtr>(anf_node)) {
    MS_LOG(DEBUG) << "anf node is node a cnode.";
    return lite::RET_ERROR;
  }
  auto cnode = anf_node->cast<CNodePtr>();
  CHECK_NULL_RETURN(cnode);

  auto node_users = manager->node_users()[anf_node];
  for (auto &node_user : node_users) {
    auto post_cnode = node_user.first;
    manager->SetEdge(post_cnode, node_user.second, cnode->input(kIndex1));
  }
  return lite::RET_OK;
}

// Pattern:
// Mul -> Add -> Round -> Cast
// Replace:
// Mul -> Add -> Quant
STATUS KVCacheQuantPass::RunQuantPass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  CHECK_NULL_RETURN(func_graph);
  CHECK_NULL_RETURN(manager);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node) || !CheckPrimitiveType(node, prim::kPrimCast)) {
      continue;
    }
    auto round_node = node->cast<CNodePtr>()->input(kIndexOne);
    if (!CheckPrimitiveType(round_node, prim::kPrimRound)) {
      continue;
    }
    auto add_node = round_node->cast<CNodePtr>()->input(kIndexOne);
    if (!CheckPrimitiveType(add_node, prim::kPrimAdd)) {
      continue;
    }
    auto mul_node = add_node->cast<CNodePtr>()->input(kIndexOne);
    if (CheckPrimitiveType(mul_node, prim::kPrimMul)) {
      MS_LOG(INFO) << "Cast node: " << node->fullname_with_scope() << " will replace to Quant node";
      auto status = ReplaceCastOpToQuantOp(func_graph, node, manager);
      if (status != lite::RET_OK) {
        MS_LOG(ERROR) << "Failed to replace cast node to quant node, cast cnode: " << node->fullname_with_scope();
        return lite::RET_ERROR;
      }
      MS_LOG(INFO) << "Remove round node: " << round_node->fullname_with_scope();
      status = RemoveOp(func_graph, round_node, manager);
      if (status != lite::RET_OK) {
        MS_LOG(ERROR) << "Failed to remove round op: " << round_node->fullname_with_scope();
        return lite::RET_ERROR;
      }
    }
  }
  return lite::RET_OK;
}

// Pattern:
// Cast -> Add -> Mul
// Replace:
// AntiQuant -> Add -> Mul
STATUS KVCacheQuantPass::RunAntiQuantPass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  CHECK_NULL_RETURN(func_graph);
  CHECK_NULL_RETURN(manager);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node) || !CheckPrimitiveType(node, prim::kPrimMul)) {
      continue;
    }
    auto add_node = node->cast<CNodePtr>()->input(kIndexOne);
    if (!CheckPrimitiveType(add_node, prim::kPrimAdd)) {
      continue;
    }
    auto cast_node = add_node->cast<CNodePtr>()->input(kIndexOne);
    if (CheckPrimitiveType(cast_node, prim::kPrimCast)) {
      MS_LOG(INFO) << "Cast node: " << cast_node->fullname_with_scope() << " will replace to AscendAntiQuant node";
      auto status = ReplaceCastOpToAntiQuantOp(func_graph, cast_node, manager);
      if (status != lite::RET_OK) {
        MS_LOG(ERROR) << "Failed to replace cast node to antiquant node, cast cnode: "
                      << cast_node->fullname_with_scope();
        return lite::RET_ERROR;
      }
    }
  }
  return lite::RET_OK;
}

bool KVCacheQuantPass::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, false);
  auto status = RunQuantPass(func_graph, manager);
  MS_CHECK_TRUE_RET(status != lite::RET_ERROR, false);
  status = RunAntiQuantPass(func_graph, manager);
  MS_CHECK_TRUE_RET(status != lite::RET_ERROR, false);
  return true;
}
}  // namespace mindspore::opt
