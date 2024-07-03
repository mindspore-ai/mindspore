/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fusion/inference_matmul_split_fusion.h"
#include <vector>
#include <set>
#include "plugin/device/ascend/optimizer/common/gllo_utils.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {

bool InferenceMatmulSplitFusion::Run(const FuncGraphPtr &graph) {
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  bool changed = false;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return false;
  }
  constexpr auto kInferenceMatmulSplitSiluName = "InferenceMatmulSplitSilu";
  constexpr auto kInferenceMatmulSplitName = "InferenceMatmulSplit";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  auto enable_fusion =
    (std::find(enable_op_list.begin(), enable_op_list.end(), kInferenceMatmulSplitName) != enable_op_list.end());
  if (!enable_fusion) {
    return false;
  }
  enable_fusion_silu =
    (std::find(enable_op_list.begin(), enable_op_list.end(), kInferenceMatmulSplitSiluName) != enable_op_list.end());

  std::string pattern_name = "";
  auto node_list = TopoSort(graph->output());
  std::reverse(node_list.begin(), node_list.end());
  for (const auto &node : node_list) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto node_name = common::AnfAlgo::GetCNodeName(cnode);
    if (node_name != prim::kPrimSplitWithSize->name() && node_name != prim::kPrimSiLU->name()) {
      continue;
    }
    if (visited_cnodes.find(cnode) != visited_cnodes.end()) {
      continue;
    }
    pattern_name = GetFusionPatternName(cnode);
    MS_LOG(DEBUG) << "fusion pattern is : " << pattern_name;
    if (!pattern_name.empty()) {
      auto new_node = Process(pattern_name, graph, node);
      changed |= new_node != nullptr;
    }
  }
  return changed;
}

bool InferenceMatmulSplitFusion::CheckReshapeNode(const AnfNodePtr &node) const {
  auto matmul_cnode = node->cast<CNodePtr>();
  auto reshape_node = common::AnfAlgo::GetInputNode(matmul_cnode, kIndex0);
  if (reshape_node == nullptr || !reshape_node->isa<CNode>()) {
    return false;
  }
  auto reshape_node_name = common::AnfAlgo::GetCNodeName(reshape_node);
  if (reshape_node_name != prim::kPrimReshape->name()) {
    MS_LOG(DEBUG) << "matmul input node is not a reshape node, it's a : " << reshape_node_name;
    return false;
  }
  return true;
}

std::string InferenceMatmulSplitFusion::GetSplitFusionPatternName(const CNodePtr &cnode) const {
  std::string pattern_name = "";
  auto reshape_node = common::AnfAlgo::GetInputNode(cnode, kIndex0);
  if (reshape_node == nullptr || !reshape_node->isa<CNode>()) {
    return "";
  }
  auto reshape_node_name = common::AnfAlgo::GetCNodeName(reshape_node);
  if (reshape_node_name != prim::kPrimReshape->name()) {
    MS_LOG(DEBUG) << "reshape node name is: " << reshape_node_name;
    return "";
  }
  auto reshape_cnode = reshape_node->cast<CNodePtr>();
  auto reshape_input_node = common::AnfAlgo::GetInputNode(reshape_cnode, kIndex0);
  if (reshape_input_node != nullptr && reshape_input_node->isa<CNode>()) {
    auto reshape_input_name = common::AnfAlgo::GetCNodeName(reshape_input_node);
    if (reshape_input_name == prim::kPrimMatMul->name()) {
      MS_LOG(DEBUG) << "process matmul reshape split fusion";
      pattern_name = CheckReshapeNode(reshape_input_node) ? kPatternNameMatMulSplit : "";
    } else if (reshape_input_name == prim::kPrimQuantBatchMatmul->name()) {
      MS_LOG(DEBUG) << "process quant_batch_matmul reshape split fusion";
      pattern_name = CheckReshapeNode(reshape_input_node) ? kPatternNameQuantbatchmatmulSplit : "";
    } else if (reshape_input_name == prim::kPrimAdd->name()) {
      auto bias_add_cnode = reshape_input_node->cast<CNodePtr>();
      auto bias_input_node = common::AnfAlgo::GetInputNode(bias_add_cnode, kIndex0);
      if (bias_input_node->isa<CNode>() &&
          common::AnfAlgo::GetCNodeName(bias_input_node) == prim::kPrimMatMul->name()) {
        MS_LOG(DEBUG) << "process matmul biasadd reshape split fusion";
        pattern_name = CheckReshapeNode(bias_input_node) ? kPatternNameMatMulBiasAddSplit : "";
      }
    }
  }
  return pattern_name;
}

std::string InferenceMatmulSplitFusion::GetFusionPatternName(const CNodePtr &cnode) const {
  std::string pattern_name = "";
  auto cnode_name = common::AnfAlgo::GetCNodeName(cnode);
  if (cnode_name == prim::kPrimSiLU->name()) {
    if (!enable_fusion_silu) {
      MS_LOG(DEBUG) << "disable matmul split silu fusion";
      return "";
    }
    auto silu_input_node = common::AnfAlgo::GetInputNode(cnode, kIndex0);
    auto silu_input_name = common::AnfAlgo::GetCNodeName(silu_input_node);
    if (silu_input_name == prim::kPrimTupleGetItem->name()) {
      auto silu_input_cnode = silu_input_node->cast<CNodePtr>();
      auto item_input_node = common::AnfAlgo::GetInputNode(silu_input_cnode, kIndex0);
      auto item_input_name = common::AnfAlgo::GetCNodeName(item_input_node);
      if (item_input_name == prim::kPrimSplitWithSize->name()) {
        auto item_input_cnode = item_input_node->cast<CNodePtr>();
        auto split_pattern_name = GetSplitFusionPatternName(item_input_cnode);
        if (!split_pattern_name.empty()) {
          pattern_name = split_pattern_name + "Silu";
        }
      }
    }
  } else if (cnode_name == prim::kPrimSplitWithSize->name()) {
    pattern_name = GetSplitFusionPatternName(cnode);
  }
  return pattern_name;
}

bool InferenceMatmulSplitFusion::CheckMatMulDataFormat(const CNodePtr &matmul_cnode) const {
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, false);
  size_t trans_a_index = 0;
  size_t trans_b_index = 0;
  auto cnode_name = common::AnfAlgo::GetCNodeName(matmul_cnode);
  if (cnode_name == prim::kPrimQuantBatchMatmul->name()) {
    trans_a_index = kIndex6;
    trans_b_index = kIndex7;
  } else if (cnode_name == prim::kPrimMatMul->name()) {
    trans_a_index = kIndex3;
    trans_b_index = kIndex4;
  }
  auto trans_a = matmul_cnode->input(trans_a_index)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_RET(trans_a != nullptr, false);
  auto trans_b = matmul_cnode->input(trans_b_index)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_RET(trans_b != nullptr, false);
  bool is_trans_a = GetValue<bool>(trans_a->value());
  bool is_trans_b = GetValue<bool>(trans_b->value());
  MS_LOG(DEBUG) << "the transpose format of matmul node is: trans_a=" << is_trans_a << " trans_b=" << is_trans_b;
  if (!is_trans_a && is_trans_b) {
    return true;
  }
  return false;
}

size_t InferenceMatmulSplitFusion::GetSplitSizeLen(const CNodePtr &split_cnode) const {
  auto split_size = split_cnode->input(kIndex2)->cast<ValueNodePtr>();
  if (split_size == nullptr || !split_size->isa<ValueNode>()) {
    MS_LOG(DEBUG) << "split size node is nullptr";
    return 0;
  }
  auto split_size_shape = GetValue<std::vector<int64_t>>(split_size->value());
  size_t split_size_len = split_size_shape.size();
  return split_size_len;
}

PrimitivePtr InferenceMatmulSplitFusion::CreateMatmulSplitPrim(const CNodePtr &split_cnode, size_t split_size_len,
                                                               const std::string &pattern_name) const {
  PrimitivePtr matmul_split_prim = nullptr;
  std::string prim_name = "";
  auto iter = PatternPrimMap.find(split_size_len);
  if (iter != PatternPrimMap.end()) {
    auto iter_n = iter->second.find(pattern_name);
    if (iter_n != iter->second.end()) {
      prim_name = iter_n->second;
    }
  }
  MS_CHECK_TRUE_RET(!prim_name.empty(), nullptr);
  matmul_split_prim = std::make_shared<Primitive>(prim_name);
  MS_CHECK_TRUE_RET(matmul_split_prim != nullptr, nullptr);
  auto split_size = split_cnode->input(kIndex2)->cast<ValueNodePtr>();
  matmul_split_prim->AddAttr("n_lens", split_size->value());
  return matmul_split_prim;
}

CNodePtr InferenceMatmulSplitFusion::CreateMatmulSplitNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                           const std::string &pattern_name) const {
  MS_LOG(DEBUG) << "start create MatmulSplit node";
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto split_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(split_cnode != nullptr, nullptr);

  auto reshape_cnode = split_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_cnode != nullptr, nullptr);
  auto tuple_node = reshape_cnode->input(kIndex2);
  MS_CHECK_TRUE_RET(tuple_node != nullptr, nullptr);

  auto matmul_cnode = reshape_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(matmul_cnode->func_graph() == split_cnode->func_graph(), nullptr);

  auto pre_reshape_cnode = matmul_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(pre_reshape_cnode != nullptr, nullptr);
  auto input_x = pre_reshape_cnode->input(kIndex1);
  MS_CHECK_TRUE_RET(input_x != nullptr, nullptr);
  auto input_w = matmul_cnode->input(kIndex2);
  MS_CHECK_TRUE_RET(input_w != nullptr, nullptr);
  const std::set<TypeId> support_dtype = {kNumberTypeFloat16, kNumberTypeBFloat16};
  if (!CheckSupportDataType(input_x, support_dtype) || !CheckMatMulDataFormat(matmul_cnode)) {
    return nullptr;
  }

  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  auto reshape_tuple = kernel_graph->NewValueNode(MakeValue((int64_t)kTuplePlaceHolderNum));
  kernel_graph->AddValueNodeToGraph(reshape_tuple);
  MS_LOG(DEBUG) << "reshape_tuple node is: " << reshape_tuple->DebugString();

  size_t split_size_len = GetSplitSizeLen(split_cnode);
  auto matmul_split_prim = CreateMatmulSplitPrim(split_cnode, split_size_len, pattern_name);
  std::vector<AnfNodePtr> matmul_split_inputs = {input_x, input_w, reshape_tuple};
  auto matmul_split_cnode = func_graph->NewCNode(matmul_split_prim, matmul_split_inputs);
  MS_EXCEPTION_IF_NULL(matmul_split_cnode);

  matmul_split_cnode->set_fullname_with_scope(matmul_cnode->fullname_with_scope() + "-SplitWithSize");
  if (node->abstract() != nullptr) {
    matmul_split_cnode->set_abstract(split_cnode->abstract()->Clone());
  }
  visited_cnodes.insert(split_cnode);
  MS_LOG(DEBUG) << "create MatmulSplit node success.";
  return matmul_split_cnode;
}

CNodePtr InferenceMatmulSplitFusion::CreateMatmulBiasAddSplitNode(const FuncGraphPtr &func_graph,
                                                                  const AnfNodePtr &node,
                                                                  const std::string &pattern_name) const {
  MS_LOG(DEBUG) << "start create MatmulBiasAddSplit node";
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto split_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(split_cnode != nullptr, nullptr);

  auto reshape_cnode = split_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_cnode != nullptr, nullptr);
  auto reshape_tuple = reshape_cnode->input(kIndex2);
  MS_CHECK_TRUE_RET(reshape_tuple != nullptr, nullptr);

  auto biasAdd_cnode = reshape_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(biasAdd_cnode != nullptr, nullptr);
  auto matmul_cnode = biasAdd_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(matmul_cnode->func_graph() == split_cnode->func_graph(), {});

  auto pre_reshape = matmul_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(pre_reshape != nullptr, nullptr);
  auto matmul_x = pre_reshape->input(kIndex1);
  MS_EXCEPTION_IF_NULL(matmul_x);
  auto matmul_w = matmul_cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(matmul_w);
  auto input_bias = biasAdd_cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(input_bias);
  const std::set<TypeId> support_dtype = {kNumberTypeFloat16};
  if (!CheckSupportDataType(matmul_x, support_dtype) || !CheckMatMulDataFormat(matmul_cnode)) {
    return nullptr;
  }

  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  auto reshape_tuple_node = kernel_graph->NewValueNode(MakeValue((int64_t)kTuplePlaceHolderNum));
  kernel_graph->AddValueNodeToGraph(reshape_tuple_node);
  MS_LOG(DEBUG) << "reshape_tuple is " << reshape_tuple_node->DebugString();

  size_t split_size_len = GetSplitSizeLen(split_cnode);
  auto matmul_split_prim = CreateMatmulSplitPrim(split_cnode, split_size_len, pattern_name);
  matmul_split_prim->AddAttr("with_bias", MakeValue<bool>(true));
  std::vector<AnfNodePtr> matmul_split_inputs = {matmul_x, matmul_w, reshape_tuple_node, input_bias};
  auto matmul_split_cnode = func_graph->NewCNode(matmul_split_prim, matmul_split_inputs);
  MS_EXCEPTION_IF_NULL(matmul_split_cnode);

  matmul_split_cnode->set_fullname_with_scope(matmul_cnode->fullname_with_scope() + "-BiasAddSplitWithSize");
  if (node->abstract() != nullptr) {
    matmul_split_cnode->set_abstract(split_cnode->abstract()->Clone());
  }
  visited_cnodes.insert(split_cnode);
  MS_LOG(DEBUG) << "create MatmulBiasAddSplit node success.";
  return matmul_split_cnode;
}

CNodePtr InferenceMatmulSplitFusion::CreateQuantbatchmatmulSplitNode(const FuncGraphPtr &func_graph,
                                                                     const AnfNodePtr &node,
                                                                     const std::string &pattern_name) const {
  MS_LOG(DEBUG) << "start create QuantbatchmatmulSplit node";
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto split_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(split_cnode != nullptr, nullptr);

  auto reshape_cnode = split_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_cnode != nullptr, nullptr);
  auto qbmm_tuple = reshape_cnode->input(kIndex2);
  MS_CHECK_TRUE_RET(qbmm_tuple != nullptr, nullptr);
  auto qbmm_cnode = reshape_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(qbmm_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(qbmm_cnode->func_graph() == split_cnode->func_graph(), nullptr);

  auto pre_reshape = qbmm_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(pre_reshape != nullptr, nullptr);
  auto input_x = pre_reshape->input(kIndex1);
  MS_EXCEPTION_IF_NULL(input_x);
  auto input_w = qbmm_cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(input_w);
  auto input_bias = qbmm_cnode->input(kIndex5);
  MS_EXCEPTION_IF_NULL(input_bias);
  auto input_scale = qbmm_cnode->input(kIndex3);
  MS_EXCEPTION_IF_NULL(input_scale);
  const std::set<TypeId> support_dtype = {kNumberTypeInt8};
  if (!CheckSupportDataType(input_x, support_dtype) || !CheckMatMulDataFormat(qbmm_cnode)) {
    return nullptr;
  }

  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  auto tuple_node = kernel_graph->NewValueNode(MakeValue((int64_t)kTuplePlaceHolderNum));
  kernel_graph->AddValueNodeToGraph(tuple_node);
  MS_LOG(DEBUG) << "tuple_node " << tuple_node->DebugString();

  size_t split_size_len = GetSplitSizeLen(split_cnode);
  auto qbmm_split_prim = CreateMatmulSplitPrim(split_cnode, split_size_len, pattern_name);
  std::vector<AnfNodePtr> qbmm_split_inputs = {input_x, input_w, tuple_node, input_bias, input_scale};
  auto qbmm_split_cnode = func_graph->NewCNode(qbmm_split_prim, qbmm_split_inputs);
  MS_EXCEPTION_IF_NULL(qbmm_split_cnode);

  qbmm_split_cnode->set_fullname_with_scope(qbmm_cnode->fullname_with_scope() + "-SplitWithSize");
  if (node->abstract() != nullptr) {
    qbmm_split_cnode->set_abstract(split_cnode->abstract()->Clone());
  }
  visited_cnodes.insert(split_cnode);
  MS_LOG(DEBUG) << "create QuantbatchmatmulSplit node success.";
  return qbmm_split_cnode;
}

CNodePtr InferenceMatmulSplitFusion::CreateGetItemNode(const FuncGraphPtr &func_graph, const CNodePtr &split_cnode,
                                                       const CNodePtr &matmul_split_cnode, const CNodePtr &silu_cnode,
                                                       const size_t output_index) const {
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto iter = manager->node_users().find(split_cnode);
  if (iter == manager->node_users().end()) {
    MS_LOG(DEBUG) << "node has no output in manager";
    return nullptr;
  }

  auto output_info_list = iter->second;
  size_t used_output_index;
  CNodePtr item_other_node = nullptr;
  for (const auto &output_info : output_info_list) {
    auto cnode_name = common::AnfAlgo::GetCNodeName(output_info.first);
    if (cnode_name == prim::kPrimTupleGetItem->name()) {
      used_output_index = common::AnfAlgo::GetTupleGetItemOutIndex(utils::cast<CNodePtr>(output_info.first));
      if (used_output_index != output_index) {
        item_other_node = utils::cast<CNodePtr>(output_info.first);
        break;
      }
    }
  }
  MS_CHECK_TRUE_RET(item_other_node != nullptr, nullptr);
  item_other_node->set_input(kRealInputNodeIndexInTupleGetItem, matmul_split_cnode);
  auto value0 = NewValueNode(MakeValue((int64_t)output_index));
  value0->set_abstract(value0->value()->ToAbstract());
  auto new_item_cnode =
    func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem->Clone()), matmul_split_cnode, value0});
  MS_CHECK_TRUE_RET(new_item_cnode != nullptr, nullptr);
  auto silu_node = silu_cnode->cast<AnfNodePtr>();
  if (silu_node->abstract() != nullptr) {
    new_item_cnode->set_abstract(silu_node->abstract()->Clone());
  }
  MS_LOG(DEBUG) << "create new get_item_node success.";
  return new_item_cnode;
}

CNodePtr InferenceMatmulSplitFusion::CreateMatmulSplitSiluNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                               const std::string &pattern_name) const {
  MS_LOG(DEBUG) << "start create MatmulSplitSilu node";
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto silu_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(silu_cnode != nullptr, nullptr);
  auto item_cnode = silu_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(item_cnode != nullptr, nullptr);
  auto split_cnode = item_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(split_cnode != nullptr, nullptr);

  auto reshape_cnode = split_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_cnode != nullptr, nullptr);
  auto tuple = reshape_cnode->input(kIndex2);
  MS_CHECK_TRUE_RET(tuple != nullptr, nullptr);
  auto matmul_cnode = reshape_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(matmul_cnode->func_graph() == split_cnode->func_graph(), nullptr);

  auto pre_reshape = matmul_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(pre_reshape != nullptr, nullptr);
  auto x_node = pre_reshape->input(kIndex1);
  MS_EXCEPTION_IF_NULL(x_node);
  auto weight_node = matmul_cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(weight_node);
  const std::set<TypeId> support_dtype = {kNumberTypeFloat16, kNumberTypeBFloat16};
  if (!CheckSupportDataType(x_node, support_dtype) || !CheckMatMulDataFormat(matmul_cnode)) {
    return nullptr;
  }
  size_t split_size_len = GetSplitSizeLen(split_cnode);
  if (split_size_len != kMatmulFfnSplitSizeLen) {
    MS_LOG(DEBUG) << "MatmulSplitSilu only support ffn output";
    return nullptr;
  }
  auto fusion_prim = CreateMatmulSplitPrim(split_cnode, split_size_len, pattern_name);
  size_t output_index = common::AnfAlgo::GetTupleGetItemOutIndex(item_cnode);
  fusion_prim->AddAttr("silu_position", MakeValue<int32_t>(output_index));
  std::vector<AnfNodePtr> matmul_split_inputs = {x_node, weight_node, tuple};
  auto matmul_split_cnode = func_graph->NewCNode(fusion_prim, matmul_split_inputs);
  MS_EXCEPTION_IF_NULL(matmul_split_cnode);

  auto new_item_cnode = CreateGetItemNode(func_graph, split_cnode, matmul_split_cnode, silu_cnode, output_index);
  MS_CHECK_TRUE_RET(new_item_cnode != nullptr, nullptr);
  matmul_split_cnode->set_fullname_with_scope(matmul_cnode->fullname_with_scope() + "-SplitWithSizeSilu");
  if (node->abstract() != nullptr) {
    matmul_split_cnode->set_abstract(split_cnode->abstract()->Clone());
  }
  visited_cnodes.insert({silu_cnode, split_cnode});
  MS_LOG(DEBUG) << "create MatmulSplitSilu node success.";
  return new_item_cnode;
}

CNodePtr InferenceMatmulSplitFusion::CreateMatmulBiasAddSplitSiluNode(const FuncGraphPtr &func_graph,
                                                                      const AnfNodePtr &node,
                                                                      const std::string &pattern_name) const {
  MS_LOG(DEBUG) << "start create MatmulBiasAddSplitSilu node";
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto silu_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(silu_cnode != nullptr, nullptr);
  auto get_item_cnode = silu_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(get_item_cnode != nullptr, nullptr);
  auto split_cnode = get_item_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(split_cnode != nullptr, nullptr);

  auto reshape_cnode = split_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_cnode != nullptr, nullptr);
  auto tuple_node = reshape_cnode->input(kIndex2);
  MS_CHECK_TRUE_RET(tuple_node != nullptr, nullptr);
  auto biasAdd_cnode = reshape_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(biasAdd_cnode != nullptr, nullptr);

  auto matmul_cnode = biasAdd_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(matmul_cnode->func_graph() == split_cnode->func_graph(), {});

  auto pre_reshape = matmul_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(pre_reshape != nullptr, nullptr);
  auto matmul_input = pre_reshape->input(kIndex1);
  MS_EXCEPTION_IF_NULL(matmul_input);
  auto input_w = matmul_cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(input_w);
  auto input_bias = biasAdd_cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(input_bias);
  const std::set<TypeId> support_dtype = {kNumberTypeFloat16};
  if (!CheckSupportDataType(matmul_input, support_dtype) || !CheckMatMulDataFormat(matmul_cnode)) {
    return nullptr;
  }
  size_t split_len = GetSplitSizeLen(split_cnode);
  if (split_len != kMatmulFfnSplitSizeLen) {
    MS_LOG(DEBUG) << "MatmulBiasAddSplitSilu only support ffn output";
    return nullptr;
  }
  auto matmul_split_prim = CreateMatmulSplitPrim(split_cnode, split_len, pattern_name);
  size_t output_index = common::AnfAlgo::GetTupleGetItemOutIndex(get_item_cnode);
  matmul_split_prim->AddAttr("silu_position", MakeValue<int32_t>(output_index));
  matmul_split_prim->AddAttr("with_bias", MakeValue<bool>(true));
  std::vector<AnfNodePtr> matmul_split_inputs = {matmul_input, input_w, tuple_node, input_bias};
  auto matmul_split_cnode = func_graph->NewCNode(matmul_split_prim, matmul_split_inputs);
  MS_EXCEPTION_IF_NULL(matmul_split_cnode);

  auto new_item_cnode = CreateGetItemNode(func_graph, split_cnode, matmul_split_cnode, silu_cnode, output_index);
  MS_CHECK_TRUE_RET(new_item_cnode != nullptr, nullptr);
  matmul_split_cnode->set_fullname_with_scope(matmul_cnode->fullname_with_scope() + "-BiasAddSplitWithSizeSilu");
  if (node->abstract() != nullptr) {
    matmul_split_cnode->set_abstract(split_cnode->abstract()->Clone());
  }
  visited_cnodes.insert({silu_cnode, split_cnode});
  MS_LOG(DEBUG) << "create MatmulBiasAddSplitSilu node success.";
  return new_item_cnode;
}

CNodePtr InferenceMatmulSplitFusion::CreateQuantbatchmatmulSplitSiluNode(const FuncGraphPtr &func_graph,
                                                                         const AnfNodePtr &node,
                                                                         const std::string &pattern_name) const {
  MS_LOG(DEBUG) << "start create QuantbatchmatmulSplitSilu node";
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto silu_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(silu_cnode != nullptr, nullptr);
  auto item_cnode = silu_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(item_cnode != nullptr, nullptr);
  auto split_cnode = item_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(split_cnode != nullptr, nullptr);

  auto reshape_cnode = split_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_cnode != nullptr, nullptr);
  auto reshape_tuple = reshape_cnode->input(kIndex2);
  MS_CHECK_TRUE_RET(reshape_tuple != nullptr, nullptr);
  auto qbmm_cnode = reshape_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(qbmm_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(qbmm_cnode->func_graph() == split_cnode->func_graph(), nullptr);

  auto pre_reshape = qbmm_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(pre_reshape != nullptr, nullptr);
  auto qbmm_x = pre_reshape->input(kIndex1);
  MS_EXCEPTION_IF_NULL(qbmm_x);
  auto qbmm_w = qbmm_cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(qbmm_w);
  auto input_bias = qbmm_cnode->input(kIndex5);
  MS_EXCEPTION_IF_NULL(input_bias);
  auto input_scale = qbmm_cnode->input(kIndex3);
  MS_EXCEPTION_IF_NULL(input_scale);
  const std::set<TypeId> support_dtype = {kNumberTypeInt8};
  if (!CheckSupportDataType(qbmm_x, support_dtype) || !CheckMatMulDataFormat(qbmm_cnode)) {
    return nullptr;
  }
  size_t split_size_len = GetSplitSizeLen(split_cnode);
  if (split_size_len != kMatmulFfnSplitSizeLen) {
    MS_LOG(DEBUG) << "QuantbatchmatmulSplitSilu only support ffn output";
    return nullptr;
  }
  auto qbmm_split_prim = CreateMatmulSplitPrim(split_cnode, split_size_len, pattern_name);
  size_t output_index = common::AnfAlgo::GetTupleGetItemOutIndex(item_cnode);
  qbmm_split_prim->AddAttr("silu_position", MakeValue<int32_t>(output_index));
  std::vector<AnfNodePtr> qbmm_split_inputs = {qbmm_x, qbmm_w, reshape_tuple, input_bias, input_scale};
  auto qbmm_split_cnode = func_graph->NewCNode(qbmm_split_prim, qbmm_split_inputs);
  MS_EXCEPTION_IF_NULL(qbmm_split_cnode);

  auto new_item_cnode = CreateGetItemNode(func_graph, split_cnode, qbmm_split_cnode, silu_cnode, output_index);
  MS_CHECK_TRUE_RET(new_item_cnode != nullptr, nullptr);
  qbmm_split_cnode->set_fullname_with_scope(qbmm_cnode->fullname_with_scope() + "-SplitWithSizeSilu");
  if (node->abstract() != nullptr) {
    qbmm_split_cnode->set_abstract(split_cnode->abstract()->Clone());
  }
  visited_cnodes.insert({silu_cnode, split_cnode});
  MS_LOG(DEBUG) << "create QuantbatchmatmulSplitSilu node success.";
  return new_item_cnode;
}

AnfNodePtr InferenceMatmulSplitFusion::Process(const std::string &pattern_name, const FuncGraphPtr &func_graph,
                                               const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  auto split_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(split_cnode != nullptr, nullptr);
  CNodePtr matmul_split_cnode = nullptr;

  if (pattern_name == kPatternNameMatMulSplit) {
    matmul_split_cnode = CreateMatmulSplitNode(func_graph, node, pattern_name);
  }
  if (pattern_name == kPatternNameMatMulBiasAddSplit) {
    matmul_split_cnode = CreateMatmulBiasAddSplitNode(func_graph, node, pattern_name);
  }
  if (pattern_name == kPatternNameQuantbatchmatmulSplit) {
    matmul_split_cnode = CreateQuantbatchmatmulSplitNode(func_graph, node, pattern_name);
  }

  if (pattern_name == kPatternNameMatMulSplitSilu) {
    matmul_split_cnode = CreateMatmulSplitSiluNode(func_graph, node, pattern_name);
  }
  if (pattern_name == kPatternNameMatMulBiasAddSplitSilu) {
    matmul_split_cnode = CreateMatmulBiasAddSplitSiluNode(func_graph, node, pattern_name);
  }
  if (pattern_name == kPatternNameQuantbatchmatmulSplitSilu) {
    matmul_split_cnode = CreateQuantbatchmatmulSplitSiluNode(func_graph, node, pattern_name);
  }
  MS_CHECK_TRUE_RET(matmul_split_cnode != nullptr, nullptr);

  (void)manager->Replace(split_cnode, matmul_split_cnode);
  MS_LOG(DEBUG) << "MatmulSplit replace success";
  return matmul_split_cnode;
}
}  // namespace opt
}  // namespace mindspore
