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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_matmul_split_fusion.h"
#include <vector>
#include <set>
#include "backend/common/pass/common/gllo_utils.h"
#include "primitive/nn_ops.h"
#include "primitive/math_ops.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "utils/convert_utils.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "ir/graph_utils.h"
#include "primitive/nn_optimizer_ops.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_q.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {

bool InferenceMatmulSplitFusion::Run(const FuncGraphPtr &graph) {
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  bool changed = false;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  constexpr auto kInferenceMatmulSplitSiluName = "InferenceMatmulSplitSilu";
  constexpr auto kInferenceMatmulSplitName = "InferenceMatmulSplit";
  constexpr auto kInferenceMatmulSplitSiluFastgeluAddMulName = "InferenceGatedFFN";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  auto enable_fuse_gated_ffn = (std::find(enable_op_list.begin(), enable_op_list.end(),
                                          kInferenceMatmulSplitSiluFastgeluAddMulName) != enable_op_list.end());
  auto enable_fusion =
    (std::find(enable_op_list.begin(), enable_op_list.end(), kInferenceMatmulSplitName) != enable_op_list.end());
  if (!enable_fusion && !enable_fuse_gated_ffn) {
    return false;
  }
  enable_fusion_silu = enable_fusion && (std::find(enable_op_list.begin(), enable_op_list.end(),
                                                   kInferenceMatmulSplitSiluName) != enable_op_list.end());

  std::string pattern_name = "";
  auto node_list = TopoSort(graph->output());
  std::reverse(node_list.begin(), node_list.end());
  for (const auto &node : node_list) {
    bool fuse_gated_ffn = false;
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto node_name = common::AnfAlgo::GetCNodeName(cnode);
    if (enable_fuse_gated_ffn && (node_name == prim::kPrimMul->name())) {
      // last node is Mul and Fusion is allowed
      fuse_gated_ffn = true;
    } else if (node_name != prim::kPrimSplitWithSize->name() && node_name != prim::kPrimSiLU->name()) {
      continue;
    }
    if (visited_cnodes.find(cnode) != visited_cnodes.end()) {
      continue;
    }
    if (fuse_gated_ffn) {
      pattern_name = GetGatedFFNFusionPatternName(cnode);
    } else if (enable_fusion) {
      pattern_name = GetFusionPatternName(cnode);
    }
    MS_LOG(DEBUG) << "fusion pattern is : " << pattern_name;
    if (!pattern_name.empty()) {
      MS_LOG(DEBUG) << "pattern name is not empty. node name is " << node_name;
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

std::string InferenceMatmulSplitFusion::GetSiluMulPattern(const CNodePtr &mul_input0_node,
                                                          const CNodePtr &mul_input1_node) const {
  // in this branch we aim to find the first pattern -- kPatternNameMatMulSplitSiluMul
  std::string pattern_name = "";
  auto silu_input_node = common::AnfAlgo::GetInputNode(mul_input0_node->cast<CNodePtr>(), kIndex0);
  auto silu_input_name = common::AnfAlgo::GetCNodeName(silu_input_node);
  auto tuple_input_node = common::AnfAlgo::GetInputNode(mul_input1_node->cast<CNodePtr>(), kIndex0);
  auto tuple_input_name = common::AnfAlgo::GetCNodeName(tuple_input_node);
  if ((silu_input_name == prim::kPrimTupleGetItem->name()) && (tuple_input_name == prim::kPrimSplitWithSize->name())) {
    auto tuple2_input_node = common::AnfAlgo::GetInputNode(silu_input_node->cast<CNodePtr>(), kIndex0);
    auto split_input_node = common::AnfAlgo::GetInputNode(tuple_input_node->cast<CNodePtr>(), kIndex0);
    auto split_input_name = common::AnfAlgo::GetCNodeName(split_input_node);
    if ((tuple_input_node == tuple2_input_node) && (split_input_name == prim::kPrimReshape->name())) {
      auto reshape_input_node = common::AnfAlgo::GetInputNode(split_input_node->cast<CNodePtr>(), kIndex0);
      auto reshape_input_name = common::AnfAlgo::GetCNodeName(reshape_input_node);
      if (reshape_input_name == prim::kPrimMatMul->name()) {
        pattern_name = kPatternNameMatMulSplitSiluMul;
      } else if (reshape_input_name == prim::kPrimQuantBatchMatmul->name()) {
        pattern_name = kPatternNameQMatMulSplitSiluMul;
      }
    }
  }
  return pattern_name;
}

std::string InferenceMatmulSplitFusion::GetSiluFastGeluAddMulSubPattern(const CNodePtr &tuple_input_node,
                                                                        const CNodePtr &tuple2_input_node,
                                                                        const CNodePtr &tuple3_input_node) const {
  std::string pattern_name = "";
  auto tuple2_input_name = common::AnfAlgo::GetCNodeName(tuple2_input_node);
  if ((tuple2_input_name == prim::kPrimSplitWithSize->name()) && (tuple2_input_node == tuple3_input_node) &&
      (tuple2_input_node == tuple_input_node)) {
    auto split_input_node = common::AnfAlgo::GetInputNode(tuple_input_node->cast<CNodePtr>(), kIndex0);
    auto split_input_name = common::AnfAlgo::GetCNodeName(split_input_node);
    if (split_input_name == prim::kPrimReshape->name()) {
      auto reshape_input_node = common::AnfAlgo::GetInputNode(split_input_node->cast<CNodePtr>(), kIndex0);
      auto reshape_input_name = common::AnfAlgo::GetCNodeName(reshape_input_node);
      if (reshape_input_name == prim::kPrimMatMul->name()) {
        pattern_name = kPatternNameMatMulSplitSiluFastgeluAddMul;
      } else if (reshape_input_name == prim::kPrimQuantBatchMatmul->name()) {
        pattern_name = kPatternNameQMatMulSplitSiluFastgeluAddMul;
      }
    }
  }
  return pattern_name;
}

std::string InferenceMatmulSplitFusion::GetSiluFastGeluAddMulPattern(const CNodePtr &mul_input0_node,
                                                                     const CNodePtr &mul_input1_node) const {
  // in this branch we aim to find the second pattern -- kPatternNameMatMulSplitSiluFastgeluAddMul
  std::string pattern_name = "";
  auto add_input0_node = common::AnfAlgo::GetInputNode(mul_input0_node->cast<CNodePtr>(), kIndex0);
  auto add_input0_name = common::AnfAlgo::GetCNodeName(add_input0_node);
  auto add_input1_node = common::AnfAlgo::GetInputNode(mul_input0_node->cast<CNodePtr>(), kIndex1);
  auto add_input1_name = common::AnfAlgo::GetCNodeName(add_input1_node);
  auto tuple_input_node = common::AnfAlgo::GetInputNode(mul_input1_node->cast<CNodePtr>(), kIndex0);
  auto tuple_input_name = common::AnfAlgo::GetCNodeName(tuple_input_node);
  if ((add_input0_name == prim::kPrimSiLU->name()) && (add_input1_name == prim::kPrimFastGeLU->name()) &&
      (tuple_input_name == prim::kPrimSplitWithSize->name())) {
    auto silu_input_node = common::AnfAlgo::GetInputNode(add_input0_node->cast<CNodePtr>(), kIndex0);
    auto silu_input_name = common::AnfAlgo::GetCNodeName(silu_input_node);
    auto fastgelu_input_node = common::AnfAlgo::GetInputNode(add_input1_node->cast<CNodePtr>(), kIndex0);
    auto fastgelu_input_name = common::AnfAlgo::GetCNodeName(fastgelu_input_node);
    if ((silu_input_name == prim::kPrimTupleGetItem->name()) &&
        (fastgelu_input_name == prim::kPrimTupleGetItem->name())) {
      auto tuple2_input_node = common::AnfAlgo::GetInputNode(silu_input_node->cast<CNodePtr>(), kIndex0);
      auto tuple3_input_node = common::AnfAlgo::GetInputNode(fastgelu_input_node->cast<CNodePtr>(), kIndex0);
      pattern_name = GetSiluFastGeluAddMulSubPattern(
        tuple_input_node->cast<CNodePtr>(), tuple2_input_node->cast<CNodePtr>(), tuple3_input_node->cast<CNodePtr>());
    }
  }
  return pattern_name;
}

std::string InferenceMatmulSplitFusion::GetGatedFFNFusionPatternName(const CNodePtr &mul_cnode) const {
  // in this Fusion Pass we are searching for two constructions:
  // silu(w1 * x) ( w3 * x ) and
  // [silu(w1 * x) + FastGeLU(w11 * x)] (w3 * x)
  // in each of these constructions, in order to speed up computation, the weights are concatenated into a single W
  // such that a single MatMul operation is performed, then the output is Reshape, Split and used accordingly
  // This modification is performed in the python level, and here, during the fusion pass we capture it and replace it
  // with a single kernel
  std::string pattern_name = "";
  auto mul_i0_node = common::AnfAlgo::GetInputNode(mul_cnode, kIndex0);
  auto mul_i1_node = common::AnfAlgo::GetInputNode(mul_cnode, kIndex1);
  if (mul_i0_node == nullptr || !mul_i0_node->isa<CNode>() || mul_i1_node == nullptr || !mul_i1_node->isa<CNode>()) {
    return "";
  }
  auto mul_i0_name = common::AnfAlgo::GetCNodeName(mul_i0_node);
  auto mul_i1_name = common::AnfAlgo::GetCNodeName(mul_i1_node);
  if ((mul_i0_name == prim::kPrimSiLU->name()) && (mul_i1_name == prim::kPrimTupleGetItem->name())) {
    pattern_name = GetSiluMulPattern(mul_i0_node->cast<CNodePtr>(), mul_i1_node->cast<CNodePtr>());
  } else if ((mul_i0_name == prim::kPrimAdd->name()) && (mul_i1_name == prim::kPrimTupleGetItem->name())) {
    pattern_name = GetSiluFastGeluAddMulPattern(mul_i0_node->cast<CNodePtr>(), mul_i1_node->cast<CNodePtr>());
  }
  MS_LOG(DEBUG) << " found pattern " << pattern_name;
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
    trans_a_index = kIndex7;
    trans_b_index = kIndex8;
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

bool InferenceMatmulSplitFusion::CheckSplitSize(const AnfNodePtr &weight_cnode, const CNodePtr &split_cnode) const {
  size_t num_split = GetSplitSizeLen(split_cnode);
  if (num_split == 0) {
    MS_LOG(DEBUG) << "split size num is zero";
    return false;
  }

  auto split_size = split_cnode->input(kIndex2)->cast<ValueNodePtr>();
  auto split_size_shape = GetValue<std::vector<int64_t>>(split_size->value());
  auto weight_shape = BaseShapeToShape(AnfAlgo::GetOutputDetailShape(weight_cnode, 0));
  auto reshape_cnode = split_cnode->input(kIndex1)->cast<CNodePtr>();
  auto reshape_shape = common::AnfAlgo::GetOutputInferShape(reshape_cnode, kIndex0);
  unsigned int total_size = 0;
  for (size_t i = 0; i < num_split; i++) {
    total_size += split_size_shape[i];
  }
  const int64_t axis_n = 0;
  if (weight_shape[axis_n] != total_size || reshape_shape.size() != kDim3) {
    MS_LOG(DEBUG) << "check split size failed, weight_shape_n: " << weight_shape[axis_n]
                  << ", total_size: " << total_size << ", reshape_size: " << reshape_shape.size();
    return false;
  }
  MS_LOG(DEBUG) << "check split size pass";
  return true;
}

bool InferenceMatmulSplitFusion::CheckSplitSizeValid(const CNodePtr &split_cnode) const {
  size_t num_split = GetSplitSizeLen(split_cnode);
  if (num_split == 0) {
    MS_LOG(DEBUG) << "split size num is zero";
    return false;
  }
  auto split_size_node = split_cnode->input(kIndex2)->cast<ValueNodePtr>();
  auto split_size_shape = GetValue<std::vector<int64_t>>(split_size_node->value());
  for (size_t i = 0; i < num_split; i++) {
    auto split_size = split_size_shape[i];
    if (split_size % kValidShape != 0) {
      MS_LOG(DEBUG) << "split size should be a multiple of 16";
      return false;
    }
  }
  return true;
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
  if (split_size_len != 1) {
    auto split_size = split_cnode->input(kIndex2)->cast<ValueNodePtr>();
    matmul_split_prim->AddAttr("n_lens", split_size->value());
  }
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
  if (!CheckSupportDataType(input_x, support_dtype) || !CheckMatMulDataFormat(matmul_cnode) ||
      !CheckSplitSize(input_w, split_cnode) || !CheckSplitSizeValid(split_cnode)) {
    return nullptr;
  }
  std::vector<AnfNodePtr> orig_nodes = {split_cnode, reshape_cnode, matmul_cnode, pre_reshape_cnode};
  if (!CheckStreamAndCoreAttrWithOrigNodes(func_graph, orig_nodes)) {
    return nullptr;
  }

  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  auto reshape_tuple = kernel_graph->NewValueNode(MakeValue((int64_t)kTuplePlaceHolderNum));
  kernel_graph->AddValueNodeToGraph(reshape_tuple);
  MS_LOG(DEBUG) << "reshape_tuple node is: " << reshape_tuple->DebugString();

  size_t split_size_len = GetSplitSizeLen(split_cnode);
  auto matmul_split_prim = CreateMatmulSplitPrim(split_cnode, split_size_len, pattern_name);
  std::vector<AnfNodePtr> matmul_split_inputs = {NewValueNode(matmul_split_prim), input_x, input_w, reshape_tuple};
  auto matmul_split_cnode = NewCNode(matmul_split_inputs, func_graph, orig_nodes);
  MS_EXCEPTION_IF_NULL(matmul_split_cnode);

  matmul_split_cnode->set_scope(matmul_cnode->scope());
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
  const std::set<TypeId> support_dtype = {kNumberTypeFloat16, kNumberTypeBFloat16};
  if (!CheckSupportDataType(matmul_x, support_dtype) || !CheckMatMulDataFormat(matmul_cnode) ||
      !CheckSplitSize(matmul_w, split_cnode) || !CheckSplitSizeValid(split_cnode)) {
    return nullptr;
  }
  std::vector<AnfNodePtr> orig_nodes = {split_cnode, reshape_cnode, biasAdd_cnode, matmul_cnode, pre_reshape};
  if (!CheckStreamAndCoreAttrWithOrigNodes(func_graph, orig_nodes)) {
    return nullptr;
  }

  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  auto reshape_tuple_node = kernel_graph->NewValueNode(MakeValue((int64_t)kTuplePlaceHolderNum));
  kernel_graph->AddValueNodeToGraph(reshape_tuple_node);
  MS_LOG(DEBUG) << "reshape_tuple is " << reshape_tuple_node->DebugString();

  size_t split_size_len = GetSplitSizeLen(split_cnode);
  auto matmul_split_prim = CreateMatmulSplitPrim(split_cnode, split_size_len, pattern_name);
  matmul_split_prim->AddAttr("with_bias", MakeValue<bool>(true));
  std::vector<AnfNodePtr> matmul_split_inputs = {NewValueNode(matmul_split_prim), matmul_x, matmul_w,
                                                 reshape_tuple_node, input_bias};
  const std::set<TypeId> weight_bf16_dtype = {kNumberTypeBFloat16};
  if (CheckSupportDataType(input_bias, weight_bf16_dtype)) {
    auto type_value_f32 = std::make_shared<Int64Imm>(static_cast<int64_t>(TypeId::kNumberTypeFloat32));
    auto type_node_f32 = kernel_graph->NewValueNode(type_value_f32);
    std::vector<AnfNodePtr> casted_bias_inputs = {NewValueNode(prim::kPrimCast), input_bias, type_node_f32};
    auto bias_cast_cnode = NewCNode(casted_bias_inputs, func_graph, orig_nodes);
    MS_EXCEPTION_IF_NULL(bias_cast_cnode);
    auto type_fp32 = TypeIdToType(TypeId::kNumberTypeFloat32);
    auto cast_abs = std::make_shared<abstract::AbstractTensor>(type_fp32, input_bias->Shape());
    bias_cast_cnode->set_abstract(cast_abs);
    bias_cast_cnode->set_scope(matmul_cnode->scope());

    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    TypeId type_id = common::AnfAlgo::GetOutputInferDataType(matmul_cnode, 0);
    builder.SetInputsDeviceType({type_id, TypeId::kNumberTypeInt});
    builder.SetInputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
    builder.SetOutputsDeviceType({TypeId::kNumberTypeFloat32});
    builder.SetOutputsFormat({kOpFormat_DEFAULT});
    auto build_info = builder.Build();
    AnfAlgo::SetSelectKernelBuildInfo(build_info, bias_cast_cnode.get());

    const auto bias_in_num = 4;
    matmul_split_inputs[bias_in_num] = bias_cast_cnode;
  }
  auto matmul_split_cnode = NewCNode(matmul_split_inputs, func_graph, orig_nodes);
  MS_EXCEPTION_IF_NULL(matmul_split_cnode);

  matmul_split_cnode->set_scope(matmul_cnode->scope());
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
  auto pertoken_scale = qbmm_cnode->input(kIndex6);
  MS_EXCEPTION_IF_NULL(pertoken_scale);
  if (!IsValueNode<None>(pertoken_scale)) {
    MS_LOG(INFO) << "Currently, do not support to fuse qbmm(pertoken) with split.";
    return nullptr;
  }
  auto input_scale = qbmm_cnode->input(kIndex3);
  MS_EXCEPTION_IF_NULL(input_scale);
  const std::set<TypeId> support_dtype = {kNumberTypeInt8};
  const std::set<TypeId> support_output_dtype = {kNumberTypeFloat16};
  if (!CheckSupportDataType(input_x, support_dtype) || !CheckSupportDataType(qbmm_cnode, support_output_dtype) ||
      !CheckMatMulDataFormat(qbmm_cnode) || !CheckSplitSize(input_w, split_cnode)) {
    return nullptr;
  }
  std::vector<AnfNodePtr> orig_nodes = {split_cnode, reshape_cnode, qbmm_cnode, pre_reshape};
  if (!CheckStreamAndCoreAttrWithOrigNodes(func_graph, orig_nodes)) {
    return nullptr;
  }

  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  auto tuple_node = kernel_graph->NewValueNode(MakeValue((int64_t)kTuplePlaceHolderNum));
  kernel_graph->AddValueNodeToGraph(tuple_node);
  MS_LOG(DEBUG) << "tuple_node " << tuple_node->DebugString();

  size_t split_size_len = GetSplitSizeLen(split_cnode);
  auto qbmm_split_prim = CreateMatmulSplitPrim(split_cnode, split_size_len, pattern_name);
  std::vector<AnfNodePtr> qbmm_split_inputs = {
    NewValueNode(qbmm_split_prim), input_x, input_w, tuple_node, input_bias, input_scale};
  auto qbmm_split_cnode = NewCNode(qbmm_split_inputs, func_graph, orig_nodes);
  MS_EXCEPTION_IF_NULL(qbmm_split_cnode);

  qbmm_split_cnode->set_scope(qbmm_cnode->scope());
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
    NewCNode({NewValueNode(prim::kPrimTupleGetItem->Clone()), matmul_split_cnode, value0}, func_graph);
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
  if (!CheckSupportDataType(x_node, support_dtype) || !CheckMatMulDataFormat(matmul_cnode) ||
      !CheckSplitSize(weight_node, split_cnode)) {
    return nullptr;
  }
  size_t split_size_len = GetSplitSizeLen(split_cnode);
  if (split_size_len != kMatmulFfnSplitSizeLen) {
    MS_LOG(DEBUG) << "MatmulSplitSilu only support ffn output";
    return nullptr;
  }
  std::vector<AnfNodePtr> orig_nodes = {silu_cnode, split_cnode, reshape_cnode, matmul_cnode, pre_reshape};
  if (!CheckStreamAndCoreAttrWithOrigNodes(func_graph, orig_nodes)) {
    return nullptr;
  }
  auto fusion_prim = CreateMatmulSplitPrim(split_cnode, split_size_len, pattern_name);
  size_t output_index = common::AnfAlgo::GetTupleGetItemOutIndex(item_cnode);
  fusion_prim->AddAttr("silu_position", MakeValue<int32_t>(output_index));
  std::vector<AnfNodePtr> matmul_split_inputs = {NewValueNode(fusion_prim), x_node, weight_node, tuple};
  auto matmul_split_cnode = NewCNode(matmul_split_inputs, func_graph, orig_nodes);
  MS_EXCEPTION_IF_NULL(matmul_split_cnode);

  auto new_item_cnode = CreateGetItemNode(func_graph, split_cnode, matmul_split_cnode, silu_cnode, output_index);
  MS_CHECK_TRUE_RET(new_item_cnode != nullptr, nullptr);
  matmul_split_cnode->set_scope(matmul_cnode->scope());
  if (node->abstract() != nullptr) {
    matmul_split_cnode->set_abstract(split_cnode->abstract()->Clone());
  }
  visited_cnodes.insert({silu_cnode, split_cnode});
  MS_LOG(DEBUG) << "create MatmulSplitSilu node success.";
  return new_item_cnode;
}

CNodePtr InferenceMatmulSplitFusion::CreateMatmulSplitSiluMulNode(const FuncGraphPtr &func_graph,
                                                                  const AnfNodePtr &node,
                                                                  const std::string &pattern_name) const {
  MS_LOG(DEBUG) << "start create MatmulSplitSiluMul node";
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto elem_mul_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(elem_mul_cnode != nullptr, nullptr);

  auto silu_cnode = elem_mul_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(silu_cnode != nullptr, nullptr);
  auto tuple_get_item_cnode = elem_mul_cnode->input(kIndex2)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(tuple_get_item_cnode != nullptr, nullptr);
  auto split_cnode = tuple_get_item_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(split_cnode != nullptr, nullptr);

  size_t split_size_len = kMatmulFfnSplitSizeLen;
  auto reshape_cnode = split_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_cnode != nullptr, nullptr);

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
  std::vector<AnfNodePtr> orig_nodes = {elem_mul_cnode, silu_cnode,   split_cnode,
                                        reshape_cnode,  matmul_cnode, pre_reshape};
  if (!CheckStreamAndCoreAttrWithOrigNodes(func_graph, orig_nodes)) {
    return nullptr;
  }
  auto fusion_prim = CreateMatmulSplitPrim(split_cnode, split_size_len, pattern_name);
  fusion_prim->AddAttr("silu_position", MakeValue<int32_t>(1));
  std::vector<AnfNodePtr> matmul_split_inputs = {NewValueNode(fusion_prim), x_node, weight_node};
  auto matmul_split_cnode = NewCNode(matmul_split_inputs, func_graph, orig_nodes);
  MS_EXCEPTION_IF_NULL(matmul_split_cnode);

  matmul_split_cnode->set_fullname_with_scope(matmul_cnode->fullname_with_scope() + "-SplitWithSiluMul");
  if (node->abstract() != nullptr) {
    matmul_split_cnode->set_abstract(elem_mul_cnode->abstract()->Clone());
  }

  visited_cnodes.insert({silu_cnode, split_cnode});
  MS_LOG(DEBUG) << "create MatmulSplitSiluMul node success.";
  return matmul_split_cnode;
}

CNodePtr InferenceMatmulSplitFusion::CreateMatmulSplitSiluFastgeluAddMulNode(const FuncGraphPtr &func_graph,
                                                                             const AnfNodePtr &node,
                                                                             const std::string &pattern_name) const {
  MS_LOG(DEBUG) << "start create MatmulSplitSiluFastgeluAddMul node";
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto elem_mul_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(elem_mul_cnode != nullptr, nullptr);

  auto add_cnode = elem_mul_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add_cnode != nullptr, nullptr);
  auto silu_cnode = add_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(silu_cnode != nullptr, nullptr);

  auto tuple_get_item_cnode = elem_mul_cnode->input(kIndex2)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(tuple_get_item_cnode != nullptr, nullptr);
  auto split_cnode = tuple_get_item_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(split_cnode != nullptr, nullptr);

  size_t split_size_len = kMatmulQkvSplitSizeLen;

  auto reshape_cnode = split_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_cnode != nullptr, nullptr);

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
  std::vector<AnfNodePtr> orig_nodes = {elem_mul_cnode, silu_cnode,   split_cnode,
                                        reshape_cnode,  matmul_cnode, pre_reshape};
  if (!CheckStreamAndCoreAttrWithOrigNodes(func_graph, orig_nodes)) {
    return nullptr;
  }
  auto fusion_prim = CreateMatmulSplitPrim(split_cnode, split_size_len, pattern_name);
  fusion_prim->AddAttr("silu_position", MakeValue<int32_t>(1));
  std::vector<AnfNodePtr> matmul_split_inputs = {NewValueNode(fusion_prim), x_node, weight_node};
  auto matmul_split_cnode = NewCNode(matmul_split_inputs, func_graph, orig_nodes);
  MS_EXCEPTION_IF_NULL(matmul_split_cnode);

  matmul_split_cnode->set_fullname_with_scope(matmul_cnode->fullname_with_scope() + "-SplitWithSiluFastGeluAddMul");
  if (node->abstract() != nullptr) {
    matmul_split_cnode->set_abstract(elem_mul_cnode->abstract()->Clone());
  }

  visited_cnodes.insert({silu_cnode, split_cnode});
  MS_LOG(DEBUG) << "create MatmulSplitSiluFastgeluAddMul node success.";
  return matmul_split_cnode;
}

CNodePtr InferenceMatmulSplitFusion::CreateQMatmulSplitSiluMulNode(const FuncGraphPtr &func_graph,
                                                                   const AnfNodePtr &node,
                                                                   const std::string &pattern_name) const {
  MS_LOG(DEBUG) << "start create MatmulSplitSiluMul node";
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto elem_mul_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(elem_mul_cnode != nullptr, nullptr);

  auto silu_cnode = elem_mul_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(silu_cnode != nullptr, nullptr);
  auto tuple_get_item_cnode = elem_mul_cnode->input(kIndex2)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(tuple_get_item_cnode != nullptr, nullptr);
  auto split_cnode = tuple_get_item_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(split_cnode != nullptr, nullptr);

  size_t split_size_len = kMatmulFfnSplitSizeLen;
  auto reshape_cnode = split_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_cnode != nullptr, nullptr);

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
  auto pertoken_scale = qbmm_cnode->input(kIndex6);
  MS_EXCEPTION_IF_NULL(pertoken_scale);
  if (!IsValueNode<None>(pertoken_scale)) {
    MS_LOG(INFO) << "Currently, do not support to fuse qbmm(pertoken) with split.";
    return nullptr;
  }
  auto input_scale = qbmm_cnode->input(kIndex3);
  MS_EXCEPTION_IF_NULL(input_scale);
  const std::set<TypeId> support_dtype = {kNumberTypeInt8};
  if (!CheckSupportDataType(qbmm_x, support_dtype) || !CheckMatMulDataFormat(qbmm_cnode) ||
      !CheckSplitSize(qbmm_w, split_cnode)) {
    return nullptr;
  }
  std::vector<AnfNodePtr> orig_nodes = {elem_mul_cnode, silu_cnode, split_cnode,
                                        reshape_cnode,  qbmm_cnode, pre_reshape};
  if (!CheckStreamAndCoreAttrWithOrigNodes(func_graph, orig_nodes)) {
    return nullptr;
  }

  auto fusion_prim = CreateMatmulSplitPrim(split_cnode, split_size_len, pattern_name);
  fusion_prim->AddAttr("silu_position", MakeValue<int32_t>(1));
  std::vector<AnfNodePtr> qbmm_split_inputs = {NewValueNode(fusion_prim), qbmm_x, qbmm_w, input_bias, input_scale};

  auto qmatmul_split_cnode = NewCNode(qbmm_split_inputs, func_graph, orig_nodes);
  MS_EXCEPTION_IF_NULL(qmatmul_split_cnode);

  qmatmul_split_cnode->set_fullname_with_scope(qbmm_cnode->fullname_with_scope() + "-SplitWithSiluMul");
  if (node->abstract() != nullptr) {
    qmatmul_split_cnode->set_abstract(elem_mul_cnode->abstract()->Clone());
  }

  visited_cnodes.insert({silu_cnode, split_cnode});
  MS_LOG(DEBUG) << "create QMatmulSplitSiluMul node success.";
  return qmatmul_split_cnode;
}

CNodePtr InferenceMatmulSplitFusion::CreateQMatmulSplitSiluFastgeluAddMulNode(const FuncGraphPtr &func_graph,
                                                                              const AnfNodePtr &node,
                                                                              const std::string &pattern_name) const {
  MS_LOG(DEBUG) << "start create MatmulSplitSiluFastgeluAddMul node";
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto elem_mul_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(elem_mul_cnode != nullptr, nullptr);

  auto add_cnode = elem_mul_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add_cnode != nullptr, nullptr);
  auto silu_cnode = add_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(silu_cnode != nullptr, nullptr);

  auto tuple_get_item_cnode = elem_mul_cnode->input(kIndex2)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(tuple_get_item_cnode != nullptr, nullptr);
  auto split_cnode = tuple_get_item_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(split_cnode != nullptr, nullptr);

  size_t split_size_len = kMatmulQkvSplitSizeLen;
  auto reshape_cnode = split_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_cnode != nullptr, nullptr);

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
  auto pertoken_scale = qbmm_cnode->input(kIndex6);
  MS_EXCEPTION_IF_NULL(pertoken_scale);
  if (!IsValueNode<None>(pertoken_scale)) {
    MS_LOG(INFO) << "Currently, do not support to fuse qbmm(pertoken) with split.";
    return nullptr;
  }
  auto input_scale = qbmm_cnode->input(kIndex3);
  MS_EXCEPTION_IF_NULL(input_scale);
  const std::set<TypeId> support_dtype = {kNumberTypeInt8};
  if (!CheckSupportDataType(qbmm_x, support_dtype) || !CheckMatMulDataFormat(qbmm_cnode) ||
      !CheckSplitSize(qbmm_w, split_cnode)) {
    return nullptr;
  }
  std::vector<AnfNodePtr> orig_nodes = {elem_mul_cnode, add_cnode,  silu_cnode, split_cnode,
                                        reshape_cnode,  qbmm_cnode, pre_reshape};
  if (!CheckStreamAndCoreAttrWithOrigNodes(func_graph, orig_nodes)) {
    return nullptr;
  }

  auto fusion_prim = CreateMatmulSplitPrim(split_cnode, split_size_len, pattern_name);
  fusion_prim->AddAttr("silu_position", MakeValue<int32_t>(1));
  std::vector<AnfNodePtr> qbmm_split_inputs = {NewValueNode(fusion_prim), qbmm_x, qbmm_w, input_bias, input_scale};
  auto qmatmul_split_cnode = NewCNode(qbmm_split_inputs, func_graph, orig_nodes);
  MS_EXCEPTION_IF_NULL(qmatmul_split_cnode);

  qmatmul_split_cnode->set_fullname_with_scope(qbmm_cnode->fullname_with_scope() + "-SplitWithSiluFastGeluAddMul");
  if (node->abstract() != nullptr) {
    qmatmul_split_cnode->set_abstract(elem_mul_cnode->abstract()->Clone());
  }

  visited_cnodes.insert({silu_cnode, split_cnode});
  MS_LOG(DEBUG) << "create QMatmulSplitSiluFastgeluAddMul node success.";
  return qmatmul_split_cnode;
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
  if (!CheckSupportDataType(matmul_input, support_dtype) || !CheckMatMulDataFormat(matmul_cnode) ||
      !CheckSplitSize(input_w, split_cnode)) {
    return nullptr;
  }
  size_t split_len = GetSplitSizeLen(split_cnode);
  if (split_len != kMatmulFfnSplitSizeLen) {
    MS_LOG(DEBUG) << "MatmulBiasAddSplitSilu only support ffn output";
    return nullptr;
  }
  std::vector<AnfNodePtr> orig_nodes = {silu_cnode,    split_cnode,  reshape_cnode,
                                        biasAdd_cnode, matmul_cnode, pre_reshape};
  if (!CheckStreamAndCoreAttrWithOrigNodes(func_graph, orig_nodes)) {
    return nullptr;
  }
  auto matmul_split_prim = CreateMatmulSplitPrim(split_cnode, split_len, pattern_name);
  size_t output_index = common::AnfAlgo::GetTupleGetItemOutIndex(get_item_cnode);
  matmul_split_prim->AddAttr("silu_position", MakeValue<int32_t>(output_index));
  matmul_split_prim->AddAttr("with_bias", MakeValue<bool>(true));
  std::vector<AnfNodePtr> matmul_split_inputs = {NewValueNode(matmul_split_prim), matmul_input, input_w, tuple_node,
                                                 input_bias};
  auto matmul_split_cnode = NewCNode(matmul_split_inputs, func_graph, orig_nodes);
  MS_EXCEPTION_IF_NULL(matmul_split_cnode);

  auto new_item_cnode = CreateGetItemNode(func_graph, split_cnode, matmul_split_cnode, silu_cnode, output_index);
  MS_CHECK_TRUE_RET(new_item_cnode != nullptr, nullptr);
  matmul_split_cnode->set_scope(matmul_cnode->scope());
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
  auto pertoken_scale = qbmm_cnode->input(kIndex6);
  MS_EXCEPTION_IF_NULL(pertoken_scale);
  if (!IsValueNode<None>(pertoken_scale)) {
    MS_LOG(INFO) << "Currently, do not support to fuse qbmm(pertoken) with split.";
    return nullptr;
  }
  auto input_scale = qbmm_cnode->input(kIndex3);
  MS_EXCEPTION_IF_NULL(input_scale);
  const std::set<TypeId> support_dtype = {kNumberTypeInt8};
  if (!CheckSupportDataType(qbmm_x, support_dtype) || !CheckMatMulDataFormat(qbmm_cnode) ||
      !CheckSplitSize(qbmm_w, split_cnode)) {
    return nullptr;
  }
  size_t split_size_len = GetSplitSizeLen(split_cnode);
  if (split_size_len != kMatmulFfnSplitSizeLen) {
    MS_LOG(DEBUG) << "QuantbatchmatmulSplitSilu only support ffn output";
    return nullptr;
  }
  std::vector<AnfNodePtr> orig_nodes = {silu_cnode, split_cnode, reshape_cnode, qbmm_cnode, pre_reshape};
  if (!CheckStreamAndCoreAttrWithOrigNodes(func_graph, orig_nodes)) {
    return nullptr;
  }
  auto qbmm_split_prim = CreateMatmulSplitPrim(split_cnode, split_size_len, pattern_name);
  size_t output_index = common::AnfAlgo::GetTupleGetItemOutIndex(item_cnode);
  qbmm_split_prim->AddAttr("silu_position", MakeValue<int32_t>(output_index));
  std::vector<AnfNodePtr> qbmm_split_inputs = {
    NewValueNode(qbmm_split_prim), qbmm_x, qbmm_w, reshape_tuple, input_bias, input_scale};
  auto qbmm_split_cnode = NewCNode(qbmm_split_inputs, func_graph, orig_nodes);
  MS_EXCEPTION_IF_NULL(qbmm_split_cnode);

  auto new_item_cnode = CreateGetItemNode(func_graph, split_cnode, qbmm_split_cnode, silu_cnode, output_index);
  MS_CHECK_TRUE_RET(new_item_cnode != nullptr, nullptr);
  qbmm_split_cnode->set_scope(qbmm_cnode->scope());
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

  auto top_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(top_cnode != nullptr, nullptr);
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
  if (pattern_name == kPatternNameMatMulSplitSiluMul) {
    matmul_split_cnode = CreateMatmulSplitSiluMulNode(func_graph, node, pattern_name);
  }
  if (pattern_name == kPatternNameMatMulSplitSiluFastgeluAddMul) {
    matmul_split_cnode = CreateMatmulSplitSiluFastgeluAddMulNode(func_graph, node, pattern_name);
  }
  if (pattern_name == kPatternNameQMatMulSplitSiluMul) {
    matmul_split_cnode = CreateQMatmulSplitSiluMulNode(func_graph, node, pattern_name);
  }
  if (pattern_name == kPatternNameQMatMulSplitSiluFastgeluAddMul) {
    matmul_split_cnode = CreateQMatmulSplitSiluFastgeluAddMulNode(func_graph, node, pattern_name);
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

  (void)manager->Replace(top_cnode, matmul_split_cnode);
  MS_LOG(DEBUG) << "MatmulSplit replace success";
  return matmul_split_cnode;
}
}  // namespace opt
}  // namespace mindspore
