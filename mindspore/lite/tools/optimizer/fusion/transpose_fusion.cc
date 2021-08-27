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

#include "tools/optimizer/fusion/transpose_fusion.h"
#include <unordered_map>
#include <memory>
#include <vector>
#include "tools/converter/quant_param_holder.h"
#include "mindspore/core/ops/transpose.h"
#include "tools/optimizer/common/format_utils.h"

namespace mindspore::opt {
bool IsBNCNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim::kPrimBatchNorm) ||
           CheckPrimitiveType(anf_node, prim::kPrimFusedBatchNorm);
  }
  return false;
}

VectorRef TransposeFusion::DefineBNPattern() const {
  auto transpose_var = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  auto conv_var = std::make_shared<CondVar>(IsConvNode);
  auto transpose_param = std::make_shared<CondVar>(IsParamNode);
  VectorRef transpose_conv_ref = VectorRef({transpose_var, conv_var, transpose_param});
  auto bn_var = std::make_shared<CondVar>(IsBNCNode);
  auto bn_mean_var = std::make_shared<CondVar>(IsParamNode);
  auto bn_variable_var = std::make_shared<CondVar>(IsParamNode);
  auto bn_other_var = std::make_shared<SeqVar>();
  VectorRef bn_ref = VectorRef({bn_var, transpose_conv_ref, bn_mean_var, bn_variable_var, bn_other_var});
  return bn_ref;
}

VectorRef TransposeFusion::DefineActivationscalePattern() const {
  auto transpose_var = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  auto conv_var = std::make_shared<CondVar>(IsConvNode);
  auto transpose_param = std::make_shared<CondVar>(IsParamNode);
  VectorRef transpose_conv_ref = VectorRef({transpose_var, conv_var, transpose_param});
  auto scale_var = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimScaleFusion>);
  auto scale_var_1 = std::make_shared<CondVar>(IsParamNode);
  auto scale_var_2 = std::make_shared<SeqVar>();
  VectorRef sclae_ref = VectorRef({scale_var, transpose_conv_ref, scale_var_1, scale_var_2});
  return sclae_ref;
}

VectorRef TransposeFusion::DefineActivationPattern() const {
  auto transpose_var = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  auto conv_var = std::make_shared<CondVar>(IsConvNode);
  auto transpose_param = std::make_shared<CondVar>(IsParamNode);
  VectorRef transpose_conv_ref = VectorRef({transpose_var, conv_var, transpose_param});
  auto act_var = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  VectorRef act_ref = VectorRef({act_var, transpose_conv_ref});
  return act_ref;
}

VectorRef TransposeFusion::DefineBiasAddPattern() const {
  auto transpose_var = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  auto conv_var = std::make_shared<CondVar>(IsConvNode);
  auto transpose_param = std::make_shared<CondVar>(IsParamNode);
  VectorRef transpose_conv_ref = VectorRef({transpose_var, conv_var, transpose_param});
  auto bias_var = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimBiasAdd>);
  auto bias_param = std::make_shared<CondVar>(IsParamNode);
  VectorRef act_ref = VectorRef({bias_var, transpose_conv_ref, bias_param});
  return act_ref;
}

VectorRef TransposeFusion::DefineTransTransPattern() const {
  auto transpose_var_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  auto transpose_var_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  auto transpose_param = std::make_shared<CondVar>(IsParamNode);
  VectorRef trans_trans_ref = VectorRef({transpose_var_2, transpose_var_1, transpose_param});
  return trans_trans_ref;
}

std::unordered_map<std::string, VectorRef> TransposeFusion::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns["BNPatternName"] = DefineBNPattern();
  patterns["ActivationPatternName"] = DefineActivationPattern();
  patterns["BiasAddPatternName"] = DefineBiasAddPattern();
  patterns["ScalePatternName"] = DefineActivationscalePattern();
  patterns["TransTransPatternName"] = DefineTransTransPattern();
  return patterns;
}

CNodePtr GenTransposeNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, const AnfNodePtr &perm,
                          const std::string &cnode_name) {
  MS_ASSERT(func_graph != nullptr && input_node != nullptr);
  auto trans_prim = std::make_shared<ops::Transpose>();
  MS_ASSERT(trans_prim != nullptr);
  auto cnode = func_graph->NewCNode(trans_prim, {input_node, perm});
  MS_ASSERT(cnode != nullptr);
  cnode->set_fullname_with_scope(cnode_name);
  auto quant_params_holder = std::make_shared<lite::QuantParamHolder>(2, 1);
  auto trans_insert_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  trans_insert_prim->AddAttr("quant_params", quant_params_holder);
  return cnode;
}

AnfNodePtr TransposeFusion::TransTransFusion(const mindspore::FuncGraphPtr &func_graph,
                                             const mindspore::AnfNodePtr &node) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto trans_cnode_2 = node->cast<CNodePtr>();
  if (!CheckPrimitiveType(trans_cnode_2, prim::kPrimTranspose) ||
      !CheckPrimitiveType(trans_cnode_2->input(1), prim::kPrimTranspose)) {
    return nullptr;
  }
  std::vector<int> post_perm;
  if (GetTransposePerm(trans_cnode_2, &post_perm) != lite::RET_OK) {
    MS_LOG(ERROR) << "get tanspose perm failed.";
    return nullptr;
  }
  std::vector<int> pre_perm;
  auto pre_node = trans_cnode_2->input(1);
  auto pre_cnode = pre_node->cast<CNodePtr>();
  if (pre_cnode == nullptr) {
    return nullptr;
  }
  if (GetTransposePerm(pre_cnode, &pre_perm) != lite::RET_OK) {
    MS_LOG(ERROR) << "get tanspose perm failed.";
    return nullptr;
  }
  if ((pre_perm == kNH2NC && post_perm == kNC2NH) || (pre_perm == kNC2NH && post_perm == kNH2NC)) {
    return pre_cnode->input(1);
  }
  return nullptr;
}

AnfNodePtr TransposeFusion::Process(const std::string &pattern_name, const mindspore::FuncGraphPtr &func_graph,
                                    const mindspore::AnfNodePtr &node, const mindspore::EquivPtr &equiv) const {
  if (pattern_name == "TransTransPatternName") {
    return TransTransFusion(func_graph, node);
  }
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  if (node->cast<CNodePtr>() == nullptr) {
    return nullptr;
  }
  auto any_cnode = node->cast<CNodePtr>();
  const auto transpose_node = any_cnode->input(1);
  if (transpose_node->cast<CNodePtr>() == nullptr) {
    return nullptr;
  }
  const CNodePtr &transpose_cnode = transpose_node->cast<CNodePtr>();
  auto perm_node = transpose_cnode->input(kInputIndexTwo);
  auto trans_post_node = GenTransposeNode(func_graph, any_cnode, perm_node, any_cnode->fullname_with_scope() + "_post");
  trans_post_node->set_abstract(any_cnode->abstract()->Clone());
  any_cnode->set_abstract(transpose_cnode->input(1)->abstract()->Clone());
  auto tr = func_graph->manager()->Transact();
  tr.SetEdge(any_cnode, 1, transpose_cnode->input(1));
  tr.Commit();
  return trans_post_node;
}
}  // namespace mindspore::opt
