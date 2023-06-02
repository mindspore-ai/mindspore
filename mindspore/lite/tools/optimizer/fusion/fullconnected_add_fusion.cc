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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/fullconnected_add_fusion.h"
#include <vector>
#include <memory>
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/lite_ops.h"
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/full_connection.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace opt {
namespace {
bool IsPrimitiveProper(const CNodePtr &add_cnode, const CNodePtr &fc_cnode, int index) {
  auto add_primc = GetValueNode<PrimitiveCPtr>(add_cnode->input(0));
  MS_CHECK_TRUE_RET(add_primc != nullptr, false);
  if (IsQuantParameterNode(add_primc)) {
    MS_LOG(INFO) << add_cnode->fullname_with_scope() << " is quant node";
    return false;
  }

  auto add_param_node = add_cnode->input(kInputSizeThree - index);
  if (!utils::isa<ValueNode>(add_param_node) &&
      (!utils::isa<Parameter>(add_param_node) || !add_param_node->cast<ParameterPtr>()->default_param())) {
    return false;
  }
  auto abstract = add_param_node->abstract();
  MS_CHECK_TRUE_RET(abstract != nullptr, false);
  std::vector<int64_t> bias_shape;
  if (FetchShapeFromAbstract(abstract, &bias_shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "Fetch shape from abstract failed.";
    return false;
  }
  if (bias_shape.size() > DIMENSION_1D) {
    MS_LOG(INFO) << "only support bias with shape size of 1.";
    return false;
  }

  if (fc_cnode->size() > kInputSizeThree) {
    auto fc_bias_node = fc_cnode->input(kInputIndexThree);
    if (!utils::isa<ValueNode>(fc_bias_node) &&
        (!utils::isa<Parameter>(fc_bias_node) || !fc_bias_node->cast<ParameterPtr>()->default_param())) {
      MS_LOG(INFO) << fc_cnode->fullname_with_scope() << "'s bias is not parameter";
      return false;
    }
  }
  auto fc_primc = ops::GetOperator<ops::FullConnection>(fc_cnode->input(0));
  MS_CHECK_TRUE_RET(fc_primc != nullptr, false);
  if (fc_primc->GetAttr(ops::kActivationType) != nullptr &&
      fc_primc->get_activation_type() != ActivationType::NO_ACTIVATION) {
    MS_LOG(INFO) << fc_cnode->fullname_with_scope() << " has activation attr";
    return false;
  }
  auto prim_c = fc_primc->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, false);
  if (IsQuantParameterNode(prim_c)) {
    MS_LOG(INFO) << fc_cnode->fullname_with_scope() << "is quant node";
    return false;
  }

  return true;
}

int CalNewCnodeBias(const AnfNodePtr &add_weight_node, const CNodePtr &fc_cnode) {
  MS_CHECK_TRUE_RET(add_weight_node != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(fc_cnode != nullptr, RET_ERROR);
  auto fc_bias_node = fc_cnode->input(kInputIndexThree);
  MS_CHECK_TRUE_RET(fc_bias_node != nullptr, RET_ERROR);
  std::shared_ptr<tensor::Tensor> fc_bias_tensor = GetTensorInfo(fc_bias_node);
  MS_CHECK_TRUE_RET(fc_bias_tensor != nullptr, RET_ERROR);
  if (fc_bias_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(INFO) << "only support float32 data type";
    return RET_ERROR;
  }
  std::vector<int64_t> fc_bias_shape = fc_bias_tensor->shape();
  auto fc_bias_data = reinterpret_cast<float *>(fc_bias_tensor->data_c());
  MS_CHECK_TRUE_RET(fc_bias_data != nullptr, RET_ERROR);

  std::shared_ptr<tensor::Tensor> add_weight_tensor = GetTensorInfo(add_weight_node);
  MS_CHECK_TRUE_RET(add_weight_tensor != nullptr, RET_ERROR);
  if (add_weight_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(INFO) << "only support float32 data type";
    return RET_ERROR;
  }
  std::vector<int64_t> add_weight_shape = add_weight_tensor->shape();
  MS_CHECK_TRUE_RET(fc_bias_shape == add_weight_shape, RET_ERROR);
  auto add_weight_data = reinterpret_cast<float *>(add_weight_tensor->data_c());
  MS_CHECK_TRUE_RET(add_weight_data != nullptr, RET_ERROR);

  for (int64_t i = 0; i < fc_bias_shape[0]; ++i) {
    fc_bias_data[i] += add_weight_data[i];
  }
  return RET_OK;
}
}  // namespace

VectorRef FullconnectedAddFusion::DefineFcAddFusionPattern() const {
  auto is_fc1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimFullConnection>);
  MS_CHECK_TRUE_RET(is_fc1 != nullptr, {});
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  return VectorRef({is_add, is_fc1, is_seq_var});
}

VectorRef FullconnectedAddFusion::DefineFcBiasAddPattern() const {
  auto is_fc1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimFullConnection>);
  MS_CHECK_TRUE_RET(is_fc1 != nullptr, {});
  auto is_bias_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimBiasAdd>);
  MS_CHECK_TRUE_RET(is_bias_add != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  return VectorRef({is_bias_add, is_fc1, is_seq_var});
}

std::unordered_map<std::string, VectorRef> FullconnectedAddFusion::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns["FcAddFusionPatternName"] = DefineFcAddFusionPattern();
  patterns["FcBiasAddPatternName"] = DefineFcBiasAddPattern();
  return patterns;
}

AnfNodePtr FullconnectedAddFusion::Process(const std::string &pattern_name, const FuncGraphPtr &func_graph,
                                           const AnfNodePtr &node, const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }

  auto add_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add_cnode != nullptr, nullptr);
  if (IsMarkedTrainOp(add_cnode)) {
    return nullptr;
  }
  if (!CheckPrimitiveType(node, prim::kPrimAddFusion) && !CheckPrimitiveType(node, prim::kPrimBiasAdd)) {
    return nullptr;
  }

  size_t index = 0;
  if (!CheckAndGetCnodeIndex(add_cnode, &index, prim::kPrimFullConnection)) {
    return nullptr;
  }
  auto fc_cnode = add_cnode->input(index)->cast<CNodePtr>();
  MS_ASSERT(fc_cnode != nullptr);
  if (IsMarkedTrainOp(fc_cnode)) {
    return nullptr;
  }

  if (IsMultiOutputTensors(func_graph, fc_cnode)) {
    return nullptr;
  }

  if (!IsPrimitiveProper(add_cnode, fc_cnode, index)) {
    return nullptr;
  }

  auto manager = func_graph->manager();
  auto add_param_node = add_cnode->input(kInputSizeThree - index);
  MS_CHECK_TRUE_RET(manager != nullptr, nullptr);
  if (fc_cnode->size() == kInputSizeThree) {
    manager->AddEdge(fc_cnode, add_param_node);
  } else if (fc_cnode->size() == kInputSizeFour) {
    if (CalNewCnodeBias(add_param_node, fc_cnode) != RET_OK) {
      MS_LOG(INFO) << add_cnode->fullname_with_scope() << " failed to fusion with " << fc_cnode->fullname_with_scope();
      return nullptr;
    }
  }

  if (CheckPrimitiveType(node, prim::kPrimAddFusion)) {
    auto add_primc = ops::GetOperator<ops::AddFusion>(add_cnode->input(0));
    MS_CHECK_TRUE_RET(add_primc != nullptr, nullptr);
    if (add_primc->GetAttr(ops::kActivationType) != nullptr &&
        add_primc->get_activation_type() != ActivationType::NO_ACTIVATION) {
      auto fc_primc = ops::GetOperator<ops::FullConnection>(fc_cnode->input(0));
      MS_CHECK_TRUE_RET(fc_primc != nullptr, nullptr);
      fc_primc->set_activation_type(add_primc->get_activation_type());
    }
  }
  (void)manager->Replace(node, fc_cnode);
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
