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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/cast_fusion.h"
#include <unordered_map>
#include <memory>
#include <vector>
#include "tools/converter/quantizer/quant_param_holder.h"
#include "mindspore/core/ops/transpose.h"
#include "tools/optimizer/common/format_utils.h"
#include "ops/fusion/scale_fusion.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "tools/lite_exporter/fetch_content.h"

namespace mindspore::opt {
namespace {
STATUS GetCastPerm(const CNodePtr &cnode, int *perm) {
  MS_CHECK_TRUE_RET(cnode != nullptr, lite::RET_NULL_PTR);
  MS_CHECK_TRUE_RET(perm != nullptr, lite::RET_NULL_PTR);
  if (cnode->size() != kInputSizeThree) {
    MS_LOG(ERROR) << "transpose op input size must be three.";
    return lite::RET_ERROR;
  }
  if (utils::isa<CNodePtr>(cnode->input(kInputIndexTwo))) {
    return lite::RET_OK;
  }
  lite::DataInfo data_info;
  int status;
  if (utils::isa<ParameterPtr>(cnode->input(kInputIndexTwo))) {
    status = lite::FetchDataFromParameterNode(cnode, kInputIndexTwo, converter::kFmkTypeMs, &data_info, true);
  } else {
    status = lite::FetchDataFromValueNode(cnode, kInputIndexTwo, converter::kFmkTypeMs, false, &data_info, true);
  }
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "fetch transpose perm data failed.";
    return lite::RET_ERROR;
  }
  if (data_info.data_type_ != kNumberTypeInt && data_info.data_type_ != kNumberTypeInt32) {
    MS_LOG(ERROR) << "transpose perm data is invalid.";
    return lite::RET_ERROR;
  }
  if (data_info.data_.size() < sizeof(int32_t)) {
    MS_LOG(ERROR) << "Data and datatype of data-info not match.";
    return false;
  }
  *perm = reinterpret_cast<int *>(data_info.data_.data())[0];
  return lite::RET_OK;
}
}  // namespace

VectorRef CastFusionPass::DefineCastCastPattern() const {
  auto is_cast1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast1 != nullptr, {});
  auto is_cast2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast2 != nullptr, {});
  auto is_weight_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_weight_param != nullptr, {});
  VectorRef cast_cast_ref = VectorRef({is_cast2, is_cast1, is_weight_param});
  return cast_cast_ref;
}

std::unordered_map<std::string, VectorRef> CastFusionPass::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns["CastCastPatternName"] = DefineCastCastPattern();
  return patterns;
}

AnfNodePtr CastFusionPass::CastCastFusion(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node) const {
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto trans_cnode_2 = node->cast<CNodePtr>();
  MS_ASSERT(trans_cnode_2 != nullptr);
  if (IsMarkedTrainOp(trans_cnode_2)) {
    return nullptr;
  }
  MS_CHECK_TRUE_RET(trans_cnode_2 != nullptr, nullptr);
  MS_CHECK_TRUE_RET(trans_cnode_2->size() == kInputSizeThree, nullptr);
  if (!CheckPrimitiveType(trans_cnode_2, prim::kPrimCast) ||
      !CheckPrimitiveType(trans_cnode_2->input(1), prim::kPrimCast)) {
    return nullptr;
  }
  int post_cast_type;
  if (GetCastPerm(trans_cnode_2, &post_cast_type) != lite::RET_OK) {
    MS_LOG(ERROR) << "get transpose perm failed.";
    return nullptr;
  }
  int pre_cast_type;
  auto pre_node = trans_cnode_2->input(1);
  MS_CHECK_TRUE_RET(pre_node != nullptr, nullptr);
  auto pre_cnode = pre_node->cast<CNodePtr>();
  if (pre_cnode == nullptr) {
    return nullptr;
  }
  if (IsMarkedTrainOp(pre_cnode)) {
    return nullptr;
  }
  if (GetCastPerm(pre_cnode, &pre_cast_type) != lite::RET_OK) {
    MS_LOG(ERROR) << "get transpose perm failed.";
    return nullptr;
  }
  auto pre_node_input = pre_cnode->input(1);
  MS_CHECK_TRUE_RET(pre_node_input != nullptr, nullptr);
  TypeId input_data_type;
  if (GetDataTypeFromAnfNode(pre_node_input, &input_data_type) != RET_OK) {
    MS_LOG(ERROR) << "get input node data type failed." << pre_node_input->fullname_with_scope();
    return nullptr;
  }

  if (static_cast<int>(input_data_type) == post_cast_type) {
    return pre_cnode->input(1);
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  manager->SetEdge(trans_cnode_2, 1, pre_cnode->input(1));
  return nullptr;
}

AnfNodePtr CastFusionPass::Process(const std::string &pattern_name, const mindspore::FuncGraphPtr &func_graph,
                                   const mindspore::AnfNodePtr &node, const mindspore::EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  if (pattern_name == "CastCastPatternName") {
    return CastCastFusion(func_graph, node);
  }
  return nullptr;
}
}  // namespace mindspore::opt
