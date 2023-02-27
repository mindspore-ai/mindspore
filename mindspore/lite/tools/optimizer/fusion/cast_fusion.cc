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
#include <set>
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/optimizer/common/format_utils.h"
#include "nnacl/op_base.h"
#include "tools/lite_exporter/fetch_content.h"

namespace mindspore::opt {
namespace {
bool IsGoodCastSplitFusion(const FuncGraphPtr &func_graph, const CNodePtr &split_cnode_2) {
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  MS_ASSERT(split_cnode_2 != nullptr);
  auto node_users = manager->node_users();
  auto split_node_users = node_users[split_cnode_2];
  for (auto &node_user : split_node_users) {
    auto post_node = node_user.first;
    if (opt::CheckPrimitiveType(post_node, prim::kPrimTupleGetItem)) {
      auto post_item_nodes = node_users[post_node];
      for (auto &post_item : post_item_nodes) {
        auto post_item_node = post_item.first;
        if (opt::CheckPrimitiveType(post_item_node, prim::kPrimGather) && post_item.second == kInputIndexTwo) {
          continue;
        }
        if (opt::CheckPrimitiveType(post_item_node, prim::kPrimCast)) {
          int post_item_cast_type;
          if (GetCastDstDataType(post_item_node->cast<CNodePtr>(), &post_item_cast_type) != lite::RET_OK) {
            MS_LOG(ERROR) << "get cast dst type failed.";
            return false;
          }
          if (post_item_cast_type == kNumberTypeInt32) {
            continue;
          }
        }
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}
bool IsAnyNode(const BaseRef &n) { return true; }
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

VectorRef CastFusionPass::DefineCastGatherPattern() const {
  auto is_any0 = std::make_shared<CondVar>(IsAnyNode);
  MS_CHECK_TRUE_RET(is_any0 != nullptr, {});
  auto is_cast1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast1 != nullptr, {});
  auto is_gather2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimGather>);
  MS_CHECK_TRUE_RET(is_gather2 != nullptr, {});
  auto is_weight_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_weight_param != nullptr, {});
  VectorRef cast_cast_ref = VectorRef({is_gather2, is_any0, is_cast1, is_weight_param});
  return cast_cast_ref;
}

VectorRef CastFusionPass::DefineCastEqualPattern() const {
  auto is_cast1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast1 != nullptr, {});
  auto is_notequal2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimNotEqual>);
  MS_CHECK_TRUE_RET(is_notequal2 != nullptr, {});
  auto is_weight_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_weight_param != nullptr, {});
  VectorRef cast_cast_ref = VectorRef({is_notequal2, is_cast1, is_weight_param});
  return cast_cast_ref;
}

VectorRef CastFusionPass::DefineCastEqual2Pattern() const {
  auto is_cast1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast1 != nullptr, {});
  auto is_notequal2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimNotEqual>);
  MS_CHECK_TRUE_RET(is_notequal2 != nullptr, {});
  auto is_weight_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_weight_param != nullptr, {});
  VectorRef cast_cast_ref = VectorRef({is_notequal2, is_weight_param, is_cast1});
  return cast_cast_ref;
}

VectorRef CastFusionPass::DefineCastSplitPattern() const {
  auto is_cast1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast1 != nullptr, {});
  auto is_split2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSplit>);
  MS_CHECK_TRUE_RET(is_split2 != nullptr, {});
  VectorRef cast_cast_ref = VectorRef({is_split2, is_cast1});
  return cast_cast_ref;
}

std::unordered_map<std::string, VectorRef> CastFusionPass::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns["CastCastPatternName"] = DefineCastCastPattern();
  patterns["CastGatherPatternName"] = DefineCastGatherPattern();
  patterns["CastNotEqualPatternName"] = DefineCastEqualPattern();
  patterns["CastNotEqual2PatternName"] = DefineCastEqual2Pattern();
  patterns["CastSplitPatternName"] = DefineCastSplitPattern();
  return patterns;
}

AnfNodePtr CastFusionPass::CastCastFusion(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node) const {
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto cast_cnode_2 = node->cast<CNodePtr>();
  MS_ASSERT(cast_cnode_2 != nullptr);
  if (IsMarkedTrainOp(cast_cnode_2)) {
    return nullptr;
  }
  MS_CHECK_TRUE_RET(cast_cnode_2 != nullptr, nullptr);
  MS_CHECK_TRUE_RET(cast_cnode_2->size() == kInputSizeThree, nullptr);
  if (!CheckPrimitiveType(cast_cnode_2, prim::kPrimCast) ||
      !CheckPrimitiveType(cast_cnode_2->input(1), prim::kPrimCast)) {
    return nullptr;
  }
  int post_cast_type;
  if (GetCastDstDataType(cast_cnode_2, &post_cast_type) != lite::RET_OK) {
    MS_LOG(ERROR) << "get cast dst type failed.";
    return nullptr;
  }
  int pre_cast_type;
  auto pre_node = cast_cnode_2->input(1);
  MS_CHECK_TRUE_RET(pre_node != nullptr, nullptr);
  auto pre_cnode = pre_node->cast<CNodePtr>();
  if (pre_cnode == nullptr) {
    return nullptr;
  }
  if (IsMarkedTrainOp(pre_cnode)) {
    return nullptr;
  }
  if (GetCastDstDataType(pre_cnode, &pre_cast_type) != lite::RET_OK) {
    MS_LOG(ERROR) << "get cast dst type failed.";
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
  manager->SetEdge(cast_cnode_2, 1, pre_cnode->input(1));
  return nullptr;
}

AnfNodePtr CastFusionPass::CastGatherFusion(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node) const {
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto gather_cnode_2 = node->cast<CNodePtr>();
  MS_ASSERT(gather_cnode_2 != nullptr);
  if (IsMarkedTrainOp(gather_cnode_2)) {
    return nullptr;
  }
  MS_CHECK_TRUE_RET(gather_cnode_2 != nullptr, nullptr);
  MS_CHECK_TRUE_RET(gather_cnode_2->size() == kInputSizeFour, nullptr);
  if (!CheckPrimitiveType(gather_cnode_2, prim::kPrimGather) ||
      !CheckPrimitiveType(gather_cnode_2->input(kInputIndexTwo), prim::kPrimCast)) {
    return nullptr;
  }
  auto pre_cast_cnode = gather_cnode_2->input(kInputIndexTwo)->cast<CNodePtr>();
  MS_ASSERT(pre_cast_cnode != nullptr);
  int post_cast_type;
  if (GetCastDstDataType(pre_cast_cnode, &post_cast_type) != lite::RET_OK) {
    MS_LOG(ERROR) << "get cast dst type failed.";
    return nullptr;
  }
  auto pre_node_input = pre_cast_cnode->input(1);
  MS_CHECK_TRUE_RET(pre_node_input != nullptr, nullptr);
  TypeId input_data_type;
  if (GetDataTypeFromAnfNode(pre_node_input, &input_data_type) != RET_OK) {
    MS_LOG(ERROR) << "get input node data type failed." << pre_node_input->fullname_with_scope();
    return nullptr;
  }
  const std::set<TypeId> support_dtype = {kNumberTypeInt64, kNumberTypeInt32, kNumberTypeBool};
  if (support_dtype.find(input_data_type) != support_dtype.end() &&
      support_dtype.find(static_cast<TypeId>(post_cast_type)) != support_dtype.end() &&
      (static_cast<TypeId>(post_cast_type) != kNumberTypeBool || input_data_type == kNumberTypeBool)) {
    auto manager = func_graph->manager();
    MS_ASSERT(manager != nullptr);
    manager->SetEdge(gather_cnode_2, kInputIndexTwo, pre_node_input);
    return nullptr;
  }
  return nullptr;
}

AnfNodePtr CastFusionPass::CastSplitFusion(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node) const {
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto split_cnode_2 = node->cast<CNodePtr>();
  MS_ASSERT(split_cnode_2 != nullptr);
  if (IsMarkedTrainOp(split_cnode_2)) {
    return nullptr;
  }
  MS_CHECK_TRUE_RET(split_cnode_2 != nullptr, nullptr);
  MS_CHECK_TRUE_RET(split_cnode_2->size() == kInputSizeTwo, nullptr);
  if (!CheckPrimitiveType(split_cnode_2, prim::kPrimSplit) ||
      !CheckPrimitiveType(split_cnode_2->input(kInputIndexOne), prim::kPrimCast)) {
    return nullptr;
  }
  auto pre_cast_cnode = split_cnode_2->input(kInputIndexOne)->cast<CNodePtr>();
  MS_ASSERT(pre_cast_cnode != nullptr);
  int post_cast_type;
  if (GetCastDstDataType(pre_cast_cnode, &post_cast_type) != lite::RET_OK) {
    MS_LOG(ERROR) << "get cast dst type failed.";
    return nullptr;
  }
  auto pre_node_input = pre_cast_cnode->input(kInputIndexOne);
  MS_CHECK_TRUE_RET(pre_node_input != nullptr, nullptr);
  TypeId input_data_type;
  if (GetDataTypeFromAnfNode(pre_node_input, &input_data_type) != RET_OK) {
    MS_LOG(ERROR) << "get input node data type failed." << pre_node_input->fullname_with_scope();
    return nullptr;
  }
  if (input_data_type == kNumberTypeInt32 && post_cast_type == kNumberTypeInt64) {
    if (IsGoodCastSplitFusion(func_graph, split_cnode_2)) {
      auto manager = func_graph->manager();
      MS_ASSERT(manager != nullptr);
      manager->SetEdge(split_cnode_2, kInputIndexOne, pre_node_input);
    }
  }
  return nullptr;
}

AnfNodePtr CastFusionPass::CastNotEqualFusion(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node) const {
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto not_equal_cnode_2 = node->cast<CNodePtr>();
  MS_ASSERT(not_equal_cnode_2 != nullptr);
  if (IsMarkedTrainOp(not_equal_cnode_2)) {
    return nullptr;
  }
  MS_CHECK_TRUE_RET(not_equal_cnode_2 != nullptr, nullptr);
  MS_CHECK_TRUE_RET(not_equal_cnode_2->size() == kInputSizeThree, nullptr);
  if (!CheckPrimitiveType(not_equal_cnode_2, prim::kPrimNotEqual)) {
    return nullptr;
  }
  CNodePtr pre_cast_cnode;
  ParameterPtr param_node;
  lite::DataInfo data_info;
  int status;
  int cast_index;
  if (CheckPrimitiveType(not_equal_cnode_2->input(kInputIndexOne), prim::kPrimCast)) {
    cast_index = kInputIndexOne;
    pre_cast_cnode = not_equal_cnode_2->input(kInputIndexOne)->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(IsParamNode(not_equal_cnode_2->input(kInputIndexTwo)), nullptr);
    param_node = not_equal_cnode_2->input(kInputIndexTwo)->cast<ParameterPtr>();
    status =
      lite::FetchDataFromParameterNode(not_equal_cnode_2, kInputIndexTwo, converter::kFmkTypeMs, &data_info, true);
  } else if (CheckPrimitiveType(not_equal_cnode_2->input(kInputIndexTwo), prim::kPrimCast)) {
    cast_index = kInputIndexTwo;
    pre_cast_cnode = not_equal_cnode_2->input(kInputIndexTwo)->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(IsParamNode(not_equal_cnode_2->input(kInputIndexOne)), nullptr);
    param_node = not_equal_cnode_2->input(kInputIndexOne)->cast<ParameterPtr>();
    status =
      lite::FetchDataFromParameterNode(not_equal_cnode_2, kInputIndexOne, converter::kFmkTypeMs, &data_info, true);
  } else {
    return nullptr;
  }
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "fetch transpose perm data failed.";
    return nullptr;
  }
  int post_cast_type;
  if (GetCastDstDataType(pre_cast_cnode, &post_cast_type) != lite::RET_OK) {
    MS_LOG(ERROR) << "get cast dst type failed.";
    return nullptr;
  }
  if (post_cast_type != data_info.data_type_) {
    return nullptr;
  }
  if (data_info.data_type_ == kNumberTypeInt64) {
    if (data_info.data_.size() < sizeof(int32_t)) {
      MS_LOG(ERROR) << "Data and datatype of data-info not match.";
      return nullptr;
    }
    auto p_data = reinterpret_cast<int64_t *>(data_info.data_.data());
    for (size_t i = 0; i < (data_info.data_.size() / sizeof(int64_t)); i++) {
      if ((p_data[i] > INT32_MAX) || (p_data[i] < INT_MIN)) {
        return nullptr;
      }
    }
    auto abstract = param_node->abstract();
    MS_CHECK_TRUE_RET(abstract != nullptr, nullptr);
    auto new_abstract = abstract->Clone();
    new_abstract->set_value(std::make_shared<AnyValue>());
    if (GenCastNode(func_graph, param_node, param_node->fullname_with_scope() + "_post_cast",
                    static_cast<TypeId>(kNumberTypeInt32), new_abstract) == nullptr) {
      MS_LOG(ERROR) << "GenCastNode failed.";
      return nullptr;
    }
    auto manager = func_graph->manager();
    MS_ASSERT(manager != nullptr);
    manager->SetEdge(not_equal_cnode_2, cast_index, pre_cast_cnode->input(kInputIndexOne));
  }

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
  } else if (pattern_name == "CastGatherPatternName") {
    return CastGatherFusion(func_graph, node);
  } else if (pattern_name == "CastSplitPatternName") {
    return CastSplitFusion(func_graph, node);
  } else if (pattern_name == "CastNotEqualPatternName") {
    return CastNotEqualFusion(func_graph, node);
  } else if (pattern_name == "CastNotEqual2PatternName") {
    return CastNotEqualFusion(func_graph, node);
  }
  return nullptr;
}
}  // namespace mindspore::opt
