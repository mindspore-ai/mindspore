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

#include "src/train/optimizer/fusion/reshape_gather_reshape_fusion_pass.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "schema/inner/model_generated.h"
#include "tools/common/meta_graph_utils.h"
#include "src/train/optimizer/common/fusion_utils.h"

namespace {
constexpr std::string_view Reshape1Name = "RESHAPE1";
constexpr std::string_view Reshape2Name = "RESHAPE2";
constexpr std::string_view GatherName = "GATHER";
}  // namespace
namespace mindspore {
namespace lite {
/*
 * The subgraph such as the following.
 *            any
 *           /  |
 *     reshape  |
 *           \  |
 *           gather
 *           /  |
 *     reshape  |
 *           \  |
 *            any
 */
STATUS ReshapeGatherReshapeFusionPass::DefinePattern() {
  auto reshape_op1 = std::make_shared<PatternOp>();
  MS_CHECK_TRUE_RET(reshape_op1 != nullptr, RET_NULL_PTR);
  reshape_op1->id = Reshape1Name;
  reshape_op1->types = {schema::PrimitiveType_Reshape};

  auto gather_op = std::make_shared<PatternOp>();
  MS_CHECK_TRUE_RET(gather_op != nullptr, RET_NULL_PTR);
  gather_op->id = GatherName;
  gather_op->types = {schema::PrimitiveType_Gather};
  gather_op->left = reshape_op1;

  auto reshape_op2 = std::make_shared<PatternOp>();
  MS_CHECK_TRUE_RET(reshape_op2 != nullptr, RET_NULL_PTR);
  reshape_op2->id = Reshape2Name;
  reshape_op2->types = {schema::PrimitiveType_Reshape};
  reshape_op2->left = gather_op;

  auto fusion_pattern = std::make_unique<FusionPattern>("ReshapeGatherReshapeFusion");
  MS_CHECK_TRUE_MSG(fusion_pattern != nullptr, RET_NULL_PTR, "new fusion_pattern failed");
  fusion_pattern->AddPatternOp(reshape_op1);
  fusion_pattern->AddPatternOp(gather_op);
  fusion_pattern->AddPatternOp(reshape_op2);
  fusion_pattern->Finish();
  this->patterns.emplace_back(fusion_pattern.release());
  return RET_OK;
}

STATUS ReshapeGatherReshapeFusionPass::DoFusion(
  MetaGraphT *graph, const std::string &pattern_name,
  const std::unordered_map<std::string, std::shared_ptr<Path>> &matched_path) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_NULL_PTR);
  MS_CHECK_TRUE_MSG(matched_path.size() == opt::kMatchPathLenThree, RET_PARAM_INVALID,
                    "Reshape-Gather-Reshape-Fusion should have three NodeIndex in matchedPair");
  size_t reshape1_index = 0;
  STATUS ret = opt::GetMatchNodeIndex(graph, matched_path, std::string(Reshape1Name), &reshape1_index);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "cannot get reshape1_index");
  auto &reshape1_node = graph->nodes.at(reshape1_index);
  MS_CHECK_TRUE_MSG(reshape1_node != nullptr, RET_NULL_PTR, "reshape1_node is nullptr");
  size_t gather_index = 0;
  ret = opt::GetMatchNodeIndex(graph, matched_path, std::string(GatherName), &gather_index);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "cannot get gather_index");
  auto &gather_node = graph->nodes.at(gather_index);
  MS_CHECK_TRUE_MSG(gather_node != nullptr, RET_NULL_PTR, "gather_node is nullptr");
  size_t reshape2_index = 0;
  ret = opt::GetMatchNodeIndex(graph, matched_path, std::string(Reshape2Name), &reshape2_index);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "cannot get reshape2_index");
  auto &reshape2_node = graph->nodes.at(reshape2_index);
  MS_CHECK_TRUE_MSG(reshape2_node != nullptr, RET_NULL_PTR, "reshape2_node is nullptr");
  if (reshape1_node->inputIndex.size() != opt::kInputSizeTwo ||
      reshape1_node->quantType == schema::QuantType_QUANT_ALL ||
      reshape1_node->quantType == schema::QuantType_QUANT_DYNAMIC ||
      reshape2_node->inputIndex.size() != opt::kInputSizeTwo ||
      reshape2_node->quantType == schema::QuantType_QUANT_ALL ||
      reshape2_node->quantType == schema::QuantType_QUANT_DYNAMIC ||
      gather_node->quantType == schema::QuantType_QUANT_ALL ||
      gather_node->quantType == schema::QuantType_QUANT_DYNAMIC) {
    MS_LOG(ERROR) << "reshape_node cannot fusion";
    return RET_NO_CHANGE;
  }
  if (opt::IsMultiOutputNode(graph, reshape1_node->outputIndex.at(0)) ||
      opt::IsMultiOutputNode(graph, gather_node->outputIndex.at(0))) {
    MS_LOG(ERROR) << "reshape node or gather node is multi-output node, cannot fusion";
    return RET_NO_CHANGE;
  }
  auto old_shape = graph->allTensors.at(reshape2_node->outputIndex.at(opt::kOutputIndexZero))->dims;
  auto gather_shape0 = graph->allTensors.at(gather_node->inputIndex.at(opt::kInputIndexZero))->dims;
  auto gather_shape1 = graph->allTensors.at(reshape1_node->inputIndex.at(opt::kInputIndexZero))->dims;
  if (old_shape.empty() || gather_shape0.empty() || gather_shape1.empty()) {
    return RET_NO_CHANGE;
  }
  int gather_axis;
  auto data = graph->allTensors.at(gather_node->inputIndex.at(opt::kInputIndexTwo))->data;
  if (data.empty()) {
    gather_axis = 0;
  } else {
    memcpy(&gather_axis, &data[0], data.size());
  }
  if (gather_axis < 0) {
    gather_axis += gather_shape1.size();
  }
  gather_shape0.erase(gather_shape0.begin() + gather_axis);
  (void)gather_shape0.insert(gather_shape0.begin() + gather_axis, gather_shape1.begin(), gather_shape1.end());
  if (gather_shape0 != old_shape) {
    return RET_NO_CHANGE;
  }
  gather_node->inputIndex.at(opt::kInputIndexOne) = reshape1_node->inputIndex.at(opt::kInputIndexZero);
  gather_node->outputIndex.at(opt::kOutputIndexZero) = reshape2_node->outputIndex.at(opt::kOutputIndexZero);
  // cannot delete node here, otherwise will destroy order in other pattern's node index
  // make it an isolated node to be removed in IsolatedNodeRemovePass
  reshape1_node->inputIndex.clear();
  reshape1_node->outputIndex.clear();
  reshape2_node->inputIndex.clear();
  reshape2_node->outputIndex.clear();
  return RET_OK;
}

ReshapeGatherReshapeFusionPass::~ReshapeGatherReshapeFusionPass() = default;
}  // namespace lite
}  // namespace mindspore
