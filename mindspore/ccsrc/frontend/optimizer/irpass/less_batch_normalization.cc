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

#include "frontend/optimizer/irpass/less_batch_normalization.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace {
const char kLessBatchNormalizationPassName[] = "less_bn";
constexpr auto kValidResidualStructureIndex = 1;
constexpr auto kBNParametersStartIndex = 2;
// Pattern 1
// Add -> BatchNorm -> Conv2D -> Relu ... -> End
//     ↘  BatchNorm -> Conv2D -> -> -> -> ↗
constexpr auto kFirstBranchPattern1 = 12;
constexpr auto kSecondBranchPattern1 = 3;
constexpr auto kFirstBranchStartIndexPattern1 = 4;
constexpr auto kFirstBranchEndIndexPattern1 = 11;
const std::vector<kStructureTuple> ResidualStructureBasePattern{
  {kFirstBranchPattern1,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu},
   {kFirstBranchStartIndexPattern1, kFirstBranchEndIndexPattern1}},
  {kSecondBranchPattern1, {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D}, {SIZE_MAX, SIZE_MAX}}};
// Pattern 2
// Add -> BatchNorm -> Conv2D -> Relu ... -> End
//     ↘  -> ->     ...   ... ...    -> -> ↗
constexpr auto kFirstBranchPattern2 = 12;
constexpr auto kSecondBranchPattern2 = 1;
constexpr auto kFirstBranchStartIndexPattern2 = 4;
constexpr auto kFirstBranchEndIndexPattern2 = 11;
const std::vector<kStructureTuple> ResidualStructureShortCutPattern{
  {kFirstBranchPattern2,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu},
   {kFirstBranchStartIndexPattern2, kFirstBranchEndIndexPattern2}},
  {kSecondBranchPattern2, {prim::kPrimRelu}, {SIZE_MAX, SIZE_MAX}}};
// Pattern 3
// Add -> BatchNorm -> Conv2D -> Relu ... BatchNorm -> Conv2D -> End
//     ↘  BatchNorm -> Conv2D -> ->   ...   ...   ...   -> -> ↗
constexpr auto kFirstBranchPattern3 = 11;
constexpr auto kSecondBranchPattern3 = 3;
constexpr auto kFirstBranchStartIndexPattern3 = 4;
constexpr auto kFirstBranchEndIndexPattern3 = 10;
const std::vector<kStructureTuple> ResidualStructureFirstStepPattern{
  {kFirstBranchPattern3,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu, prim::kPrimTupleGetItem,
    prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu, prim::kPrimTupleGetItem, prim::kPrimBatchNorm,
    prim::kPrimConv2D},
   {kFirstBranchStartIndexPattern3, kFirstBranchEndIndexPattern3}},
  {kSecondBranchPattern3, {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D}, {SIZE_MAX, SIZE_MAX}}};
static const std::vector<std::vector<kStructureTuple>> kNeedMatchPattern = {
  ResidualStructureBasePattern, ResidualStructureShortCutPattern, ResidualStructureFirstStepPattern};

bool NeedRemove(const ParameterPtr &a, const std::vector<AnfNodePtr> &parameter_list) {
  if (a == nullptr) {
    return false;
  }
  return std::any_of(parameter_list.begin(), parameter_list.end(), [&a](const AnfNodePtr &b) {
    return (b->isa<Parameter>() && a->name() == b->cast<ParameterPtr>()->name());
  });
}

bool IsRealRemoveParameterNode(const FuncGraphManagerPtr &manager, const AnfNodePtr &parameter) {
  auto param_output = manager->node_users().find(parameter);
  if (param_output == manager->node_users().end()) {
    return true;
  }

  return false;
}

void RemoveBatchNormalizetionNotUseParameters(const FuncGraphManagerPtr &manager,
                                              const std::vector<AnfNodePtr> &remove_parameter_list) {
  auto roots = manager->roots();
  if (roots.size() != 1) {
    MS_LOG(ERROR) << "The size of roots " << roots.size() << " is not valid.";
    return;
  }
  auto root_graph = *(roots.begin());
  MS_EXCEPTION_IF_NULL(root_graph);

  std::vector<AnfNodePtr> real_remove_parameter_list;
  std::copy_if(remove_parameter_list.begin(), remove_parameter_list.end(),
               std::back_inserter(real_remove_parameter_list),
               [&manager](const AnfNodePtr &param) { return IsRealRemoveParameterNode(manager, param); });

  auto root_parameters = root_graph->parameters();
  root_parameters.erase(std::remove_if(root_parameters.begin(), root_parameters.end(),
                                       [&real_remove_parameter_list](const AnfNodePtr &node) {
                                         return NeedRemove(node->cast<ParameterPtr>(), real_remove_parameter_list);
                                       }),
                        root_parameters.end());

  manager->SetParameters(root_graph, root_parameters);
}
}  // namespace

bool LessBatchNormalization::MatchStructureNode(const CNodePtr &cnode, const int32_t index,
                                                const kStructureTuple &patternTuple) {
  if (index < 0) {
    return false;
  }
  const auto &use_pattern = std::get<1>(patternTuple);
  int32_t use_index = index % use_pattern.size();
  if (!IsPrimitiveCNode(cnode, use_pattern[use_index])) {
    return false;
  }
  return true;
}

bool LessBatchNormalization::MatchGraphStructure(const CNodePtr &cnode,
                                                 const std::vector<kStructureTuple> &match_pattern) {
  if ((match_branch_ + 1 >= total_match_node_.size()) || (match_branch_ >= match_pattern.size())) {
    return false;
  }

  int32_t index = static_cast<int32_t>(match_node_) - static_cast<int32_t>(total_match_node_[match_branch_]);
  const auto &pattern = match_pattern[match_branch_];
  if (!MatchStructureNode(cnode, index, pattern)) {
    return false;
  }

  match_node_++;
  if (match_node_ == total_match_node_.back()) {
    is_match_ = true;
    return false;
  }
  if (match_node_ == total_match_node_[match_branch_ + 1]) {
    match_branch_++;
    return false;
  }
  return true;
}

void LessBatchNormalization::IsRemoveNode(const CNodePtr &cnode, const std::vector<kStructureTuple> &match_pattern) {
  if (!IsPrimitiveCNode(cnode, prim::kPrimBatchNorm) && !IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
    return;
  }
  if (match_pattern.empty()) {
    return;
  }
  const auto &start_end_pair = std::get<2>(match_pattern.at(match_branch_));
  if (match_node_ >= start_end_pair.first && match_node_ <= start_end_pair.second) {
    remove_node_list_.insert(cnode);
  }
}

AnfNodePtr LessBatchNormalization::operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
  const auto &fg = node->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  if (!fg->has_attr(kLessBatchNormalizationPassName)) {
    return nullptr;
  }
  match_pattern_ = 0;
  while (match_pattern_ < kNeedMatchPattern.size()) {
    Reset();
    const auto &current_pattern = kNeedMatchPattern.at(match_pattern_);
    size_t sum_match_node = 0;
    std::for_each(current_pattern.begin(), current_pattern.end(), [&](const kStructureTuple &t) {
      sum_match_node += std::get<0>(t);
      total_match_node_.emplace_back(sum_match_node);
    });
    AnfVisitor::Match(prim::kPrimAdd, {IsCNode, IsCNode})(node);
    if (is_match_) {
      break;
    }
    match_pattern_++;
  }

  if (!is_match_ || remove_node_list_.empty()) {
    return nullptr;
  }

  auto manager = optimizer->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<AnfNodePtr> remove_load_list;
  std::vector<AnfNodePtr> remove_parameter_list;
  for (auto &iter : remove_node_list_) {
    // Need to remove batchnorm's parameter input.
    if (IsPrimitiveCNode(iter, prim::kPrimBatchNorm)) {
      std::copy_if(iter->inputs().begin() + kBNParametersStartIndex, iter->inputs().end(),
                   std::back_inserter(remove_load_list),
                   [](const AnfNodePtr &node) { return IsPrimitiveCNode(node, prim::kPrimLoad); });
      std::transform(
        remove_load_list.begin(), remove_load_list.end(), std::back_inserter(remove_parameter_list),
        [](const AnfNodePtr &node) { return node->cast<CNodePtr>()->input(kValidResidualStructureIndex); });
    }
    // Remove useless node.
    auto input_cnode = iter->input(kValidResidualStructureIndex);
    manager->Replace(iter, input_cnode);
  }
  RemoveBatchNormalizetionNotUseParameters(manager, remove_parameter_list);

  return node;
}

void LessBatchNormalization::Visit(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    return;
  }

  const auto &current_pattern = kNeedMatchPattern.at(match_pattern_);
  IsRemoveNode(cnode, current_pattern);
  if (!MatchGraphStructure(cnode, current_pattern)) {
    return;
  }

  auto search_input = cnode->input(kValidResidualStructureIndex);
  if (search_input != nullptr && search_input->isa<CNode>()) {
    this->Visit(search_input->cast<CNodePtr>());
  }
  return;
}

void LessBatchNormalization::Reset() {
  remove_node_list_.clear();
  total_match_node_.clear();
  total_match_node_.emplace_back(0);
  match_node_ = 0;
  match_branch_ = 0;
  is_match_ = false;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
