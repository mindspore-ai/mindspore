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

#include <set>
#include "utils/hash_map.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace {
enum class RemoveNodeType { kOtherNode = 0, kOptimizerNode };
const char kLessBatchNormalizationPassName[] = "less_bn";
constexpr auto kValidResidualStructureIndex = 1;
constexpr auto kBNParametersStartIndex = 2;
// Pattern 1
// Add -> BatchNorm -> Conv2D -> Relu ... -> End
//     BatchNorm -> Conv2D -> -> -> ->
constexpr auto kFirstBranchPattern1 = 12;
constexpr auto kSecondBranchPattern1 = 3;
constexpr auto kFirstBranchStartIndexPattern1 = 4;
constexpr auto kFirstBranchEndIndexPattern1 = 11;
constexpr auto kSecondBranchStartIndexPattern1 = kFirstBranchPattern1;
constexpr auto kSecondBranchEndIndexPattern1 = 2 + kFirstBranchPattern1;
const std::vector<kStructureTuple> ResidualStructureBasePattern{
  {kFirstBranchPattern1,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu},
   {kFirstBranchStartIndexPattern1, kFirstBranchEndIndexPattern1}},
  {kSecondBranchPattern1,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D},
   {kSecondBranchStartIndexPattern1, kSecondBranchEndIndexPattern1}}};
// Pattern 2
// Add -> BatchNorm -> Conv2D -> Relu ... -> End
//       -> ->     ...   ... ...    -> ->
constexpr auto kFirstBranchPattern2 = 12;
constexpr auto kSecondBranchPattern2 = 1;
constexpr auto kFirstBranchStartIndexPattern2 = 4;
constexpr auto kFirstBranchEndIndexPattern2 = 11;
constexpr auto kSecondBranchStartIndexPattern2 = kFirstBranchPattern2;
constexpr auto kSecondBranchEndIndexPattern2 = 1 + kSecondBranchPattern2;
const std::vector<kStructureTuple> ResidualStructureShortCutPattern{
  {kFirstBranchPattern2,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu},
   {kFirstBranchStartIndexPattern2, kFirstBranchEndIndexPattern2}},
  {kSecondBranchPattern2, {prim::kPrimRelu}, {kSecondBranchStartIndexPattern2, kSecondBranchEndIndexPattern2}}};
// Pattern 3
// Add -> BatchNorm -> Conv2D -> Relu ... BatchNorm -> Conv2D -> End
//       BatchNorm -> Conv2D -> ->   ...   ...   ...   -> ->
constexpr auto kFirstBranchPattern3 = 11;
constexpr auto kSecondBranchPattern3 = 3;
constexpr auto kFirstBranchStartIndexPattern3 = 4;
constexpr auto kFirstBranchEndIndexPattern3 = 10;
constexpr auto kSecondBranchStartIndexPattern3 = kFirstBranchPattern3;
constexpr auto kSecondBranchEndIndexPattern3 = 2 + kFirstBranchPattern3;
const std::vector<kStructureTuple> ResidualStructureFirstStepPattern{
  {kFirstBranchPattern3,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu, prim::kPrimTupleGetItem,
    prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu, prim::kPrimTupleGetItem, prim::kPrimBatchNorm,
    prim::kPrimConv2D},
   {kFirstBranchStartIndexPattern3, kFirstBranchEndIndexPattern3}},
  {kSecondBranchPattern3,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D},
   {kSecondBranchStartIndexPattern3, kSecondBranchEndIndexPattern3}}};
// Pattern 4
constexpr auto kFirstBranchPattern4 = 8;
constexpr auto kSecondBranchPattern4 = 3;
constexpr auto kFirstBranchStartIndexPattern4 = 4;
constexpr auto kFirstBranchEndIndexPattern4 = 6;
constexpr auto kSecondBranchStartIndexPattern4 = kFirstBranchPattern4;
constexpr auto kSecondBranchEndIndexPattern4 = 3 + kFirstBranchPattern4;
const std::vector<kStructureTuple> BasicStructBasePattern{
  {kFirstBranchPattern4,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu},
   {kFirstBranchStartIndexPattern4, kFirstBranchEndIndexPattern4}},
  {kSecondBranchPattern4,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D},
   {kSecondBranchStartIndexPattern4, kSecondBranchEndIndexPattern4}}};
// Pattern 5
constexpr auto kFirstBranchPattern5 = 7;
constexpr auto kSecondBranchPattern5 = 1;
constexpr auto kFirstBranchStartIndexPattern5 = 4;
constexpr auto kFirstBranchEndIndexPattern5 = 6;
constexpr auto kSecondBranchStartIndexPattern5 = kFirstBranchPattern5;
constexpr auto kSecondBranchEndIndexPattern5 = 3 + kFirstBranchPattern5;
const std::vector<kStructureTuple> BasicStructFirstStepPattern{
  {kFirstBranchPattern5,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu, prim::kPrimTupleGetItem,
    prim::kPrimBatchNorm, prim::kPrimConv2D},
   {kFirstBranchStartIndexPattern5, kFirstBranchEndIndexPattern5}},
  {kSecondBranchPattern5, {prim::kPrimMaxPool}, {kSecondBranchStartIndexPattern5, kSecondBranchEndIndexPattern5}}};
// Pattern 6
constexpr auto kFirstBranchPattern6 = 8;
constexpr auto kSecondBranchPattern6 = 1;
constexpr auto kFirstBranchStartIndexPattern6 = 4;
constexpr auto kFirstBranchEndIndexPattern6 = 6;
constexpr auto kSecondBranchStartIndexPattern6 = kFirstBranchPattern6;
constexpr auto kSecondBranchEndIndexPattern6 = 3 + kFirstBranchPattern6;
const std::vector<kStructureTuple> BasicStructShortCutPattern{
  {kFirstBranchPattern6,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu},
   {kFirstBranchStartIndexPattern6, kFirstBranchEndIndexPattern6}},
  {kSecondBranchPattern6, {prim::kPrimRelu}, {kSecondBranchStartIndexPattern6, kSecondBranchEndIndexPattern6}}};
// Pattern 7
constexpr auto kFirstBranchPattern7 = 1;
constexpr auto kSecondBranchPattern7 = 13;
constexpr auto kFirstBranchStartIndexPattern7 = SIZE_MAX;
constexpr auto kFirstBranchEndIndexPattern7 = SIZE_MAX;
constexpr auto kSecondBranchStartIndexPattern7 = 7;
constexpr auto kSecondBranchEndIndexPattern7 = 10;
const std::vector<kStructureTuple> InvertedResidualShortCutPattern{
  {kFirstBranchPattern7,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm},
   {kFirstBranchStartIndexPattern7, kFirstBranchEndIndexPattern7}},
  {kSecondBranchPattern7,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu6, prim::kPrimTupleGetItem,
    prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu6, prim::kPrimTupleGetItem, prim::kPrimBatchNorm,
    prim::kPrimConv2D, prim::kPrimTupleGetItem, prim::kPrimBatchNorm},
   {kSecondBranchStartIndexPattern7, kSecondBranchEndIndexPattern7}}};
// Pattern 8
constexpr auto kFirstBranchPattern8 = 4;
constexpr auto kFirstBranchStartIndexPattern8 = 0;
constexpr auto kFirstBranchEndIndexPattern8 = 3;
const std::vector<kStructureTuple> InvertedResidualPattern{
  {kFirstBranchPattern8,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimAdd},
   {kFirstBranchStartIndexPattern8, kFirstBranchEndIndexPattern8}}};
// Pattern 9
constexpr auto kFirstBranchPattern9 = 1;
constexpr auto kSecondBranchPattern9 = 12;
constexpr auto kFirstBranchStartIndexPattern9 = SIZE_MAX;
constexpr auto kFirstBranchEndIndexPattern9 = SIZE_MAX;
constexpr auto kSecondBranchStartIndexPattern9 = 7;
constexpr auto kSecondBranchEndIndexPattern9 = 10;
const std::vector<kStructureTuple> InvertedResidualShortCutPattern2{
  {kFirstBranchPattern9, {prim::kPrimAdd}, {kFirstBranchStartIndexPattern9, kFirstBranchEndIndexPattern9}},
  {kSecondBranchPattern9,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu6, prim::kPrimTupleGetItem,
    prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu6, prim::kPrimTupleGetItem, prim::kPrimBatchNorm,
    prim::kPrimConv2D, prim::kPrimAdd},
   {kSecondBranchStartIndexPattern9, kSecondBranchEndIndexPattern9}}};
// Pattern 10
constexpr auto kFirstBranchPattern10 = 5;
constexpr auto kFirstBranchStartIndexPattern10 = 0;
constexpr auto kFirstBranchEndIndexPattern10 = 4;
const std::vector<kStructureTuple> InvertedResidualPattern2{
  {kFirstBranchPattern10,
   {prim::kPrimReduceMean, prim::kPrimRelu6, prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D},
   {kFirstBranchStartIndexPattern10, kFirstBranchEndIndexPattern10}}};
// Pattern 11
constexpr auto kFirstBranchPattern11 = 17;
constexpr auto kFirstBranchStartIndexPattern11 = 3;
constexpr auto kFirstBranchEndIndexPattern11 = 6;
const std::vector<kStructureTuple> InvertedResidualPattern3{
  {kFirstBranchPattern11,
   {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu6, prim::kPrimTupleGetItem,
    prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D,
    prim::kPrimRelu6, prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D, prim::kPrimRelu6,
    prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D},
   {kFirstBranchStartIndexPattern11, kFirstBranchEndIndexPattern11}}};
// Pattern 12
constexpr auto kFirstBranchPattern12 = 1;
constexpr auto kSecondBranchPattern12 = 9;
constexpr auto kFirstBranchStartIndexPattern12 = SIZE_MAX;
constexpr auto kFirstBranchEndIndexPattern12 = SIZE_MAX;
constexpr auto kSecondBranchStartIndexPattern12 = kFirstBranchPattern12 + 5;
constexpr auto kSecondBranchEndIndexPattern12 = kFirstBranchPattern12 + 8;
const std::vector<kStructureTuple> DenseBlockShortCutPattern{
  {kFirstBranchPattern12, {prim::kPrimConcat}, {kFirstBranchStartIndexPattern12, kFirstBranchEndIndexPattern12}},
  {kSecondBranchPattern12,
   {prim::kPrimConv2D, prim::kPrimRelu, prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D,
    prim::kPrimRelu, prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConcat},
   {kSecondBranchStartIndexPattern12, kSecondBranchEndIndexPattern12}}};
// Pattern 13
constexpr auto kFirstBranchPattern13 = 5;
constexpr auto kFirstBranchStartIndexPattern13 = 0;
constexpr auto kFirstBranchEndIndexPattern13 = 4;
const std::vector<kStructureTuple> DenseBlockPattern{
  {kFirstBranchPattern13,
   {prim::kPrimConv2D, prim::kPrimRelu, prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConcat},
   {kFirstBranchStartIndexPattern13, kFirstBranchEndIndexPattern13}}};
// Pattern 14
constexpr auto kFirstBranchPattern14 = 9;
constexpr auto kSecondBranchPattern14 = 1;
constexpr auto kFirstBranchStartIndexPattern14 = 5;
constexpr auto kFirstBranchEndIndexPattern14 = 8;
constexpr auto kSecondBranchStartIndexPattern14 = SIZE_MAX;
constexpr auto kSecondBranchEndIndexPattern14 = SIZE_MAX;
const std::vector<kStructureTuple> DenseBlockShortCutPattern2{
  {kFirstBranchPattern14,
   {prim::kPrimConv2D, prim::kPrimRelu, prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D,
    prim::kPrimRelu, prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConcat},
   {kFirstBranchStartIndexPattern14, kFirstBranchEndIndexPattern14}},
  {kSecondBranchPattern14, {prim::kPrimConcat}, {kSecondBranchStartIndexPattern14, kSecondBranchEndIndexPattern14}}};
// Pattern 15
constexpr auto kFirstBranchPattern15 = 9;
constexpr auto kSecondBranchPattern15 = 1;
constexpr auto kFirstBranchStartIndexPattern15 = 0;
constexpr auto kFirstBranchEndIndexPattern15 = 4;
constexpr auto kSecondBranchStartIndexPattern15 = SIZE_MAX;
constexpr auto kSecondBranchEndIndexPattern15 = SIZE_MAX;
const std::vector<kStructureTuple> DenseBlockPoolPattern{
  {kFirstBranchPattern15,
   {prim::kPrimConv2D, prim::kPrimRelu, prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D,
    prim::kPrimRelu, prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimMaxPool},
   {kFirstBranchStartIndexPattern15, kFirstBranchEndIndexPattern15}},
  {kSecondBranchPattern15, {prim::kPrimConcat}, {kSecondBranchStartIndexPattern15, kSecondBranchEndIndexPattern15}}};
// Pattern 16
constexpr auto kFirstBranchPattern16 = 1;
constexpr auto kSecondBranchPattern16 = 9;
constexpr auto kFirstBranchStartIndexPattern16 = SIZE_MAX;
constexpr auto kFirstBranchEndIndexPattern16 = SIZE_MAX;
constexpr auto kSecondBranchStartIndexPattern16 = kFirstBranchPattern16;
constexpr auto kSecondBranchEndIndexPattern16 = kFirstBranchPattern16 + 4;
const std::vector<kStructureTuple> DenseBlockPoolPatter2{
  {kFirstBranchPattern16, {prim::kPrimConcat}, {kFirstBranchStartIndexPattern16, kFirstBranchEndIndexPattern16}},
  {kSecondBranchPattern16,
   {prim::kPrimConv2D, prim::kPrimRelu, prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimConv2D,
    prim::kPrimRelu, prim::kPrimTupleGetItem, prim::kPrimBatchNorm, prim::kPrimMaxPool},
   {kSecondBranchStartIndexPattern16, kSecondBranchEndIndexPattern16}}};
static const std::vector<std::vector<kStructureTuple>> kNeedMatchPattern = {ResidualStructureBasePattern,
                                                                            ResidualStructureShortCutPattern,
                                                                            ResidualStructureFirstStepPattern,
                                                                            BasicStructBasePattern,
                                                                            BasicStructFirstStepPattern,
                                                                            BasicStructShortCutPattern,
                                                                            InvertedResidualShortCutPattern,
                                                                            InvertedResidualPattern,
                                                                            InvertedResidualShortCutPattern2,
                                                                            InvertedResidualPattern2,
                                                                            InvertedResidualPattern3,
                                                                            DenseBlockShortCutPattern,
                                                                            DenseBlockPattern,
                                                                            DenseBlockShortCutPattern2,
                                                                            DenseBlockPoolPattern,
                                                                            DenseBlockPoolPatter2};
const std::set<PrimitivePtr> kNeedRemoveNodeSet{
  prim::kPrimLoad,      prim::kPrimRefToEmbed, prim::kPrimApplyMomentum, prim::kPrimMomentum,
  prim::kPrimApplyFtrl, prim::kPrimSGD,        prim::kPrimApplyRMSProp,  prim::kPrimAdam};
static mindspore::HashMap<RemoveNodeType, mindspore::HashSet<size_t>> kRemoveIndex{
  {RemoveNodeType::kOtherNode, {2}}, {RemoveNodeType::kOptimizerNode, {3, 5, 6}}};

bool NeedRemove(const ParameterPtr &a, const std::vector<AnfNodePtr> &parameter_list) {
  if (a == nullptr) {
    return false;
  }
  return std::any_of(parameter_list.begin(), parameter_list.end(), [&a](const AnfNodePtr &b) {
    return (b->isa<Parameter>() && a->name() == b->cast<ParameterPtr>()->name());
  });
}

bool IsNotRealUseNode(const AnfNodePtr &node) {
  for (const auto &prim : kNeedRemoveNodeSet) {
    if (IsPrimitiveCNode(node, prim)) {
      return true;
    }
  }
  return false;
}

CNodePtr ConvertRemoveNodeToVirtualNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> args;
  size_t index = 0;
  const auto &inputs = cnode->inputs();
  auto remove_index = kRemoveIndex[RemoveNodeType::kOptimizerNode];
  if (IsPrimitiveCNode(cnode, prim::kPrimLoad) || IsPrimitiveCNode(cnode, prim::kPrimRefToEmbed)) {
    remove_index = kRemoveIndex[RemoveNodeType::kOtherNode];
  }

  (void)std::copy_if(
    inputs.cbegin(), inputs.cend(), std::back_inserter(args),
    [&remove_index, &index](const AnfNodePtr &) { return remove_index.find(index++) != remove_index.end(); });

  (void)args.insert(args.begin(), NewValueNode(prim::kPrimMakeTuple));
  const auto &fg = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto new_make_tuple = fg->NewCNode(args);
  return new_make_tuple;
}

bool IsRealRemoveParameterNode(const FuncGraphManagerPtr &manager, const AnfNodePtr &parameter) {
  auto param_output = manager->node_users().find(parameter);
  if (param_output == manager->node_users().end()) {
    return true;
  }

  bool need_remove = true;
  auto output_info_list = param_output->second;
  for (const auto &output_info : output_info_list) {
    const auto &node = output_info.first;
    if (IsNotRealUseNode(node)) {
      const auto &cnode = node->cast<CNodePtr>();
      const auto &new_cnode = ConvertRemoveNodeToVirtualNode(cnode);
      (void)manager->Replace(cnode, new_cnode);
      continue;
    }
    need_remove = false;
  }

  return need_remove;
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
  (void)std::copy_if(remove_parameter_list.cbegin(), remove_parameter_list.cend(),
                     std::back_inserter(real_remove_parameter_list),
                     [&manager](const AnfNodePtr &param) { return IsRealRemoveParameterNode(manager, param); });

  auto root_parameters = root_graph->parameters();
  size_t origin_param_count = root_parameters.size();
  (void)root_parameters.erase(std::remove_if(root_parameters.begin(), root_parameters.end(),
                                             [&real_remove_parameter_list](const AnfNodePtr &node) {
                                               return NeedRemove(node->cast<ParameterPtr>(),
                                                                 real_remove_parameter_list);
                                             }),
                              root_parameters.cend());
  size_t remove_param_count = origin_param_count - root_parameters.size();
  size_t fv_param_count = root_graph->fv_param_count();
  if (remove_param_count > fv_param_count) {
    MS_LOG(ERROR) << "The number of deleted parameters cannot exceed the number of original parameters.";
    return;
  }
  fv_param_count = fv_param_count - remove_param_count;
  root_graph->set_fv_param_count(fv_param_count);
  manager->SetParameters(root_graph, root_parameters);
}
}  // namespace

bool LessBatchNormalization::MatchStructureNode(const CNodePtr &cnode, const int32_t index,
                                                const kStructureTuple &patternTuple) const {
  if (index < 0) {
    return false;
  }
  const auto &use_pattern = std::get<1>(patternTuple);
  int32_t use_index = index % static_cast<int32_t>(use_pattern.size());
  if (!IsPrimitiveCNode(cnode, use_pattern[IntToSize(use_index)]) &&
      use_pattern[IntToSize(use_index)] != prim::kPrimTupleGetItem) {
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
  if (!IsPrimitiveCNode(cnode, prim::kPrimBatchNorm) && !IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem) &&
      !IsValueNode<FuncGraph>(cnode->input(0))) {
    return;
  }
  if (match_pattern.empty()) {
    return;
  }
  const auto &start_end_pair = std::get<2>(match_pattern.at(match_branch_));
  if (match_node_ >= start_end_pair.first && match_node_ <= start_end_pair.second) {
    (void)remove_node_list_.insert(cnode);
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
    (void)std::for_each(current_pattern.begin(), current_pattern.end(), [&, this](const kStructureTuple &t) {
      sum_match_node += std::get<0>(t);
      (void)this->total_match_node_.emplace_back(sum_match_node);
    });
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || cnode->inputs().empty()) {
      return nullptr;
    }
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    std::vector<PredicateFuncType> funcs(cnode->inputs().size() - 1, IsCNode);
    AnfVisitor::Match(prim, funcs)(node);
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
      (void)std::copy_if(iter->inputs().begin() + kBNParametersStartIndex, iter->inputs().end(),
                         std::back_inserter(remove_load_list),
                         [](const AnfNodePtr &node) { return IsPrimitiveCNode(node, prim::kPrimLoad); });
      (void)std::transform(
        remove_load_list.begin(), remove_load_list.end(), std::back_inserter(remove_parameter_list),
        [](const AnfNodePtr &node) { return node->cast<CNodePtr>()->input(kValidResidualStructureIndex); });
    }
    // Remove useless node.
    auto input_cnode = iter->input(kValidResidualStructureIndex);
    (void)manager->Replace(iter, input_cnode);
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
  (void)total_match_node_.emplace_back(0);
  match_node_ = 0;
  match_branch_ = 0;
  is_match_ = false;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
