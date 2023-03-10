/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/mindir/all_to_all_unify_mindir.h"
#include <vector>
#include <string>
#include "utils/trace_base.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/comm_manager.h"
#include "include/backend/optimizer/helper.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kCNodePrimitiveIdx = 0;
constexpr size_t kAllToAllInputIdx = 1;

void ChangePrimitiveToAllToAllV(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto neighbor_exchange = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(neighbor_exchange);

  if (neighbor_exchange->size() == kCNodePrimitiveIdx) {
    MS_LOG(EXCEPTION) << "Inputs should not be empty for cnode " << node->DebugString()
                      << trace::DumpSourceLines(neighbor_exchange);
  }

  auto prim = GetValueNode<PrimitivePtr>(neighbor_exchange->input(kCNodePrimitiveIdx));
  MS_EXCEPTION_IF_NULL(prim);
  prim->Named::operator=(Named(kAllToAllvOpName));
}

uint32_t GetRankSize(const std::string &group) {
  uint32_t rank_size;
  if (!CommManager::GetInstance().GetRankSize(group, &rank_size)) {
    MS_LOG(EXCEPTION) << "Get hccl rank size for group " << group << " failed.";
  }
  return rank_size;
}
}  // namespace

CNodePtr AllToAllUnifyMindIR::CreateSplitNode(const FuncGraphPtr &graph, const CNodePtr &all_to_all) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(all_to_all);
  int64_t split_count = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  int64_t split_dim = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitDim);

  if (all_to_all->size() <= kAllToAllInputIdx) {
    MS_LOG(EXCEPTION) << "Inputs should not be empty for cnode " << all_to_all->DebugString()
                      << trace::DumpSourceLines(all_to_all);
  }
  auto all_to_all_input = all_to_all->input(kAllToAllInputIdx);
  std::vector<AnfNodePtr> split_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitV->name())),
                                         all_to_all_input};
  auto split_v = NewCNode(split_input, graph);
  MS_EXCEPTION_IF_NULL(split_v);
  auto dtype = common::AnfAlgo::GetOutputInferDataType(all_to_all_input, 0);
  auto shape = common::AnfAlgo::GetOutputInferShape(all_to_all_input, 0);
  auto shape_size = SizeToLong(shape.size());
  if (split_dim >= shape_size || split_dim < -shape_size) {
    MS_LOG(EXCEPTION) << "Invalid split dim " << split_dim << " is over the shape size " << shape.size()
                      << trace::DumpSourceLines(all_to_all);
  }
  size_t split_idx = split_dim < 0 ? LongToSize(split_dim + shape_size) : LongToSize(split_dim);
  if (split_count == 0 || shape[split_idx] % split_count != 0) {
    MS_LOG(EXCEPTION) << "Invalid split count " << split_count << " cannot be divisible by shape[" << split_idx
                      << "] = " << shape[split_idx] << trace::DumpSourceLines(all_to_all);
  }
  shape[split_idx] /= split_count;
  std::vector<TypeId> dtypes(split_count, dtype);
  std::vector<ShapeVector> shapes(split_count, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split_v.get());

  common::AnfAlgo::SetNodeAttr(kAttrSplitDim, MakeValue<int64_t>(split_dim), split_v);
  common::AnfAlgo::SetNodeAttr(kAttrNumSplit, MakeValue<int64_t>(split_count), split_v);
  common::AnfAlgo::SetNodeAttr(kAttrSizeSplits,
                               MakeValue(std::vector<int64_t>(split_count, shape[LongToSize(split_dim)])), split_v);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), split_v);
  return split_v;
}

CNodePtr AllToAllUnifyMindIR::CreateAllToAllvNode(const FuncGraphPtr &graph, const CNodePtr &all_to_all,
                                                  const CNodePtr &split) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(all_to_all);
  MS_EXCEPTION_IF_NULL(split);
  int64_t split_count = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  std::string group = common::AnfAlgo::GetNodeAttr<std::string>(all_to_all, kAttrGroup);
  std::vector<uint32_t> group_rank_ids =
    common::AnfAlgo::HasNodeAttr(kAttrGroupRankIds, all_to_all)
      ? common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(all_to_all, kAttrGroupRankIds)
      : std::vector<uint32_t>();
  std::vector<AnfNodePtr> split_outputs;
  CreateMultipleOutputsOfAnfNode(graph, split, static_cast<size_t>(split_count), &split_outputs);
  if (split_outputs.empty()) {
    MS_LOG(EXCEPTION) << "The node " << split->DebugString() << " should have at least one output, but got 0."
                      << trace::DumpSourceLines(split);
  }
  std::vector<AnfNodePtr> all_to_all_v_input = {NewValueNode(std::make_shared<Primitive>(kAllToAllvOpName))};
  (void)all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs.begin(), split_outputs.end());
  auto all_to_all_v = NewCNode(all_to_all_v_input, graph);
  MS_EXCEPTION_IF_NULL(all_to_all_v);
  auto single_shape = AnfAlgo::GetOutputDetailShape(split_outputs[0], 0UL);
  auto single_type = common::AnfAlgo::GetOutputInferDataType(split_outputs[0], 0UL);
  std::vector<TypeId> dtypes(split_count, single_type);
  std::vector<BaseShapePtr> shapes(split_count, single_shape);
  common::AnfAlgo::SetOutputTypeAndDetailShape(dtypes, shapes, all_to_all_v.get());
  uint32_t rank_size = GetRankSize(group);
  std::vector<int64_t> rank_ids(rank_size, 0);
  for (uint32_t i = 0; i < rank_size; ++i) {
    rank_ids[i] = static_cast<int64_t>(i);
  }

  common::AnfAlgo::SetNodeAttr(kAttrSendRankIds, MakeValue<std::vector<int64_t>>(rank_ids), all_to_all_v);
  common::AnfAlgo::SetNodeAttr(kAttrRecvRankIds, MakeValue<std::vector<int64_t>>(rank_ids), all_to_all_v);
  common::AnfAlgo::SetNodeAttr(kAttrGroup, MakeValue<std::string>(group), all_to_all_v);
  common::AnfAlgo::SetNodeAttr(kAttrGroupRankIds, MakeValue<std::vector<uint32_t>>(group_rank_ids), all_to_all_v);
  auto all_to_all_prim = GetCNodePrimitive(all_to_all);
  MS_EXCEPTION_IF_NULL(all_to_all_prim);
  if (all_to_all_prim->HasAttr(parallel::COMM_REUSE) &&
      GetValue<bool>(all_to_all_prim->GetAttr(parallel::COMM_REUSE))) {
    auto all_to_all_v_prim = GetCNodePrimitive(all_to_all_v);
    MS_EXCEPTION_IF_NULL(all_to_all_v_prim);
    (void)all_to_all_v_prim->AddAttr(parallel::COMM_REUSE, MakeValue(true));
  }
  MS_LOG(INFO) << "Create AllToAllv success, split count " << split_count << ", rank size " << rank_size;
  return all_to_all_v;
}

CNodePtr AllToAllUnifyMindIR::CreateConcatNode(const FuncGraphPtr &graph, const CNodePtr &all_to_all,
                                               const CNodePtr &all_to_all_v) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(all_to_all);
  MS_EXCEPTION_IF_NULL(all_to_all_v);
  int64_t split_count = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  int64_t concat_dim = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrConcatDim);
  std::vector<AnfNodePtr> all_to_all_v_outputs;
  CreateMultipleOutputsOfAnfNode(graph, all_to_all_v, static_cast<size_t>(split_count), &all_to_all_v_outputs);
  if (all_to_all_v_outputs.empty()) {
    MS_LOG(EXCEPTION) << "The node " << all_to_all_v->DebugString() << " should have at least one output, but got 0."
                      << trace::DumpSourceLines(all_to_all_v);
  }
  std::vector<AnfNodePtr> concat_input = {NewValueNode(std::make_shared<Primitive>(kConcatDOpName))};
  (void)concat_input.insert(concat_input.end(), all_to_all_v_outputs.begin(), all_to_all_v_outputs.end());
  auto concat = NewCNode(concat_input, graph);
  MS_EXCEPTION_IF_NULL(concat);
  auto single_shape = common::AnfAlgo::GetOutputInferShape(all_to_all_v_outputs[0], 0);
  auto shape_size = SizeToLong(single_shape.size());
  if (concat_dim >= shape_size || concat_dim < -shape_size) {
    MS_LOG(EXCEPTION) << "Invalid concat dim " << concat_dim << " is greater than shape size " << single_shape.size()
                      << trace::DumpSourceLines(all_to_all);
  }
  size_t concat_idx = concat_dim < 0 ? LongToSize(concat_dim + shape_size) : LongToSize(concat_dim);
  single_shape[concat_idx] *= split_count;
  common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(all_to_all_v_outputs[0], 0UL)},
                                              {single_shape}, concat.get());

  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue<int64_t>(concat_dim), concat);
  common::AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue(split_count), concat);
  std::vector<int64_t> dyn_input_size{split_count};
  common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_size), concat);
  return concat;
}

std::vector<std::string> NeighborExchangeUnifyMindIR::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimNeighborExchange->name());
  return ret;
}

const BaseRef NeighborExchangeUnifyMindIR::DefinePattern() const {
  return VectorRef({prim::kPrimNeighborExchange, std::make_shared<SeqVar>()});
}

const AnfNodePtr NeighborExchangeUnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                      const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  ChangePrimitiveToAllToAllV(node);
  return node;
}

std::vector<std::string> AllToAllUnifyMindIR::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimAllToAll->name());
  return ret;
}

const BaseRef AllToAllUnifyMindIR::DefinePattern() const {
  return VectorRef({prim::kPrimAllToAll, std::make_shared<SeqVar>()});
}

const AnfNodePtr AllToAllUnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto all_to_all = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(all_to_all);
  auto split = CreateSplitNode(graph, all_to_all);
  auto all_to_all_v = CreateAllToAllvNode(graph, all_to_all, split);
  auto concat = CreateConcatNode(graph, all_to_all, all_to_all_v);
  return concat;
}
}  // namespace opt
}  // namespace mindspore
