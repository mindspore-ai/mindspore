/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/ascend/mindir/all_to_all_unify_mindir.h"
#include <vector>
#include <string>
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/hccl_adapter/hccl_adapter.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kCNodePrimitiveIdx = 0;
constexpr size_t kAllToAllInputIdx = 1;

inline int64_t NormalizeDim(const std::vector<size_t> &shape, int64_t dim) {
  return dim < 0 ? SizeToLong(shape.size()) + dim : dim;
}

void ChangePrimitiveToAllToAllV(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto neighbor_exchange = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(neighbor_exchange);

  if (neighbor_exchange->size() == kCNodePrimitiveIdx) {
    MS_LOG(EXCEPTION) << "Invalid cnode " << node->DebugString() << " input size " << neighbor_exchange->size();
  }

  auto prim = GetValueNode<PrimitivePtr>(neighbor_exchange->input(kCNodePrimitiveIdx));
  MS_EXCEPTION_IF_NULL(prim);
  prim->Named::operator=(Named(kAllToAllVOpName));
}

uint32_t GetRankSize(const std::string &group) {
  uint32_t rank_size;
  auto hccl_ret = hccl::HcclAdapter::GetInstance().HcclGetRankSize(group, &rank_size);
  if (hccl_ret != ::HcclResult::HCCL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Get hccl rank size for group " << group << " failed, ret = " << hccl_ret;
  }
  return rank_size;
}

CNodePtr CreateSplitNode(const FuncGraphPtr &graph, const CNodePtr &all_to_all) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(all_to_all);
  int64_t split_count = AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  int64_t split_dim = AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitDim);

  if (all_to_all->size() <= kAllToAllInputIdx) {
    MS_LOG(EXCEPTION) << "Invalid cnode " << all_to_all->DebugString() << " input size " << all_to_all->size();
  }
  auto all_to_all_input = all_to_all->input(kAllToAllInputIdx);
  std::vector<AnfNodePtr> split_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitV->name())),
                                         all_to_all_input};
  auto split_v = graph->NewCNode(split_input);
  MS_EXCEPTION_IF_NULL(split_v);
  auto dtype = AnfAlgo::GetOutputInferDataType(all_to_all_input, 0);
  auto shape = AnfAlgo::GetOutputInferShape(all_to_all_input, 0);
  split_dim = NormalizeDim(shape, split_dim);
  if (SizeToLong(shape.size()) <= split_dim) {
    MS_LOG(EXCEPTION) << "Invalid split dim " << split_dim << " is over the shape size " << shape.size();
  }
  if (split_count == 0 || shape[LongToSize(split_dim)] % static_cast<size_t>(split_count) != 0) {
    MS_LOG(EXCEPTION) << "Invalid split count " << split_count << " cannot be divisible by shape[" << split_dim
                      << "] = " << shape[LongToSize(split_dim)];
  }
  shape[LongToSize(split_dim)] /= static_cast<size_t>(split_count);
  std::vector<TypeId> dtypes(split_count, dtype);
  std::vector<std::vector<size_t>> shapes(split_count, shape);
  AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split_v.get());
  AnfAlgo::SetNodeAttr(kAttrSplitDim, MakeValue<int64_t>(split_dim), split_v);
  AnfAlgo::SetNodeAttr(kAttrNumSplit, MakeValue<int64_t>(split_count), split_v);
  AnfAlgo::SetNodeAttr(kAttrSizeSplits, MakeValue(std::vector<int64_t>(split_count, shape[LongToSize(split_dim)])),
                       split_v);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), split_v);
  return split_v;
}

CNodePtr CreateAllToAllvNode(const FuncGraphPtr &graph, const CNodePtr &all_to_all, const CNodePtr &split) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(all_to_all);
  MS_EXCEPTION_IF_NULL(split);
  int64_t split_count = AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  std::string group = AnfAlgo::GetNodeAttr<std::string>(all_to_all, kAttrGroup);
  std::vector<AnfNodePtr> split_outputs;
  CreateMultipleOutputsOfAnfNode(graph, split, static_cast<size_t>(split_count), &split_outputs);
  if (split_outputs.empty()) {
    MS_LOG(EXCEPTION) << "The node " << split->DebugString() << " should have at least one output, but got 0.";
  }
  std::vector<AnfNodePtr> all_to_all_v_input = {NewValueNode(std::make_shared<Primitive>(kAllToAllVOpName))};
  (void)all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs.begin(), split_outputs.end());
  auto all_to_all_v = graph->NewCNode(all_to_all_v_input);
  MS_EXCEPTION_IF_NULL(all_to_all_v);
  auto single_shape = AnfAlgo::GetOutputInferShape(split_outputs[0], 0);
  auto single_type = AnfAlgo::GetOutputInferDataType(split_outputs[0], 0);
  std::vector<TypeId> dtypes(split_count, single_type);
  std::vector<std::vector<size_t>> shapes(split_count, single_shape);
  AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, all_to_all_v.get());
  uint32_t rank_size = GetRankSize(group);
  std::vector<int64_t> rank_ids(rank_size, 0);
  for (uint32_t i = 0; i < rank_size; ++i) {
    rank_ids[i] = static_cast<int64_t>(i);
  }

  AnfAlgo::SetNodeAttr(kAttrSendRankIds, MakeValue<std::vector<int64_t>>(rank_ids), all_to_all_v);
  AnfAlgo::SetNodeAttr(kAttrRecvRankIds, MakeValue<std::vector<int64_t>>(rank_ids), all_to_all_v);
  AnfAlgo::SetNodeAttr(kAttrGroup, MakeValue<std::string>(group), all_to_all_v);
  MS_LOG(INFO) << "Create AllToAllv success, split count " << split_count << ", rank size " << rank_size;
  return all_to_all_v;
}

CNodePtr CreateConcatNode(const FuncGraphPtr &graph, const CNodePtr &all_to_all, const CNodePtr &all_to_all_v) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(all_to_all);
  MS_EXCEPTION_IF_NULL(all_to_all_v);
  int64_t split_count = AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  int64_t concat_dim = AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrConcatDim);
  std::vector<AnfNodePtr> all_to_all_v_outputs;
  CreateMultipleOutputsOfAnfNode(graph, all_to_all_v, static_cast<size_t>(split_count), &all_to_all_v_outputs);
  if (all_to_all_v_outputs.empty()) {
    MS_LOG(EXCEPTION) << "The node " << all_to_all_v->DebugString() << " should have at least one output, but got 0.";
  }
  std::vector<AnfNodePtr> concat_input = {NewValueNode(std::make_shared<Primitive>(kConcatOpName))};
  (void)concat_input.insert(concat_input.end(), all_to_all_v_outputs.begin(), all_to_all_v_outputs.end());
  auto concat = graph->NewCNode(concat_input);
  MS_EXCEPTION_IF_NULL(concat);
  auto single_shape = AnfAlgo::GetOutputInferShape(all_to_all_v_outputs[0], 0);
  concat_dim = NormalizeDim(single_shape, concat_dim);
  if (LongToSize(concat_dim) >= single_shape.size()) {
    MS_LOG(EXCEPTION) << "Invalid concat dim " << concat_dim << " is greater than shape size " << single_shape.size();
  }
  single_shape[LongToSize(concat_dim)] *= static_cast<size_t>(split_count);
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(all_to_all_v_outputs[0], 0)}, {single_shape},
                                      concat.get());
  AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue<int64_t>(concat_dim), concat);
  AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue(split_count), concat);
  std::vector<int64_t> dyn_input_size{split_count};
  AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_size), concat);
  return concat;
}
}  // namespace

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
