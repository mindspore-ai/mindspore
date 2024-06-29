/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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
#include "ops/other_ops.h"
#include "ops/array_ops.h"
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
constexpr auto kAttrIrUnified = "ir_unified";
constexpr auto kAttrFlashIndex = "FLASH_INDEX";

void ChangePrimitiveToAllToAllV(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto neighbor_exchange = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(neighbor_exchange);

  if (neighbor_exchange->size() == kCNodePrimitiveIdx) {
    MS_LOG(INTERNAL_EXCEPTION) << "Inputs should not be empty for cnode " << node->DebugString()
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

CNodePtr AllToAllUnifyMindIR::CreateSplitNode(const KernelGraphPtr &graph, const CNodePtr &all_to_all,
                                              const AnfNodePtr &input_node, int64_t split_count,
                                              int64_t split_dim) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(all_to_all);

  std::vector<AnfNodePtr> split_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplit->name())),
                                         input_node, graph->NewValueNode(MakeValue(split_dim)),
                                         graph->NewValueNode(MakeValue(split_count))};
  auto split = NewCNode(split_input, graph);
  MS_EXCEPTION_IF_NULL(split);
  split->set_scope(all_to_all->scope());
  auto dtype = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
  auto shape = common::AnfAlgo::GetOutputInferShape(input_node, 0);
  auto shape_size = SizeToLong(shape.size());
  if (split_dim >= shape_size || split_dim < -shape_size) {
    MS_LOG(INTERNAL_EXCEPTION) << "Invalid split dim " << split_dim << " is over the shape size " << shape.size()
                               << trace::DumpSourceLines(all_to_all);
  }
  size_t split_idx = split_dim < 0 ? LongToSize(split_dim + shape_size) : LongToSize(split_dim);
  if (shape[split_idx] >= 0 && (split_count == 0 || shape[split_idx] % split_count != 0)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Invalid split count " << split_count << " cannot be divisible by shape[" << split_idx
                               << "] = " << shape[split_idx] << trace::DumpSourceLines(all_to_all);
  }
  shape[split_idx] = shape[split_idx] >= 0 ? shape[split_idx] / split_count : shape[split_idx];
  std::vector<TypeId> dtypes(split_count, dtype);
  std::vector<ShapeVector> shapes(split_count, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split.get());

  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), split);
  return split;
}

CNodePtr NeighborExchangeUnifyMindIR::CreateAllToAllvNode(const FuncGraphPtr &graph,
                                                          const CNodePtr &neighbor_exchange) const {
  std::string group = common::AnfAlgo::GetNodeAttr<std::string>(neighbor_exchange, kAttrGroup);
  std::vector<uint32_t> group_rank_ids =
    common::AnfAlgo::HasNodeAttr(kAttrGroupRankIds, neighbor_exchange)
      ? common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(neighbor_exchange, kAttrGroupRankIds)
      : std::vector<uint32_t>();
  std::vector<int64_t> send_rank_ids =
    common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange, kAttrSendRankIds);
  std::vector<int64_t> recv_rank_ids =
    common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange, kAttrRecvRankIds);

  int64_t send_count = send_rank_ids.size(), recv_count = recv_rank_ids.size();
  auto tuple_input = neighbor_exchange->input(1);
  std::vector<AnfNodePtr> split_outputs;
  CreateMultipleOutputsOfAnfNode(graph, tuple_input, static_cast<size_t>(send_count), &split_outputs);
  if (split_outputs.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "The node " << tuple_input->DebugString()
                               << " should have at least one output, but got 0." << trace::DumpSourceLines(tuple_input);
  }
  std::vector<AnfNodePtr> all_to_all_v_input = {NewValueNode(std::make_shared<Primitive>(kAllToAllvOpName))};
  (void)all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs.begin(), split_outputs.end());
  auto all_to_all_v = NewCNode(all_to_all_v_input, graph);
  MS_EXCEPTION_IF_NULL(all_to_all_v);

  auto single_shape = AnfAlgo::GetOutputDetailShape(split_outputs[0], 0UL);
  auto single_type = common::AnfAlgo::GetOutputInferDataType(split_outputs[0], 0UL);
  std::vector<TypeId> dtypes(recv_count, single_type);
  std::vector<BaseShapePtr> shapes(recv_count, single_shape);
  common::AnfAlgo::SetSingleOutputTypeAndDetailShape(dtypes, shapes, all_to_all_v.get());

  common::AnfAlgo::SetNodeAttr(kAttrSendRankIds, MakeValue<std::vector<int64_t>>(send_rank_ids), all_to_all_v);
  common::AnfAlgo::SetNodeAttr(kAttrRecvRankIds, MakeValue<std::vector<int64_t>>(recv_rank_ids), all_to_all_v);
  common::AnfAlgo::SetNodeAttr(kAttrGroup, MakeValue<std::string>(group), all_to_all_v);
  common::AnfAlgo::SetNodeAttr(kAttrGroupRankIds, MakeValue<std::vector<uint32_t>>(group_rank_ids), all_to_all_v);

  auto neighbor_exchange_prim = GetCNodePrimitive(neighbor_exchange);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_prim);
  if (neighbor_exchange_prim->HasAttr(parallel::COMM_REUSE) &&
      GetValue<bool>(neighbor_exchange_prim->GetAttr(parallel::COMM_REUSE))) {
    auto all_to_all_v_prim = GetCNodePrimitive(all_to_all_v);
    MS_EXCEPTION_IF_NULL(all_to_all_v_prim);
    (void)all_to_all_v_prim->AddAttr(parallel::COMM_REUSE, MakeValue(true));
  }

  if (neighbor_exchange_prim->HasAttr("FLASH_INDEX")) {
    auto flash_index = GetValue<std::string>(neighbor_exchange_prim->GetAttr("FLASH_INDEX"));
    auto all_to_all_v_prim = GetCNodePrimitive(all_to_all_v);
    MS_EXCEPTION_IF_NULL(all_to_all_v_prim);
    (void)all_to_all_v_prim->AddAttr("FLASH_INDEX", MakeValue<std::string>(flash_index));
  }
  return all_to_all_v;
}

CNodePtr AllToAllUnifyMindIR::CreateSplitNodeWithSplitDim(const KernelGraphPtr &graph,
                                                          const CNodePtr &all_to_all) const {
  MS_EXCEPTION_IF_NULL(all_to_all);
  int64_t split_count = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  int64_t split_dim = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitDim);

  if (all_to_all->size() <= kAllToAllInputIdx) {
    MS_LOG(EXCEPTION) << "Inputs should not be empty for cnode " << all_to_all->DebugString()
                      << trace::DumpSourceLines(all_to_all);
  }
  auto all_to_all_input = all_to_all->input(kAllToAllInputIdx);
  return CreateSplitNode(graph, all_to_all, all_to_all_input, split_count, split_dim);
}

CNodePtr AllToAllUnifyMindIR::CreateSplitNodeWithDim0(const KernelGraphPtr &graph, const CNodePtr &all_to_all,
                                                      const CNodePtr &input_node) const {
  MS_EXCEPTION_IF_NULL(all_to_all);
  int64_t split_count = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  return CreateSplitNode(graph, all_to_all, input_node, split_count, 0);
}

CNodePtr AllToAllUnifyMindIR::CreateAllToAllvNode(const KernelGraphPtr &graph, const CNodePtr &all_to_all,
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
    MS_LOG(INTERNAL_EXCEPTION) << "The node " << split->DebugString() << " should have at least one output, but got 0."
                               << trace::DumpSourceLines(split);
  }
  std::vector<AnfNodePtr> new_ata_input = {NewValueNode(std::make_shared<Primitive>(kAllToAllvOpName))};
  (void)new_ata_input.insert(new_ata_input.end(), split_outputs.begin(), split_outputs.end());
  auto new_ata = NewCNode(new_ata_input, graph);
  MS_EXCEPTION_IF_NULL(new_ata);
  new_ata->set_scope(all_to_all->scope());
  auto single_shape = AnfAlgo::GetOutputDetailShape(split_outputs[0], 0UL);
  auto single_type = common::AnfAlgo::GetOutputInferDataType(split_outputs[0], 0UL);
  std::vector<TypeId> dtypes(split_count, single_type);
  std::vector<BaseShapePtr> shapes(split_count, single_shape);
  common::AnfAlgo::SetOutputTypeAndDetailShape(dtypes, shapes, new_ata.get());
  uint32_t rank_size = GetRankSize(group);
  std::vector<int64_t> rank_ids(rank_size, 0);
  for (uint32_t i = 0; i < rank_size; ++i) {
    rank_ids[i] = static_cast<int64_t>(i);
  }

  common::AnfAlgo::SetNodeAttr(kAttrSendRankIds, MakeValue<std::vector<int64_t>>(rank_ids), new_ata);
  common::AnfAlgo::SetNodeAttr(kAttrRecvRankIds, MakeValue<std::vector<int64_t>>(rank_ids), new_ata);
  common::AnfAlgo::SetNodeAttr(kAttrGroup, MakeValue<std::string>(group), new_ata);
  common::AnfAlgo::SetNodeAttr(kAttrGroupRankIds, MakeValue<std::vector<uint32_t>>(group_rank_ids), new_ata);
  auto all_to_all_prim = GetCNodePrimitive(all_to_all);
  MS_EXCEPTION_IF_NULL(all_to_all_prim);
  if (all_to_all_prim->HasAttr(parallel::COMM_REUSE) &&
      GetValue<bool>(all_to_all_prim->GetAttr(parallel::COMM_REUSE))) {
    auto new_ata_prim = GetCNodePrimitive(new_ata);
    MS_EXCEPTION_IF_NULL(new_ata_prim);
    (void)new_ata_prim->AddAttr(parallel::COMM_REUSE, MakeValue(true));
  }
  MS_LOG(INFO) << "Create AllToAllv success, split count " << split_count << ", rank size " << rank_size;
  return new_ata;
}

CNodePtr AllToAllUnifyMindIR::CreateAllToAllNode(const KernelGraphPtr &graph, const CNodePtr &all_to_all,
                                                 const CNodePtr &concat) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(all_to_all);
  MS_EXCEPTION_IF_NULL(concat);
  int64_t split_count = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  std::string group = common::AnfAlgo::GetNodeAttr<std::string>(all_to_all, kAttrGroup);
  std::vector<AnfNodePtr> new_ata_input = {NewValueNode(std::make_shared<Primitive>(kAllToAllOpName))};
  (void)new_ata_input.insert(new_ata_input.end(), concat);
  auto new_ata = NewCNode(new_ata_input, graph);
  MS_EXCEPTION_IF_NULL(new_ata);
  new_ata->set_scope(all_to_all->scope());
  new_ata->set_abstract(concat->abstract());
  common::AnfAlgo::CopyNodeAttr(kAttrGroup, all_to_all, new_ata);
  auto all_to_all_prim = GetCNodePrimitive(all_to_all);
  MS_EXCEPTION_IF_NULL(all_to_all_prim);
  if (all_to_all_prim->HasAttr(parallel::COMM_REUSE) &&
      GetValue<bool>(all_to_all_prim->GetAttr(parallel::COMM_REUSE))) {
    auto new_ata_prim = GetCNodePrimitive(new_ata);
    MS_EXCEPTION_IF_NULL(new_ata_prim);
    (void)new_ata_prim->AddAttr(parallel::COMM_REUSE, MakeValue(true));
  }
  common::AnfAlgo::SetNodeAttr(kAttrIrUnified, MakeValue(true), new_ata);
  uint32_t rank_size = GetRankSize(group);
  MS_LOG(INFO) << "Create AlltoAll success, split count " << split_count << ", rank size " << rank_size;
  return new_ata;
}

CNodePtr AllToAllUnifyMindIR::CreateConcatNode(const KernelGraphPtr &graph, const CNodePtr &all_to_all,
                                               const CNodePtr &input_node, int64_t split_count,
                                               int64_t concat_dim) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(all_to_all);
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> input_node_outputs;
  CreateMultipleOutputsOfAnfNode(graph, input_node, static_cast<size_t>(split_count), &input_node_outputs);
  if (input_node_outputs.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "The node " << input_node->DebugString()
                               << " should have at least one output, but got 0." << trace::DumpSourceLines(input_node);
  }
  std::vector<AnfNodePtr> concat_input = {NewValueNode(std::make_shared<Primitive>(kConcatOpName)), input_node,
                                          graph->NewValueNode(MakeValue(concat_dim))};
  auto concat = NewCNode(concat_input, graph);
  MS_EXCEPTION_IF_NULL(concat);
  concat->set_scope(all_to_all->scope());
  auto single_shape = common::AnfAlgo::GetOutputInferShape(input_node_outputs[0], 0);
  auto shape_size = SizeToLong(single_shape.size());
  if (concat_dim >= shape_size || concat_dim < -shape_size) {
    MS_LOG(INTERNAL_EXCEPTION) << "Invalid concat dim " << concat_dim << " is greater than shape size "
                               << single_shape.size() << trace::DumpSourceLines(all_to_all);
  }
  size_t concat_idx = concat_dim < 0 ? LongToSize(concat_dim + shape_size) : LongToSize(concat_dim);
  single_shape[concat_idx] =
    single_shape[concat_idx] >= 0 ? single_shape[concat_idx] * split_count : single_shape[concat_idx];
  common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(input_node_outputs[0], 0UL)},
                                              {single_shape}, concat.get());
  return concat;
}

CNodePtr AllToAllUnifyMindIR::CreateConcatNodeWithConcatDim(const KernelGraphPtr &graph, const CNodePtr &all_to_all,
                                                            const CNodePtr &input_node) const {
  MS_EXCEPTION_IF_NULL(all_to_all);
  int64_t split_count = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  int64_t concat_dim = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrConcatDim);
  return CreateConcatNode(graph, all_to_all, input_node, split_count, concat_dim);
}

CNodePtr AllToAllUnifyMindIR::CreateConcatNodeWithDim0(const KernelGraphPtr &graph, const CNodePtr &all_to_all,
                                                       const CNodePtr &input_node) const {
  MS_EXCEPTION_IF_NULL(all_to_all);
  int64_t split_count = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  return CreateConcatNode(graph, all_to_all, input_node, split_count, 0);
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
  auto neighbor_exchange = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(neighbor_exchange);
  auto neighbor_exchange_prim = GetCNodePrimitive(neighbor_exchange);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_prim);
  if (!neighbor_exchange_prim->HasAttr(kAttrFlashIndex)) {
    ChangePrimitiveToAllToAllV(node);
    return node;
  }
  auto all_to_all_v = CreateAllToAllvNode(graph, neighbor_exchange);
  return all_to_all_v;
}

std::vector<std::string> AllToAllUnifyMindIR::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimAlltoAll->name());
  return ret;
}

const BaseRef AllToAllUnifyMindIR::DefinePattern() const {
  return VectorRef({prim::kPrimAlltoAll, std::make_shared<SeqVar>()});
}

const AnfNodePtr AllToAllUnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto all_to_all = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(all_to_all);
  if (GetBoolAttr(all_to_all, kAttrIrUnified)) {
    return nullptr;
  }
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto ms_context = MsContext::GetInstance();
  bool is_kbk = ms_context->IsKByKExecutorMode() || ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK) == false;
  AnfNodePtr ret_node = nullptr;
  if (is_kbk) {
    auto split = CreateSplitNodeWithSplitDim(kernel_graph, all_to_all);
    auto concat_dim0 = CreateConcatNodeWithDim0(kernel_graph, all_to_all, split);
    auto new_ata = CreateAllToAllNode(kernel_graph, all_to_all, concat_dim0);
    auto split_dim0 = CreateSplitNodeWithDim0(kernel_graph, all_to_all, new_ata);
    auto concat = CreateConcatNodeWithConcatDim(kernel_graph, all_to_all, split_dim0);
    ret_node = concat;
  } else {
    auto split = CreateSplitNodeWithSplitDim(kernel_graph, all_to_all);
    auto new_ata = CreateAllToAllvNode(kernel_graph, all_to_all, split);
    auto concat = CreateConcatNodeWithConcatDim(kernel_graph, all_to_all, new_ata);
    ret_node = concat;
  }
  return ret_node;
}
}  // namespace opt
}  // namespace mindspore
