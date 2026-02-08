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

#include "backend/common/pass/mindir/all_to_all_unify_mindir.h"
#include <vector>
#include <string>
#include "primitive/other_ops.h"
#include "primitive/array_ops.h"
#include "utils/trace_base.h"
#include "include/utils/anfalgo.h"
#include "include/utils/comm_manager.h"
#include "include/backend/common/pass_manager/helper.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "backend/common/pass/common/optimizer_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kCNodePrimitiveIdx = 0;
constexpr size_t kAllToAllInputIdx = 1;
constexpr auto kAttrIrUnified = "ir_unified";
constexpr auto kAttrFlashIndex = "FLASH_INDEX";
bool CheckNoNeedTranspose(const ShapeVector &shape, size_t dim) {
  if (shape.size() > dim && dim > 0) {
    for (size_t i = 0; i < dim; i++) {
      if (shape[i] != 1) {
        return false;
      }
    }
    return true;
  }
  return false;
}
bool CheckStaticShape(const ShapeVector &shape) {
  for (size_t i = 0; i < shape.size(); i++) {
    if (shape[i] <= 0) {
      return false;
    }
  }
  return true;
}
size_t GetIdx(const CNodePtr &all_to_all, const std::string &attr, size_t shape_size) {
  int64_t dim = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, attr);
  size_t idx = dim < 0 ? LongToSize(dim + shape_size) : LongToSize(dim);
  if (idx >= shape_size) {
    MS_LOG(INTERNAL_EXCEPTION) << "Invalid split or concat dim " << dim << " is over the shape size " << shape_size;
  }
  return idx;
}
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
  prim->Named::operator=(Named(kAlltoAllVOpName));
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

CNodePtr NeighborExchangeUnifyMindIR::CreateAlltoAllVNode(const FuncGraphPtr &graph,
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
  std::vector<AnfNodePtr> all_to_all_v_input = {NewValueNode(std::make_shared<Primitive>(kAlltoAllVOpName))};
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
  common::AnfAlgo::SetNodeAttr(kAttrBlockSize, MakeValue<int64_t>(1), all_to_all_v);
  common::AnfAlgo::SetNodeAttr(kAttrGroupRankIds, MakeValue<std::vector<uint32_t>>(group_rank_ids), all_to_all_v);
  if (common::AnfAlgo::HasNodeAttr(parallel::COMM_REUSE, neighbor_exchange)) {
    common::AnfAlgo::CopyNodeAttr(parallel::COMM_REUSE, neighbor_exchange, all_to_all_v);
  }
  if (common::AnfAlgo::HasNodeAttr("FLASH_INDEX", neighbor_exchange)) {
    common::AnfAlgo::CopyNodeAttr("FLASH_INDEX", neighbor_exchange, all_to_all_v);
  }
  return all_to_all_v;
}

CNodePtr AllToAllUnifyMindIR::CreateSplitNodeWithSplitDim(const KernelGraphPtr &graph,
                                                          const CNodePtr &all_to_all) const {
  MS_EXCEPTION_IF_NULL(all_to_all);
  int64_t split_count = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  int64_t split_dim = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitDim);

  auto all_to_all_input = all_to_all->input(kAllToAllInputIdx);
  return CreateSplitNode(graph, all_to_all, all_to_all_input, split_count, split_dim);
}

CNodePtr AllToAllUnifyMindIR::CreateSplitNodeWithDim0(const KernelGraphPtr &graph, const CNodePtr &all_to_all,
                                                      const CNodePtr &input_node) const {
  MS_EXCEPTION_IF_NULL(all_to_all);
  int64_t split_count = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  return CreateSplitNode(graph, all_to_all, input_node, split_count, 0);
}

const CNodePtr AllToAllUnifyMindIR::CreateTransposeNode(const KernelGraphPtr &graph, const AnfNodePtr &input_node,
                                                        ShapeVector shape, ShapeVector dims) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input_node);
  auto prim = std::make_shared<Primitive>(kTransposeOpName);
  MS_EXCEPTION_IF_NULL(prim);
  auto transpose_perm_node = CreateValueNodeWithKernelInfo(graph, MakeValue(dims));
  MS_EXCEPTION_IF_NULL(transpose_perm_node);
  std::vector<AnfNodePtr> transpose_inputs = {NewValueNode(prim), input_node, transpose_perm_node};
  auto transpose_node = NewCNode(transpose_inputs, graph);
  MS_EXCEPTION_IF_NULL(transpose_node);
  common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(input_node, 0)}, {shape},
                                              transpose_node.get());
  return transpose_node;
}

CNodePtr AllToAllUnifyMindIR::CreateAlltoAllVNode(const KernelGraphPtr &graph, const CNodePtr &all_to_all,
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
  std::vector<AnfNodePtr> new_ata_input = {NewValueNode(std::make_shared<Primitive>(kAlltoAllVOpName))};
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
  common::AnfAlgo::SetNodeAttr(kAttrBlockSize, MakeValue<int64_t>(1), new_ata);
  common::AnfAlgo::SetNodeAttr(kAttrGroupRankIds, MakeValue<std::vector<uint32_t>>(group_rank_ids), new_ata);
  if (common::AnfAlgo::HasNodeAttr(parallel::COMM_REUSE, all_to_all)) {
    common::AnfAlgo::CopyNodeAttr(parallel::COMM_REUSE, all_to_all, new_ata);
  }
  MS_LOG(INFO) << "Create AlltoAllV success, split count " << split_count << ", rank size " << rank_size;
  return new_ata;
}

CNodePtr AllToAllUnifyMindIR::CreateAllToAllNode(const KernelGraphPtr &graph, const CNodePtr &all_to_all,
                                                 const AnfNodePtr &all_to_all_input) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(all_to_all);
  MS_EXCEPTION_IF_NULL(all_to_all_input);
  int64_t split_count = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  std::string group = common::AnfAlgo::GetNodeAttr<std::string>(all_to_all, kAttrGroup);
  std::vector<AnfNodePtr> new_ata_input = {NewValueNode(std::make_shared<Primitive>(kAllToAllOpName))};
  (void)new_ata_input.insert(new_ata_input.end(), all_to_all_input);
  auto new_ata = NewCNode(new_ata_input, graph);
  MS_EXCEPTION_IF_NULL(new_ata);
  new_ata->set_scope(all_to_all->scope());
  new_ata->set_abstract(all_to_all_input->abstract());
  common::AnfAlgo::CopyNodeAttr(kAttrGroup, all_to_all, new_ata);
  if (common::AnfAlgo::HasNodeAttr(parallel::COMM_REUSE, all_to_all)) {
    common::AnfAlgo::CopyNodeAttr(parallel::COMM_REUSE, all_to_all, new_ata);
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

const CNodePtr AllToAllUnifyMindIR::CreateReshapeNode(const KernelGraphPtr &graph, const AnfNodePtr &input_node,
                                                      const ShapeVector &shape) const {
  auto prim = std::make_shared<Primitive>(kReshapeOpName);
  MS_EXCEPTION_IF_NULL(prim);
  auto shape_value_node = CreateValueNodeWithKernelInfo(graph, MakeValue(shape));
  MS_EXCEPTION_IF_NULL(shape_value_node);
  AnfNodePtrList reshape_inputs = {NewValueNode(prim), input_node, shape_value_node};
  auto reshape_node = NewCNode(reshape_inputs, graph);
  MS_EXCEPTION_IF_NULL(reshape_node);
  auto abs = InferAbstract(prim, {input_node, shape_value_node});
  MS_EXCEPTION_IF_NULL(abs);
  reshape_node->set_abstract(abs);
  return reshape_node;
}

CNodePtr AllToAllUnifyMindIR::CreateAntecedentTransposeNode(const KernelGraphPtr &kernel_graph,
                                                            const AnfNodePtr &all_to_all_input,
                                                            const ShapeVector &shape, int64_t split_count,
                                                            size_t split_idx) const {
  ShapeVector reshape_shape;
  ShapeVector trans_shape;
  ShapeVector trans_dims;
  ShapeVector reshape_shape2;
  trans_shape.push_back(split_count);
  trans_dims.push_back(split_idx);
  for (size_t i = 0, idx = 0; i < shape.size(); i++, idx++) {
    if (i == LongToSize(split_idx)) {
      reshape_shape.push_back(split_count);
      reshape_shape.push_back(-1);
      idx++;
      trans_shape.push_back(shape[i] / split_count);
      trans_dims.push_back(idx);

      reshape_shape2.push_back(-1);
    } else {
      reshape_shape.push_back(shape[i]);
      trans_shape.push_back(reshape_shape[idx]);
      trans_dims.push_back(idx);
      if (i == 0) {
        reshape_shape2.push_back(shape[0] * split_count);
      } else {
        reshape_shape2.push_back(shape[i]);
      }
    }
  }
  auto reshape_node = CreateReshapeNode(kernel_graph, all_to_all_input, reshape_shape);
  auto transpose = CreateTransposeNode(kernel_graph, reshape_node, trans_shape, trans_dims);
  auto reshape_node2 = CreateReshapeNode(kernel_graph, transpose, reshape_shape2);

  return reshape_node2;
}

CNodePtr AllToAllUnifyMindIR::CreateSuccessorTransposeNode(const KernelGraphPtr &kernel_graph,
                                                           const AnfNodePtr &new_ata, const ShapeVector &out_shape,
                                                           int64_t split_count, size_t split_idx,
                                                           size_t concat_idx) const {
  auto in_shape = common::AnfAlgo::GetOutputInferShape(new_ata, kIndex0);
  ShapeVector reshape_shape;
  ShapeVector trans_shape;
  ShapeVector trans_dims;
  for (size_t i = 0; i < in_shape.size(); i++) {
    if (i == 0) {
      reshape_shape.push_back(split_count);
      reshape_shape.push_back(in_shape[i] / split_count);
      trans_shape.push_back(in_shape[i] / split_count);
    } else if (i == LongToSize(concat_idx)) {
      reshape_shape.push_back(in_shape[i]);
      trans_shape.push_back(reshape_shape[0]);
      trans_shape.push_back(reshape_shape[i + 1]);
      trans_dims.push_back(0);
    } else {
      reshape_shape.push_back(in_shape[i]);
      trans_shape.push_back(reshape_shape[i + 1]);
    }
    trans_dims.push_back(i + 1);
  }
  reshape_shape[split_idx + 1] = -1;
  auto reshape_node = CreateReshapeNode(kernel_graph, new_ata, reshape_shape);
  auto transpose = CreateTransposeNode(kernel_graph, reshape_node, trans_shape, trans_dims);
  auto reshape_node2 = CreateReshapeNode(kernel_graph, transpose, out_shape);
  return reshape_node2;
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
  auto all_to_all_v = CreateAlltoAllVNode(graph, neighbor_exchange);
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
  if (all_to_all->size() <= kAllToAllInputIdx) {
    MS_LOG_WITH_NODE(EXCEPTION, all_to_all) << "Inputs should not be empty for cnode "
                                            << all_to_all->fullname_with_scope() << trace::DumpSourceLines(all_to_all);
  }
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  bool is_kbk = !kernel_graph->is_graph_run_mode();
  AnfNodePtr ret_node = nullptr;
  if (is_kbk) {
    MS_LOG(INFO) << "AlltoAll pass in KernelMode, node: " << node->fullname_with_scope()
                 << ", graph: " << graph->ToString();
    AnfNodePtr all_to_all_input = all_to_all->input(kAllToAllInputIdx);
    auto shape = common::AnfAlgo::GetOutputInferShape(all_to_all->input(kIndex1), kIndex0);
    int64_t split_count = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
    size_t split_idx = GetIdx(all_to_all, kAttrSplitDim, shape.size());
    size_t concat_idx = GetIdx(all_to_all, kAttrConcatDim, shape.size());
    bool is_static_shape = CheckStaticShape(shape);
    if (is_static_shape && CheckNoNeedTranspose(shape, split_idx)) {
      auto new_shape = shape;
      if (shape[split_idx] % split_count != 0) {
        MS_LOG(INTERNAL_EXCEPTION) << "Invalid split count " << split_count << " cannot be divisible by shape["
                                   << split_idx << "] = " << shape[split_idx] << trace::DumpSourceLines(all_to_all);
      }
      new_shape[0] = shape[0] * split_count;
      new_shape[split_idx] = -1;
      auto reshape_node = CreateReshapeNode(kernel_graph, all_to_all_input, new_shape);
      all_to_all_input = reshape_node;
    } else if (is_static_shape && split_idx != 0) {
      all_to_all_input = CreateAntecedentTransposeNode(kernel_graph, all_to_all_input, shape, split_count, split_idx);
      OptimizerUtils::MoveContrlDepend(graph, all_to_all->input(1), all_to_all_input);
    } else if (split_idx != 0) {
      auto split = CreateSplitNodeWithSplitDim(kernel_graph, all_to_all);
      auto concat_dim0 = CreateConcatNodeWithDim0(kernel_graph, all_to_all, split);
      all_to_all_input = concat_dim0;
      OptimizerUtils::MoveContrlDepend(graph, all_to_all->input(1), all_to_all_input);
    }
    auto new_ata = CreateAllToAllNode(kernel_graph, all_to_all, all_to_all_input);
    ret_node = new_ata;
    auto out_shape = common::AnfAlgo::GetOutputInferShape(all_to_all, kIndex0);
    if (is_static_shape && CheckNoNeedTranspose(out_shape, static_cast<size_t>(concat_idx))) {
      out_shape[split_idx] = -1;
      auto reshape_node = CreateReshapeNode(kernel_graph, new_ata, out_shape);
      ret_node = reshape_node;
    } else if (is_static_shape && concat_idx != 0) {
      out_shape[split_idx] = -1;
      OptimizerUtils::MoveContrlDepend(graph, node, new_ata);
      auto moved_depends = OptimizerUtils::MoveDataDepend(graph, node, new_ata);
      auto pre_node = new_ata;
      if (!moved_depends.empty()) {
        pre_node = moved_depends[0];
      }
      auto transpose =
        CreateSuccessorTransposeNode(kernel_graph, pre_node, out_shape, split_count, split_idx, concat_idx);
      OptimizerUtils::ReplaceDataDepend(graph, moved_depends, transpose);
      ret_node = transpose;
    } else if (concat_idx != 0) {
      OptimizerUtils::MoveContrlDepend(graph, node, new_ata);
      auto moved_depends = OptimizerUtils::MoveDataDepend(graph, node, new_ata);
      auto pre_node = new_ata;
      if (!moved_depends.empty()) {
        pre_node = moved_depends[0];
      }
      auto split_dim0 = CreateSplitNodeWithDim0(kernel_graph, all_to_all, pre_node);
      auto concat = CreateConcatNodeWithConcatDim(kernel_graph, all_to_all, split_dim0);
      OptimizerUtils::ReplaceDataDepend(graph, moved_depends, concat);
      ret_node = concat;
    }
  } else {
    MS_LOG(INFO) << "AlltoAll pass in GraphMode, node: " << node->fullname_with_scope()
                 << ", graph: " << graph->ToString();
    auto split = CreateSplitNodeWithSplitDim(kernel_graph, all_to_all);
    OptimizerUtils::MoveContrlDepend(graph, all_to_all->input(1), split);
    auto new_ata = CreateAlltoAllVNode(kernel_graph, all_to_all, split);
    OptimizerUtils::MoveContrlDepend(graph, node, new_ata);
    auto moved_depends = OptimizerUtils::MoveDataDepend(graph, node, new_ata);
    auto pre_node = new_ata;
    if (!moved_depends.empty()) {
      pre_node = moved_depends[0];
    }
    auto concat = CreateConcatNodeWithConcatDim(kernel_graph, all_to_all, pre_node);
    OptimizerUtils::ReplaceDataDepend(graph, moved_depends, concat);
    ret_node = concat;
  }
  return ret_node;
}
}  // namespace opt
}  // namespace mindspore
