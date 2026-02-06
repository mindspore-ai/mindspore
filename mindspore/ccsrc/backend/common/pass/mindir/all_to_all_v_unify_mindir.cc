/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "backend/common/pass/mindir/all_to_all_v_unify_mindir.h"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <tuple>
#include <algorithm>
#include "include/utils/anfalgo.h"
#include "include/utils/comm_manager.h"
#include "include/backend/common/pass_manager/helper.h"
#include "primitive/array_ops.h"
#include "primitive/other_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "backend/common/pass/common/optimizer_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace opt {
namespace {
class MemRange {
 public:
  explicit MemRange(uint32_t rank_size) : counts(rank_size, 0), displs(rank_size, 0) {}
  std::vector<int64_t> counts;
  std::vector<int64_t> displs;
};

uint32_t GetRankSize(const std::string &group) {
  uint32_t rank_size;
  if (!CommManager::GetInstance().GetRankSize(group, &rank_size)) {
    MS_LOG(EXCEPTION) << "Get hccl rank size for group " << group << " failed.";
  }
  return rank_size;
}

bool IsInTheOrder(const std::vector<int64_t> &vec) {
  for (size_t i = 1; i < vec.size(); ++i) {
    if (vec[i] <= vec[i - 1]) {
      return false;
    }
  }
  return true;
}

MemRange CalcAlltoAllVUnifyMindIRMemRange(const CNodePtr &origin_node, uint32_t rank_size,
                                          const std::vector<size_t> &mem_sizes, const std::string &rank_ids_attr) {
  MS_EXCEPTION_IF_NULL(origin_node);
  MemRange mem_range(rank_size);

  auto rank_ids = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(origin_node, rank_ids_attr);
  if (!IsInTheOrder(rank_ids)) {
    std::vector<size_t> mem_offset(mem_sizes.size(), 0);
    for (size_t i = 1; i < mem_sizes.size(); ++i) {
      mem_offset[i] = mem_offset[i - 1] + mem_sizes[i - 1];
    }
    for (size_t i = 0; i < rank_ids.size(); ++i) {
      if (rank_ids[i] < 0 || static_cast<size_t>(rank_ids[i]) >= rank_size) {
        MS_LOG(INTERNAL_EXCEPTION) << "Invalid rank id " << rank_ids[i] << " at index " << i << " as rank size "
                                   << rank_size;
      }
      mem_range.counts[LongToSize(rank_ids[i])] = static_cast<int64_t>(mem_sizes[i]);
      mem_range.displs[LongToSize(rank_ids[i])] = static_cast<int64_t>(mem_offset[i]);
    }
    return mem_range;
  }

  std::map<int64_t, size_t> rank_id_map;
  for (size_t i = 0; i < rank_ids.size(); ++i) {
    auto rank_id = rank_ids.at(i);
    if (rank_id < 0 || LongToSize(rank_id) >= rank_size) {
      MS_LOG(INTERNAL_EXCEPTION) << "Invalid rank id " << rank_id << " at index " << i << " as rank size " << rank_size;
    }
    (void)rank_id_map.emplace(rank_id, i);
  }

  size_t offset = 0;
  for (uint32_t i = 0; i < rank_size; ++i) {
    mem_range.displs[i] = SizeToLong(offset);
    decltype(rank_id_map)::const_iterator iter = rank_id_map.find(i);
    if (iter != rank_id_map.end()) {
      mem_range.counts[i] = static_cast<int64_t>(mem_sizes[iter->second]);
      offset += mem_sizes[iter->second];
    } else {
      mem_range.counts[i] = 0;
    }
  }
  return mem_range;
}

std::tuple<MemRange, MemRange> CalcAlltoAllVAttr(const CNodePtr &origin_node, uint32_t rank_size,
                                                 const std::vector<ShapeVector> &origin_output_shapes) {
  MS_EXCEPTION_IF_NULL(origin_node);

  auto need_drop_input = common::AnfAlgo::GetBooleanAttr(origin_node, kAttrNeedDropInput);
  size_t input_num = need_drop_input ? 0 : common::AnfAlgo::GetInputTensorNum(origin_node);
  size_t output_num = origin_output_shapes.size();
  std::vector<size_t> input_mem_size(input_num);
  std::vector<size_t> output_mem_size(output_num);
  for (size_t i = 0; i < input_num; ++i) {
    auto ms_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(origin_node, i);
    input_mem_size[i] = SizeOf(ms_shape);
  }
  for (size_t i = 0; i < output_num; ++i) {
    auto ms_shape = origin_output_shapes[i];
    output_mem_size[i] = SizeOf(ms_shape);
  }
  auto send_mem_range = CalcAlltoAllVUnifyMindIRMemRange(origin_node, rank_size, input_mem_size, kAttrSendRankIds);
  auto recv_mem_range = CalcAlltoAllVUnifyMindIRMemRange(origin_node, rank_size, output_mem_size, kAttrRecvRankIds);
  return {send_mem_range, recv_mem_range};
}

std::vector<ShapeVector> GetAlltoAllVOutputShapes(const CNodePtr &all_to_all_v) {
  MS_EXCEPTION_IF_NULL(all_to_all_v);
  std::vector<ShapeVector> output_shapes;
  size_t output_num = common::AnfAlgo::GetOutputNumByAbstract(all_to_all_v->abstract());
  for (size_t i = 0; i < output_num; ++i) {
    auto shape = common::AnfAlgo::GetOutputInferShape(all_to_all_v, i);
    if (shape.empty()) {
      continue;
    }
    (void)output_shapes.emplace_back(shape);
  }
  return output_shapes;
}

AnfNodePtrList CreateFlattenReshapeNodes(const FuncGraphPtr &graph, const CNodePtr &all_to_all_v) {
  MS_EXCEPTION_IF_NULL(graph);
  auto all_to_all_v_inputs = all_to_all_v->inputs();
  AnfNodePtrList flatten_reshape_nodes;
  (void)std::transform(all_to_all_v_inputs.begin() + 1, all_to_all_v_inputs.end(),
                       std::back_inserter(flatten_reshape_nodes), [&graph, &all_to_all_v](const auto &input_node) {
                         MS_EXCEPTION_IF_NULL(input_node);
                         auto input_shape = common::AnfAlgo::GetOutputInferShape(input_node, kIndex0);
                         auto input_shape_size = SizeToLong(SizeOf(input_shape));
                         ShapeVector shape = {input_shape_size};
                         auto reshape = CreateReshapeNode(graph, input_node, shape);
                         reshape->set_scope(all_to_all_v->scope());
                         return reshape;
                       });
  return flatten_reshape_nodes;
}

CNodePtr CreateConcatNode(const KernelGraphPtr &graph, const AnfNodePtrList &input_nodes) {
  MS_EXCEPTION_IF_NULL(graph);
  if (input_nodes.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Need at least 1 input to create Concat node.";
  }

  // infer concat output shape
  int64_t concat_size = 0;
  for (const auto &input_node : input_nodes) {
    MS_EXCEPTION_IF_NULL(input_node);
    auto input_shape = common::AnfAlgo::GetOutputInferShape(input_node, kIndex0);
    // as the inputs of concat are flattened, accumulate at 0
    concat_size += input_shape.at(kIndex0);
  }
  ShapeVector shape = {concat_size};

  auto maketuple_node = CreateMakeTupleNode(graph, input_nodes);
  AnfNodePtrList concat_inputs = {NewValueNode(std::make_shared<Primitive>(kConcatOpName)), maketuple_node,
                                  graph->NewValueNode(MakeValue<int64_t>(0))};
  auto concat = NewCNode(concat_inputs, graph);
  MS_EXCEPTION_IF_NULL(concat);

  auto data_type = common::AnfAlgo::GetOutputInferDataType(input_nodes[kIndex0], kIndex0);
  common::AnfAlgo::SetOutputInferTypeAndShape({data_type}, {shape}, concat.get());
  return concat;
}

CNodePtr CreateAlltoAllVUnifyMindIRNode(const FuncGraphPtr &graph, const AnfNodePtr &input_node,
                                        const CNodePtr &origin_node,
                                        const std::vector<ShapeVector> &origin_output_shapes) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(origin_node);
  auto group = common::AnfAlgo::GetNodeAttr<std::string>(origin_node, kAttrGroup);
  std::vector<uint32_t> group_rank_ids =
    common::AnfAlgo::HasNodeAttr(kAttrGroupRankIds, origin_node)
      ? common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(origin_node, kAttrGroupRankIds)
      : std::vector<uint32_t>();
  auto rank_size = GetRankSize(group);
  auto [send_mem_range, recv_mem_range] = CalcAlltoAllVAttr(origin_node, rank_size, origin_output_shapes);

  AnfNodePtrList atav_inputs = {NewValueNode(std::make_shared<Primitive>(kAlltoAllVOpName)), input_node,
                                CreateShapeValueNode(graph, send_mem_range.counts),
                                CreateShapeValueNode(graph, recv_mem_range.counts)};
  auto atav_node = NewCNode(atav_inputs, graph);
  MS_EXCEPTION_IF_NULL(atav_node);
  atav_node->set_scope(origin_node->scope());
  atav_node->set_fullname_with_scope(origin_node->fullname_with_scope());
  common::AnfAlgo::SetNodeAttr(kAttrSendOffsetList, MakeValue(send_mem_range.displs), atav_node);
  common::AnfAlgo::SetNodeAttr(kAttrRecvOffsetList, MakeValue(recv_mem_range.displs), atav_node);
  common::AnfAlgo::SetNodeAttr(kAttrGroup, MakeValue<std::string>(group), atav_node);
  common::AnfAlgo::SetNodeAttr(kAttrBlockSize, MakeValue<int64_t>(1), atav_node);
  common::AnfAlgo::SetNodeAttr(kAttrGroupRankIds, MakeValue<std::vector<uint32_t>>(group_rank_ids), atav_node);
  if (common::AnfAlgo::HasNodeAttr(kAttrRecvType, origin_node)) {
    common::AnfAlgo::CopyNodeAttr(kAttrRecvType, origin_node, atav_node);
  }
  if (common::AnfAlgo::HasNodeAttr(parallel::COMM_REUSE, origin_node)) {
    common::AnfAlgo::CopyNodeAttr(parallel::COMM_REUSE, origin_node, atav_node);
  }
  if (common::AnfAlgo::HasNodeAttr("FLASH_INDEX", origin_node)) {
    common::AnfAlgo::CopyNodeAttr("FLASH_INDEX", origin_node, atav_node);
  }

  auto data_type = common::AnfAlgo::GetOutputInferDataType(origin_node, kIndex0);
  if (data_type == TypeId::kMetaTypeNone) {
    atav_node->set_abstract(origin_node->abstract());
  } else {
    int64_t flatten_size =
      std::accumulate(origin_output_shapes.cbegin(), origin_output_shapes.cend(), (int64_t)0,
                      [](const int64_t &acc, const ShapeVector &shape) { return acc + SizeToLong(SizeOf(shape)); });
    auto shape = flatten_size == 0 ? ShapeVector{} : ShapeVector{flatten_size};
    common::AnfAlgo::SetOutputInferTypeAndShape({data_type}, {shape}, atav_node.get());
  }
  MS_LOG(INFO) << "Create AlltoAllVUnifyMindIR node: " << atav_node->fullname_with_scope()
               << " success, send counts: " << send_mem_range.counts
               << ", send displacements: " << send_mem_range.displs << ", recv counts: " << recv_mem_range.counts
               << ", recv displacements: " << recv_mem_range.displs;
  return atav_node;
}

CNodePtr CreateSplitNode(const KernelGraphPtr &graph, const AnfNodePtr &input_node,
                         const std::vector<ShapeVector> &origin_output_shapes) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input_node);

  auto base_dtype = common::AnfAlgo::GetOutputInferDataType(input_node, kIndex0);
  std::vector<TypeId> data_types(origin_output_shapes.size(), base_dtype);
  std::vector<ShapeVector> shapes;
  ShapeVector split_lens;
  for (const auto &shape : origin_output_shapes) {
    auto shape_size = SizeToLong(SizeOf(shape));
    (void)split_lens.emplace_back(shape_size);
    (void)shapes.emplace_back(ShapeVector{shape_size});
  }

  AnfNodePtrList split_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitV->name())), input_node};
  auto split_node = NewCNode(split_inputs, graph);
  MS_EXCEPTION_IF_NULL(split_node);
  common::AnfAlgo::SetNodeAttr(kAttrSplitDim, MakeValue<int64_t>(0), split_node);
  common::AnfAlgo::SetNodeAttr(kAttrNumSplit, MakeValue<int64_t>(split_lens.size()), split_node);
  common::AnfAlgo::SetNodeAttr(kAttrSizeSplits, MakeValue(split_lens), split_node);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), split_node);
  common::AnfAlgo::SetOutputInferTypeAndShape(data_types, shapes, split_node.get());
  return split_node;
}

AnfNodePtrList CreateReshapeNodes(const FuncGraphPtr &graph, const AnfNodePtrList &input_nodes,
                                  const std::vector<ShapeVector> &origin_output_shapes) {
  MS_EXCEPTION_IF_NULL(graph);
  if (input_nodes.size() != origin_output_shapes.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "The number of input nodes to reshape must match shapes, but got "
                               << input_nodes.size() << " and " << origin_output_shapes.size();
  }

  AnfNodePtrList reshape_nodes;
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    (void)reshape_nodes.emplace_back(CreateReshapeNode(graph, input_nodes[i], origin_output_shapes[i]));
  }
  return reshape_nodes;
}
}  // namespace

std::vector<std::string> AlltoAllVUnifyMindIR::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimAlltoAllV->name());
  return ret;
}

const BaseRef AlltoAllVUnifyMindIR::DefinePattern() const {
  return VectorRef({prim::kPrimAlltoAllV, std::make_shared<SeqVar>()});
}

const AnfNodePtr AlltoAllVUnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto all_to_all_v = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(all_to_all_v);
  if (!common::AnfAlgo::HasNodeAttr(kAttrSendRankIds, all_to_all_v)) {
    return nullptr;
  }
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  auto flatten_reshape_nodes = CreateFlattenReshapeNodes(graph, all_to_all_v);
  AnfNodePtr atav_input;
  if (flatten_reshape_nodes.size() == 1) {
    atav_input = flatten_reshape_nodes[0];
  } else {
    auto concat_node = CreateConcatNode(kernel_graph, flatten_reshape_nodes);
    concat_node->set_scope(node->scope());
    atav_input = concat_node;
    OptimizerUtils::MoveContrlDepend(graph, all_to_all_v->input(1), concat_node);
  }
  // get the outputs shapes of origin node to restore outputs of new node
  auto origin_output_shapes = GetAlltoAllVOutputShapes(all_to_all_v);
  auto new_atav_node = CreateAlltoAllVUnifyMindIRNode(graph, atav_input, all_to_all_v, origin_output_shapes);
  // skip post processes if AlltoAllV has no output
  if (origin_output_shapes.empty()) {
    return new_atav_node;
  }
  AnfNodePtrList reshape_inputs;
  std::vector<CNodePtr> moved_depends;
  if (origin_output_shapes.size() == 1) {
    (void)reshape_inputs.emplace_back(new_atav_node);
  } else {
    OptimizerUtils::MoveContrlDepend(graph, node, new_atav_node);
    moved_depends = OptimizerUtils::MoveDataDepend(graph, node, new_atav_node);
    auto pre_node = new_atav_node;
    if (!moved_depends.empty()) {
      pre_node = moved_depends[0];
    }
    auto split_node = CreateSplitNode(kernel_graph, pre_node, origin_output_shapes);
    split_node->set_scope(node->scope());
    CreateMultipleOutputsOfAnfNode(graph, split_node, origin_output_shapes.size(), &reshape_inputs);
  }
  auto reshape_nodes = CreateReshapeNodes(graph, reshape_inputs, origin_output_shapes);
  auto maketuple_node = CreateMakeTupleNode(graph, reshape_nodes);
  maketuple_node->set_scope(node->scope());

  if (origin_output_shapes.size() != 1) {
    OptimizerUtils::ReplaceDataDepend(graph, moved_depends, maketuple_node);
  }

  return maketuple_node;
}
}  // namespace opt
}  // namespace mindspore
