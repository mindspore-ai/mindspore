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

#include "plugin/device/ascend/optimizer/ge/all_to_all_v_for_ge.h"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <tuple>
#include <algorithm>
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/comm_manager.h"
#include "include/backend/optimizer/helper.h"
#include "ops/array_ops.h"
#include "ops/other_ops.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kAttrModifiedForGE = "modified_for_ge";
constexpr auto kAttrIsInsertedByGE = "is_inserted_by_ge";

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

MemRange CalcAllToAllvForGEMemRange(const CNodePtr &origin_node, uint32_t rank_size,
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

std::tuple<MemRange, MemRange> CalcAllToAllvForGEInput(const CNodePtr &origin_node, uint32_t rank_size,
                                                       const std::vector<ShapeVector> &origin_output_shapes) {
  MS_EXCEPTION_IF_NULL(origin_node);

  auto need_drop_input = common::AnfAlgo::HasNodeAttr(kAttrNeedDropInput, origin_node) &&
                         common::AnfAlgo::GetNodeAttr<bool>(origin_node, kAttrNeedDropInput);
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
  auto send_mem_range = CalcAllToAllvForGEMemRange(origin_node, rank_size, input_mem_size, kAttrSendRankIds);
  auto recv_mem_range = CalcAllToAllvForGEMemRange(origin_node, rank_size, output_mem_size, kAttrRecvRankIds);
  return {send_mem_range, recv_mem_range};
}

std::vector<ShapeVector> GetAllToAllvOutputShapes(const CNodePtr &all_to_all_v) {
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

AnfNodePtrList CreateFlattenReshapeNodes(const FuncGraphPtr &graph, const AnfNodePtrList &input_nodes) {
  MS_EXCEPTION_IF_NULL(graph);
  AnfNodePtrList flatten_reshape_nodes;
  (void)std::transform(input_nodes.begin(), input_nodes.end(), std::back_inserter(flatten_reshape_nodes),
                       [&graph](const auto &input_node) {
                         MS_EXCEPTION_IF_NULL(input_node);
                         auto input_shape = common::AnfAlgo::GetOutputInferShape(input_node, kIndex0);
                         auto input_shape_size = SizeToLong(SizeOf(input_shape));
                         ShapeVector shape = {input_shape_size};
                         return mindspore::common::CreateReshapeNode(graph, input_node, shape);
                       });
  return flatten_reshape_nodes;
}

CNodePtr CreateConcatNode(const FuncGraphPtr &graph, const AnfNodePtrList &input_nodes) {
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
  AnfNodePtrList concat_inputs = {NewValueNode(std::make_shared<Primitive>(kConcatDOpName)), maketuple_node};
  auto concat = NewCNode(concat_inputs, graph);
  MS_EXCEPTION_IF_NULL(concat);

  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue<int64_t>(0), concat);
  auto data_type = common::AnfAlgo::GetOutputInferDataType(input_nodes[kIndex0], kIndex0);
  common::AnfAlgo::SetOutputInferTypeAndShape({data_type}, {shape}, concat.get());
  return concat;
}

CNodePtr CreateAllToAllvForGENode(const FuncGraphPtr &graph, const AnfNodePtr &input_node, const CNodePtr &origin_node,
                                  const std::vector<ShapeVector> &origin_output_shapes) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(origin_node);
  auto group = common::AnfAlgo::GetNodeAttr<std::string>(origin_node, kAttrGroup);
  auto rank_size = GetRankSize(group);
  auto [send_mem_range, recv_mem_range] = CalcAllToAllvForGEInput(origin_node, rank_size, origin_output_shapes);

  AnfNodePtrList atav_inputs = {NewValueNode(std::make_shared<Primitive>(kAllToAllvOpName)),
                                input_node,
                                mindspore::common::CreateShapeVectorNode(send_mem_range.counts),
                                mindspore::common::CreateShapeVectorNode(send_mem_range.displs),
                                mindspore::common::CreateShapeVectorNode(recv_mem_range.counts),
                                mindspore::common::CreateShapeVectorNode(recv_mem_range.displs)};
  auto atav_node = NewCNode(atav_inputs, graph);
  MS_EXCEPTION_IF_NULL(atav_node);
  atav_node->set_scope(origin_node->scope());
  atav_node->set_fullname_with_scope(origin_node->fullname_with_scope());
  common::AnfAlgo::CopyNodeAttrs(origin_node, atav_node);
  common::AnfAlgo::SetNodeAttr(kAttrModifiedForGE, MakeValue(true), atav_node);
  auto data_type = common::AnfAlgo::GetOutputInferDataType(origin_node, kIndex0);
  // ge do not support None data type
  if (data_type == TypeId::kMetaTypeNone) {
    data_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(origin_node, kIndex0);
  }
  int64_t flatten_size =
    std::accumulate(origin_output_shapes.cbegin(), origin_output_shapes.cend(), 0,
                    [](const int64_t &acc, const ShapeVector &shape) { return acc + SizeToLong(SizeOf(shape)); });
  auto shape = flatten_size == 0 ? ShapeVector{} : ShapeVector{flatten_size};
  common::AnfAlgo::SetOutputInferTypeAndShape({data_type}, {shape}, atav_node.get());
  MS_LOG(INFO) << "Create AllToAllvForGE node: " << atav_node->fullname_with_scope()
               << " success, send counts: " << send_mem_range.counts
               << ", send displacements: " << send_mem_range.displs << ", recv counts: " << recv_mem_range.counts
               << ", recv displacements: " << recv_mem_range.displs;
  return atav_node;
}

CNodePtr CreateSplitNode(const FuncGraphPtr &graph, const AnfNodePtr &input_node,
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

  AnfNodePtrList split_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitVD->name())), input_node};
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
    (void)reshape_nodes.emplace_back(
      mindspore::common::CreateReshapeNode(graph, input_nodes[i], origin_output_shapes[i]));
  }
  return reshape_nodes;
}
}  // namespace

std::vector<std::string> AllToAllvForGE::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimAllToAllv->name());
  return ret;
}

const BaseRef AllToAllvForGE::DefinePattern() const {
  return VectorRef({prim::kPrimAllToAllv, std::make_shared<SeqVar>()});
}

const AnfNodePtr AllToAllvForGE::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto all_to_all_v = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(all_to_all_v);
  if (common::AnfAlgo::HasNodeAttr(kAttrModifiedForGE, all_to_all_v) &&
      common::AnfAlgo::GetNodeAttr<bool>(all_to_all_v, kAttrModifiedForGE)) {
    return nullptr;
  }

  auto all_to_all_v_inputs = all_to_all_v->inputs();
  auto flatten_reshape_nodes =
    CreateFlattenReshapeNodes(graph, {all_to_all_v_inputs.begin() + 1, all_to_all_v_inputs.end()});
  auto concat_node = CreateConcatNode(graph, flatten_reshape_nodes);
  concat_node->set_scope(node->scope());
  // get the outputs shapes of origin node to restore outputs of new node
  auto origin_output_shapes = GetAllToAllvOutputShapes(all_to_all_v);
  auto new_atav_node = CreateAllToAllvForGENode(graph, concat_node, all_to_all_v, origin_output_shapes);
  // skip post processes if AllToAllv has no output
  if (origin_output_shapes.empty()) {
    common::AnfAlgo::SetNodeAttr(kAttrIsInsertedByGE, MakeValue(true), new_atav_node);
    return new_atav_node;
  }
  auto split_node = CreateSplitNode(graph, new_atav_node, origin_output_shapes);
  split_node->set_scope(node->scope());
  AnfNodePtrList split_outputs;
  CreateMultipleOutputsOfAnfNode(graph, split_node, origin_output_shapes.size(), &split_outputs);
  auto reshape_nodes = CreateReshapeNodes(graph, split_outputs, origin_output_shapes);
  auto maketuple_node = CreateMakeTupleNode(graph, reshape_nodes);
  maketuple_node->set_scope(node->scope());
  return maketuple_node;
}
}  // namespace opt
}  // namespace mindspore
