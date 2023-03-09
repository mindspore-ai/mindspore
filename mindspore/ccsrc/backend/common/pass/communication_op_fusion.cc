/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/communication_op_fusion.h"

#include <vector>
#include <set>
#include <memory>

#include "utils/hash_map.h"
#include "ir/graph_utils.h"
#include "mindspore/core/ops/core_ops.h"
#include "include/backend/kernel_info.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/kernel_build_info.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kAttrDefaultGroup = "default_group";
constexpr auto kAttrDefaultOp = "default_op";
constexpr size_t kAlignSize = 2 << 9;
constexpr int64_t kDefaultThresholdMb2Byte = 262144;

kernel::KernelBuildInfoPtr GenerateKernelBuildInfo(const CommunicationOpInfo &communication_op_info, size_t start_index,
                                                   size_t end_index) {
  if (end_index >= communication_op_info.communication_op_nodes.size()) {
    MS_LOG(EXCEPTION) << "end index out of communication_op_nodes size";
  }
  std::vector<std::string> inputs_device_format;
  std::vector<std::string> outputs_device_format;
  std::vector<TypeId> inputs_device_type;
  std::vector<TypeId> outputs_device_type;
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  for (size_t idx = start_index; idx <= end_index; ++idx) {
    auto cnode = communication_op_info.communication_op_nodes[idx];
    int64_t rank_size = 1;
    if (common::AnfAlgo::HasNodeAttr(kAttrRankSize, cnode) &&
        common::AnfAlgo::GetCNodeName(cnode) == kAllGatherOpName) {
      rank_size = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrRankSize);
    }
    if (rank_size == 0) {
      MS_LOG(EXCEPTION) << "Rank size should not be zero.";
    }
    MS_EXCEPTION_IF_NULL(cnode);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
    for (size_t input_index = 0; input_index < input_num; ++input_index) {
      inputs_device_format.push_back(AnfAlgo::GetInputFormat(cnode, input_index));
      inputs_device_type.push_back(AnfAlgo::GetInputDeviceDataType(cnode, input_index));
    }
    for (int64_t rank_index = 0; rank_index < rank_size; ++rank_index) {
      size_t output_num = AnfAlgo::GetOutputTensorNum(cnode);
      for (size_t output_index = 0; output_index < output_num; ++output_index) {
        outputs_device_format.push_back(AnfAlgo::GetOutputFormat(cnode, output_index));
        outputs_device_type.push_back(AnfAlgo::GetOutputDeviceDataType(cnode, output_index));
      }
    }
    builder.SetFusionType(AnfAlgo::GetFusionType(cnode));
    builder.SetProcessor(AnfAlgo::GetProcessor(cnode));
    builder.SetKernelType(AnfAlgo::GetKernelType(cnode));
  }
  builder.SetInputsFormat(inputs_device_format);
  builder.SetOutputsFormat(outputs_device_format);
  builder.SetInputsDeviceType(inputs_device_type);
  builder.SetOutputsDeviceType(outputs_device_type);
  return builder.Build();
}

std::string GetFusionGroupKey(const AnfNodePtr &node) {
  auto primitive = common::AnfAlgo::GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  ValuePtr attr_fusion = primitive->GetAttr(kAttrFusion);
  if (attr_fusion == nullptr) {
    return "";
  }
  auto fusion = GetValue<int64_t>(attr_fusion);
  if (fusion == 0) {
    return "";
  }
  std::string group = kAttrDefaultGroup;
  ValuePtr attr_group = primitive->GetAttr(kAttrGroup);
  if (attr_group != nullptr) {
    group = GetValue<std::string>(attr_group);
  }
  std::string op = kAttrDefaultOp;
  ValuePtr attr_op = primitive->GetAttr(kAttrOp);
  if (attr_op != nullptr) {
    op = GetValue<std::string>(attr_op);
  }
  auto dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0);
  return group + op + std::to_string(fusion) + TypeIdLabel(dtype);
}

void CheckInputs(const std::vector<AnfNodePtr> &fusion_inputs) {
  std::set<AnfNodePtr> inputs_set(fusion_inputs.begin(), fusion_inputs.end());
  if (inputs_set.size() < fusion_inputs.size()) {
    MS_LOG(EXCEPTION) << "Different communication op in one segment cannot share the same input";
  }
}

bool CheckSegments(size_t communication_op_node_size, const std::vector<size_t> *segment_index) {
  MS_EXCEPTION_IF_NULL(segment_index);
  auto segments = segment_index->size();
  if (segment_index->at(segments - 1) != communication_op_node_size - 1) {
    MS_LOG(EXCEPTION) << "the last segment index is invalid.";
  }
  for (size_t i = 0; i < segments - 1; ++i) {
    if (segment_index->at(i) > segment_index->at(i + 1)) {
      MS_LOG(EXCEPTION) << "illegal split: segment_index[" << i << "]=" << segment_index->at(i) << ", segment_index[ "
                        << (i + 1) << "]=" << segment_index->at(i + 1);
    }
  }
  return true;
}
}  // namespace

bool CommunicationOpFusion::GetSplitSegments(const CommunicationOpInfo &communication_op_info,
                                             std::vector<size_t> *segment_index, const std::string &group) const {
  MS_EXCEPTION_IF_NULL(segment_index);
  size_t communication_op_node_size = communication_op_info.communication_op_nodes.size();
  MS_LOG(INFO) << "graph " << op_name_ << " node size " << communication_op_node_size;

  if (op_name_ == kHcomSendOpName || op_name_ == kReceiveOpName) {
    if (communication_op_node_size == 0) {
      return false;
    }
    (void)segment_index->emplace_back(communication_op_node_size - 1);
    return true;
  }

  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  std::vector<uint32_t> split_indices;
  if (!parallel_context->enable_parallel_optimizer()) {
    split_indices = parallel_context->GetAllReduceFusionSplitIndices(group);
  }

  if (!split_indices.empty()) {
    uint32_t last_index = 0;
    for (size_t i = 0; i < split_indices.size(); ++i) {
      uint32_t index = split_indices[i];
      if (index <= last_index && i != 0) {
        MS_LOG(EXCEPTION) << "invalid " << op_name_ << " split index " << i << " " << index;
      }
      if (index >= communication_op_node_size) {
        MS_LOG(WARNING) << op_name_ << "'s split index " << index
                        << " is Greater than or equal to total gradient's number " << communication_op_node_size;
        continue;
      }
      segment_index->push_back(index);
      last_index = index;
    }
    if (last_index != communication_op_node_size - 1) {
      segment_index->push_back(communication_op_node_size - 1);
    }
  } else {
    for (size_t i = 0; i < groups_ - 1; ++i) {
      segment_index->push_back((i + 1) * (communication_op_node_size / groups_) - 1);
    }
    segment_index->push_back(communication_op_node_size - 1);
  }
  auto parallel_mode = parallel_context->parallel_mode();
  if (parallel_mode == parallel::kDataParallel && op_name_ == kAllReduceOpName) {
    auto threshold = parallel_context->dp_fusion_threshold_mb();
    GetAllReduceSplitSegment(communication_op_info.communication_op_nodes, threshold, segment_index);
    MS_LOG(INFO) << "The split threshold for AllReduce is " << threshold << ", the segment num is "
                 << segment_index->size();
  }
  return CheckSegments(communication_op_node_size, segment_index);
}

void CommunicationOpFusion::GetAllReduceSplitSegment(const std::vector<CNodePtr> &nodes, int64_t threshold,
                                                     std::vector<size_t> *segment_index) const {
  MS_EXCEPTION_IF_NULL(segment_index);
  if (threshold < 0) {
    MS_LOG(INFO) << "Split threshold is " << threshold << ". AllReduce nodes will take default fusion strategy.";
    return;
  }
  threshold *= kDefaultThresholdMb2Byte;
  std::vector<size_t> real_segment_index;
  size_t start_index = 0;
  for (auto index : *segment_index) {
    if (index >= nodes.size()) {
      MS_LOG(WARNING) << "split index is greater than or equal to total gradient's number " << nodes.size();
      continue;
    }
    size_t accumulate = 0;
    for (size_t j = start_index; j <= index; ++j) {
      auto tensor_size = AnfAlgo::GetOutputTensorMemSize(nodes[j], 0);
      if (accumulate + tensor_size > LongToSize(threshold)) {
        real_segment_index.push_back(j);
        accumulate = 0;
      } else {
        accumulate += tensor_size;
      }
    }
    if (accumulate != 0) {
      real_segment_index.push_back(index);
    }
    start_index = index + 1;
  }
  *segment_index = std::move(real_segment_index);
}

// Hard coded Load(%paraxxx, cnode()) to Load(%paraxxx, U) to prevent
// cycle after AllReduce fused. It's a workaround.
// case 1:
// cnode_load = Load(%para2, cnode_u)
// %100 = UpdateState(cnode_u, cnode_load)
// ...
// %109 = AssignAdd(%para485, Tensor(34), %100)
// %110 = UpdateState(%100, xxx)
// will convert to:
// cnode_load = Load(%para2, U)
// ...
// %109 = AssignAdd(%para485, Tensor(34), cnode_u)
// %110 = UpdateState(cnode_u, xxx)
//
// case 2:
// cnode_load = Load(%para2, cnode_u)
// %99 = make_tuple(yyy, ..., cnode_load, ...)
// %100 = UpdateState(cnode_u, %99)
// ...
// %109 = AssignAdd(%para485, Tensor(34), %100)
// %110 = UpdateState(%100, xxx)
// will convert to:
// cnode_load = Load(%para2, U)
// %99 = make_tuple(yyy, ...)
// %100 = UpdateState(cnode_u, %99)
// ...
// %109 = AssignAdd(%para485, Tensor(34), %100)
// %110 = UpdateState(%100, xxx)
//
// case 3:
// cnode_load = Load(%para2, cnode_u)
// %99 = make_tuple(cnode_load)
// %100 = UpdateState(cnode_u, %99)
// ...
// %109 = AssignAdd(%para485, Tensor(34), %100)
// %110 = UpdateState(%100, xxx)
// will convert to:
// cnode_load = Load(%para2, U)
// ...
// %109 = AssignAdd(%para485, Tensor(34), cnode_u)
// %110 = UpdateState(cnode_u, xxx)
static void AdjustAllReduceInputWithLoad(const CNodePtr &cnode) {
  const size_t monad_index = 2;
  const size_t tuple_inputs_size = 2;
  const size_t load_inputs_size = 3;
  auto cnode_load = BroadFirstSearchFirstOf({cnode}, [&](const CNodePtr &search_cnode) {
    if (!IsPrimitiveCNode(search_cnode, prim::kPrimLoad)) {
      return false;
    }
    if (search_cnode->inputs().size() != load_inputs_size) {
      MS_LOG(EXCEPTION) << "Load CNode should have 3 inputs, but: " << search_cnode->DebugString();
    }
    return search_cnode->input(monad_index)->isa<CNode>();
  });
  if (cnode_load != nullptr) {
    auto const_u_monad = NewValueNode(kUMonad);
    const_u_monad->set_abstract(kUMonad->ToAbstract());
    const auto &cnode_u = cnode_load->input(monad_index);
    MS_LOG(DEBUG) << "Replace Load with CNode U to constant U for cnode: " << cnode_load->DebugString();
    MS_EXCEPTION_IF_NULL(cnode->func_graph());
    MS_EXCEPTION_IF_NULL(cnode->func_graph()->manager());
    auto manager = cnode->func_graph()->manager();
    manager->SetEdge(cnode_load, monad_index, const_u_monad);
    // Update the u_monad input of UpdateState from CNode U same as Load to constant U.
    CNodePtr cnode_update_state = nullptr;
    CNodePtr cnode_make_tuple = nullptr;
    const auto &cnode_load_users = manager->node_users()[cnode_load];
    for (auto &load_user : cnode_load_users) {
      if (IsPrimitiveCNode(load_user.first, prim::kPrimMakeTuple)) {
        const auto &cnode_make_tuple_users = manager->node_users()[load_user.first];
        for (auto &make_tuple_user : cnode_make_tuple_users) {
          if (IsPrimitiveCNode(make_tuple_user.first, prim::kPrimUpdateState)) {
            const auto &cnode_user = make_tuple_user.first->cast<CNodePtr>();
            if (cnode_user->input(1) == cnode_u) {
              cnode_update_state = cnode_user;
              cnode_make_tuple = load_user.first->cast<CNodePtr>();
              break;
            }
          }
        }
        if (cnode_update_state != nullptr) {
          break;
        }
      }
      if (IsPrimitiveCNode(load_user.first, prim::kPrimUpdateState)) {
        const auto &cnode_user = load_user.first->cast<CNodePtr>();
        if (cnode_user->input(1) == cnode_u) {
          cnode_update_state = cnode_user;
          break;
        }
      }
    }
    if (cnode_update_state != nullptr) {
      if (cnode_make_tuple == nullptr || cnode_make_tuple->inputs().size() == tuple_inputs_size) {
        // case 1 and case 3: Replace cnode_update_state to cnode_u;
        MS_LOG(DEBUG) << "Replace UpdateState with CNode U: " << cnode_update_state->DebugString()
                      << " ::TO:: " << cnode_u->DebugString();
        manager->Replace(cnode_update_state, cnode_u);
      } else if (cnode_make_tuple->inputs().size() > tuple_inputs_size) {
        // case 2: remove cnode_load from cnode_make_tuple;
        MS_LOG(DEBUG) << "Drop " << cnode_load->DebugString() << " from " << cnode_make_tuple->DebugString();
        const auto &make_tuple_inputs = cnode_make_tuple->inputs();
        AnfNodePtrList new_tuple_inputs(make_tuple_inputs.size() - 1);
        std::copy_if(make_tuple_inputs.cbegin(), make_tuple_inputs.cend(), new_tuple_inputs.begin(),
                     [cnode_load](const auto &inp) { return inp != cnode_load; });
        auto new_cnode_make_tuple = cnode_make_tuple->func_graph()->NewCNode(new_tuple_inputs);
        manager->Replace(cnode_make_tuple, new_cnode_make_tuple);
      } else {
        MS_LOG(EXCEPTION) << "Cannot replace UpdateState with CNode U: " << cnode_update_state->DebugString()
                          << " as make_tuple CNode cannot match " << cnode_make_tuple->DebugString();
      }
    }
  }
}

AnfNodePtr CommunicationOpFusion::CreateFusedCommunicationOp(const FuncGraphPtr &func_graph,
                                                             const CommunicationOpInfo &communication_op_info,
                                                             size_t start_index, size_t end_index) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto prim = std::make_shared<Primitive>(op_name_);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> fusion_inputs = {NewValueNode(prim)};
  // get all inputs of current segment
  if (end_index >= communication_op_info.communication_op_nodes.size()) {
    MS_LOG(EXCEPTION) << "End index is out of communication_op_nodes size";
  }
  std::vector<AnfNodePtr> orig_nodes;
  for (size_t idx = start_index; idx <= end_index; ++idx) {
    auto cnode = communication_op_info.communication_op_nodes[idx];
    MS_EXCEPTION_IF_NULL(cnode);
    if (idx != start_index) {
      AdjustAllReduceInputWithLoad(cnode);
    }
    (void)fusion_inputs.insert(fusion_inputs.cend(), cnode->inputs().cbegin() + 1, cnode->inputs().cend());
    (void)orig_nodes.emplace_back(cnode);
  }
  CheckInputs(fusion_inputs);
  AnfNodePtr fused_node = NewCNode(fusion_inputs, func_graph, orig_nodes);
  MS_EXCEPTION_IF_NULL(fused_node);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  fused_node->set_kernel_info(kernel_info);
  auto final_node = communication_op_info.communication_op_nodes[end_index];
  size_t node_num = end_index - start_index + 1;
  int64_t rank_size = 1;
  if (common::AnfAlgo::HasNodeAttr(kAttrRankSize, final_node) &&
      common::AnfAlgo::GetCNodeName(final_node) == kAllGatherOpName) {
    rank_size = common::AnfAlgo::GetNodeAttr<int64_t>(final_node, kAttrRankSize);
  }

  if (rank_size == 0) {
    MS_LOG(EXCEPTION) << "Rank size should not be zero.";
  }
  size_t output_num = node_num * LongToSize(rank_size);
  std::vector<TypeId> dtypes(output_num, common::AnfAlgo::GetOutputInferDataType(final_node, 0));
  std::vector<ShapeVector> shapes;
  int64_t fusion_total_size = 0;
  for (int64_t i = 0; i < rank_size; ++i) {
    for (size_t idx = start_index; idx <= end_index; ++idx) {
      auto input_node = communication_op_info.communication_op_nodes[idx];
      MS_EXCEPTION_IF_NULL(input_node);
      auto shape = common::AnfAlgo::GetOutputInferShape(input_node, 0);
      if (!shape.empty()) {
        shape[0] /= rank_size;
      }
      shapes.push_back(shape);
      size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(input_node, 0);
      TypeId output_type = AnfAlgo::GetOutputDeviceDataType(input_node, 0);
      size_t type_size = GetTypeByte(TypeIdToType(output_type));
      if (type_size == 0) {
        MS_LOG(EXCEPTION) << "Divisor 'type_size' should not be 0.";
      }
      tensor_size = (tensor_size / kAlignSize + 1) * kAlignSize / type_size;
      fusion_total_size += static_cast<int64_t>(tensor_size);
    }
  }
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, fused_node.get());
  auto kernel_build_info = GenerateKernelBuildInfo(communication_op_info, start_index, end_index);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info, fused_node.get());
  const std::vector<std::string> kHcclFusionAttrs = {
    kAttrFusion, kAttrGroup, kAttrGroupBack, kAttrSrTag,        kAttrDestRank,          kAttrSrcRank,
    kAttrDType,  kAttrOp,    kAttrRankSize,  kAttrGroupRankIds, kAttrReuseCommunication};
  for (const auto &attr : kHcclFusionAttrs) {
    if (common::AnfAlgo::HasNodeAttr(attr, final_node)) {
      common::AnfAlgo::CopyNodeAttr(attr, final_node, fused_node);
    }
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrShape, final_node)) {
    std::vector<int64_t> fusion_total_shape{fusion_total_size};
    common::AnfAlgo::SetNodeAttr(kAttrShape, MakeValue(fusion_total_shape), fused_node);
  }
  bool is_recompute =
    final_node->GetAttr(kAttrDuplicated) != nullptr && GetValue<bool>(final_node->GetAttr(kAttrDuplicated));
  if (common::AnfAlgo::GetCNodeName(final_node) == kAllGatherOpName && is_recompute) {
    auto fused_cnode = fused_node->cast<CNodePtr>();
    fused_cnode->AddAttr("duplicated", MakeValue(true));
    auto fused_prim = GetCNodePrimitive(fused_cnode);
    auto final_node_prim = GetCNodePrimitive(final_node);
    fused_prim->set_instance_name(final_node_prim->instance_name());
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrNotDelayFusion, final_node)) {
    common::AnfAlgo::CopyNodeAttr(kAttrNotDelayFusion, final_node, fused_node);
  }
  return fused_node;
}

bool CommunicationOpFusion::DoFusion(const FuncGraphPtr &func_graph, const CommunicationOpInfo &communication_op_info,
                                     const std::vector<size_t> &segment_index) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  bool changed = false;
  size_t start_index = 0;
  for (size_t segment_idx = 0; segment_idx < segment_index.size(); ++segment_idx) {
    size_t end_index = segment_index.at(segment_idx);
    if (end_index - start_index < 1) {
      start_index = end_index + 1;
      continue;
    }
    auto kernel_graph = func_graph->cast<KernelGraphPtr>();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto graph_id = kernel_graph->graph_id();
    AnfNodePtr new_communication_op =
      CreateFusedCommunicationOp(func_graph, communication_op_info, start_index, end_index);
    AnfAlgo::SetGraphId(graph_id, new_communication_op.get());
    // replace old communication op with new communication op
    for (auto idx = start_index; idx <= end_index; ++idx) {
      std::vector<AnfNodePtr> tuple_getitem_input;
      tuple_getitem_input.push_back(NewValueNode(prim::kPrimTupleGetItem));
      tuple_getitem_input.push_back(new_communication_op);
      auto offset = SizeToLong(idx - start_index);
      auto index = NewValueNode(offset);
      MS_EXCEPTION_IF_NULL(index);
      auto imm = std::make_shared<Int64Imm>(idx - start_index);
      MS_EXCEPTION_IF_NULL(imm);
      auto abstract_scalar = std::make_shared<abstract::AbstractScalar>();
      MS_EXCEPTION_IF_NULL(abstract_scalar);
      index->set_abstract(abstract_scalar);
      tuple_getitem_input.push_back(index);
      AnfNodePtr tuple_getitem = func_graph->NewCNode(tuple_getitem_input);
      MS_EXCEPTION_IF_NULL(tuple_getitem);
      auto communication_op_node_item = communication_op_info.communication_op_nodes.at(idx);
      MS_EXCEPTION_IF_NULL(communication_op_node_item);
      tuple_getitem->set_abstract(communication_op_node_item->abstract());
      if (kernel_graph->IsInternalOutput(communication_op_node_item, 0)) {
        kernel_graph->ReplaceInternalOutput(communication_op_node_item, new_communication_op, 0, LongToSize(offset));
      }
      if (!manager->Replace(communication_op_node_item, tuple_getitem)) {
        MS_LOG(EXCEPTION) << "Manager replace node failed";
      }
    }
    start_index = end_index + 1;
    changed = true;
  }
  return changed;
}

bool CommunicationOpFusion::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto threshold = parallel_context->dp_fusion_threshold_mb();
  if (threshold == 0) {
    return false;
  }
  const float input_grad_size_num = 0.0;
  const float input_grad_time_num = 0.0;
  // divide candidate fusion groups with same (group,op,fusion,dtype) attrs, fusion==0 means not fusion
  mindspore::HashMap<std::string, CommunicationOpInfo> candidate_groups;
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (node != nullptr && node->isa<CNode>() && common::AnfAlgo::GetCNodeName(node) == op_name_) {
      std::string key = GetFusionGroupKey(node);
      if (key.empty()) {
        continue;
      }
      if (candidate_groups.find(key) == candidate_groups.end()) {
        CommunicationOpInfo communication_op_info;
        candidate_groups[key] = communication_op_info;
      }
      candidate_groups[key].communication_op_nodes.push_back(node->cast<CNodePtr>());
      candidate_groups[key].input_grad_size.push_back(input_grad_size_num);
      candidate_groups[key].input_grad_time.push_back(input_grad_time_num);
    }
  }
  // split candidate group to segments according to _group class member
  bool changed = false;
  for (auto &it : candidate_groups) {
    if (it.second.communication_op_nodes.size() <= 1) {
      continue;
    }
    auto first_node = it.second.communication_op_nodes[0];
    TraceGuard guard(std::make_shared<TraceOpt>(first_node->debug_info()));
    if (common::AnfAlgo::HasNodeAttr(kAttrIndex, first_node) &&
        common::AnfAlgo::GetNodeAttr<int64_t>(first_node, kAttrIndex) > 0) {
      std::stable_sort(it.second.communication_op_nodes.begin(), it.second.communication_op_nodes.end(),
                       [](const CNodePtr &a, const CNodePtr &b) {
                         return common::AnfAlgo::GetNodeAttr<int64_t>(a, kAttrIndex) <
                                common::AnfAlgo::GetNodeAttr<int64_t>(b, kAttrIndex);
                       });
    }
    std::vector<size_t> segment_index;
    if (GetSplitSegments(it.second, &segment_index, it.first)) {
      if (DoFusion(func_graph, it.second, segment_index)) {
        changed = true;
      }
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
