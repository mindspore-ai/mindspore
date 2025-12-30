/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/graph_optimizer/stream_assign/acl_stream_assign.h"
#include <algorithm>
#include <unordered_set>
#include <utility>
#include <set>
#include <string>
#include <memory>
#include <map>
#include <vector>
#include <tuple>

#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/utils/anfalgo.h"
#include "include/utils/parallel_context.h"
#include "include/utils/utils.h"
#include "include/runtime/utils/runtime_conf/runtime_env.h"
#include "ir/anf.h"
#include "primitive/ascend_op_name.h"
#include "primitive/framework_op_name.h"
#include "plugin/ascend/graph_optimizer/gpto/gpto.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "ops_utils/op_utils.h"
#include "utils/compile_config.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
void SetForSwitchInline(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &send_cnode,
                        const CNodePtr &recv_cnode, const AnfNodePtr &pre_node, const AnfNodePtr &next_node) {
  if (pre_node == nullptr || next_node == nullptr) {
    return;
  }
  std::string branch_name = "";
  const auto &node_before_iter = kernel_graph->inline_sub_graph_kernels().find(pre_node);
  if (node_before_iter != kernel_graph->inline_sub_graph_kernels().end()) {
    branch_name = node_before_iter->second;
  } else {
    const auto &node_after_iter = kernel_graph->inline_sub_graph_kernels().find(next_node);
    if (node_after_iter != kernel_graph->inline_sub_graph_kernels().end()) {
      branch_name = node_after_iter->second;
    }
  }
  if (branch_name == "") {
    return;
  }
  kernel_graph->AddInlineSubgraphKernel(send_cnode, branch_name);
  MS_LOG(DEBUG) << "Add inline subgraph send kernel:" << send_cnode->fullname_with_scope()
                << " by before send node:" << pre_node->fullname_with_scope() << " branch name:" << branch_name
                << " for kernel graph:" << kernel_graph->ToString();

  kernel_graph->AddInlineSubgraphKernel(recv_cnode, branch_name);
  MS_LOG(DEBUG) << "Add inline subgraph recv kernel:" << recv_cnode->fullname_with_scope()
                << " by after receive node:" << next_node->fullname_with_scope() << " branch name:" << branch_name
                << " for kernel graph:" << kernel_graph->ToString();
}

void AddStreamIdForCommunicationOp(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  AnfAlgo::SetStreamId(kWorldGroupStreamIndex, node.get());
  common::AnfAlgo::SetNodeAttrSafely(kAttrStreamId, MakeValue<uint32_t>(kWorldGroupStreamIndex), node);
}

void AssignStreamForCopyTo(const AnfNodePtr &node) {
  auto copy_to_stream = AscendStreamMng::GetInstance().GetCopyToStream();
  size_t copy_to_stream_id;
  if (copy_to_stream == nullptr) {
    AscendStreamMng::GetInstance().CreateStream(&copy_to_stream_id);
    MS_LOG(INFO) << "Create ascend copy to stream, stream id: " << copy_to_stream_id;
    copy_to_stream = AscendStreamMng::GetInstance().GetStream(copy_to_stream_id);
    AscendStreamMng::GetInstance().SetCopyToStream(copy_to_stream);
  }
  copy_to_stream_id = AscendStreamMng::GetInstance().GetStreamId(copy_to_stream);
  AnfAlgo::SetStreamId(copy_to_stream_id, node.get());
  common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue<uint32_t>(copy_to_stream_id), node);
  common::AnfAlgo::SetNodeAttr(kAttrCopyStreamId, MakeValue<uint32_t>(copy_to_stream_id), node);
}

void AssignStreamForCopyOut(const AnfNodePtr &node) {
  auto copy_out_stream = AscendStreamMng::GetInstance().GetCopyOutStream();
  size_t copy_out_stream_id;
  if (copy_out_stream == nullptr) {
    AscendStreamMng::GetInstance().CreateStream(&copy_out_stream_id);
    MS_LOG(INFO) << "Create ascend copy out stream, stream id: " << copy_out_stream_id;
    copy_out_stream = AscendStreamMng::GetInstance().GetStream(copy_out_stream_id);
    AscendStreamMng::GetInstance().SetCopyOutStream(copy_out_stream);
  }
  copy_out_stream_id = AscendStreamMng::GetInstance().GetStreamId(copy_out_stream);
  AnfAlgo::SetStreamId(copy_out_stream_id, node.get());
  common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue<uint32_t>(copy_out_stream_id), node);
  common::AnfAlgo::SetNodeAttr(kAttrCopyStreamId, MakeValue<uint32_t>(copy_out_stream_id), node);
}

void AssignStreamForCopyIn(const AnfNodePtr &node) {
  auto copy_in_stream = AscendStreamMng::GetInstance().GetCopyInStream();
  size_t copy_in_stream_id;
  if (copy_in_stream == nullptr) {
    AscendStreamMng::GetInstance().CreateStream(&copy_in_stream_id);
    MS_LOG(INFO) << "Create ascend copy in stream, stream id: " << copy_in_stream_id;
    copy_in_stream = AscendStreamMng::GetInstance().GetStream(copy_in_stream_id);
    AscendStreamMng::GetInstance().SetCopyInStream(copy_in_stream);
  }
  copy_in_stream_id = AscendStreamMng::GetInstance().GetStreamId(copy_in_stream);
  AnfAlgo::SetStreamId(copy_in_stream_id, node.get());
  common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue<uint32_t>(copy_in_stream_id), node);
  common::AnfAlgo::SetNodeAttr(kAttrCopyStreamId, MakeValue<uint32_t>(copy_in_stream_id), node);
}

void AssignStreamForMoveTo(const AnfNodePtr &node) {
  const auto &dst_str = common::AnfAlgo::GetMoveToDstStr(node);
  if (dst_str == kToCpu) {
    AssignStreamForCopyOut(node);
  } else if (dst_str == kToNpu) {
    AssignStreamForCopyIn(node);
  } else {
    MS_LOG(EXCEPTION) << "Get error MoveTo dst string: " << dst_str;
  }
}

std::string GetGroupName(const CNodePtr &cnode) {
  auto prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  if (common::AnfAlgo::HasNodeAttr(kAttrGroup, cnode)) {
    auto group_value = prim->GetAttr(kAttrGroup);
    if (group_value == nullptr) {
      MS_LOG(EXCEPTION) << "Group value is nullptr, node: " << cnode->fullname_with_scope();
    }
    if (group_value->isa<StringImm>()) {
      return common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrGroup);
    }
  } else {
    auto name = common::AnfAlgo::GetCNodeName(cnode);
    size_t input_size = ops::GetOpInputsNum(name);
    size_t group_idx = ops::GetInputIndexByName(name, kAttrGroup);
    if (group_idx != SIZE_MAX) {
      size_t real_input_num = common::AnfAlgo::GetInputTensorNum(cnode);
      // for all_gather and reducescatter, input has list tensor.
      group_idx = group_idx + real_input_num - input_size;
      auto group_node = common::AnfAlgo::GetInputNode(cnode, group_idx);
      auto group = GetScalarValue<std::string>(group_node->cast<ValueNodePtr>()->value());
      if (group.has_value()) {
        MS_LOG(DEBUG) << "Group name is " << group.value();
        return group.value();
      } else {
        MS_LOG(EXCEPTION) << "Group value is nullptr, node: " << cnode->fullname_with_scope();
      }
    }
  }
  return "";
}

bool IsUsersSetStreamsOp(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  if (common::AnfAlgo::IsCommunicationOp(node)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode->HasAttr(kAttrStreamId)) {
    int64_t stream_id = GetValue<int64_t>(cnode->GetAttr(kAttrStreamId));
    if (stream_id != kDefaultStreamIndex) {
      MS_LOG(DEBUG) << "User set attr stream id for node " << node->fullname_with_scope() << ", stream id is "
                    << stream_id;
      return true;
    }
  }
  return false;
}

bool IsUsersSetResLimitOp(const AnfNodePtr &node, const string &attr, uint32_t *pnum) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode->HasAttr(attr)) {
    int64_t num = GetValue<int64_t>(cnode->GetAttr(attr));
    MS_LOG(DEBUG) << "User set res limit num for node " << node->fullname_with_scope() << ", core num is " << num;
    if (num > 0) {
      *pnum = static_cast<uint32_t>(num);
      return true;
    }
  }
  return false;
}

void AddStreamIdForUsersSetStreamsOp(const AnfNodePtr &node, mindspore::HashMap<int64_t, size_t> *stream_map) {
  // Add backend stream_id for operators where users have specified streams.
  auto cnode = node->cast<CNodePtr>();
  int64_t stream_id = GetValue<int64_t>(cnode->GetAttr(kAttrStreamId));
  size_t new_stream_id;
  const auto &iter = stream_map->find(stream_id);
  if (iter != stream_map->end()) {
    new_stream_id = iter->second;
  } else {
    AscendStreamMng::GetInstance().CreateStream(&new_stream_id);
    (*stream_map)[stream_id] = new_stream_id;
    MS_LOG(INFO) << "Create ascend copy out stream, stream id: " << new_stream_id;
  }
  AnfAlgo::SetStreamId(new_stream_id, node.get());
  common::AnfAlgo::SetNodeAttrSafely(kAttrStreamId, MakeValue<uint32_t>(new_stream_id), node);
  MS_LOG(INFO) << "Set stream_id: " << new_stream_id << " by no group for node " << node->fullname_with_scope();
}

void AddStreamIdByGroup(const AnfNodePtr &node, DeviceResManager *device_res_manager,
                        mindspore::HashMap<int64_t, size_t> *stream_map) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Node is not a cnode: " << node->DebugString();
  }
  auto cnode = node->cast<CNodePtr>();
  if (!common::AnfAlgo::IsCommunicationOp(cnode)) {
    if (IsUsersSetStreamsOp(node)) {
      AddStreamIdForUsersSetStreamsOp(node, stream_map);
      MS_LOG(INFO) << "Set stream id by default for node " << node->fullname_with_scope()
                   << ", because it is users set stream operator.";
    } else if (IsOneOfPrimitiveCNode(node,
                                     {prim::kPrimCopyToHost, prim::kPrimCopyToHostExt, prim::kPrimCopyToDevice})) {
      AssignStreamForCopyTo(node);
    } else if (IsPrimitiveCNode(node, prim::kPrimMoveTo) || IsPrimitiveCNode(node, prim::kPrimMoveAssign)) {
      AssignStreamForMoveTo(node);
    } else {
      AnfAlgo::SetStreamId(kDefaultStreamIndex, node.get());
      common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue<uint32_t>(kDefaultStreamIndex), node);
    }
  } else if (common::AnfAlgo::IsLcclCommunicationOp(cnode)) {
    AnfAlgo::SetStreamId(kDefaultStreamIndex, node.get());
    common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue<uint32_t>(kDefaultStreamIndex), node);
    MS_LOG(INFO) << "Set stream id by default for node " << node->fullname_with_scope()
                 << ", because it is an LCCL operator.";
  } else {
    auto group_name = GetGroupName(cnode);
    if (!group_name.empty()) {
      size_t comm_stream_id = device_res_manager->GetCommunicationStreamIDByGroup(group_name);
      AnfAlgo::SetStreamId(comm_stream_id, node.get());
      common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue<uint32_t>(comm_stream_id), node);
      MS_LOG(INFO) << "Set stream id by group " << comm_stream_id << " for node " << node->fullname_with_scope()
                   << ", group: " << group_name;
    } else {
      AnfAlgo::SetStreamId(kDefaultStreamIndex, node.get());
      common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue<uint32_t>(kDefaultStreamIndex), node);
      MS_LOG(INFO) << "Set stream id by default for node " << node->fullname_with_scope()
                   << ", because group value is not string.";
    }
  }
}

void AddEventForUsersSetEvent(const AnfNodePtr &node, std::map<uint32_t, uint32_t> *event_map,
                              std::map<uint32_t, std::pair<aclrtEvent, uint32_t>> *event_info_map) {
  // Add backend event_id and aclrtEvent for operators where users have specified events.
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(event_map);
  MS_EXCEPTION_IF_NULL(event_info_map);

  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Node is not a cnode: " << node->DebugString();
  }
  const auto &cnode = node->cast<CNodePtr>();
  const auto &name = common::AnfAlgo::GetCNodeName(cnode);
  if (name != kStreamSendOpName && name != kStreamRecvOpName) {
    return;
  }
  if (cnode->GetAttr(kAttrEventId) == nullptr) {
    return;
  }

  auto &resource_manager = AscendStreamMng::GetInstance();
  const uint32_t event_id = GetValue<uint32_t>(cnode->GetAttr(kAttrEventId));
  uint32_t new_event_id;
  aclrtEvent event = nullptr;

  const auto iter = event_map->find(event_id);
  if (iter != event_map->end()) {
    new_event_id = iter->second;
    auto info_iter = event_info_map->find(new_event_id);
    MS_EXCEPTION_IF_NULL(info_iter->second.first);
    event = info_iter->second.first;
  } else {
    new_event_id = resource_manager.ApplyNewEvent();
    event = resource_manager.ApplyRtEvent();
    event_map->emplace(event_id, new_event_id);
    event_info_map->emplace(new_event_id, std::make_pair(event, UINT32_MAX));
    MS_LOG(INFO) << "Create ascend copy out event, event id: " << new_event_id;
  }
  common::AnfAlgo::SetNodeAttrSafely(kAttrEventId, MakeValue(new_event_id), node);
  MS_LOG(INFO) << "Set event_id: " << new_event_id << " for user set event node " << node->fullname_with_scope();

  if (name == kStreamSendOpName) {
    // Record event_id to stream_id corresponding to user-inserted 'event.record', and store them in event_info_map.
    // This is for subsequent use of record_stream_id when processing user-inserted 'event.wait'.
    auto prim = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    auto stream_value = prim->GetAttr(kAttrStreamId);
    MS_EXCEPTION_IF_NULL(stream_value);
    auto stream_id = GetValue<uint32_t>(stream_value);
    (*event_info_map)[new_event_id].second = stream_id;
    common::AnfAlgo::SetNodeAttrSafely(kAttrRecordEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), node);
  } else {
    // Add record_stream_id attribute to 'event.wait', which is the stream_id of 'event.record' stored in
    // event_info_map. If no StreamSend with the same event_id is found before StreamRecv, skip this streamRecv.
    auto &event_info = (*event_info_map)[new_event_id];
    if (event_info.second != UINT32_MAX) {
      auto record_stream_id = event_info.second;
      common::AnfAlgo::SetNodeAttrSafely(kAttrRecordEventStream, MakeValue(static_cast<uint32_t>(record_stream_id)),
                                         node);
      common::AnfAlgo::SetNodeAttrSafely(kAttrWaitEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), node);
    } else {
      MS_LOG(WARNING) << "No paired StreamSend with event_id: " << new_event_id
                      << " found, the launch of this node: " << node->fullname_with_scope() << " will be skipped.";
    }
  }
}

void SetRemoteCopyIdForKernelGraph(const KernelGraphPtr &kernel_graph) {
  static const bool use_hyper_offload =
    common::GetEnv("MS_DEV_HYPER_OFFLOAD") == "1" && common::GetCompileConfig("ENABLE_HYPER_OFFLOAD_SLIDE") == "1";
  if (!use_hyper_offload) {
    return;
  }
  // Create Copy In Stream
  auto remote_copy_in_stream = AscendStreamMng::GetInstance().GetRemoteCopyInStream();
  size_t remote_copy_in_stream_id;
  if (remote_copy_in_stream == nullptr) {
    AscendStreamMng::GetInstance().CreateStream(&remote_copy_in_stream_id);
    MS_LOG(INFO) << "Create ascend remote copy in stream, stream id: " << remote_copy_in_stream_id;
    remote_copy_in_stream = AscendStreamMng::GetInstance().GetStream(remote_copy_in_stream_id);
    AscendStreamMng::GetInstance().SetCopyInStream(remote_copy_in_stream);
  }
  remote_copy_in_stream_id = AscendStreamMng::GetInstance().GetStreamId(remote_copy_in_stream);
  kernel_graph->SetRemoteCopyInStreamId(remote_copy_in_stream_id);
  // Create Copy Out Stream
  auto remote_copy_out_stream = AscendStreamMng::GetInstance().GetRemoteCopyOutStream();
  size_t remote_copy_out_stream_id;
  if (remote_copy_out_stream == nullptr) {
    AscendStreamMng::GetInstance().CreateStream(&remote_copy_out_stream_id);
    MS_LOG(INFO) << "Create ascend remote copy out stream, stream id: " << remote_copy_out_stream_id;
    remote_copy_out_stream = AscendStreamMng::GetInstance().GetStream(remote_copy_out_stream_id);
    AscendStreamMng::GetInstance().SetCopyOutStream(remote_copy_out_stream);
  }
  remote_copy_out_stream_id = AscendStreamMng::GetInstance().GetStreamId(remote_copy_out_stream);
  kernel_graph->SetRemoteCopyOutStreamId(remote_copy_out_stream_id);
  kernel_graph->set_enable_multi_stream(true);
}
}  // namespace

CNodePtr AclStreamAssign::CreateLimitApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr,
                                                 const mindspore::HashMap<std::string, uint32_t> &res_limit_map) {
  auto limit_op = std::make_shared<Primitive>(kResLimitOpName);
  MS_EXCEPTION_IF_NULL(limit_op);
  auto limit_apply = std::make_shared<ValueNode>(limit_op);
  MS_EXCEPTION_IF_NULL(limit_apply);
  auto limit_node_ptr = graph_ptr->NewCNode({limit_apply});
  MS_EXCEPTION_IF_NULL(limit_node_ptr);
  for (const auto &iter : res_limit_map) {
    common::AnfAlgo::SetNodeAttrSafely(iter.first, MakeValue(iter.second), limit_node_ptr);
  }
  return limit_node_ptr;
}

void AclStreamAssign::InsertResLimit(const NotNull<KernelGraphPtr> &kernel_graph, DeviceResManager *device_res_manager,
                                     const mindspore::HashMap<size_t, ResLimitInfoPtr> &stream_res_limit_map,
                                     bool is_dyn_graph) {
  auto kernels = kernel_graph->execution_order();
  std::vector<CNodePtr> new_exec_orders;
  uint32_t device_cube_num = 0;
  uint32_t device_vector_num = 0;
  int32_t default_device_id = -1;
  MS_EXCEPTION_IF_NULL(device_res_manager);
  device_res_manager->SetEnableStreamLimit();
  device_res_manager->GetDeviceLimit(default_device_id, &device_cube_num, &device_vector_num);
  std::set<size_t> streams_set;
  for (const auto &node : kernels) {
    uint32_t cube_num = device_cube_num;
    uint32_t vector_num = device_vector_num;
    mindspore::HashMap<std::string, uint32_t> res_limit_map;
    auto process_stream_id = AnfAlgo::GetStreamId(node);
    auto iter = stream_res_limit_map.find(process_stream_id);
    if (iter == stream_res_limit_map.end()) {
      MS_LOG(EXCEPTION) << "Can't get process_stream_id for  " << process_stream_id;
    }
    if (IsUsersSetResLimitOp(node, kAttrCubeNum, &cube_num)) {
      iter->second->cube_num_modify_flag = true;
    }
    if (iter->second->cube_num_modify_flag && iter->second->cube_num != cube_num) {
      iter->second->cube_num = cube_num;
      res_limit_map[kAttrCubeNum] = cube_num;
      MS_LOG(DEBUG) << "Set StreamLimit with cube_num " << cube_num << " for stream id: " << process_stream_id;
    }
    if (IsUsersSetResLimitOp(node, kAttrVectorNum, &vector_num)) {
      iter->second->vector_num_modify_flag = true;
    }
    if (iter->second->vector_num_modify_flag && iter->second->vector_num != vector_num) {
      iter->second->vector_num = vector_num;
      res_limit_map[kAttrVectorNum] = vector_num;
      MS_LOG(DEBUG) << "Set StreamLimit with vector_num " << vector_num << " for stream id: " << process_stream_id;
    }
    if (res_limit_map.size() != 0) {
      auto limit_node = CreateLimitApplyKernel(kernel_graph, res_limit_map);
      AnfAlgo::SetStreamId(process_stream_id, limit_node.get());
      streams_set.insert(process_stream_id);
      common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue<uint32_t>(process_stream_id), limit_node);
      // Reuse dynamic shape process, operators are not dynamic shape operators.
      if (is_dyn_graph) {
        limit_node->cast<CNodePtr>()->AddAttr(mindspore::ops::kHasDynamicValue, MakeValue<bool>(is_dyn_graph));
      }
      MS_LOG(DEBUG) << "Create Limit node " << limit_node->fullname_with_scope()
                    << ", stream id: " << process_stream_id;
      new_exec_orders.push_back(limit_node);
    }
    new_exec_orders.push_back(node);
  }
  mindspore::HashMap<std::string, uint32_t> restore_limit_map;
  restore_limit_map[kAttrCubeNum] = device_cube_num;
  restore_limit_map[kAttrVectorNum] = device_vector_num;
  for (const auto &stream : streams_set) {
    auto limit_node = CreateLimitApplyKernel(kernel_graph, restore_limit_map);
    AnfAlgo::SetStreamId(stream, limit_node.get());
    streams_set.insert(stream);
    common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue<uint32_t>(stream), limit_node);
    // Reuse dynamic shape process, operators are not dynamic shape operators.
    if (is_dyn_graph) {
      limit_node->cast<CNodePtr>()->AddAttr(mindspore::ops::kHasDynamicValue, MakeValue<bool>(is_dyn_graph));
    }
    MS_LOG(DEBUG) << "Create Restore Limit node " << limit_node->fullname_with_scope() << ", stream id: " << stream;
    new_exec_orders.push_back(limit_node);
  }
  MS_LOG(INFO) << "End to Insert ResLimit node.";
  kernel_graph->set_execution_order(new_exec_orders);
}

void AclStreamAssign::InsertResLimitForNonTaskSink(const NotNull<KernelGraphPtr> &kernel_graph,
                                                   DeviceResManager *device_res_manager) {
  bool enable_with_stream = false;
  bool is_dyn_graph = false;
  mindspore::HashMap<size_t, ResLimitInfoPtr> stream_res_limit_map;
  auto kernels = kernel_graph->execution_order();
  for (auto &kernel : kernels) {
    auto process_stream_id = AnfAlgo::GetStreamId(kernel);
    auto iter = stream_res_limit_map.find(process_stream_id);
    if (iter == stream_res_limit_map.end()) {
      auto limit_info = std::make_shared<ResLimitInfo>();
      MS_EXCEPTION_IF_NULL(limit_info);
      limit_info->cube_num = UINT32_MAX;
      limit_info->vector_num = UINT32_MAX;
      limit_info->cube_num_modify_flag = false;
      limit_info->vector_num_modify_flag = false;
      stream_res_limit_map[process_stream_id] = limit_info;
    }
    uint32_t cube_num = 0;
    uint32_t vector_num = 0;
    if (IsUsersSetResLimitOp(kernel, kAttrCubeNum, &cube_num) ||
        IsUsersSetResLimitOp(kernel, kAttrVectorNum, &vector_num)) {
      enable_with_stream = true;
    }
    if (common::AnfAlgo::IsDynamicShape(kernel) || common::AnfAlgo::IsDynamicSequence(kernel) ||
        common::AnfAlgo::IsAnyTypeOutput(kernel) || common::AnfAlgo::IsDynamicValue(kernel)) {
      is_dyn_graph = true;
    }
  }
  if (!enable_with_stream) {
    return;
  }
  MS_LOG(INFO) << "Begin to Insert ResLimit node.";
  InsertResLimit(kernel_graph, device_res_manager, stream_res_limit_map, is_dyn_graph);
}

void AclStreamAssign::AssignStream(const NotNull<KernelGraphPtr> &kernel_graph, DeviceResManager *device_res_manager) {
  auto kernels = kernel_graph->execution_order();
  if (kernels.empty()) {
    return;
  }
  if (kernel_graph->is_from_single_op()) {
    MS_LOG(INFO) << "Not stream assign when pynative forward.";
    return;
  }
  uint32_t max_stream_id = kDefaultStreamIndex;
  // Frontend's stream_id --> backend's stream_id
  static mindspore::HashMap<int64_t, size_t> stream_map;
  // Frontend's event_id --> backend's event_id
  static std::map<uint32_t, uint32_t> event_map;
  // Backend's event_id --> (aclrt event, StreamSend's stream_id)
  static std::map<uint32_t, std::pair<aclrtEvent, uint32_t>> event_info_map;
  for (const auto &node : kernels) {
    if (AnfAlgo::IsKernelSelectBackoffOp(node)) {
      continue;
    }
    auto input_tensor_num = common::AnfAlgo::GetInputTensorNum(node);
    // for runtime speed up
    bool input_multi_graph_safe = true;
    for (size_t i = 0; i < input_tensor_num; i++) {
      auto input_node = node->input(i + 1);
      MS_EXCEPTION_IF_NULL(input_node);
      auto kernel_with_index = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, true);
      auto real_input = kernel_with_index.first;
      // input from other graph
      if (real_input->isa<Parameter>()) {
        input_multi_graph_safe = false;
      }
    }
    if (input_multi_graph_safe) {
      node->AddAttr(kAttrInputMultiStreamSafe, MakeValue(true));
    }
    auto parallel_context = parallel::ParallelContext::GetInstance();
    MS_EXCEPTION_IF_NULL(parallel_context);
    if (runtime::IsEnableRuntimeConfig(runtime::kRuntimeMultiStream)) {
      // multi_stream:true, all communication op use the same communication stream.
      MS_LOG(INFO) << "Set stream id by no group for node " << node->fullname_with_scope();
      if (common::AnfAlgo::IsCommunicationOp(node) && !common::AnfAlgo::IsLcclCommunicationOp(node)) {
        AddStreamIdForCommunicationOp(node);
      } else if (IsUsersSetStreamsOp(node)) {
        AddStreamIdForUsersSetStreamsOp(node, &stream_map);
      } else if (IsPrimitiveCNode(node, prim::kPrimMoveTo)) {
        AssignStreamForMoveTo(node);
      } else if (IsOneOfPrimitiveCNode(node,
                                       {prim::kPrimCopyToHost, prim::kPrimCopyToHostExt, prim::kPrimCopyToDevice})) {
        AssignStreamForCopyTo(node);
      } else {
        AnfAlgo::SetStreamId(kDefaultStreamIndex, node.get());
        common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue<uint32_t>(kDefaultStreamIndex), node);
      }
    } else {
      // Default scene, multi_stream:group, all communication op use the communication stream by group
      MS_LOG(INFO) << "Set stream id by group for node " << node->fullname_with_scope();
      AddStreamIdByGroup(node, device_res_manager, &stream_map);
    }
    max_stream_id = std::max(max_stream_id, AnfAlgo::GetStreamId(node));

    AddEventForUsersSetEvent(node, &event_map, &event_info_map);
  }
  kernel_graph->set_enable_multi_stream(max_stream_id != kDefaultStreamIndex);
  SetRemoteCopyIdForKernelGraph(kernel_graph);

  for (size_t i = 1; i < kernels.size(); ++i) {
    if (common::AnfAlgo::GetCNodeName(kernels[i - 1]) == kMemSetOpName) {
      auto stream_id = AnfAlgo::GetStreamId(kernels[i]);
      AnfAlgo::SetStreamId(stream_id, kernels[i - 1].get());
      common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue<uint32_t>(stream_id), kernels[i - 1]);
    }
  }

  InsertEventForNonTaskSink(kernel_graph);
  InsertResLimitForNonTaskSink(kernel_graph, device_res_manager);
}

void AclStreamAssign::CreateEvent(const NotNull<KernelGraphPtr> &kernel_graph) {
  std::map<uint32_t, CNodePtr> event_send_map;
  std::map<uint32_t, CNodePtr> event_recv_map;
  auto nodes = kernel_graph->execution_order();
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    auto name = common::AnfAlgo::GetCNodeName(node);
    if (name == kStreamRecvOpName) {
      auto event_id = common::AnfAlgo::GetNodeAttr<uint32_t>(node, kAttrEventId);
      event_recv_map[event_id] = node;
    }
    if (name == kStreamSendOpName) {
      auto event_id = common::AnfAlgo::GetNodeAttr<uint32_t>(node, kAttrEventId);
      event_send_map[event_id] = node;
    }
  }
  auto &resource_manager = AscendStreamMng::GetInstance();
  for (auto iter : event_send_map) {
    auto event = resource_manager.ApplyRtEvent();
    auto send_node = iter.second;
    common::AnfAlgo::SetNodeAttr(kAttrRecordEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), send_node);
    auto recv_node = event_recv_map.find(iter.first)->second;
    common::AnfAlgo::SetNodeAttr(kAttrWaitEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), recv_node);
  }
}

void AclStreamAssign::GenKernelIoExecInfoMap(
  const NotNull<KernelGraphPtr> &kernel_graph,
  mindspore::HashMap<CNodePtr, NodeIoExecInfoPtr> *kernel_io_exec_info_map) const {
  auto &exec_kernels = kernel_graph->execution_order();
  for (size_t i = 0; i < exec_kernels.size(); ++i) {
    auto &process_kernel = exec_kernels[i];
    MS_EXCEPTION_IF_NULL(process_kernel);
    auto process_exec_info = std::make_shared<NodeExecInfo>();
    MS_EXCEPTION_IF_NULL(process_exec_info);
    process_exec_info->node = process_kernel;
    process_exec_info->stream_id = AnfAlgo::GetStreamId(process_kernel);
    process_exec_info->execution_order_index = i;
    auto process_io_exec_info = std::make_shared<NodeIoExecInfo>();
    MS_EXCEPTION_IF_NULL(process_io_exec_info);
    process_io_exec_info->node_exec_info = process_exec_info;
    process_io_exec_info->inputs = {};
    process_io_exec_info->outputs = {};
    (*kernel_io_exec_info_map)[process_kernel] = process_io_exec_info;
  }

  for (auto &process_kernel : exec_kernels) {
    MS_EXCEPTION_IF_NULL(process_kernel);
    auto process_iter = kernel_io_exec_info_map->find(process_kernel);
    if (process_iter == kernel_io_exec_info_map->end()) {
      MS_LOG(INFO) << "Can't get kernel io execution info for  " << process_kernel->fullname_with_scope();
      continue;
    }
    auto process_io_exec_info = process_iter->second;
    MS_EXCEPTION_IF_NULL(process_io_exec_info);
    auto process_exec_info = process_iter->second->node_exec_info;
    MS_EXCEPTION_IF_NULL(process_exec_info);
    auto inputs = process_kernel->inputs();
    for (size_t i = 1; i < inputs.size(); i++) {
      auto input_node = common::AnfAlgo::VisitKernelWithReturnType(inputs[i], 0).first;
      MS_EXCEPTION_IF_NULL(input_node);
      if (AnfUtils::IsRealCNodeKernel(input_node)) {
        auto input_kernel = input_node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(input_kernel);
        auto iter = kernel_io_exec_info_map->find(input_kernel);
        if (iter == kernel_io_exec_info_map->end()) {
          MS_LOG(INFO) << "Can't get kernel io execution info for " << process_kernel->fullname_with_scope()
                       << "'s input node " << input_kernel->fullname_with_scope();
          continue;
        }
        auto input_io_exec_info = iter->second;
        auto input_exec_info = iter->second->node_exec_info;
        MS_EXCEPTION_IF_NULL(input_io_exec_info);
        process_io_exec_info->inputs.push_back(input_exec_info);
        input_io_exec_info->outputs.push_back(process_exec_info);
      }
    }
  }
}

std::pair<CNodePtr, CNodePtr> AclStreamAssign::CreateSendReceive(const NotNull<KernelGraphPtr> &kernel_graph,
                                                                 uint32_t record_stream_id, uint32_t wait_stream_id) {
  auto &resource_manager = AscendStreamMng::GetInstance();
  uint32_t event_id = resource_manager.ApplyNewEvent();
  auto event = resource_manager.ApplyRtEvent();
  auto event_generate_id = ++event_generate_id_;
  auto send_node = CreateSendApplyKernel(kernel_graph, event_id, record_stream_id, event_generate_id);
  common::AnfAlgo::SetNodeAttr(kAttrRecordEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), send_node);
  auto recv_node = CreateRecvApplyKernel(kernel_graph, event_id, record_stream_id, wait_stream_id, event_generate_id);
  common::AnfAlgo::SetNodeAttr(kAttrWaitEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), recv_node);
  return std::pair<CNodePtr, CNodePtr>(send_node, recv_node);
}

void AclStreamAssign::AddBoundarySendRecvKernel(const NotNull<KernelGraphPtr> &kernel_graph, uint32_t record_stream_id,
                                                uint32_t wait_stream_id, std::vector<CNodePtr> *exec_order,
                                                std::map<size_t, std::set<size_t>> *no_event_streams,
                                                CNodePtr pre_cnode, CNodePtr next_cnode) {
  auto &resource_manager = AscendStreamMng::GetInstance();
  uint32_t event_id = resource_manager.ApplyNewEvent();
  auto event = resource_manager.ApplyRtEvent();
  auto event_generate_id = ++event_generate_id_;
  auto send_node = CreateSendApplyKernel(kernel_graph, event_id, record_stream_id, event_generate_id);
  common::AnfAlgo::SetNodeAttr(kAttrRecordEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), send_node);
  auto recv_node = CreateRecvApplyKernel(kernel_graph, event_id, record_stream_id, wait_stream_id, event_generate_id);
  common::AnfAlgo::SetNodeAttr(kAttrWaitEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), recv_node);
  exec_order->push_back(send_node);
  exec_order->push_back(recv_node);
  (*no_event_streams)[wait_stream_id].erase(record_stream_id);
  SetForSwitchInline(kernel_graph, send_node, recv_node, pre_cnode, next_cnode);
}

void AclStreamAssign::AddDelayedSendRecvKernel(const NotNull<mindspore::KernelGraphPtr> &kernel_graph,
                                               const CNodePtr &kernel, size_t exec_idx, uint32_t record_stream_id,
                                               const std::vector<CNodePtr> &origin_exec_order,
                                               std::vector<CNodePtr> *exec_order,
                                               mindspore::HashMap<size_t, std::vector<CNodePtr>> *delayed_recv_nodes) {
  constexpr int64_t kDefaultDelayNum = 1;
  constexpr int64_t kDefaultDelayFactor = 3;
  if (IsPrimitiveCNode(kernel, prim::kPrimMoveTo) && common::AnfAlgo::GetMoveToDstStr(kernel) == kToCpu) {
    // Get pre_fetch size.
    int64_t pre_fetch = kDefaultDelayNum;
    const auto &prim = GetCNodePrimitive(kernel);
    MS_EXCEPTION_IF_NULL(prim);
    const auto &attrs = prim->attrs();
    const auto &attr_iter = attrs.find(kAttrBackwardPrefetch);
    if (attr_iter != attrs.end()) {
      const auto &pre_fetch_value = attr_iter->second;
      if (pre_fetch_value->isa<Int64Imm>()) {
        pre_fetch = GetValue<int64_t>(pre_fetch_value) * kDefaultDelayFactor;
      }
    }
    // Create send and recv kernels.
    auto &resource_manager = AscendStreamMng::GetInstance();
    uint32_t event_id = resource_manager.ApplyNewEvent();
    auto event = resource_manager.ApplyRtEvent();
    auto event_generate_id = ++event_generate_id_;
    auto send_node = CreateSendApplyKernel(kernel_graph, event_id, record_stream_id, event_generate_id);
    common::AnfAlgo::SetNodeAttr(kAttrRecordEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), send_node);
    auto recv_node =
      CreateRecvApplyKernel(kernel_graph, event_id, record_stream_id, kDefaultStreamIndex, event_generate_id);
    common::AnfAlgo::SetNodeAttr(kAttrWaitEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), recv_node);
    exec_order->push_back(send_node);

    const auto max_exec_idx = origin_exec_order.size();
    auto node_before_recv_index = exec_idx + pre_fetch >= max_exec_idx ? max_exec_idx - 1 : exec_idx + pre_fetch;
    while (node_before_recv_index < max_exec_idx - 1) {
      const auto &node_before_recv = origin_exec_order[node_before_recv_index];
      if (AnfAlgo::GetStreamId(node_before_recv) == kDefaultStreamIndex) {
        break;
      }
      node_before_recv_index += 1;
    }
    (*delayed_recv_nodes)[node_before_recv_index].emplace_back(recv_node);
    MS_LOG(DEBUG) << "Add send and recv for MoveTo, send: " << exec_idx << ", recv: " << node_before_recv_index;
  }
  const auto &iter = delayed_recv_nodes->find(exec_idx);
  if (iter != delayed_recv_nodes->end()) {
    const auto &recv_nodes = iter->second;
    std::copy(recv_nodes.begin(), recv_nodes.end(), std::back_inserter(*exec_order));
  }
}

void AclStreamAssign::ProcessSideEffect(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr kernel,
                                        size_t process_stream_id, const CNodePtr last_kernel,
                                        std::vector<AnfNodePtr> *real_inputs,
                                        std::map<AnfNodePtr, std::set<size_t>> *side_effect_map,
                                        std::map<size_t, std::set<size_t>> *no_event_streams,
                                        std::vector<CNodePtr> *new_exec_orders) {
  bool has_side_effect = false;
  auto input_tensor_num = common::AnfAlgo::GetInputTensorNum(kernel);
  for (size_t i = 0; i < input_tensor_num; i++) {
    auto input_node = kernel->input(i + 1);
    MS_EXCEPTION_IF_NULL(input_node);
    if (HasAbstractMonad(input_node)) {
      has_side_effect = true;
      continue;
    }
    auto kernel_with_index = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, true);
    auto real_input = kernel_with_index.first;
    real_inputs->push_back(real_input);
  }
  auto prim = GetValueNode<PrimitivePtr>(kernel->input(0));
  if (prim != nullptr &&
      (GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_MEM) || GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_IO))) {
    has_side_effect = true;
  }

  if (has_side_effect) {
    for (auto real_input : (*real_inputs)) {
      auto &stream_set = (*side_effect_map)[real_input];
      for (auto stream_id : stream_set) {
        if (stream_id == process_stream_id) {
          continue;
        }
        auto &no_event_stream_set = (*no_event_streams)[process_stream_id];
        auto no_event_iter = no_event_stream_set.find(stream_id);
        if (no_event_iter == no_event_stream_set.end()) {
          continue;
        }
        MS_LOG(INFO) << "Add side effect event " << stream_id << " to " << process_stream_id << " for kernel "
                     << kernel->fullname_with_scope();
        AddBoundarySendRecvKernel(kernel_graph, stream_id, process_stream_id, new_exec_orders, no_event_streams,
                                  last_kernel, kernel);
      }
    }
  }
}

StreamInfo AclStreamAssign::AddInitialBoundarySync(const NotNull<KernelGraphPtr> &kernel_graph,
                                                   std::vector<CNodePtr> *new_exec_orders) {
  std::set<size_t> streams_set;
  std::set<size_t> streams_usr_set;
  std::map<size_t, std::set<size_t>> no_event_streams;
  auto exec_kernels = kernel_graph->execution_order();
  for (auto &kernel : exec_kernels) {
    auto process_stream_id = AnfAlgo::GetStreamId(kernel);
    if (process_stream_id != kDefaultStreamIndex) {
      streams_set.insert(process_stream_id);
    }
    if (IsUsersSetStreamsOp(kernel)) {
      streams_usr_set.insert(process_stream_id);
    }
    no_event_streams[process_stream_id] = {};
  }
  for (const auto &stream : streams_set) {
    auto it = streams_usr_set.find(stream);
    if (it != streams_usr_set.end()) {
      continue;
    }
    AddBoundarySendRecvKernel(kernel_graph, kDefaultStreamIndex, stream, new_exec_orders, &no_event_streams);
  }
  return {streams_set, streams_usr_set, no_event_streams};
}

void AclStreamAssign::AddFinalBoundarySync(const NotNull<KernelGraphPtr> &kernel_graph,
                                           const std::set<size_t> &streams_set, const std::set<size_t> &streams_usr_set,
                                           std::vector<CNodePtr> *new_exec_orders,
                                           std::map<size_t, std::set<size_t>> *no_event_streams) {
  for (const auto &stream : streams_set) {
    if (streams_usr_set.find(stream) != streams_usr_set.end()) {
      continue;
    }
    AddBoundarySendRecvKernel(kernel_graph, stream, kDefaultStreamIndex, new_exec_orders, no_event_streams);
  }
}

void AclStreamAssign::UpdateEventsToExecutionOrder(
  const NotNull<KernelGraphPtr> &kernel_graph,
  const mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> &send_after_node,
  const mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> &recv_before_node,
  const mindspore::HashMap<AnfNodePtr, std::set<size_t>> &producer_streams) {
  MS_LOG(DEBUG) << "Start UpdateEventsToExecutionOrder...";
  std::map<AnfNodePtr, std::set<size_t>> side_effect_map;
  auto exec_kernels = kernel_graph->execution_order();
  mindspore::HashMap<size_t, std::vector<CNodePtr>> delayed_recv_nodes;
  std::vector<CNodePtr> new_exec_orders;
  auto [streams_set, streams_usr_set, no_event_streams] = AddInitialBoundarySync(kernel_graph, &new_exec_orders);
  CNodePtr last_kernel = nullptr;
  size_t cur_idx = 0;
  for (auto &kernel : exec_kernels) {
    auto before_iter = recv_before_node.find(kernel);
    if (before_iter != recv_before_node.end()) {
      (void)std::copy(before_iter->second.begin(), before_iter->second.end(), std::back_inserter(new_exec_orders));
    }
    if (IsUsersSetStreamsOp(kernel)) {
      new_exec_orders.push_back(kernel);
      continue;
    }
    auto process_stream_id = AnfAlgo::GetStreamId(kernel);
    if (process_stream_id != kDefaultStreamIndex) {
      AddBoundarySendRecvKernel(kernel_graph, kDefaultStreamIndex, process_stream_id, &new_exec_orders,
                                &no_event_streams, last_kernel, kernel);
      auto it = producer_streams.find(kernel);
      if (it != producer_streams.end()) {
        for (auto record_stream_id : it->second) {
          if (record_stream_id == kDefaultStreamIndex) {
            continue;
          }
          AddBoundarySendRecvKernel(kernel_graph, record_stream_id, process_stream_id, &new_exec_orders,
                                    &no_event_streams, last_kernel, kernel);
        }
      }
    }
    if (kernel_graph->enable_multi_stream() && IsPrimitiveCNode(kernel, std::make_shared<Primitive>(kSendOpName))) {
      AddBoundarySendRecvKernel(kernel_graph, process_stream_id, kDefaultStreamIndex, &new_exec_orders,
                                &no_event_streams, last_kernel, kernel);
    }
    std::vector<AnfNodePtr> real_inputs;
    ProcessSideEffect(kernel_graph, kernel, process_stream_id, last_kernel, &real_inputs, &side_effect_map,
                      &no_event_streams, &new_exec_orders);

    for (auto real_input : real_inputs) {
      side_effect_map[real_input].insert(process_stream_id);
    }
    for (auto &kv : no_event_streams) {
      if (kv.first != process_stream_id) {
        kv.second.insert(process_stream_id);
      }
    }
    new_exec_orders.push_back(kernel);
    AddDelayedSendRecvKernel(kernel_graph, kernel, cur_idx, process_stream_id, exec_kernels, &new_exec_orders,
                             &delayed_recv_nodes);
    last_kernel = kernel;
    auto after_iter = send_after_node.find(kernel);
    if (after_iter != send_after_node.end()) {
      (void)std::copy(after_iter->second.begin(), after_iter->second.end(), std::back_inserter(new_exec_orders));
    }
    cur_idx += 1;
  }
  auto graph_output = kernel_graph->output();
  auto graph_output_iter = recv_before_node.find(graph_output);
  if (graph_output_iter != recv_before_node.end()) {
    (void)std::copy(graph_output_iter->second.begin(), graph_output_iter->second.end(),
                    std::back_inserter(new_exec_orders));
  }
  AddFinalBoundarySync(kernel_graph, streams_set, streams_usr_set, &new_exec_orders, &no_event_streams);
  kernel_graph->set_execution_order(new_exec_orders);
  MS_LOG(DEBUG) << "Finish UpdateEventsToExecutionOrder.";
}

void AclStreamAssign::ProcessStreamForInputs(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &kernel,
                                             const NodeIoExecInfoPtr &io_exec_info,
                                             mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
                                             mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv,
                                             mindspore::HashMap<AnfNodePtr, std::set<size_t>> *producer_streams) {
  MS_EXCEPTION_IF_NULL(io_exec_info);
  auto process_stream_id = AnfAlgo::GetStreamId(kernel);
  auto input_exec_info_list = io_exec_info->inputs;
  mindspore::HashMap<uint32_t, NodeExecInfoPtr> stream_max_exec_node_map;

  for (auto &input : input_exec_info_list) {
    MS_EXCEPTION_IF_NULL(input);
    auto input_stream_id = input->stream_id;
    auto iter = stream_max_exec_node_map.find(input_stream_id);
    if (iter == stream_max_exec_node_map.end()) {
      stream_max_exec_node_map[input_stream_id] = input;
    } else {
      MS_EXCEPTION_IF_NULL(iter->second);
      if (input->execution_order_index > iter->second->execution_order_index) {
        iter->second = input;
      }
    }
  }

  for (auto input_exec : stream_max_exec_node_map) {
    MS_EXCEPTION_IF_NULL(input_exec.second);
    if (input_exec.second->stream_id == process_stream_id) {
      continue;
    }
    (*producer_streams)[kernel].insert(input_exec.second->stream_id);
  }
}

void AclStreamAssign::InsertEventsForOutputs(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &kernel,
                                             const NodeIoExecInfoPtr &io_exec_info,
                                             mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
                                             mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv) {
  MS_EXCEPTION_IF_NULL(io_exec_info);
  auto process_stream_id = AnfAlgo::GetStreamId(kernel);
  auto output_exec_info_list = io_exec_info->outputs;
  mindspore::HashMap<uint32_t, NodeExecInfoPtr> stream_min_exec_node_map;
  for (auto &output : output_exec_info_list) {
    MS_EXCEPTION_IF_NULL(output);
    auto output_stream_id = output->stream_id;
    auto iter = stream_min_exec_node_map.find(output_stream_id);
    if (iter == stream_min_exec_node_map.end()) {
      stream_min_exec_node_map[output_stream_id] = output;
    } else {
      MS_EXCEPTION_IF_NULL(iter->second);
      if (output->execution_order_index < iter->second->execution_order_index) {
        iter->second = output;
      }
    }
  }

  for (auto output_exec : stream_min_exec_node_map) {
    MS_EXCEPTION_IF_NULL(output_exec.second);
    if (output_exec.second->stream_id == process_stream_id || IsUsersSetStreamsOp(kernel)) {
      continue;
    }
    if (process_stream_id == kDefaultStreamIndex && IsUsersSetStreamsOp(output_exec.second->node)) {
      continue;
    }
    InsertEvents(kernel_graph, kernel, kernel, kernel_send, kernel_recv, output_exec.second->node);
  }
}

CNodePtr AclStreamAssign::CreateSendApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id,
                                                uint32_t stream_id, uint32_t event_generate_id) {
  auto send_op = std::make_shared<Primitive>(kStreamSendOpName);
  MS_EXCEPTION_IF_NULL(send_op);
  auto send_apply = std::make_shared<ValueNode>(send_op);
  MS_EXCEPTION_IF_NULL(send_apply);
  auto send_node_ptr = graph_ptr->NewCNode({send_apply});
  MS_EXCEPTION_IF_NULL(send_node_ptr);
  common::AnfAlgo::SetNodeAttr(kAttrEventId, MakeValue(event_id), send_node_ptr);
  common::AnfAlgo::SetNodeAttr(kAttrRecordWaitEventStreamPairId, MakeValue(event_generate_id), send_node_ptr);
  AnfAlgo::SetStreamId(stream_id, send_node_ptr.get());
  return send_node_ptr;
}

CNodePtr AclStreamAssign::CreateRecvApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id,
                                                uint32_t record_stream_id, uint32_t stream_id,
                                                uint32_t event_generate_id) {
  auto recv_op = std::make_shared<Primitive>(kStreamRecvOpName);
  MS_EXCEPTION_IF_NULL(recv_op);
  auto recv_apply = std::make_shared<ValueNode>(recv_op);
  MS_EXCEPTION_IF_NULL(recv_apply);
  auto recv_node_ptr = graph_ptr->NewCNode({recv_apply});
  MS_EXCEPTION_IF_NULL(recv_node_ptr);
  common::AnfAlgo::SetNodeAttr(kAttrEventId, MakeValue(event_id), recv_node_ptr);
  common::AnfAlgo::SetNodeAttr(kAttrRecordEventStream, MakeValue(record_stream_id), recv_node_ptr);
  common::AnfAlgo::SetNodeAttr(kAttrRecordWaitEventStreamPairId, MakeValue(event_generate_id), recv_node_ptr);
  AnfAlgo::SetStreamId(stream_id, recv_node_ptr.get());
  return recv_node_ptr;
}

void AclStreamAssign::InsertEvents(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &parallel_cnode,
                                   const AnfNodePtr &node_before_send,
                                   mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
                                   mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv,
                                   const AnfNodePtr &node_after_recv) {
  MS_EXCEPTION_IF_NULL(kernel_send);
  MS_EXCEPTION_IF_NULL(kernel_recv);
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  uint32_t event_id = resource_manager.ApplyNewEvent();
  auto event = resource_manager.ApplyRtEvent();
  auto send_stream_id = AnfAlgo::GetStreamId(node_before_send);
  auto event_generate_id = ++event_generate_id_;
  auto send_cnode = CreateSendApplyKernel(kernel_graph, event_id, send_stream_id, event_generate_id);
  common::AnfAlgo::SetNodeAttr(kAttrRecordEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), send_cnode);
  auto send_iter = kernel_send->find(node_before_send);
  if (send_iter == kernel_send->end()) {
    (*kernel_send)[node_before_send] = {send_cnode};
  } else {
    send_iter->second.push_back(send_cnode);
  }

  CNodePtr recv_cnode = CreateRecvApplyKernel(kernel_graph, event_id, send_stream_id,
                                              AnfAlgo::GetStreamId(node_after_recv), event_generate_id);
  common::AnfAlgo::SetNodeAttr(kAttrWaitEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), recv_cnode);
  auto process_iter = kernel_recv->find(node_after_recv);
  if (process_iter == kernel_recv->end()) {
    (*kernel_recv)[node_after_recv] = {recv_cnode};
  } else {
    process_iter->second.push_back(recv_cnode);
  }
  SetForSwitchInline(kernel_graph, send_cnode, recv_cnode, node_before_send, node_after_recv);
}

void AclStreamAssign::GenEventsForParallelOp(const NotNull<KernelGraphPtr> &kernel_graph,
                                             mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
                                             mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv,
                                             mindspore::HashMap<AnfNodePtr, std::set<size_t>> *producer_streams) {
  MS_LOG(DEBUG) << "Start GenEventsForParallelOp...";
  auto exec_kernels = kernel_graph->execution_order();
  mindspore::HashMap<CNodePtr, NodeIoExecInfoPtr> kernel_io_exec_info_map;
  GenKernelIoExecInfoMap(kernel_graph, &kernel_io_exec_info_map);
  for (auto &process_kernel : exec_kernels) {
    if (AnfAlgo::IsKernelSelectBackoffOp(process_kernel)) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(process_kernel);
    auto process_stream_id = AnfAlgo::GetStreamId(process_kernel);
    if (process_stream_id == kDefaultStreamIndex) {
      continue;
    }
    MS_LOG(DEBUG) << "Start GenEvents For ParallelOp " << process_kernel->fullname_with_scope();
    auto process_iter = kernel_io_exec_info_map.find(process_kernel);
    if (process_iter == kernel_io_exec_info_map.end()) {
      MS_LOG(INFO) << "Can't get node io execution info for  " << process_kernel->fullname_with_scope();
      continue;
    }
    auto process_io_exec_info = process_iter->second;
    ProcessStreamForInputs(kernel_graph, process_kernel, process_io_exec_info, kernel_send, kernel_recv,
                           producer_streams);
    InsertEventsForOutputs(kernel_graph, process_kernel, process_io_exec_info, kernel_send, kernel_recv);
  }
  MS_LOG(DEBUG) << "Finish GenEventsForParallelOp.";
}

std::pair<CNodePtr, CNodePtr> AclStreamAssign::CreateSendRecvEventsPair(const NotNull<KernelGraphPtr> &kernel_graph,
                                                                        size_t send_stream_id, size_t wait_stream_id) {
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  uint32_t event_id = resource_manager.ApplyNewEvent();
  auto event = resource_manager.ApplyRtEvent();
  auto event_generate_id = ++event_generate_id_;

  auto send_cnode = CreateSendApplyKernel(kernel_graph, event_id, send_stream_id, event_generate_id);
  common::AnfAlgo::SetNodeAttr(kAttrRecordEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), send_cnode);

  auto recv_cnode = CreateRecvApplyKernel(kernel_graph, event_id, send_stream_id, wait_stream_id, event_generate_id);
  common::AnfAlgo::SetNodeAttr(kAttrWaitEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), recv_cnode);

  return std::make_pair(send_cnode, recv_cnode);
}

void AclStreamAssign::UpdateGPTOEventsToExecutionOrder(
  const NotNull<KernelGraphPtr> &kernel_graph,
  const std::vector<std::pair<CNodePtr, std::tuple<char, size_t, size_t, size_t>>> &mock_exec_order) {
  std::vector<CNodePtr> new_exec_order;
  auto exec_order_list = kernel_graph->execution_order();

  // Find all streams to protect between graphs
  std::set<size_t> streams_set;
  for (auto &kernel : exec_order_list) {
    auto process_stream_id = AnfAlgo::GetStreamId(kernel);
    if (process_stream_id != kDefaultStreamIndex) {
      streams_set.insert(process_stream_id);
    }
  }

  // Add stream events at the beginning of the graph
  for (const auto &stream : streams_set) {
    std::pair<CNodePtr, CNodePtr> send_recv_nodes = CreateSendRecvEventsPair(kernel_graph, kDefaultStreamIndex, stream);
    new_exec_order.push_back(send_recv_nodes.first);
    new_exec_order.push_back(send_recv_nodes.second);
  }

  // Materialize mock execution order
  std::map<size_t, CNodePtr> recv_event;
  for (auto pair : mock_exec_order) {
    const auto &cnode_ptr = pair.first;
    const auto &send_rcv = pair.second;

    if (cnode_ptr != nullptr) {
      if (common::AnfAlgo::GetCNodeName(cnode_ptr) == kSendOpName) {
        std::pair<CNodePtr, CNodePtr> events_pair =
          CreateSendRecvEventsPair(kernel_graph, AnfAlgo::GetStreamId(cnode_ptr), kDefaultStreamIndex);
        new_exec_order.push_back(events_pair.first);
        new_exec_order.push_back(events_pair.second);
      }
      new_exec_order.push_back(cnode_ptr);
      continue;
    }

    // Case of send/recv event
    const auto &event_type = std::get<0>(send_rcv);
    const auto &send_stream_id = std::get<1>(send_rcv);
    const auto &recv_stream_id = std::get<2>(send_rcv);
    const auto &event_id = std::get<3>(send_rcv);

    if (event_type == 's') {  // send case
      std::pair<CNodePtr, CNodePtr> events_pair =
        CreateSendRecvEventsPair(kernel_graph, send_stream_id, recv_stream_id);
      new_exec_order.push_back(events_pair.first);
      recv_event[event_id] = events_pair.second;
    } else {  // recv case
      new_exec_order.push_back(recv_event[event_id]);
    }
  }

  // Add stream events at the end of the graph
  for (const auto &stream : streams_set) {
    std::pair<CNodePtr, CNodePtr> send_recv_nodes = CreateSendRecvEventsPair(kernel_graph, stream, kDefaultStreamIndex);
    new_exec_order.push_back(send_recv_nodes.first);
    new_exec_order.push_back(send_recv_nodes.second);
  }

  kernel_graph->set_execution_order(new_exec_order);
}

void AclStreamAssign::InsertEventForNonTaskSink(const NotNull<KernelGraphPtr> &kernel_graph) {
  mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> kernel_send;
  mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> kernel_recv;
  mindspore::HashMap<AnfNodePtr, std::set<size_t>> producer_streams;
  AnfAlgo::SetStreamId(kDefaultStreamIndex, kernel_graph->output().get());

  std::vector<std::pair<CNodePtr, std::tuple<char, size_t, size_t, size_t>>> mock_exec_order;
  mindspore::gpto::RunGPTO(kernel_graph, mock_exec_order);
  if (!mock_exec_order.empty()) {
    MS_LOG(INFO) << "Current event generation is GPTO algo";
    UpdateGPTOEventsToExecutionOrder(kernel_graph, mock_exec_order);
    return;
  }

  MS_LOG(INFO) << "Current event generation is default algo";
  GenEventsForParallelOp(kernel_graph, &kernel_send, &kernel_recv, &producer_streams);
  UpdateEventsToExecutionOrder(kernel_graph, kernel_send, kernel_recv, producer_streams);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
