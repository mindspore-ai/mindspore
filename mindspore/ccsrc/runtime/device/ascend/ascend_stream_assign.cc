/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "runtime/device/ascend/ascend_stream_assign.h"

#include <algorithm>
#include <utility>

#include "ir/manager.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "frontend/parallel/context.h"
#include "frontend/parallel/device_manager.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/kernel_adjust.h"
#include "backend/optimizer/common/helper.h"
#include "backend/kernel_compiler/oplib/oplib.h"
#include "utils/utils.h"

#ifdef ENABLE_DUMP_IR
#include "debug/rdr/running_data_recorder.h"
#endif

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr uint32_t kDeviceNumOfServer = 8;
constexpr uint32_t kDeviceNumThreshold = 1024;
const char kDefaultGroup[] = "__default_group";

constexpr uint32_t kMaxStreamNum = 1024;
constexpr uint32_t kHcomSecondaryStreamNum = 3;

constexpr uint32_t kMaxTaskNumPerStream = 1010;
constexpr uint32_t kMaxCommonNodeNumPerStream = 350;

constexpr uint32_t kTaskNumPerHcomNode = 200;
constexpr uint32_t kTaskNumPerWorldHcomNode = 250;
constexpr uint32_t kTaskNumPerSameServerHcomNode = 125;
constexpr uint32_t kTaskNumPerHcomSendRecvNode = 15;

bool IsSameServer(const std::vector<uint32_t> &rank_ids) {
  auto min_iter = min_element(rank_ids.begin(), rank_ids.end());
  uint32_t min = (min_iter != rank_ids.end()) ? *min_iter : 0;
  auto max_iter = max_element(rank_ids.begin(), rank_ids.end());
  uint32_t max = (max_iter != rank_ids.end()) ? *max_iter : 0;
  return ((max - min < kDeviceNumOfServer) && (min / kDeviceNumOfServer == max / kDeviceNumOfServer));
}

string DoGetHcomGroup(const string &original_group) {
  string communi_parallel_mode = parallel::ParallelContext::GetInstance()->communi_parallel_mode();
  if (communi_parallel_mode == parallel::ALL_GROUP_PARALLEL) {
    return original_group;
  }

  if (communi_parallel_mode == parallel::NO_GROUP_PARALLEL) {
    return kDefaultGroup;
  }

  MS_EXCEPTION_IF_NULL(parallel::g_device_manager);
  auto group_info = parallel::g_device_manager->group_info();
  for (const auto &info : group_info) {
    if (info.first != original_group) {
      continue;
    }

    const auto &rank_ids = info.second;
    if (IsSameServer(rank_ids)) {
      return original_group;
    } else {
      return kDefaultGroup;
    }
  }

  // world group is not in group_info.
  return kDefaultGroup;
}

string GetHcomGroup(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!AnfAlgo::HasNodeAttr(kAttrGroup, cnode)) {
    MS_LOG_EXCEPTION << "Hcom node " << cnode->fullname_with_scope() << " has no group attribute.";
  }

  auto group_name = AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrGroup);
  auto new_group = DoGetHcomGroup(group_name);
  MS_LOG_INFO << "hcom node: " << cnode->fullname_with_scope() << ", old group: " << group_name
              << ", new group: " << new_group;

  return new_group;
}

uint32_t GetHcomTaskNum(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!AnfAlgo::HasNodeAttr(kAttrGroup, cnode)) {
    MS_LOG_EXCEPTION << "Hcom node " << cnode->fullname_with_scope() << " has no group attribute.";
  }

  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "Device manager is nullptr.";
    return kTaskNumPerHcomNode;
  }

  auto node_name = AnfAlgo::GetCNodeName(cnode);
  if (node_name == kHcomSendOpName || node_name == kReceiveOpName) {
    return kTaskNumPerHcomSendRecvNode;
  }

  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());
  auto device_num = parallel::ParallelContext::GetInstance()->device_num();
  auto group_name = AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrGroup);
  auto group_info = parallel::g_device_manager->group_info();
  for (const auto &info : group_info) {
    if (info.first != group_name) {
      continue;
    }
    const auto &rank_ids = info.second;
    if (IsSameServer(rank_ids)) {
      return kTaskNumPerSameServerHcomNode;
    } else if (rank_ids.size() == static_cast<size_t>(device_num) && device_num >= kDeviceNumThreshold) {
      return kTaskNumPerWorldHcomNode;
    } else {
      return kTaskNumPerHcomNode;
    }
  }

  // world group is not in group_info.
  if (device_num >= kDeviceNumThreshold) {
    return kTaskNumPerWorldHcomNode;
  } else {
    return kTaskNumPerHcomNode;
  }
}

CNodePtr GetHcomAndOverflowMarker(const NotNull<KernelGraphPtr> &graph_ptr, vector<CNodePtr> *hcom_nodes) {
  auto cnode_ptr_list = graph_ptr->execution_order();
  CNodePtr overflow_marker = nullptr;
  std::string kNPUGetFloatStatusOpName = "NPUGetFloatStatus";
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    auto cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (AnfAlgo::GetCNodeName(cur_cnode_ptr) == kNPUGetFloatStatusOpName) {
      overflow_marker = cur_cnode_ptr;
    } else if (AnfAlgo::GetKernelType(cur_cnode_ptr) == HCCL_KERNEL) {
      hcom_nodes->emplace_back(cur_cnode_ptr);
    } else if (i > 0 && AnfAlgo::GetCNodeName(cnode_ptr_list[i - 1]) == kAtomicAddrCleanOpName) {
      auto graph_id = AnfAlgo::GetGraphId(cur_cnode_ptr.get());
      AnfAlgo::SetGraphId(graph_id, cnode_ptr_list[i - 1].get());
    }
  }
  return overflow_marker;
}

bool HasRefNodes(const vector<CNodePtr> &moved_backward_cnodes) {
  for (auto &cnode : moved_backward_cnodes) {
    std::string op_name = AnfAlgo::GetCNodeName(cnode);
    auto op_info = mindspore::kernel::OpLib::FindOp(op_name, kernel::kTBE);
    if (op_info != nullptr && op_info->is_ref()) {
      MS_LOG(INFO) << "Find RefNode: " << op_name << ", full name: " << cnode->fullname_with_scope();
      return true;
    }
  }
  return false;
}

StreamActiveKind GetStreamKind(uint32_t cur_stream_id, uint32_t pre_stream_id, uint32_t next_stream_id) {
  // pre_stream_id equal to UINT32_MAX means no node active current StreamActive
  // next_stream_id equal to UINT32_MAX means current StreamActive active no node
  if (pre_stream_id == UINT32_MAX || next_stream_id == UINT32_MAX) {
    return kInvalid;
  }

  if (cur_stream_id == pre_stream_id && cur_stream_id == next_stream_id) {
    return kMiddle;
  }

  if (cur_stream_id == pre_stream_id) {
    return kTail;
  }

  if (cur_stream_id == next_stream_id) {
    return kHead;
  }

  return kInvalid;
}
}  // namespace

void AscendStreamAssign::AssignStream(const NotNull<KernelGraphPtr> &graph_ptr) {
  if (IsTaskSink() && !graph_ptr->is_dynamic_shape()) {
    MS_LOG(INFO) << "Communication parallel mode: " << parallel::ParallelContext::GetInstance()->communi_parallel_mode()
                 << ".";

    Reset();
    SetLoopSink();
    ReorderIndependentOrders(graph_ptr);
    TrailingTimeOptimizationByReorder(graph_ptr);

    AssignAllNodesStream(graph_ptr);
    UpdateAtomicAddrCleanStreamId(graph_ptr);
    InsertStreamActive(graph_ptr);
    InsertEventForHcomParallel(graph_ptr);
    InsertEventForIndependentParallel(graph_ptr);
    GetIndependentMaxTarget(graph_ptr);
    InsertCtrlForIndependentParallel(graph_ptr);
    AdjustAtomicAddrCleanOrder(graph_ptr);

    GetNeedActiveStreams(graph_ptr);

    MS_LOG(INFO) << "Before check resource assign";
    graph_ptr->PrintGraphExecuteOrder();

    CheckResourceAssign(graph_ptr);
    MS_LOG(INFO) << "After finish stream assign";
#ifdef ENABLE_DUMP_IR
    SubModuleId module = SubModuleId::SM_SESSION;
    std::string name = "assign_stream." + std::to_string(graph_ptr->graph_id());
    const std::vector<CNodePtr> &exec_order = graph_ptr->execution_order();
    mindspore::RDR::RecordStreamExecOrder(module, name, exec_order);
#endif
    graph_ptr->PrintGraphExecuteOrder();

    FindStreamRelations(graph_ptr);
    PrintStreamRelations();
    GetStreamRelations();
    PrintStreamGroups();
    FindEventRelations(graph_ptr);
  }
}

void AscendStreamAssign::SetLoopSink() {
  if (KernelAdjust::NeedInsertSwitch()) {
    loop_sink_ = true;
  } else {
    loop_sink_ = false;
  }
}

// section 1
void AscendStreamAssign::ReorderIndependentOrders(const NotNull<KernelGraphPtr> &graph_ptr) {
  std::vector<CNodePtr> exe_orders;
  std::vector<CNodePtr> independents;
  std::vector<CNodePtr> others;

  auto cnode_ptr_list = graph_ptr->execution_order();
  MS_LOG(INFO) << "Before reorder, graph orders size:" << cnode_ptr_list.size();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    auto cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (AnfAlgo::IsIndependentNode(cur_cnode_ptr)) {
      independents.emplace_back(cur_cnode_ptr);
    } else {
      others.emplace_back(cur_cnode_ptr);
    }
  }

  if (others.empty() || independents.empty()) {
    MS_LOG(INFO) << "Independent or others is empty, no need reorder";
    return;
  }

  std::set<CNode *> processed;
  for (size_t i = 0; i < others.size(); i++) {
    auto begin = others.begin() + i;
    auto end = begin + 1;
    bool flag = false;
    for (size_t j = 0; j < independents.size(); j++) {
      auto cur_independent = independents[j];
      auto it = std::find(processed.begin(), processed.end(), cur_independent.get());
      if (it != processed.end()) {
        continue;
      }

      auto res = FindTargetOp(begin, end, cur_independent, false);
      if (res != end) {
        flag = true;
        exe_orders.emplace_back(cur_independent);
        exe_orders.emplace_back(*begin);
        processed.emplace(cur_independent.get());
        break;
      }
    }

    if (!flag) {
      exe_orders.emplace_back(*begin);
    }
  }

  MS_LOG(INFO) << "After reorder, graph orders size:" << exe_orders.size();
  if (processed.size() != independents.size()) {
    MS_LOG(WARNING) << "Processed independent nodes size is not equal to exiting independent nodes size";
    return;
  }

  graph_ptr->set_execution_order(exe_orders);
}

void AscendStreamAssign::CheckScenario(const NotNull<KernelGraphPtr> &graph_ptr,
                                       vector<CNodePtr> *last_grad_and_status) {
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> hcom_nodes;
  auto overflow_marker = GetHcomAndOverflowMarker(graph_ptr, &hcom_nodes);
  if (hcom_nodes.size() < 2 || overflow_marker == nullptr) {
    MS_LOG(INFO) << "Current model isn't in distribute or mix-precision mode, no optimization needed";
    last_grad_and_status->clear();
    return;
  }

  auto overflow_marker_pos = find(cnode_ptr_list.begin(), cnode_ptr_list.end(), overflow_marker);
  auto last_hcom_ptr = hcom_nodes[hcom_nodes.size() - 1];
  auto last_hcom_pos = find(cnode_ptr_list.begin(), cnode_ptr_list.end(), last_hcom_ptr);
  auto last_grad_hcom_ptr = hcom_nodes[hcom_nodes.size() - 2];
  auto last_grad_hcom_pos = find(cnode_ptr_list.begin(), cnode_ptr_list.end(), last_grad_hcom_ptr);
  if (last_grad_hcom_pos > overflow_marker_pos || last_hcom_pos < overflow_marker_pos) {
    MS_LOG(INFO) << "Grads average done after overflow judgement or status aren't allgathered, no optimization needed";
    last_grad_and_status->clear();
    return;
  }

  auto last_inputs = GetLastInputCnode(graph_ptr, last_grad_hcom_ptr);
  if (last_inputs.empty() || last_inputs.size() > 1 || IsHcom(last_inputs[0])) {
    MS_LOG(INFO) << "Inputs of last gradients allreduce is empty or include other allreduce, no optimization needed";
    last_grad_and_status->clear();
    return;
  }
  auto last_grad_ptr = last_inputs[0];
  MS_LOG(DEBUG) << "Last Hcom: " << last_grad_hcom_ptr->fullname_with_scope()
                << "; last input: " << last_grad_ptr->fullname_with_scope();
  auto last_grad_hcom_graph_id = AnfAlgo::GetGraphId(last_grad_hcom_ptr.get());
  auto last_grad_graph_id = AnfAlgo::GetGraphId(last_grad_ptr.get());
  auto overflow_marker_graph_id = AnfAlgo::GetGraphId(overflow_marker.get());
  if (last_grad_graph_id != last_grad_hcom_graph_id || last_grad_graph_id != overflow_marker_graph_id) {
    MS_LOG(INFO) << "The grads and grad_hcom or overflow marker were not on the same subgraph, no optimization needed";
    last_grad_and_status->clear();
    return;
  }

  auto label_switch_pos = find_if(last_grad_hcom_pos, cnode_ptr_list.end(),
                                  [](CNodePtr &node) -> bool { return AnfAlgo::GetCNodeName(node) == "LabelSwitch"; });
  if (label_switch_pos == cnode_ptr_list.end()) {
    MS_LOG(INFO) << "No branches after getting overflow status, no optimization needed";
    last_grad_and_status->clear();
    return;
  }
  last_grad_and_status->emplace_back(last_grad_ptr);
  last_grad_and_status->emplace_back(overflow_marker);
  return;
}

CNodePtr AscendStreamAssign::GetCNodesNeededMoved(vector<CNodePtr> *moved_backward_cnodes,
                                                  vector<CNodePtr> *moved_forward_cnodes,
                                                  const vector<CNodePtr> &last_grad_and_status,
                                                  const NotNull<KernelGraphPtr> &graph_ptr) {
  auto cnode_ptr_list = graph_ptr->execution_order();
  if (last_grad_and_status.size() != 2) {
    return nullptr;
  }
  auto last_grad_ptr = last_grad_and_status[0];
  auto float_status_ptr = last_grad_and_status[1];
  auto last_grad_pos = find(cnode_ptr_list.begin(), cnode_ptr_list.end(), last_grad_ptr);
  auto float_status_pos = find(cnode_ptr_list.begin(), cnode_ptr_list.end(), float_status_ptr);
  if (last_grad_pos == cnode_ptr_list.end() || float_status_pos == cnode_ptr_list.end()) {
    return nullptr;
  }
  auto graph_id = AnfAlgo::GetGraphId(last_grad_ptr.get());
  moved_backward_cnodes->insert(moved_backward_cnodes->end(), last_grad_pos + 1, float_status_pos);

  auto it = float_status_pos;
  while (AnfAlgo::GetGraphId((*it).get()) == graph_id && it < cnode_ptr_list.end()) {
    if (AnfAlgo::GetCNodeName(*it) == kAtomicAddrCleanOpName) {
      it++;
      continue;
    }
    auto inputs = GetInputKernels(*it);
    bool is_independent = true;
    for (auto &input : inputs) {
      if (find(moved_backward_cnodes->begin(), moved_backward_cnodes->end(), input) != moved_backward_cnodes->end()) {
        is_independent = false;
        break;
      }
    }
    if (is_independent) {
      if (AnfAlgo::GetCNodeName(*(it - 1)) == kAtomicAddrCleanOpName) {
        moved_forward_cnodes->emplace_back(*(it - 1));
      }
      moved_forward_cnodes->emplace_back(*it);
    } else {
      if (AnfAlgo::GetCNodeName(*(it - 1)) == kAtomicAddrCleanOpName) {
        moved_backward_cnodes->emplace_back(*(it - 1));
      }
      moved_backward_cnodes->emplace_back(*it);
    }
    it++;
  }

  size_t total_moved_size = it - last_grad_pos - 1;
  if (HasRefNodes(*moved_backward_cnodes) ||
      moved_backward_cnodes->size() + moved_forward_cnodes->size() != total_moved_size) {
    MS_LOG(INFO) << "Ref node was found or invalid number of moved nodes, give up optimization";
    return nullptr;
  }
  return GetTargetOutputNode(*moved_backward_cnodes, *it, graph_ptr);
}

CNodePtr AscendStreamAssign::GetTargetOutputNode(const vector<CNodePtr> &moved_backward_cnodes,
                                                 const CNodePtr first_node, const NotNull<KernelGraphPtr> &graph_ptr) {
  auto cnode_ptr_list = graph_ptr->execution_order();
  if (moved_backward_cnodes.empty() || !first_node) {
    return nullptr;
  }
  uint32_t subgraph_id = 0;
  bool get_subgraph_id = false;
  auto it = find(cnode_ptr_list.begin(), cnode_ptr_list.end(), first_node);
  CNodePtr first_output_node_ptr = nullptr;
  while (!get_subgraph_id && it < cnode_ptr_list.end()) {
    auto inputs = GetInputKernels(*it);
    for (auto &input : inputs) {
      if (find(moved_backward_cnodes.begin(), moved_backward_cnodes.end(), input) != moved_backward_cnodes.end()) {
        get_subgraph_id = true;
        subgraph_id = AnfAlgo::GetGraphId((*it).get());
        first_output_node_ptr = *it;
        break;
      }
    }
    it++;
  }
  if (subgraph_id == 0) {
    MS_LOG(INFO) << "The nodes moved backward were not used by any other nodes, no need moved";
    return nullptr;
  }

  for (; it < cnode_ptr_list.end() && AnfAlgo::GetGraphId((*it).get()) != subgraph_id; it++) {
    auto inputs = GetInputKernels(*it);
    for (auto &input : inputs) {
      if (find(moved_backward_cnodes.begin(), moved_backward_cnodes.end(), input) != moved_backward_cnodes.end()) {
        MS_LOG(INFO) << "The nodes moved backward were used by nodes on different subgraphs, no need moved";
        return nullptr;
      }
    }
  }
  return first_output_node_ptr;
}

bool AscendStreamAssign::FinetuneSubgraphExecOrder(vector<CNodePtr> *cnodes) {
  MS_EXCEPTION_IF_NULL(cnodes);
  auto hcom_pos = find_if(cnodes->begin(), cnodes->end(),
                          [](CNodePtr &node_ptr) -> bool { return AnfAlgo::GetCNodeName(node_ptr) == "AllReduce"; });
  if (hcom_pos == cnodes->end()) {
    return false;
  }
  CNodePtr hcom_ptr = *hcom_pos;

  vector<CNodePtr> ori_cnodes(cnodes->begin(), cnodes->end());
  cnodes->clear();
  vector<CNodePtr> atomic_addr_clean;
  for (auto iter = ori_cnodes.begin(); iter < ori_cnodes.end(); iter++) {
    if (AnfAlgo::GetCNodeName(*iter) == kAtomicAddrCleanOpName) {
      atomic_addr_clean.emplace_back(*iter);
      continue;
    }
    auto last_input_pos = cnodes->end();
    for (auto &input : GetInputKernels(*iter)) {
      auto pos = find(cnodes->begin(), cnodes->end(), input);
      if (pos != cnodes->end()) {
        last_input_pos = (last_input_pos == cnodes->end() || last_input_pos < pos) ? pos : last_input_pos;
      }
    }
    if (last_input_pos == cnodes->end()) {
      auto hcom_it = find(cnodes->begin(), cnodes->end(), hcom_ptr);
      if (hcom_it == cnodes->end() || AnfAlgo::GetCNodeName(*iter) == kLabelGotoOpName ||
          AnfAlgo::GetCNodeName(*iter) == kLabelSetOpName || AnfAlgo::GetCNodeName(*iter) == kLabelSwitchOpName) {
        cnodes->emplace_back(*iter);
      } else {
        cnodes->insert(hcom_it, *iter);
      }
    } else {
      cnodes->insert(last_input_pos + 1, *iter);
    }
  }

  for (auto &node : atomic_addr_clean) {
    auto first_input_pos = cnodes->end();
    for (auto &input : GetInputKernels(node)) {
      auto pos = find(cnodes->begin(), cnodes->end(), input);
      first_input_pos = (first_input_pos == cnodes->end() || first_input_pos > pos) ? pos : first_input_pos;
    }
    if (first_input_pos == cnodes->end()) {
      return false;
    } else {
      cnodes->insert(first_input_pos, node);
    }
  }
  return cnodes->size() == ori_cnodes.size();
}

// performance optimization for trailing time in distribute mode
// allreduce of the last batch of gradients and the optimizer can be done parallel
void AscendStreamAssign::TrailingTimeOptimizationByReorder(const NotNull<KernelGraphPtr> &graph_ptr) {
  vector<CNodePtr> last_grad_and_status;
  CheckScenario(graph_ptr, &last_grad_and_status);
  if (last_grad_and_status.empty()) {
    MS_LOG(INFO) << "Unsuitable scenario, no optimization needed";
    return;
  }

  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> moved_forward_cnodes;
  vector<CNodePtr> moved_backward_cnodes;
  CNodePtr first_output_ptr =
    GetCNodesNeededMoved(&moved_backward_cnodes, &moved_forward_cnodes, last_grad_and_status, graph_ptr);
  if (moved_backward_cnodes.empty() || first_output_ptr == nullptr) {
    MS_LOG(INFO) << "Unsuitable scenario, no optimization needed";
    return;
  }

  uint32_t subgraph_id = AnfAlgo::GetGraphId(first_output_ptr.get());
  auto last_grad_ptr = last_grad_and_status[0];
  auto last_grad_pos = find(cnode_ptr_list.begin(), cnode_ptr_list.end(), last_grad_ptr);
  vector<CNodePtr> cnodes(cnode_ptr_list.begin(), last_grad_pos + 1);
  cnodes.insert(cnodes.end(), moved_forward_cnodes.begin(), moved_forward_cnodes.end());
  auto pos = last_grad_pos + moved_forward_cnodes.size() + moved_backward_cnodes.size() + 1;
  while (pos < cnode_ptr_list.end() && AnfAlgo::GetGraphId((*pos).get()) != subgraph_id) {
    cnodes.emplace_back(*pos);
    pos++;
  }

  vector<CNodePtr> subgraph_cnodes;
  while (pos < cnode_ptr_list.end() && AnfAlgo::GetGraphId((*pos).get()) == subgraph_id) {
    if (AnfAlgo::GetCNodeName(*pos) == kLabelGotoOpName) {
      break;
    }
    if (*pos != first_output_ptr) {
      subgraph_cnodes.emplace_back(*pos);
    } else {
      subgraph_cnodes.insert(subgraph_cnodes.end(), moved_backward_cnodes.begin(), moved_backward_cnodes.end());
      subgraph_cnodes.emplace_back(*pos);
    }
    pos++;
  }

  if (!FinetuneSubgraphExecOrder(&subgraph_cnodes) || subgraph_cnodes.empty()) {
    MS_LOG(INFO) << "Finetune subgraph execute order failed, no optimization needed";
    return;
  }

  cnodes.insert(cnodes.end(), subgraph_cnodes.begin(), subgraph_cnodes.end());
  cnodes.insert(cnodes.end(), pos, cnode_ptr_list.end());
  if (cnodes.size() != cnode_ptr_list.size()) {
    return;
  }
  for (auto &node : subgraph_cnodes) {
    AnfAlgo::SetGraphId(subgraph_id, node.get());
  }

  graph_ptr->set_execution_order(cnodes);
}

// section 2
void AscendStreamAssign::AssignAllNodesStream(const NotNull<KernelGraphPtr> &graph_ptr) {
  auto cnode_ptr_list = graph_ptr->execution_order();
  bool exit_independent = false;
  bool exit_hcom = false;
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    // node has been assigned stream before
    if (AnfAlgo::GetStreamId(cur_cnode_ptr) != kInvalidStreamId) {
      continue;
    }

    if (IsHcom(cur_cnode_ptr)) {
      exit_hcom = true;
      continue;
    }

    if (AnfAlgo::IsIndependentNode(cur_cnode_ptr)) {
      exit_independent = true;
      continue;
    }

    AssignCommonStreamId(cur_cnode_ptr);
  }

  auto common_stream_num = resource_manager.get_cur_stream_num();

  if (exit_hcom) {
    AssignHcom(graph_ptr);
  }
  auto hcom_stream_num = resource_manager.get_cur_stream_num() - common_stream_num;

  if (exit_independent) {
    AssignIndependent(graph_ptr);
  }
  auto independent_stream_num = resource_manager.get_cur_stream_num() - common_stream_num - hcom_stream_num;
  auto total_stream_num = resource_manager.get_cur_stream_num() + hcom_stream_num * kHcomSecondaryStreamNum;
  MS_LOG(INFO) << "Total stream number: " << total_stream_num << ", common stream number: " << common_stream_num
               << ", hcom stream number: " << hcom_stream_num << "*" << kHcomSecondaryStreamNum + 1
               << ", independent stream number: " << independent_stream_num << ".";

  if (total_stream_num > kMaxStreamNum) {
    MS_LOG(EXCEPTION) << "Total stream number " << total_stream_num << " exceeds the limit of " << kMaxStreamNum << ".";
  }

  MS_LOG(INFO) << "After stream assign, total stream nums:" << resource_manager.get_cur_stream_num();
}

void AscendStreamAssign::AssignCommonStreamId(const CNodePtr &cur_cnode_ptr) {
  MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  uint32_t cur_common_stream_id = 0;
  uint32_t cur_stream_num = resource_manager.get_cur_stream_num();
  if (cur_stream_num == 0) {
    cur_common_stream_id = resource_manager.ApplyNewStream();
  } else {
    cur_common_stream_id = resource_manager.GetCurAllocStreamId();
  }

  auto it = common_stream_map_.find(cur_common_stream_id);
  if (it == common_stream_map_.end()) {
    AnfAlgo::SetStreamId(cur_common_stream_id, cur_cnode_ptr.get());
    common_stream_map_.insert(std::make_pair(cur_common_stream_id, 1));
  } else {
    if (it->second < kMaxCommonNodeNumPerStream) {
      AnfAlgo::SetStreamId(it->first, cur_cnode_ptr.get());
      it->second++;
    } else {
      cur_common_stream_id = resource_manager.ApplyNewStream();
      AnfAlgo::SetStreamId(cur_common_stream_id, cur_cnode_ptr.get());
      common_stream_map_.insert(std::make_pair(cur_common_stream_id, 1));
    }
  }
}

void AscendStreamAssign::AssignHcom(const NotNull<KernelGraphPtr> &graph_ptr) {
  auto cnode_ptr_list = graph_ptr->execution_order();
  std::map<std::string, std::map<uint32_t, std::vector<CNodePtr>>> group_graph_nodes_map;
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    // node has been assigned stream before
    if (AnfAlgo::GetStreamId(cur_cnode_ptr) != kInvalidStreamId) {
      continue;
    }

    if (IsHcom(cur_cnode_ptr)) {
      auto group_name = GetHcomGroup(cur_cnode_ptr);
      auto hcom_graph_id = AnfAlgo::GetGraphId(cur_cnode_ptr.get());
      auto iter = group_graph_nodes_map.find(group_name);
      if (iter == group_graph_nodes_map.end()) {
        std::map<uint32_t, std::vector<CNodePtr>> graph_nodes_map;
        graph_nodes_map[hcom_graph_id] = {cur_cnode_ptr};
        group_graph_nodes_map[group_name] = graph_nodes_map;
      } else {
        auto &graph_nodes_map = iter->second;
        auto it = graph_nodes_map.find(hcom_graph_id);
        if (it == graph_nodes_map.end()) {
          graph_nodes_map[hcom_graph_id] = {cur_cnode_ptr};
        } else {
          it->second.emplace_back(cur_cnode_ptr);
        }
      }
    }
  }

  MS_LOG(INFO) << "hcom diff group size:" << group_graph_nodes_map.size();
  for (const auto &item : group_graph_nodes_map) {
    MS_LOG_INFO << "group id:" << item.first << "; diff graph id size:" << item.second.size();
  }

  for (const auto &diff_group : group_graph_nodes_map) {
    // group id:
    std::map<uint32_t, std::set<uint32_t>> hcom_graph_map;
    for (const auto &item : diff_group.second) {
      bool new_graph = true;
      auto graph_id = item.first;
      hcom_graph_map[graph_id] = {};
      for (const auto &hcom_node_ptr : item.second) {
        auto assigned_stream_id = AssignHcomStreamId(hcom_node_ptr, new_graph);
        hcom_graph_map[graph_id].emplace(assigned_stream_id);
        new_graph = false;
      }
    }
    group_hcom_graph_map_[diff_group.first] = hcom_graph_map;
  }
}

uint32_t AscendStreamAssign::AssignHcomStreamId(const CNodePtr &cur_cnode_ptr, bool new_graph) {
  MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  auto task_num = GetHcomTaskNum(cur_cnode_ptr);

  uint32_t cur_hcom_stream_id;
  if (new_graph) {
    cur_hcom_stream_id = resource_manager.ApplyNewStream();
  } else {
    cur_hcom_stream_id = resource_manager.GetCurAllocStreamId();
  }
  auto it = hcom_stream_map_.find(cur_hcom_stream_id);
  if (it == hcom_stream_map_.end()) {
    AnfAlgo::SetStreamId(cur_hcom_stream_id, cur_cnode_ptr.get());
    hcom_stream_map_.emplace(cur_hcom_stream_id, task_num);
  } else {
    if (it->second <= kMaxTaskNumPerStream - task_num) {
      AnfAlgo::SetStreamId(it->first, cur_cnode_ptr.get());
      it->second += task_num;
    } else {
      cur_hcom_stream_id = resource_manager.ApplyNewStream();
      AnfAlgo::SetStreamId(cur_hcom_stream_id, cur_cnode_ptr.get());
      hcom_stream_map_.emplace(cur_hcom_stream_id, task_num);
    }
  }
  return cur_hcom_stream_id;
}

void AscendStreamAssign::AssignIndependent(const NotNull<KernelGraphPtr> &graph_ptr) {
  auto cnode_ptr_list = graph_ptr->execution_order();
  std::map<uint32_t, std::vector<CNodePtr>> graph_nodes_map;
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    if (AnfAlgo::GetStreamId(cur_cnode_ptr) != kInvalidStreamId) {
      continue;
    }
    if (AnfAlgo::IsIndependentNode(cur_cnode_ptr)) {
      auto independent_graph_id = AnfAlgo::GetGraphId(cur_cnode_ptr.get());
      auto it = graph_nodes_map.find(independent_graph_id);
      if (it == graph_nodes_map.end()) {
        graph_nodes_map[independent_graph_id] = {cur_cnode_ptr};
      } else {
        it->second.emplace_back(cur_cnode_ptr);
      }
    }
  }

  MS_LOG(INFO) << "independent diff graph id size:" << graph_nodes_map.size();
  for (const auto &item : graph_nodes_map) {
    bool new_graph = true;
    auto graph_id = item.first;
    independent_graph_map_[graph_id] = {};
    for (const auto &independent_node_ptr : item.second) {
      auto assigned_stream_id = AssignIndependentStreamId(independent_node_ptr, new_graph);
      independent_graph_map_[graph_id].emplace(assigned_stream_id);
      new_graph = false;
    }
  }
  MS_LOG(INFO) << "stream nums:" << independent_stream_map_.size();
}

uint32_t AscendStreamAssign::AssignIndependentStreamId(const CNodePtr &cur_cnode_ptr, bool new_graph) {
  MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  uint32_t cur_independent_stream_id;
  if (new_graph) {
    cur_independent_stream_id = resource_manager.ApplyNewStream();
  } else {
    cur_independent_stream_id = resource_manager.GetCurAllocStreamId();
  }
  auto it = independent_stream_map_.find(cur_independent_stream_id);
  if (it == independent_stream_map_.end()) {
    AnfAlgo::SetStreamId(cur_independent_stream_id, cur_cnode_ptr.get());
    independent_stream_map_.insert(std::make_pair(cur_independent_stream_id, 1));
  } else {
    if (it->second < kMaxCommonNodeNumPerStream) {
      AnfAlgo::SetStreamId(it->first, cur_cnode_ptr.get());
      it->second++;
    } else {
      cur_independent_stream_id = resource_manager.ApplyNewStream();
      AnfAlgo::SetStreamId(cur_independent_stream_id, cur_cnode_ptr.get());
      independent_stream_map_.insert(std::make_pair(cur_independent_stream_id, 1));
    }
  }

  return cur_independent_stream_id;
}

// section 3:
void AscendStreamAssign::UpdateAtomicAddrCleanStreamId(const NotNull<KernelGraphPtr> &graph_ptr) {
  MS_LOG(INFO) << "Start";
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    // update AtomicAddrClean stream same with the next node
    if (i > 0 && AnfAlgo::GetCNodeName(cnode_ptr_list[i - 1]) == kAtomicAddrCleanOpName) {
      AnfAlgo::SetStreamId(AnfAlgo::GetStreamId(cur_cnode_ptr), cnode_ptr_list[i - 1].get());
    }
  }
  MS_LOG(INFO) << "End";
}

// section 4
void AscendStreamAssign::InsertStreamActive(const NotNull<KernelGraphPtr> &graph_ptr) {
  InsertStreamActiveForCommon(graph_ptr);
  InsertStreamActiveForIndependent(graph_ptr);
  InsertStreamActiveForParallel(graph_ptr);
}

void AscendStreamAssign::InsertStreamActiveForParallel(const NotNull<KernelGraphPtr> &graph_ptr) {
  if (group_hcom_graph_map_.empty() && independent_graph_map_.empty()) {
    MS_LOG(INFO) << "Hcom and independent is empty";
    return;
  }
  auto root_graph_id = graph_ptr->graph_id();
  if (root_graph_id == kInvalidGraphId) {
    MS_LOG(INFO) << "Root graph id is invalid";
    return;
  }

  std::map<uint32_t, std::set<uint32_t>> other_graph;
  std::set<uint32_t> hcom_streams;
  for (const auto &graph_nodes : group_hcom_graph_map_) {
    for (const auto &item : graph_nodes.second) {
      MS_LOG(INFO) << "Graph id:" << item.first;
      if (item.first == root_graph_id) {
        if (loop_sink_) {
          hcom_streams.insert(item.second.begin(), item.second.end());
        }
      } else {
        auto it = other_graph.find(item.first);
        if (it == other_graph.end()) {
          other_graph[item.first] = item.second;
        } else {
          for (const auto &stream : item.second) {
            it->second.emplace(stream);
          }
        }
      }
    }
  }

  if (!hcom_streams.empty()) {
    ActiveRootGraphHcom(graph_ptr, hcom_streams);
  }

  MS_LOG(INFO) << "Independent graph map size:" << independent_graph_map_.size();
  for (const auto &item : independent_graph_map_) {
    MS_LOG(DEBUG) << "Graph id:" << item.first;
    if (item.first == root_graph_id) {
      if (loop_sink_) {
        ActiveRootGraphIndependent(graph_ptr, item.second);
      }
    } else {
      auto it = other_graph.find(item.first);
      if (it == other_graph.end()) {
        other_graph[item.first] = item.second;
      } else {
        for (const auto &stream : item.second) {
          it->second.emplace(stream);
        }
      }
    }
  }

  ActiveOtherGraphParallel(graph_ptr, other_graph);
}

void AscendStreamAssign::ActiveOtherGraphParallel(const NotNull<KernelGraphPtr> &graph_ptr,
                                                  std::map<uint32_t, std::set<uint32_t>> other_graph) {
  MS_LOG(INFO) << "Other graph size:" << other_graph.size();
  if (other_graph.empty()) {
    return;
  }

  auto root_graph_id = graph_ptr->graph_id();

  std::vector<CNodePtr> update_stream_list;
  auto exe_order = graph_ptr->execution_order();
  for (size_t i = 0; i < exe_order.size(); i++) {
    auto cur_cnode_ptr = exe_order[i];
    auto cur_graph_id = AnfAlgo::GetGraphId(cur_cnode_ptr.get());
    if (cur_graph_id == root_graph_id) {
      update_stream_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    auto it = other_graph.find(cur_graph_id);
    if (it == other_graph.end()) {
      update_stream_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    auto cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
    // 1.set stream id
    AnfAlgo::SetStreamId(cur_stream_id, active_ptr.get());
    // 2.set active stream ids
    std::vector<uint32_t> active_index_list;
    std::copy(it->second.begin(), it->second.end(), std::back_inserter(active_index_list));
    AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_index_list), active_ptr);

    // find position for insert streamactive
    if (AnfAlgo::GetCNodeName(cur_cnode_ptr) == kLabelSetOpName) {
      update_stream_list.emplace_back(cur_cnode_ptr);
      update_stream_list.emplace_back(active_ptr);
    } else {
      update_stream_list.emplace_back(active_ptr);
      update_stream_list.emplace_back(cur_cnode_ptr);
    }
    other_graph.erase(it);
  }
  graph_ptr->set_execution_order(update_stream_list);
}

void AscendStreamAssign::ActiveRootGraphHcom(const NotNull<KernelGraphPtr> &graph_ptr,
                                             const std::set<uint32_t> &hcom_streams) {
  MS_LOG(INFO) << "Active root graph hcom start";
  std::vector<CNodePtr> update_cnode_list;
  auto exe_orders = graph_ptr->execution_order();
  for (size_t i = 0; i < exe_orders.size(); i++) {
    CNodePtr cur_cnode_ptr = exe_orders[i];
    if (AnfAlgo::GetCNodeName(cur_cnode_ptr) != kStreamSwitchOpName) {
      update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    if (!AnfAlgo::HasNodeAttr(kAttrStreamSwitchKind, cur_cnode_ptr)) {
      update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    auto kind = AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode_ptr, kAttrStreamSwitchKind);
    if (kind != kFpBpStreamSwitch) {
      update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    auto true_stream_id = AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode_ptr, kAttrTrueBranchStream);
    MS_LOG(INFO) << "FpBpStreamswtich stream id:" << AnfAlgo::GetStreamId(cur_cnode_ptr)
                 << "; true branch stream id:" << true_stream_id;
    CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
    AnfAlgo::SetStreamId(true_stream_id, active_ptr.get());
    vector<uint32_t> active_ids;
    // active hcom stream
    std::copy(hcom_streams.begin(), hcom_streams.end(), std::back_inserter(active_ids));
    AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_ids), active_ptr);
    update_cnode_list.emplace_back(cur_cnode_ptr);
    update_cnode_list.emplace_back(active_ptr);
    std::copy(exe_orders.begin() + i + 1, exe_orders.end(), std::back_inserter(update_cnode_list));
    break;
  }

  hcom_stream_activated_ = true;
  graph_ptr->set_execution_order(update_cnode_list);
}

void AscendStreamAssign::ActiveRootGraphIndependent(const NotNull<KernelGraphPtr> &graph_ptr,
                                                    std::set<uint32_t> independent_streams) {
  MS_LOG(DEBUG) << "Start active root graph independent";
  std::vector<CNodePtr> update_cnode_list;
  auto exe_orders = graph_ptr->execution_order();
  for (size_t i = 0; i < exe_orders.size(); i++) {
    CNodePtr cur_cnode_ptr = exe_orders[i];
    if (AnfAlgo::GetCNodeName(cur_cnode_ptr) != kStreamSwitchOpName) {
      update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    if (!AnfAlgo::HasNodeAttr(kAttrStreamSwitchKind, cur_cnode_ptr)) {
      update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    auto kind = AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode_ptr, kAttrStreamSwitchKind);
    if (kind != kIndependentStreamSwitch) {
      update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    // first independetn stream id is minimum and order by std map;
    auto first_independent_stream = *(independent_streams.begin());
    AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, MakeValue<uint32_t>(first_independent_stream), cur_cnode_ptr);
    update_cnode_list.emplace_back(cur_cnode_ptr);
    std::copy(exe_orders.begin() + i + 1, exe_orders.end(), std::back_inserter(update_cnode_list));
    break;
  }

  independent_stream_activated_ = true;
  graph_ptr->set_execution_order(update_cnode_list);
}
void AscendStreamAssign::InsertStreamActiveForCommon(const NotNull<KernelGraphPtr> &graph_ptr) {
  MS_LOG(INFO) << "Start";
  GetProcessedStream(graph_ptr);
  std::vector<CNodePtr> update_cnode_list;
  CNodePtr cur_cnode_ptr = nullptr;
  CNodePtr pre_cnode_ptr = nullptr;
  uint32_t pre_stream_id = UINT32_MAX;

  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (AnfAlgo::IsIndependentNode(cur_cnode_ptr)) {
      update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    if (IsHcom(cur_cnode_ptr)) {
      update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }
    uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    bool processed = IsProcessedStream(cur_stream_id);
    // 1)inner stream assign, need insert active op
    if (!processed) {
      MS_LOG(INFO) << "Common stream active info:" << pre_stream_id << "->active" << cur_stream_id;
      CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
      // 1.set stream id
      AnfAlgo::SetStreamId(pre_stream_id, active_ptr.get());
      // 2.set active stream ids
      std::vector<uint32_t> active_index_list{cur_stream_id};
      AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_index_list), active_ptr);
      update_cnode_list.emplace_back(active_ptr);
    }

    if (AnfAlgo::GetCNodeName(cur_cnode_ptr) == kStreamSwitchOpName) {
      MS_LOG(INFO) << "Insert StreamActive op after FP StreamSwitch for stream parallel";
      update_cnode_list.emplace_back(cur_cnode_ptr);
    } else {
      update_cnode_list.emplace_back(cur_cnode_ptr);
    }

    processed_streams_.emplace(cur_stream_id);
    pre_stream_id = cur_stream_id;
    pre_cnode_ptr = cur_cnode_ptr;
  }
  graph_ptr->set_execution_order(update_cnode_list);
}

void AscendStreamAssign::InsertStreamActiveForIndependent(const NotNull<KernelGraphPtr> &graph_ptr) {
  auto root_graph_id = graph_ptr->graph_id();
  if (root_graph_id == kInvalidGraphId) {
    return;
  }
  std::set<uint32_t> independent_streams;
  for (const auto &item : independent_graph_map_) {
    if (item.first == root_graph_id) {
      independent_streams = item.second;
    }
  }

  // Root graph independent stream size is not more than one, no need insert active
  if (independent_streams.size() <= 1) {
    return;
  }
  std::vector<CNodePtr> update_cnode_list;
  auto exe_orders = graph_ptr->execution_order();

  // first independent is been activated, active other independent stream
  std::vector<uint32_t> streams;
  std::copy(independent_streams.begin(), independent_streams.end(), std::back_inserter(streams));
  std::sort(streams.begin(), streams.end());
  uint32_t node_num = 0;
  for (size_t i = 0; i < exe_orders.size(); i++) {
    auto cur_cnode_ptr = exe_orders[i];
    update_cnode_list.emplace_back(cur_cnode_ptr);
    if (!AnfAlgo::IsIndependentNode(cur_cnode_ptr)) {
      continue;
    }

    if (AnfAlgo::GetGraphId(cur_cnode_ptr.get()) != root_graph_id) {
      continue;
    }

    node_num++;
    auto cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    auto it = std::find(streams.begin(), streams.end(), cur_stream_id);
    if (it == streams.end()) {
      MS_LOG(EXCEPTION) << "Can't find independent stream id:" << cur_stream_id;
    } else if (it == streams.end() - 1) {
      std::copy(exe_orders.begin() + i + 1, exe_orders.end(), std::back_inserter(update_cnode_list));
      break;
    } else {
      if (node_num == kMaxCommonNodeNumPerStream) {
        CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
        // 1.set stream id
        AnfAlgo::SetStreamId(cur_stream_id, active_ptr.get());
        // 2.set active stream ids
        std::vector<uint32_t> active_index_list{*(it + 1)};
        AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_index_list), active_ptr);
        update_cnode_list.emplace_back(active_ptr);
        node_num = 0;
      }
    }
  }
  graph_ptr->set_execution_order(update_cnode_list);
}

void AscendStreamAssign::GetProcessedStream(const NotNull<KernelGraphPtr> &graph_ptr) {
  // 0 stream is activated at first
  processed_streams_.emplace(0);
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    auto cur_cnode_ptr = cnode_ptr_list[i];
    uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);

    if (AnfAlgo::GetCNodeName(cur_cnode_ptr) == kStreamSwitchOpName) {
      if (AnfAlgo::HasNodeAttr(kAttrTrueBranchStream, cur_cnode_ptr)) {
        auto true_stream_id = AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode_ptr, kAttrTrueBranchStream);
        processed_streams_.emplace(true_stream_id);
      }

      if (!AnfAlgo::HasNodeAttr(kStreamNeedActivedFirst, cur_cnode_ptr)) {
        continue;
      }
      auto need_active = AnfAlgo::GetNodeAttr<bool>(cur_cnode_ptr, kStreamNeedActivedFirst);
      if (need_active) {
        processed_streams_.emplace(cur_stream_id);
      }
    }
  }
  for (const auto &item : processed_streams_) {
    MS_LOG(INFO) << "Before active:" << item << " is been processed";
  }
}

bool AscendStreamAssign::CheckStreamSwitch(const CNodePtr &switch_ptr) {
  if (!AnfAlgo::HasNodeAttr(kStreamNeedActivedFirst, switch_ptr)) {
    return false;
  }

  auto need_active = AnfAlgo::GetNodeAttr<bool>(switch_ptr, kStreamNeedActivedFirst);
  if (!need_active) {
    return false;
  }

  if (!AnfAlgo::HasNodeAttr(kAttrStreamSwitchKind, switch_ptr)) {
    return false;
  }

  auto kind = AnfAlgo::GetNodeAttr<uint32_t>(switch_ptr, kAttrStreamSwitchKind);
  if (kind == kEosStreamSwitch || kind == kGetNextStreamSwitch) {
    return false;
  }

  return true;
}

void AscendStreamAssign::UpdateStreamSwitch(const NotNull<KernelGraphPtr> &graph_ptr, const CNodePtr &switch_ptr,
                                            vector<CNodePtr> *orders) {
  if (!CheckStreamSwitch(switch_ptr)) {
    orders->emplace_back(switch_ptr);
    return;
  }

  auto kind = AnfAlgo::GetNodeAttr<uint32_t>(switch_ptr, kAttrStreamSwitchKind);
  if (kind == kIndependentStreamSwitch) {
    bool independent_empty = independent_stream_map_.empty();
    // if independent empty: delete independent streamswitch
    if (!independent_empty) {
      for (const auto &item : independent_stream_map_) {
        // first independent stream id is minimum and order by std map;
        auto first_independent_stream = item.first;
        AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, MakeValue<uint32_t>(first_independent_stream), switch_ptr);
        orders->emplace_back(switch_ptr);
        break;
      }
    } else {
      MS_LOG(ERROR) << "Independent stream switch exit, but independent stream is empty";
    }

    // update processed stream
    independent_stream_activated_ = true;
    for (const auto &item : independent_stream_map_) {
      processed_streams_.emplace(item.first);
    }
  } else if (kind == kFpBpStreamSwitch) {
    if (hcom_stream_map_.empty()) {
      orders->emplace_back(switch_ptr);
      return;
    }
    if (!AnfAlgo::HasNodeAttr(kAttrTrueBranchStream, switch_ptr)) {
      // FpBp StreamSwitch has no true branch attr
      orders->emplace_back(switch_ptr);
      return;
    }
    auto true_stream_id = AnfAlgo::GetNodeAttr<uint32_t>(switch_ptr, kAttrTrueBranchStream);
    MS_LOG(INFO) << "Switch stream id:" << AnfAlgo::GetStreamId(switch_ptr) << "; active stream id:" << true_stream_id;
    CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
    AnfAlgo::SetStreamId(true_stream_id, active_ptr.get());
    vector<uint32_t> active_ids;
    // active hcom stream
    for (const auto &item : hcom_stream_map_) {
      active_ids.emplace_back(item.first);
    }
    AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_ids), active_ptr);
    hcom_stream_activated_ = true;
    for (const auto &item : hcom_stream_map_) {
      processed_streams_.emplace(item.first);
    }
    orders->emplace_back(switch_ptr);
    orders->emplace_back(active_ptr);
  }
}

bool AscendStreamAssign::IsProcessedStream(uint32_t stream_id) {
  auto it = std::find(processed_streams_.begin(), processed_streams_.end(), stream_id);
  if (it != processed_streams_.end()) {
    return true;
  }
  return false;
}

bool AscendStreamAssign::IsAllOutGraphOut(const KernelGraphPtr &graph, const CNodePtr &cnode) {
  auto cnode_out_num = AnfAlgo::GetOutputTensorNum(cnode);
  auto nodes = AnfAlgo::GetAllOutput(graph->output(), {prim::kPrimTupleGetItem});
  std::set<int> output_index_set;
  // Assign Communicate Op Memory firstly.
  for (const auto &node : nodes) {
    auto item_with_index = AnfAlgo::VisitKernelWithReturnType(node, 0, true);
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    if (!item_with_index.first->isa<CNode>() || !AnfAlgo::IsRealKernel(item_with_index.first)) {
      continue;
    }
    if (item_with_index.first == cnode) {
      output_index_set.insert(item_with_index.second);
    }
  }

  MS_LOG(INFO) << "Node " << cnode->fullname_with_scope() << " has " << cnode_out_num
               << " outputs, in graph output num:" << output_index_set.size();
  return cnode_out_num == output_index_set.size();
}

vector<CNodePtr>::iterator AscendStreamAssign::FindGraphEnd(vector<CNodePtr>::iterator begin,
                                                            vector<CNodePtr>::iterator end) {
  while (begin != end) {
    if (AnfAlgo::HasNodeAttr(kAttrFpBpEnd, *begin)) {
      MS_LOG(INFO) << "FpBp end op is " << (*begin)->fullname_with_scope();
      return begin;
    }
    ++begin;
  }
  return end;
}

// section5
void AscendStreamAssign::InsertEventForHcomParallel(const NotNull<KernelGraphPtr> &graph_ptr) {
  MS_LOG(INFO) << "Start";
  InsertEventCommonDependHcom(graph_ptr);
  InsertEventHcomDependCommonBak(graph_ptr);
  InsertEventHcomDependHcom(graph_ptr);
  MS_LOG(INFO) << "End";
}

void AscendStreamAssign::InsertEventCommonDependHcom(const NotNull<KernelGraphPtr> &graph_ptr) {
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> cnodes = cnode_ptr_list;
  uint32_t cur_event_id = resource_manager.ApplyNewEvent();
  auto it = cnodes.begin();
  while (it != cnodes.end()) {
    MS_EXCEPTION_IF_NULL(*it);
    if (IsHcom(*it)) {
      auto cur_hcom_node = *it;
      CNodePtr send_cnode_ptr = CreateSendApplyKernel(graph_ptr, cur_event_id, AnfAlgo::GetStreamId(*it));
      it = cnodes.insert(it + 1, send_cnode_ptr);

      auto target = FindTargetOp(it, cnodes.end(), cur_hcom_node, true);
      if (target == cnodes.end()) {
        if (IsAllOutGraphOut(graph_ptr, cur_hcom_node)) {
          // if hcom's all output is graph output, we need to insert send/recv to fpbp end in data sink mode
          target = FindGraphEnd(it, cnodes.end());
        }

        if (target == cnodes.end()) {
          MS_LOG(WARNING) << "Hcom node:" << (*(it - 1))->fullname_with_scope()
                          << ", can't find target for insert recv op, no insert send/recv";
          it = cnodes.erase(it);
          continue;
        }
      }

      // deal recv op
      uint32_t stream_id = AnfAlgo::GetStreamId(*target);
      CNodePtr recv_cnode_ptr = CreateRecvApplyKernel(graph_ptr, cur_event_id, stream_id);
      (void)cnodes.insert(target, recv_cnode_ptr);
      cur_event_id = resource_manager.ApplyNewEvent();
    }
    ++it;
  }
  // one event allocated additional, should delete
  resource_manager.DeleteEvent();
  graph_ptr->set_execution_order(cnodes);
  MS_LOG(INFO) << "After common depend hcom, total event nums:" << resource_manager.get_cur_event_num();
}

// after memory reuse is correct, use this function
void AscendStreamAssign::InsertEventHcomDependCommonBak(const NotNull<KernelGraphPtr> &graph_ptr) {
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> cnodes;
  CNodePtr cur_cnode_ptr = nullptr;
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (i == 0) {
      cnodes.emplace_back(cur_cnode_ptr);
      continue;
    }

    if (!IsHcom(cur_cnode_ptr)) {
      cnodes.emplace_back(cur_cnode_ptr);
      continue;
    }

    // get the input which located in the last exe orders
    vector<CNodePtr> inputs_cnode = GetLastInputCnode(graph_ptr, cur_cnode_ptr);
    if (inputs_cnode.empty()) {
      cnodes.emplace_back(cur_cnode_ptr);
      MS_LOG(WARNING) << "Hcom op:" << AnfAlgo::GetCNodeName(cur_cnode_ptr) << " can't find inputs nodes";
      continue;
    }

    MS_LOG(INFO) << "Current hcom:" << AnfAlgo::GetCNodeName(cur_cnode_ptr)
                 << "; inputs cnode size:" << inputs_cnode.size();

    for (size_t j = 0; j < inputs_cnode.size(); j++) {
      auto &cur_input = inputs_cnode.at(j);
      MS_LOG(INFO) << "The index:" << j << " input, name:" << AnfAlgo::GetCNodeName(cur_input);
      uint32_t cur_event_id = resource_manager.ApplyNewEvent();
      auto pre_stream_id = AnfAlgo::GetStreamId(cur_input);
      auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, pre_stream_id);
      auto it = std::find(cnodes.begin(), cnodes.end(), cur_input);
      if (it == cnodes.end()) {
        MS_LOG_EXCEPTION << "Hcom:" << AnfAlgo::GetCNodeName(cur_cnode_ptr)
                         << " can't find input node:" << AnfAlgo::GetCNodeName(cur_input);
      }
      cnodes.insert(it + 1, send);
      uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
      auto recv = CreateRecvApplyKernel(graph_ptr, cur_event_id, cur_stream_id);
      cnodes.emplace_back(recv);
      cnodes.emplace_back(cur_cnode_ptr);
    }
  }

  graph_ptr->set_execution_order(cnodes);
  MS_LOG(INFO) << "After hcom depend common, total event nums:" << resource_manager.get_cur_event_num();
}

vector<CNodePtr> AscendStreamAssign::GetLastInputCnode(const NotNull<KernelGraphPtr> &graph_ptr,
                                                       const CNodePtr &cur_cnode_ptr) {
  auto cnode_ptr_list = graph_ptr->execution_order();
  auto group_name = GetHcomGroup(cur_cnode_ptr);
  auto input_cnodes = GetInputKernels(cur_cnode_ptr);
  if (input_cnodes.empty()) {
    return {};
  }
  // record max index node for each stream
  std::map<uint32_t, std::pair<CNodePtr, uint32_t>> result;
  for (size_t i = 0; i < input_cnodes.size(); i++) {
    auto &cur_input = input_cnodes.at(i);
    auto stream_id = AnfAlgo::GetStreamId(cur_input);
    auto cur_index = GetIndexByKey(graph_ptr, cur_input.get());
    if (cur_index == UINT32_MAX) {
      MS_LOG_EXCEPTION << "The input node:" << AnfAlgo::GetCNodeName(cur_input) << " is not found in graph";
    }
    auto it = result.find(stream_id);
    if (it == result.end()) {
      result[stream_id] = std::make_pair(cur_input, cur_index);
    } else {
      auto max_index = it->second.second;
      if (cur_index > max_index) {
        result[stream_id] = std::make_pair(cur_input, cur_index);
      }
    }
  }

  vector<CNodePtr> final_inputs;
  const uint32_t max = 0;
  CNodePtr max_common_cnode = nullptr;
  for (const auto &item : result) {
    if (IsHcom(item.second.first)) {
      auto cur_group = GetHcomGroup(item.second.first);
      if (cur_group == group_name) {
        continue;
      } else {
        final_inputs.emplace_back(item.second.first);
      }
    } else {
      if (item.second.second > max) {
        max_common_cnode = item.second.first;
      }
    }
  }

  if (max_common_cnode != nullptr) {
    final_inputs.emplace_back(max_common_cnode);
  }
  return final_inputs;
}

vector<CNodePtr> AscendStreamAssign::GetInputKernels(const CNodePtr &node) {
  vector<CNodePtr> input_cnodes;
  queue<CNodePtr> nop_nodes;
  auto inputs = node->inputs();
  for (size_t i = 1; i < inputs.size(); i++) {
    auto real_input = AnfAlgo::VisitKernel(inputs[i], 0);
    auto node = real_input.first;
    if (opt::IsNopNode(node)) {
      nop_nodes.push(node->cast<CNodePtr>());
      while (!nop_nodes.empty()) {
        auto cur_node = nop_nodes.front();
        nop_nodes.pop();
        auto new_inputs = cur_node->inputs();
        for (size_t j = 1; j < new_inputs.size(); j++) {
          auto new_real_input = AnfAlgo::VisitKernel(new_inputs[j], 0);
          auto new_node = new_real_input.first;
          if (opt::IsNopNode(new_node)) {
            nop_nodes.push(new_node->cast<CNodePtr>());
          } else if (new_node->isa<CNode>()) {
            input_cnodes.emplace_back(new_node->cast<CNodePtr>());
          }
        }
      }
    } else if (node->isa<CNode>()) {
      input_cnodes.emplace_back(node->cast<CNodePtr>());
    }
  }
  return input_cnodes;
}

void AscendStreamAssign::InsertEventHcomDependCommon(const NotNull<KernelGraphPtr> &graph_ptr) {
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> cnodes;
  CNodePtr cur_cnode_ptr = nullptr;
  uint32_t pre_stream_id = UINT32_MAX;
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (i == 0) {
      cnodes.emplace_back(cur_cnode_ptr);
      pre_stream_id = cur_stream_id;
      continue;
    }

    if (!IsHcom(cur_cnode_ptr)) {
      cnodes.emplace_back(cur_cnode_ptr);
      pre_stream_id = cur_stream_id;
      continue;
    }

    if (cur_stream_id == pre_stream_id) {
      cnodes.emplace_back(cur_cnode_ptr);
      pre_stream_id = cur_stream_id;
      continue;
    }

    if (!IsHcom(cnode_ptr_list[i - 1])) {
      uint32_t cur_event_id = resource_manager.ApplyNewEvent();
      auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, pre_stream_id);
      cnodes.emplace_back(send);
      auto recv = CreateRecvApplyKernel(graph_ptr, cur_event_id, cur_stream_id);
      cnodes.emplace_back(recv);
      cnodes.emplace_back(cur_cnode_ptr);
    } else {
      cnodes.emplace_back(cur_cnode_ptr);
    }
    pre_stream_id = cur_stream_id;
  }

  graph_ptr->set_execution_order(cnodes);
  MS_LOG(INFO) << "After hcom depend common, total event nums:" << resource_manager.get_cur_event_num();
}

void AscendStreamAssign::InsertEventHcomDependHcom(const NotNull<KernelGraphPtr> &graph_ptr) {
  if (group_hcom_graph_map_.empty()) {
    return;
  }
  std::vector<string> groups;
  for (const auto &item : group_hcom_graph_map_) {
    groups.emplace_back(item.first);
  }
  for (const auto &group : groups) {
    auto cnode_ptr_list = graph_ptr->execution_order();
    std::vector<std::pair<uint32_t, vector<size_t>>> stream_indices;
    for (size_t i = 0; i < cnode_ptr_list.size(); i++) {
      auto cur_cnode = cnode_ptr_list[i];
      if (!IsHcom(cur_cnode)) {
        continue;
      }

      uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode);
      auto group_name = GetHcomGroup(cur_cnode);
      MS_LOG(INFO) << "Hcom node name:" << AnfAlgo::GetCNodeName(cur_cnode) << "; group:" << group_name
                   << "; stream id:" << cur_stream_id;
      if (group_name != group) {
        continue;
      }

      if (stream_indices.empty()) {
        stream_indices.emplace_back(std::make_pair(cur_stream_id, std::vector<size_t>{i}));
      } else {
        bool exit = false;
        for (auto &item : stream_indices) {
          if (item.first == cur_stream_id) {
            item.second.emplace_back(i);
            exit = true;
            break;
          }
        }
        if (!exit) {
          stream_indices.emplace_back(std::make_pair(cur_stream_id, std::vector<size_t>{i}));
        }
      }
    }

    if (stream_indices.size() < 2) {
      MS_LOG(INFO) << "Group:" << group
                   << "; different stream hcom size is less than 2, no need insert event between them";
      continue;
    }
    InsertEventBetweenHcom(graph_ptr, stream_indices);
  }
}

void AscendStreamAssign::InsertEventBetweenHcom(const NotNull<KernelGraphPtr> &graph_ptr,
                                                const std::vector<std::pair<uint32_t, vector<size_t>>> &hcom_index) {
  vector<CNodePtr> orders;
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  auto cnode_ptr_list = graph_ptr->execution_order();
  uint32_t cur_event_id = resource_manager.ApplyNewEvent();
  size_t first_stream_last_index = hcom_index[0].second.back();
  size_t last_stream_first_index = hcom_index.back().second.front();
  MS_LOG(INFO) << "First stream last index:" << first_stream_last_index
               << "; last stream first index:" << last_stream_first_index;
  std::copy(cnode_ptr_list.begin(), cnode_ptr_list.begin() + first_stream_last_index, std::back_inserter(orders));
  for (size_t i = first_stream_last_index; i <= last_stream_first_index; i++) {
    auto cur_cnode = cnode_ptr_list[i];
    if (!IsSatisfiedHcom(hcom_index, cur_cnode, i)) {
      orders.emplace_back(cur_cnode);
      continue;
    }
    auto cur_hcom_stream_id = AnfAlgo::GetStreamId(cur_cnode);
    if (i == first_stream_last_index) {
      // first fusion hcom
      orders.emplace_back(cur_cnode);
      auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
      orders.emplace_back(send);
    } else if (i == last_stream_first_index) {
      // last fusion hcom
      auto recv = CreateRecvApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
      orders.emplace_back(recv);
      orders.emplace_back(cur_cnode);
    } else {
      size_t cur_stream_hcom_size = UINT32_MAX;
      size_t first_index = UINT32_MAX;
      size_t last_index = UINT32_MAX;
      for (const auto &item : hcom_index) {
        if (item.first == cur_hcom_stream_id) {
          cur_stream_hcom_size = item.second.size();
          first_index = item.second.front();
          last_index = item.second.back();
        }
      }

      if (cur_stream_hcom_size == 1) {
        auto recv = CreateRecvApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
        orders.emplace_back(recv);
        cur_event_id = resource_manager.ApplyNewEvent();
        orders.emplace_back(cur_cnode);
        auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
        orders.emplace_back(send);
      } else {
        // current stream, first hcom:add recv op
        if (i == first_index) {
          auto recv = CreateRecvApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
          orders.emplace_back(recv);
          cur_event_id = resource_manager.ApplyNewEvent();
          orders.emplace_back(cur_cnode);
        } else if (i == last_index) {
          // current stream, last hcom:add send op
          orders.emplace_back(cur_cnode);
          auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
          orders.emplace_back(send);
        } else {
          // current stream, not first and last op
          orders.emplace_back(cur_cnode);
        }
      }
    }
  }
  std::copy(cnode_ptr_list.begin() + last_stream_first_index + 1, cnode_ptr_list.end(), std::back_inserter(orders));
  graph_ptr->set_execution_order(orders);
}

bool AscendStreamAssign::IsSatisfiedHcom(const std::vector<std::pair<uint32_t, vector<size_t>>> &hcom_index,
                                         const CNodePtr &node_ptr, size_t index) {
  MS_EXCEPTION_IF_NULL(node_ptr);
  auto cur_hcom_stream_id = AnfAlgo::GetStreamId(node_ptr);
  for (const auto &item : hcom_index) {
    if (item.first == cur_hcom_stream_id) {
      auto it = std::find(item.second.begin(), item.second.end(), index);
      if (it != item.second.end()) {
        return true;
      }
    }
  }
  return false;
}

// section6
void AscendStreamAssign::InsertEventForIndependentParallel(const NotNull<KernelGraphPtr> &graph_ptr) {
  MS_LOG(INFO) << "Start";
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> cnodes = cnode_ptr_list;
  uint32_t cur_event_id = resource_manager.ApplyNewEvent();
  auto it = cnodes.begin();
  while (it != cnodes.end()) {
    MS_EXCEPTION_IF_NULL(*it);
    if (AnfAlgo::IsIndependentNode(*it)) {
      MS_LOG(DEBUG) << "Deal independent op[" << (*it)->DebugString() << "]";
      CNodePtr send_cnode_ptr = CreateSendApplyKernel(graph_ptr, cur_event_id, AnfAlgo::GetStreamId(*it));
      it = cnodes.insert(it + 1, send_cnode_ptr);

      auto target = FindTargetOp(it, cnodes.end(), *(it - 1), false);
      if (target == cnodes.end()) {
        MS_LOG(DEBUG) << "Independent node[" << (*(it - 1))->fullname_with_scope()
                      << "] can't find target for insert recv op, no insert send/recv";
        it = cnodes.erase(it);
        continue;
      }

      // deal recv op
      uint32_t stream_id = AnfAlgo::GetStreamId(*target);
      CNodePtr recv_cnode_ptr = CreateRecvApplyKernel(graph_ptr, cur_event_id, stream_id);
      (void)cnodes.insert(target, recv_cnode_ptr);
      cur_event_id = resource_manager.ApplyNewEvent();
    }
    ++it;
  }
  // one event allocated additional, should delete
  resource_manager.DeleteEvent();
  graph_ptr->set_execution_order(cnodes);
  MS_LOG(INFO) << "After independent parallel, total event nums:" << resource_manager.get_cur_event_num();
  MS_LOG(INFO) << "End";
}

void AscendStreamAssign::GetIndependentMaxTarget(const NotNull<KernelGraphPtr> &graph_ptr) {
  MS_LOG(INFO) << "Start";
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); i++) {
    auto cur_node = cnode_ptr_list[i];
    auto key = cur_node.get();
    if (!AnfAlgo::IsIndependentNode(cur_node)) {
      continue;
    }

    bool flag = false;
    for (size_t j = cnode_ptr_list.size() - 1; j > i; j--) {
      auto target_node = cnode_ptr_list[j];
      auto inputs = target_node->inputs();
      for (size_t m = 1; m < inputs.size(); m++) {
        auto input = inputs[m];
        if (opt::IsNopNode(input)) {
          CNodePtr cnode = input->cast<CNodePtr>();
          auto new_inputs = cnode->inputs();
          for (size_t k = 1; k < new_inputs.size(); k++) {
            auto new_real_input = AnfAlgo::VisitKernel(new_inputs[k], 0);
            if (key == new_real_input.first.get()) {
              MS_LOG(DEBUG) << "Nop node find max target op:" << AnfAlgo::GetCNodeName(cur_node);
              independent_targets_.emplace(target_node.get());
              flag = true;
              break;
            }
          }
        } else {
          auto real_input = AnfAlgo::VisitKernel(input, 0);
          if (key == real_input.first.get()) {
            MS_LOG(DEBUG) << "Find max target op:" << AnfAlgo::GetCNodeName(cur_node);
            independent_targets_.emplace(target_node.get());
            flag = true;
          }
        }
        if (flag) {
          break;
        }
      }
    }
  }

  MS_LOG(INFO) << "End";
}

uint32_t AscendStreamAssign::GetIndexByKey(const NotNull<KernelGraphPtr> &graph_ptr, const CNodeKey &key) {
  auto &exe_orders = graph_ptr->execution_order();
  for (uint32_t i = 0; i < exe_orders.size(); i++) {
    CNodeKey node_key = exe_orders[i].get();
    if (node_key == key) {
      return i;
    }
  }

  return UINT32_MAX;
}

uint32_t AscendStreamAssign::GetMaxIndexTarget(const NotNull<KernelGraphPtr> &graph_ptr) {
  if (independent_targets_.empty()) {
    return UINT32_MAX;
  }

  std::set<uint32_t> indices;
  for (const auto &key : independent_targets_) {
    auto index = GetIndexByKey(graph_ptr, key);
    if (index == UINT32_MAX) {
      MS_LOG(EXCEPTION) << "graph has no correspond key";
    }
    indices.emplace(index);
  }

  return *(std::max_element(indices.begin(), indices.end()));
}

uint32_t AscendStreamAssign::GetIndependentStreamSwitchStreamId(const NotNull<KernelGraphPtr> &graph_ptr) {
  auto &exe_orders = graph_ptr->execution_order();
  for (const auto &item : exe_orders) {
    if (AnfAlgo::GetCNodeName(item) == kStreamSwitchOpName) {
      if (!AnfAlgo::HasNodeAttr(kAttrStreamSwitchKind, item)) {
        continue;
      }
      auto kind = AnfAlgo::GetNodeAttr<uint32_t>(item, kAttrStreamSwitchKind);
      if (kind == kIndependentStreamSwitch) {
        return AnfAlgo::GetStreamId(item);
      }
    }
  }
  return kInvalidStreamId;
}

void AscendStreamAssign::InsertCtrlForIndependentParallel(const NotNull<KernelGraphPtr> &graph_ptr) {
  if (independent_targets_.empty()) {
    return;
  }

  uint32_t independent_switch_stream = GetIndependentStreamSwitchStreamId(graph_ptr);
  if (independent_switch_stream == kInvalidStreamId) {
    return;
  }

  auto max_index = GetMaxIndexTarget(graph_ptr);
  auto &exe_orders = graph_ptr->execution_order();
  if (max_index >= exe_orders.size()) {
    MS_LOG(EXCEPTION) << "Max target index:" << max_index << " is greater than graph orders size:" << exe_orders.size();
  }

  auto max_node_stream = AnfAlgo::GetStreamId(exe_orders[max_index]);

  CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
  // 1.set stream id
  AnfAlgo::SetStreamId(max_node_stream, active_ptr.get());
  // 2.set active stream ids
  std::vector<uint32_t> active_index_list{independent_switch_stream};
  AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_index_list), active_ptr);

  std::vector<CNodePtr> update_cnode_list;
  std::copy(exe_orders.begin(), exe_orders.begin() + max_index + 1, std::back_inserter(update_cnode_list));
  update_cnode_list.emplace_back(active_ptr);
  std::copy(exe_orders.begin() + max_index + 1, exe_orders.end(), std::back_inserter(update_cnode_list));
  graph_ptr->set_execution_order(update_cnode_list);
}

// section7
void AscendStreamAssign::GetNeedActiveStreams(const NotNull<KernelGraphPtr> &graph_ptr) {
  CNodePtr cur_cnode_ptr = nullptr;
  auto cnode_ptr_list = graph_ptr->execution_order();

  // 1)stream witch kStreamNeedActivedFirst attr should be activated;
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (!AnfAlgo::HasNodeAttr(kStreamNeedActivedFirst, cur_cnode_ptr)) {
      continue;
    }

    auto need_active = AnfAlgo::GetNodeAttr<bool>(cur_cnode_ptr, kStreamNeedActivedFirst);
    if (need_active) {
      auto stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
      MS_LOG(INFO) << "Stream id:" << stream_id << " is need activated at first";
      need_first_active_streams_.push_back(stream_id);
    }
  }

  // 2)independent stream:if has not been activate, push to need active vector
  auto root_graph_id = graph_ptr->graph_id();
  if (!independent_stream_activated_) {
    auto it = independent_graph_map_.find(root_graph_id);
    if (it != independent_graph_map_.end()) {
      need_first_active_streams_.push_back(*(it->second.begin()));
    }
  }

  // 3)hcom stream:if has not been activate, push to need active vector
  if (!hcom_stream_activated_) {
    for (const auto &item : group_hcom_graph_map_) {
      auto &hcom_graph_map = item.second;
      auto it = hcom_graph_map.find(root_graph_id);
      if (it != hcom_graph_map.end()) {
        std::copy(it->second.begin(), it->second.end(), std::back_inserter(need_first_active_streams_));
      }
    }
  }

  // 4)first stream 0 should be activated first;
  auto it = std::find(need_first_active_streams_.begin(), need_first_active_streams_.end(), 0);
  if (it == need_first_active_streams_.end()) {
    need_first_active_streams_.emplace_back(0);
  }
  MS_LOG(INFO) << "Finally, need active first stream include:";
  for (const auto &item : need_first_active_streams_) {
    MS_LOG(INFO) << "stream id:" << item;
  }
}

// section8
void AscendStreamAssign::CheckResourceAssign(const NotNull<KernelGraphPtr> &graph_ptr) {
  CheckStreamAssign(graph_ptr);
  CheckEventAssign(graph_ptr);
}

void AscendStreamAssign::CheckStreamAssign(const NotNull<KernelGraphPtr> &graph_ptr) {
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  std::set<uint32_t> streams;
  uint32_t max_stream = 0;
  uint32_t min_stream = kInvalidStreamId;
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    uint32_t stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    if (stream_id == kInvalidStreamId) {
      MS_LOG(EXCEPTION) << "Node:" << AnfAlgo::GetCNodeName(cur_cnode_ptr) << "had not been assigned stream";
    }

    (void)streams.emplace(stream_id);
    if (stream_id > max_stream) {
      max_stream = stream_id;
    }
    if (stream_id < min_stream) {
      min_stream = stream_id;
    }
  }

  // check stream assign
  if (!streams.empty()) {
    if (min_stream != 0) {
      MS_LOG(EXCEPTION) << "Stream should start from 0, now is from " << min_stream;
    }
    uint32_t assigned_stream_num = resource_manager.get_cur_stream_num();
    if ((max_stream != assigned_stream_num - 1) || (streams.size() != assigned_stream_num)) {
      MS_LOG(EXCEPTION) << "Stream should be consecutive, max stream id:" << max_stream
                        << "; alloc stream nums:" << assigned_stream_num << "; streams size:" << streams.size();
    }
  }
}

void AscendStreamAssign::CheckEventAssign(const NotNull<KernelGraphPtr> &graph_ptr) {
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  std::map<uint32_t, std::vector<CNodePtr>> event_map;
  uint32_t max_event_id = 0;
  uint32_t min_event_id = kInvalidEventId;
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    auto name = AnfAlgo::GetCNodeName(cur_cnode_ptr);
    if (name == kSendOpName || name == kRecvOpName) {
      uint32_t event_id = AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode_ptr, kAttrEventId);
      if (event_id > max_event_id) {
        max_event_id = event_id;
      }

      if (event_id < min_event_id) {
        min_event_id = event_id;
      }
      auto it = event_map.find(event_id);
      if (it == event_map.end()) {
        event_map[event_id] = {cur_cnode_ptr};
      } else {
        event_map[event_id].emplace_back(cur_cnode_ptr);
      }
    }
  }
  // check event assign
  if (!event_map.empty()) {
    if (min_event_id != 0) {
      MS_LOG(EXCEPTION) << "Event should start from 0, now is from " << min_event_id;
    }
    uint32_t assigned_event_num = resource_manager.get_cur_event_num();
    if ((max_event_id != assigned_event_num - 1) || (event_map.size() != assigned_event_num)) {
      MS_LOG(EXCEPTION) << "Event should be consecutive";
    }
    for (const auto &item : event_map) {
      if (item.second.size() != 2) {
        MS_LOG(EXCEPTION) << "Send/recv should be in pair and share one event id";
      }
      auto first_name = AnfAlgo::GetCNodeName(item.second[0]);
      auto second_name = AnfAlgo::GetCNodeName(item.second[1]);
      if (!(first_name == kSendOpName && second_name == kRecvOpName)) {
        MS_LOG(EXCEPTION) << "Send should be before recv";
      }
    }
  }
}

// section9
CNodePtr AscendStreamAssign::CreateSendApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id,
                                                   uint32_t stream_id) {
  auto send_op = std::make_shared<Primitive>(kSendOpName);
  MS_EXCEPTION_IF_NULL(send_op);
  auto send_apply = std::make_shared<ValueNode>(send_op);
  MS_EXCEPTION_IF_NULL(send_apply);
  std::vector<AnfNodePtr> send_input_list = {send_apply};
  CNodePtr send_node_ptr = graph_ptr->NewCNode(send_input_list);
  MS_EXCEPTION_IF_NULL(send_node_ptr);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder;
  selected_kernel_builder.SetKernelType(KernelType::RT_KERNEL);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), send_node_ptr.get());
  AnfAlgo::SetNodeAttr(kAttrEventId, MakeValue(event_id), send_node_ptr);
  auto abstract_none = std::make_shared<abstract::AbstractNone>();
  MS_EXCEPTION_IF_NULL(abstract_none);
  send_node_ptr->set_abstract(abstract_none);
  AnfAlgo::SetStreamId(stream_id, send_node_ptr.get());
  return send_node_ptr;
}

CNodePtr AscendStreamAssign::CreateRecvApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id,
                                                   uint32_t stream_id) {
  auto recv_op = std::make_shared<Primitive>(kRecvOpName);
  MS_EXCEPTION_IF_NULL(recv_op);
  auto recv_apply = std::make_shared<ValueNode>(recv_op);
  MS_EXCEPTION_IF_NULL(recv_apply);
  std::vector<AnfNodePtr> recv_input_list = {recv_apply};
  CNodePtr recv_node_ptr = graph_ptr->NewCNode(recv_input_list);
  MS_EXCEPTION_IF_NULL(recv_node_ptr);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder;
  selected_kernel_builder.SetKernelType(KernelType::RT_KERNEL);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), recv_node_ptr.get());
  AnfAlgo::SetNodeAttr(kAttrEventId, MakeValue(event_id), recv_node_ptr);
  AnfAlgo::SetStreamId(stream_id, recv_node_ptr.get());
  auto abstract_none = std::make_shared<abstract::AbstractNone>();
  MS_EXCEPTION_IF_NULL(abstract_none);
  recv_node_ptr->set_abstract(abstract_none);
  return recv_node_ptr;
}

vector<CNodePtr>::iterator AscendStreamAssign::FindTargetOp(vector<CNodePtr>::iterator begin,
                                                            vector<CNodePtr>::iterator end, const CNodePtr &node,
                                                            bool exclude_hcom) {
  while (begin != end) {
    auto inputs = (*begin)->inputs();
    for (size_t i = 1; i < inputs.size(); i++) {
      auto input = inputs[i];
      if (opt::IsNopNode(input)) {
        CNodePtr cnode = input->cast<CNodePtr>();
        auto new_inputs = cnode->inputs();
        for (size_t j = 1; j < new_inputs.size(); j++) {
          auto new_real_input = AnfAlgo::VisitKernel(new_inputs[j], 0);
          // find target node except hcom op. insert event for hcom in:InsertEventHcomDependCommonBak function
          // only insert one time
          if (node == new_real_input.first) {
            if (!(exclude_hcom && IsHcom(*begin))) {
              MS_LOG(DEBUG) << "Nop node find target op[" << (*begin)->DebugString() << "]";
              return begin;
            }
          }
        }
      } else {
        auto real_input = AnfAlgo::VisitKernel(input, 0);
        if (node == real_input.first) {
          if (!(exclude_hcom && IsHcom(*begin))) {
            MS_LOG(DEBUG) << "Nop node find target op[" << (*begin)->DebugString() << "]";
            return begin;
          }
        }
      }
    }
    ++begin;
  }
  return end;
}

bool AscendStreamAssign::IsTaskSink() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK)) {
    MS_LOG(INFO) << "Task sink mode is not enable";
    return false;
  } else {
    MS_LOG(INFO) << "Task sink mode is enable";
    return true;
  }
}

void AscendStreamAssign::GetWaitStreams(vector<uint32_t> *wait_active_stream_list) {
  MS_EXCEPTION_IF_NULL(wait_active_stream_list);
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  uint32_t total_stream_num = resource_manager.get_cur_stream_num();
  if (total_stream_num == 0) {
    MS_LOG(INFO) << "The total_common_stream_num is zero";
    return;
  }

  // common stream:active first common stream
  for (uint32_t i = 0; i < total_stream_num; i++) {
    auto it = std::find(need_first_active_streams_.begin(), need_first_active_streams_.end(), i);
    if (it == need_first_active_streams_.end()) {
      MS_LOG(INFO) << "Wait common stream id = " << i;
      wait_active_stream_list->push_back(i);
    }
  }
}

bool AscendStreamAssign::IsHcom(const CNodePtr &apply_kernel) {
  MS_EXCEPTION_IF_NULL(apply_kernel);
  return AnfAlgo::GetKernelType(apply_kernel) == HCCL_KERNEL;
}

void AscendStreamAssign::GetHcomStreams(std::vector<uint32_t> *streams) {
  MS_EXCEPTION_IF_NULL(streams);
  for (const auto &item : hcom_stream_map_) {
    streams->emplace_back(item.first);
  }
}

void AscendStreamAssign::Reset() {
  independent_stream_activated_ = false;
  hcom_stream_activated_ = false;
  loop_sink_ = false;
  independent_stream_map_.clear();
  hcom_stream_map_.clear();
  common_stream_map_.clear();
  processed_streams_.clear();
  need_first_active_streams_.clear();
  stream_groups_.clear();
  stream_relations_.clear();
  event_map_.clear();
  independent_targets_.clear();
  independent_graph_map_.clear();
  group_hcom_graph_map_.clear();
  middle_active_streams_.clear();
}

// section 10
bool AscendStreamAssign::IsVecExist(const std::vector<uint32_t> &group) {
  auto group_size = group.size();
  if (group_size == 0) {
    return false;
  }
  for (const auto &item : stream_groups_) {
    if (item.size() < group.size()) {
      continue;
    }

    bool flag = true;
    for (size_t i = 0; i < group_size; i++) {
      if (item[i] != group.at(i)) {
        flag = false;
        break;
      }
    }

    if (flag) {
      return true;
    } else {
      continue;
    }
  }

  return false;
}

void AscendStreamAssign::DFS(uint32_t start, std::vector<uint32_t> *group) {
  auto it = stream_relations_.find(start);
  if (it == stream_relations_.end()) {
    if (!IsVecExist(*group)) {
      stream_groups_.emplace_back(*group);
    } else {
      MS_LOG(WARNING) << "DFS find same stream group, Not expected";
    }
    return;
  }

  vector<uint32_t> active_streams = stream_relations_[start];

  for (const auto &item : active_streams) {
    group->emplace_back(item);
    DFS(item, group);
    group->pop_back();
  }
}

void AscendStreamAssign::GetStreamRelations() {
  auto starts = middle_active_streams_;
  for (const auto &stream : need_first_active_streams_) {
    starts.emplace(stream);
  }

  for (const auto &start : starts) {
    vector<uint32_t> group{start};
    DFS(start, &group);
  }
}

void AscendStreamAssign::FindStreamRelations(const NotNull<KernelGraphPtr> &graph_ptr) {
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  auto stream_num = resource_manager.get_cur_stream_num();
  if (stream_num <= 1) {
    return;
  }

  auto exe_orders = graph_ptr->execution_order();
  for (size_t i = 0; i < exe_orders.size(); i++) {
    auto cur_cnode = exe_orders[i];
    auto name = AnfAlgo::GetCNodeName(cur_cnode);
    if (name != kStreamSwitchOpName && name != kStreamActiveOpName) {
      continue;
    }

    // support:streamswitch is begin of the stream
    if (name == kStreamSwitchOpName) {
      GetStreamSwitchStreamRelation(cur_cnode);
    }

    if (name == kStreamActiveOpName) {
      GetStreamActiveStreamRelation(graph_ptr, i);
    }
  }
}

void AscendStreamAssign::GetStreamSwitchStreamRelation(const CNodePtr &node_ptr) {
  MS_EXCEPTION_IF_NULL(node_ptr);
  auto cur_stream_id = AnfAlgo::GetStreamId(node_ptr);
  auto true_stream_id = AnfAlgo::GetNodeAttr<uint32_t>(node_ptr, kAttrTrueBranchStream);
  if (true_stream_id <= cur_stream_id) {
    MS_LOG(ERROR) << "StreamSwitch self stream id " << cur_stream_id
                  << " is greater than true branch stream id:" << true_stream_id;
  }
  auto it = stream_relations_.find(cur_stream_id);
  if (it == stream_relations_.end()) {
    stream_relations_[cur_stream_id] = {true_stream_id};
  } else {
    auto iter =
      std::find(stream_relations_[cur_stream_id].begin(), stream_relations_[cur_stream_id].end(), true_stream_id);
    if (iter == stream_relations_[cur_stream_id].end()) {
      stream_relations_[cur_stream_id].emplace_back(true_stream_id);
    }
  }
}

void AscendStreamAssign::GetStreamActiveStreamRelation(const NotNull<KernelGraphPtr> &graph_ptr, size_t index) {
  StreamActiveKind kind = GetStreamActiveKind(graph_ptr, index);
  if (kind == kInvalid) {
    MS_LOG(INFO) << "Invalid streamActive kind";
    return;
  }

  auto orders = graph_ptr->execution_order();
  auto cur_cnode = orders[index];
  auto cur_stream_id = AnfAlgo::GetStreamId(cur_cnode);
  auto active_list = AnfAlgo::GetNodeAttr<vector<uint32_t>>(cur_cnode, kAttrActiveStreamList);
  if (kind == kHead) {
    uint32_t active_current_node = GetStreamByActivedStream(cur_stream_id);
    if (active_current_node == kInvalidStreamId) {
      MS_LOG(EXCEPTION) << "No stream to active streamactive stream";
    }

    for (const auto &item : active_list) {
      if (item <= active_current_node) {
        MS_LOG(WARNING) << "Activated stream is less than activing stream";
        continue;
      }
      auto it =
        std::find(stream_relations_[active_current_node].begin(), stream_relations_[active_current_node].end(), item);
      if (it == stream_relations_[active_current_node].end()) {
        stream_relations_[active_current_node].emplace_back(item);
      }
    }
  }

  if (kind == kMiddle) {
    for (const auto &stream : active_list) {
      if (stream <= cur_stream_id) {
        MS_LOG(INFO) << "MIDDLE StreamActive active stream is less than self stream, no need deal";
      } else {
        MS_LOG(INFO) << "MIDDLE StreamActive :" << cur_stream_id << ", active target stream:" << stream;
        middle_active_streams_.emplace(stream);
      }
    }
  }

  if (kind == kTail) {
    auto it = stream_relations_.find(cur_stream_id);
    if (it == stream_relations_.end()) {
      stream_relations_[cur_stream_id] = active_list;
    } else {
      for (const auto &stream : active_list) {
        if (stream <= cur_stream_id) {
          MS_LOG(WARNING) << "Activated stream is less than activing stream";
          continue;
        }
        auto iter = std::find(stream_relations_[cur_stream_id].begin(), stream_relations_[cur_stream_id].end(), stream);
        if (iter == stream_relations_[cur_stream_id].end()) {
          stream_relations_[cur_stream_id].emplace_back(stream);
        }
      }
    }
  }
}

StreamActiveKind AscendStreamAssign::GetStreamActiveKind(const NotNull<KernelGraphPtr> &graph_ptr, size_t index) {
  auto exe_orders = graph_ptr->execution_order();
  if (index >= exe_orders.size()) {
    MS_LOG(EXCEPTION) << "Invalid op index:" << index;
  }

  auto cur_cnode = exe_orders[index];
  auto cur_stream_id = AnfAlgo::GetStreamId(cur_cnode);
  if (AnfAlgo::GetCNodeName(cur_cnode) != kStreamActiveOpName) {
    MS_LOG(EXCEPTION) << "Current node name is not StreamActive";
  }

  if (index == 0) {
    return kInvalid;
  }

  if (index == exe_orders.size() - 1) {
    return kInvalid;
  }

  uint32_t pre_stream_id = UINT32_MAX;
  uint32_t next_stream_id = UINT32_MAX;
  int32_t start = SizeToInt(index) - 1;
  for (int32_t i = start; i >= 0; i--) {
    auto cnode = exe_orders[IntToSize(i)];
    auto name = AnfAlgo::GetCNodeName(cnode);
    if (name == kSendOpName || name == kRecvOpName) {
      continue;
    }
    auto stream = AnfAlgo::GetStreamId(cnode);
    auto it = hcom_stream_map_.find(stream);
    if (it != hcom_stream_map_.end()) {
      continue;
    }

    it = independent_stream_map_.find(stream);
    if (it != independent_stream_map_.end()) {
      continue;
    }

    pre_stream_id = stream;
    break;
  }

  for (size_t i = index + 1; i < exe_orders.size(); i++) {
    auto cnode = exe_orders[i];
    if (AnfAlgo::GetCNodeName(cnode) == kSendOpName || AnfAlgo::GetCNodeName(cnode) == kRecvOpName) {
      continue;
    }

    auto stream = AnfAlgo::GetStreamId(cnode);
    auto it = hcom_stream_map_.find(stream);
    if (it != hcom_stream_map_.end()) {
      continue;
    }

    it = independent_stream_map_.find(stream);
    if (it != independent_stream_map_.end()) {
      continue;
    }

    next_stream_id = stream;
    break;
  }

  return GetStreamKind(cur_stream_id, pre_stream_id, next_stream_id);
}

uint32_t AscendStreamAssign::GetStreamByActivedStream(uint32_t actived_stream_id) {
  if (stream_relations_.empty()) {
    return kInvalidStreamId;
  }

  for (const auto &item : stream_relations_) {
    auto it = std::find(item.second.begin(), item.second.end(), actived_stream_id);
    if (it != item.second.end()) {
      return item.first;
    }
  }

  return kInvalidStreamId;
}

void AscendStreamAssign::PrintStreamRelations() {
  MS_LOG(INFO) << "Stream relations size:" << stream_relations_.size();
  for (const auto &item : stream_relations_) {
    MS_LOG(INFO) << "Stream:" << item.first;
    for (const auto &stream : item.second) {
      MS_LOG(INFO) << "--activated stream id:" << stream;
    }
  }
}

void AscendStreamAssign::PrintStreamGroups() {
  MS_LOG(INFO) << "Stream group size:" << stream_groups_.size();
  for (const auto &item : stream_groups_) {
    MS_LOG(INFO) << "Group:";
    for (const auto &stream : item) {
      MS_LOG(INFO) << "Stream id:" << stream;
    }
  }
}

// section 11
bool AscendStreamAssign::IsSatisfiedEvent(uint32_t send_stream_id, uint32_t recv_stream_id) const {
  size_t send_group = 0;
  size_t recv_group = 0;
  bool send_flag = true;
  bool recv_flag = true;
  for (size_t i = 0; i < stream_groups_.size(); i++) {
    auto group = stream_groups_[i];
    if (send_flag) {
      auto it = std::find(group.begin(), group.end(), send_stream_id);
      if (it != group.end()) {
        send_group = i;
        send_flag = false;
      }
    }

    if (recv_flag) {
      auto it = std::find(group.begin(), group.end(), recv_stream_id);
      if (it != group.end()) {
        recv_group = i;
        recv_flag = false;
      }
    }
  }

  if (!(send_flag || recv_flag)) {
    return (send_group != recv_group);
  }

  return false;
}

void AscendStreamAssign::FindEventRelations(const NotNull<KernelGraphPtr> &graph_ptr) {
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  auto event_nums = resource_manager.get_cur_event_num();
  if (event_nums == 0) {
    return;
  }
  auto exe_orders = graph_ptr->execution_order();
  // find all event info
  for (size_t i = 0; i < exe_orders.size(); i++) {
    auto cur_cnode = exe_orders[i];
    auto name = AnfAlgo::GetCNodeName(cur_cnode);
    if (name == kSendOpName) {
      event_map_[cur_cnode] = {};
    }

    if (name == kRecvOpName) {
      auto recv_event_id = AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode, kAttrEventId);
      for (auto &item : event_map_) {
        auto send_event_id = AnfAlgo::GetNodeAttr<uint32_t>(item.first, kAttrEventId);
        if (recv_event_id == send_event_id) {
          item.second = cur_cnode;
          break;
        }
      }
    }
  }

  // delete useless event info
  auto begin = event_map_.begin();
  while (begin != event_map_.end()) {
    auto send_stream_id = AnfAlgo::GetStreamId(begin->first);
    auto recv_stream_id = AnfAlgo::GetStreamId(begin->second);
    bool flag = IsSatisfiedEvent(send_stream_id, recv_stream_id);
    if (!flag) {
      begin = event_map_.erase(begin);
    } else {
      begin++;
    }
  }

  MS_LOG(INFO) << "Satisfied event info";
  for (const auto &item : event_map_) {
    MS_LOG(INFO) << "Event_id:" << AnfAlgo::GetNodeAttr<uint32_t>(item.first, kAttrEventId);
  }
}

// section12
void AscendStreamAssign::AdjustAtomicAddrCleanOrder(const NotNull<KernelGraphPtr> &graph_ptr) {
  // Eg:[atomic, recv, memcpy] should be [recv, atomic, memcpy]
  std::vector<CNodePtr> update_orders;
  auto &exe_orders = graph_ptr->execution_order();
  for (size_t i = 0; i < exe_orders.size(); i++) {
    auto cur_cnode = exe_orders.at(i);
    if (AnfAlgo::GetCNodeName(cur_cnode) != kAtomicAddrCleanOpName) {
      update_orders.emplace_back(cur_cnode);
      continue;
    }

    for (size_t j = i + 1; j < exe_orders.size(); j++) {
      auto next_cnode = exe_orders[j];
      auto next_cnode_name = AnfAlgo::GetCNodeName(next_cnode);
      if (next_cnode_name == kSendOpName || next_cnode_name == kRecvOpName) {
        update_orders.emplace_back(next_cnode);
      } else {
        update_orders.emplace_back(cur_cnode);
        // attention:i will be executed later i++;
        i = j - 1;
        break;
      }
    }
  }

  graph_ptr->set_execution_order(update_orders);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
