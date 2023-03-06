/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/kernel_adjust.h"

#include <map>
#include <algorithm>
#include <string>
#include <vector>
#include <utility>
#include <queue>
#include <set>

#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "include/common/utils/config_manager.h"
#include "utils/ms_utils.h"
#include "kernel/kernel_build_info.h"
#include "include/common/utils/utils.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#include "runtime/base.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include "utils/shape_utils.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#endif

namespace {
constexpr auto kGradients = "Gradients";
constexpr auto kSpecifyParameter = "accu_status";
constexpr auto kSplitOverFlow = "split_overflow";
constexpr auto kLayerOverFlow = "layer_overflow";
constexpr auto kMixLayerStatusParameter = "mix_layer_status";
int64_t kNPUShape = 8;
constexpr size_t kLastHandleDiff = 2;
}  // namespace
namespace mindspore {
namespace device {
#ifndef ENABLE_SECURITY
using device::ascend::ProfilingUtils;
#endif
void KernelAdjust::ReorderGetNext(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  const std::vector<CNodePtr> &origin_cnode_list = kernel_graph_ptr->execution_order();
  std::vector<CNodePtr> getnext_list;
  std::vector<CNodePtr> other_list;
  for (const auto &cnode : origin_cnode_list) {
    if (common::AnfAlgo::GetCNodeName(cnode) == kGetNextOpName) {
      (void)getnext_list.emplace_back(cnode);
    } else {
      (void)other_list.emplace_back(cnode);
    }
  }
  std::vector<CNodePtr> new_order_list;
  (void)new_order_list.insert(new_order_list.end(), getnext_list.begin(), getnext_list.end());
  (void)new_order_list.insert(new_order_list.end(), other_list.begin(), other_list.end());
  kernel_graph_ptr->set_execution_order(new_order_list);
}

bool KernelAdjust::NeedLoopSink() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  return (context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK) &&
          context_ptr->get_param<bool>(MS_CTX_ENABLE_LOOP_SINK) && ConfigManager::GetInstance().iter_num() > 1);
}

CNodePtr CreateEventApplyKernel(const std::shared_ptr<session::KernelGraph> &graph_ptr, uint32_t event_id,
                                const std::vector<AnfNodePtr> &input_list) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  CNodePtr event_node_ptr = graph_ptr->NewCNode(input_list);
  MS_EXCEPTION_IF_NULL(event_node_ptr);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder;
  selected_kernel_builder.SetKernelType(KernelType::RT_KERNEL);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), event_node_ptr.get());
  common::AnfAlgo::SetNodeAttr(kAttrEventId, MakeValue(event_id), event_node_ptr);
  auto abstract_none = std::make_shared<abstract::AbstractNone>();
  event_node_ptr->set_abstract(abstract_none);
  return event_node_ptr;
}

CNodePtr KernelAdjust::CreateSendApplyKernel(const std::shared_ptr<session::KernelGraph> &graph_ptr,
                                             uint32_t event_id) const {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  auto send_op = std::make_shared<Primitive>(kSendOpName);
  MS_EXCEPTION_IF_NULL(send_op);
  auto send_apply = std::make_shared<ValueNode>(send_op);
  MS_EXCEPTION_IF_NULL(send_apply);
  return CreateEventApplyKernel(graph_ptr, event_id, {send_apply});
}

CNodePtr KernelAdjust::CreateRecvApplyKernel(const std::shared_ptr<session::KernelGraph> &graph_ptr,
                                             uint32_t event_id) const {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  auto recv_op = std::make_shared<Primitive>(kRecvOpName);
  MS_EXCEPTION_IF_NULL(recv_op);
  auto recv_apply = std::make_shared<ValueNode>(recv_op);
  MS_EXCEPTION_IF_NULL(recv_apply);
  return CreateEventApplyKernel(graph_ptr, event_id, {recv_apply});
}

bool KernelAdjust::ExistGetNext(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  const std::vector<CNodePtr> &cnode_list = kernel_graph_ptr->execution_order();
  return std::any_of(cnode_list.begin(), cnode_list.end(),
                     [](const CNodePtr &cnode) { return common::AnfAlgo::GetCNodeName(cnode) == kGetNextOpName; });
}

bool KernelAdjust::ExistIndependent(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  const auto &exe_orders = kernel_graph_ptr->execution_order();
  return std::any_of(exe_orders.begin(), exe_orders.end(), [&kernel_graph_ptr](const CNodePtr &node) {
    return AnfAlgo::IsIndependentNode(node) && AnfAlgo::GetGraphId(node.get()) == kernel_graph_ptr->graph_id();
  });
}

void KernelAdjust::InsertIndepentParallel(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                          const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input,
                                          std::vector<CNodePtr> *exec_order) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(exec_order);
  device::ascend::AscendStreamMng &resource_manager = device::ascend::AscendStreamMng::GetInstance();
  CNodePtr independent_switch_app = CreateStreamSwitchOp(kernel_graph_ptr, switch_loop_input, kIndependentStreamSwitch);
  MS_EXCEPTION_IF_NULL(independent_switch_app);
  uint32_t independent_switch_stream_id = resource_manager.ApplyNewStream();
  AnfAlgo::SetStreamId(independent_switch_stream_id, independent_switch_app.get());
  common::AnfAlgo::SetNodeAttr(kStreamNeedActivedFirst, MakeValue<bool>(true), independent_switch_app);
  common::AnfAlgo::SetNodeAttr(kAttrStreamSwitchKind, MakeValue<uint32_t>(kIndependentStreamSwitch),
                               independent_switch_app);
  (*exec_order).push_back(independent_switch_app);
  MS_LOG(INFO) << "Independent op loop insert Stream Switch " << independent_switch_app->fullname_with_scope();
}

void KernelAdjust::InsertFpBpLoopStreamSwitch(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                              const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input,
                                              std::vector<CNodePtr> *exec_order, uint32_t *fpbp_stream_id,
                                              uint32_t *fpbp_switch_stream_id) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(exec_order);
  MS_EXCEPTION_IF_NULL(fpbp_stream_id);
  MS_EXCEPTION_IF_NULL(fpbp_switch_stream_id);
  device::ascend::AscendStreamMng &resource_manager = device::ascend::AscendStreamMng::GetInstance();
  *fpbp_switch_stream_id = resource_manager.ApplyNewStream();
  *fpbp_stream_id = resource_manager.ApplyNewStream();
  CNodePtr fpbp_switch_app = CreateStreamSwitchOp(kernel_graph_ptr, switch_loop_input, kFpBpStreamSwitch);
  MS_EXCEPTION_IF_NULL(fpbp_switch_app);
  AnfAlgo::SetStreamId(*fpbp_switch_stream_id, fpbp_switch_app.get());
  common::AnfAlgo::SetNodeAttr(kStreamNeedActivedFirst, MakeValue<bool>(true), fpbp_switch_app);
  // update fpbp loop stream switch true_branch_stream attr
  common::AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, MakeValue<uint32_t>(*fpbp_stream_id), fpbp_switch_app);
  common::AnfAlgo::SetNodeAttr(kAttrStreamSwitchKind, MakeValue<uint32_t>(kFpBpStreamSwitch), fpbp_switch_app);
  (*exec_order).push_back(fpbp_switch_app);
  MS_LOG(INFO) << "FpBp loop insert Stream Switch " << fpbp_switch_app->fullname_with_scope();
}

bool KernelAdjust::IsTaskSink() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  return ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
}
bool KernelAdjust::ExistInitDataSetQueue(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  const std::vector<CNodePtr> &cnode_list = kernel_graph_ptr->execution_order();
  return std::any_of(cnode_list.begin(), cnode_list.end(), [](const CNodePtr &cnode) {
    return common::AnfAlgo::GetCNodeName(cnode) == kInitDatasetQueueOpName;
  });
}

void KernelAdjust::InsertEndGraphTaskSink(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  if (IsTaskSink()) {
    auto exec_order = kernel_graph_ptr->execution_order();
    if (exec_order.empty() || ExistInitDataSetQueue(kernel_graph_ptr)) {
      return;
    }
    CNodePtr fpbp_endgraph_app = CreateEndGraphOp(kernel_graph_ptr);
    MS_EXCEPTION_IF_NULL(fpbp_endgraph_app);
    exec_order.push_back(fpbp_endgraph_app);
    MS_LOG(INFO) << "Insert End Graph " << fpbp_endgraph_app->fullname_with_scope();
    kernel_graph_ptr->set_execution_order(exec_order);
  }
}

void KernelAdjust::InsertEndGraph(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                  std::vector<CNodePtr> *exec_order, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(exec_order);
  CNodePtr end_graph_op = CreateEndGraphOp(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(end_graph_op);
  AnfAlgo::SetStreamId(stream_id, end_graph_op.get());
  exec_order->push_back(end_graph_op);
  MS_LOG(INFO) << "Insert End Graph " << end_graph_op->fullname_with_scope();
}

void KernelAdjust::CopyMemcpyList(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                  const std::vector<CNodePtr> &orders, size_t order_index,
                                  std::vector<CNodePtr> *memcpy_list, std::vector<CNodePtr> *other_list) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(memcpy_list);
  MS_EXCEPTION_IF_NULL(other_list);
  CNodePtr cur_cnode = nullptr;
  for (size_t idx = order_index + 1; idx < orders.size(); idx++) {
    cur_cnode = orders[idx];
    if (common::AnfAlgo::HasNodeAttr(kAttrLabelForInsertStreamActive, cur_cnode)) {
      auto pre_node = orders[idx - 1];
      auto pre_kernel_name = common::AnfAlgo::GetCNodeName(pre_node);
      if (pre_kernel_name == kAtomicAddrCleanOpName) {
        (*other_list).pop_back();
        (*memcpy_list).push_back(pre_node);
      }
      (void)(*memcpy_list).emplace_back(cur_cnode);
    } else {
      (void)(*other_list).emplace_back(cur_cnode);
    }
  }
}

void KernelAdjust::InsertEosDoneRecv(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                     std::vector<CNodePtr> *exec_order, uint32_t eos_done_event_id,
                                     uint32_t fpbp_stream_id) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(exec_order);
  CNodePtr eos_done_recv = CreateRecvApplyKernel(kernel_graph_ptr, eos_done_event_id);
  AnfAlgo::SetStreamId(fpbp_stream_id, eos_done_recv.get());
  (*exec_order).push_back(eos_done_recv);
  MS_LOG(INFO) << "FpBp loop insert EoS done Recv " << eos_done_recv->fullname_with_scope();
}

void KernelAdjust::InsertGetNextLoopStreamActive(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                                 std::vector<CNodePtr> *exec_order,
                                                 const std::vector<uint32_t> &getnext_active_streams) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(exec_order);
  CNodePtr getnext_active_app = CreateStreamActiveOp(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(getnext_active_app);
  common::AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(getnext_active_streams),
                               getnext_active_app);
  (*exec_order).push_back(getnext_active_app);
  MS_LOG(INFO) << "FpBp loop insert GetNext loop Stream Active " << getnext_active_app->fullname_with_scope();
}

void KernelAdjust::InsertFpBpStartRecv(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                       std::vector<CNodePtr> *exec_order, uint32_t fpbp_start_event_id,
                                       uint32_t fpbp_stream_id) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(exec_order);
  CNodePtr fpbp_start_recv = CreateRecvApplyKernel(kernel_graph_ptr, fpbp_start_event_id);
  AnfAlgo::SetStreamId(fpbp_stream_id, fpbp_start_recv.get());
  (*exec_order).push_back(fpbp_start_recv);
  MS_LOG(INFO) << "FpBp loop insert FpBp start Recv " << fpbp_start_recv->fullname_with_scope();
}

void KernelAdjust::InsertNextLoopAssignAdd(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                           std::vector<CNodePtr> *exec_order,
                                           const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input,
                                           uint32_t fpbp_stream_id) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(exec_order);
  CNodePtr assign_add_one = CreateStreamAssignAddnOP(kernel_graph_ptr, switch_loop_input, false);
  MS_EXCEPTION_IF_NULL(assign_add_one);
  AnfAlgo::SetStreamId(fpbp_stream_id, assign_add_one.get());
  (*exec_order).push_back(assign_add_one);
  MS_LOG(INFO) << "FpBp loop insert next loop AssignAdd " << assign_add_one->fullname_with_scope();
}

void KernelAdjust::InsertCurrentLoopAssignAdd(
  const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr, std::vector<CNodePtr> *exec_order,
  const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(exec_order);
  CNodePtr cur_assign_add = CreateStreamAssignAddnOP(kernel_graph_ptr, switch_loop_input, true);
  MS_EXCEPTION_IF_NULL(cur_assign_add);
  common::AnfAlgo::SetNodeAttr(kAttrFpBpEnd, MakeValue<bool>(true), cur_assign_add);
  (*exec_order).push_back(cur_assign_add);
  MS_LOG(INFO) << "FpBp loop insert current loop AssignAdd " << cur_assign_add->fullname_with_scope();
}

void KernelAdjust::InsertFpBpAndEosLoopStreamActive(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                                    std::vector<CNodePtr> *exec_order,
                                                    const std::vector<uint32_t> &fpbp_active_streams) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(exec_order);
  CNodePtr fpbp_active_app = CreateStreamActiveOp(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(fpbp_active_app);
  common::AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(fpbp_active_streams),
                               fpbp_active_app);
  (*exec_order).push_back(fpbp_active_app);
  MS_LOG(INFO) << "FpBp loop insert FpBp loop and Eos loop Stream Active " << fpbp_active_app->fullname_with_scope();
}

void KernelAdjust::InsertGetNextLoopStreamSwitch(
  const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr, std::vector<CNodePtr> *exec_order,
  uint32_t *getnext_switch_stream_id, uint32_t *getnext_stream_id,
  const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(exec_order);
  MS_EXCEPTION_IF_NULL(getnext_switch_stream_id);
  MS_EXCEPTION_IF_NULL(getnext_stream_id);
  device::ascend::AscendStreamMng &resource_manager = device::ascend::AscendStreamMng::GetInstance();
  *getnext_switch_stream_id = resource_manager.ApplyNewStream();
  *getnext_stream_id = resource_manager.ApplyNewStream();
  CNodePtr getnext_switch_app = CreateStreamSwitchOp(kernel_graph_ptr, switch_loop_input, kGetNextStreamSwitch);
  MS_EXCEPTION_IF_NULL(getnext_switch_app);
  AnfAlgo::SetStreamId(*getnext_switch_stream_id, getnext_switch_app.get());
  // update getnext loop stream switch true_branch_stream attr
  common::AnfAlgo::SetNodeAttr(kStreamNeedActivedFirst, MakeValue<bool>(true), getnext_switch_app);
  common::AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, MakeValue<uint32_t>(*getnext_stream_id), getnext_switch_app);
  common::AnfAlgo::SetNodeAttr(kAttrStreamSwitchKind, MakeValue<uint32_t>(kGetNextStreamSwitch), getnext_switch_app);
  (*exec_order).push_back(getnext_switch_app);
  MS_LOG(INFO) << "GetNext loop insert Stream Switch " << getnext_switch_app->fullname_with_scope();
}

void KernelAdjust::SetBeforeGetNextStreamID(std::vector<CNodePtr> *exec_order, const std::vector<CNodePtr> &orders,
                                            size_t *order_index, CNodePtr *getnext_cnode,
                                            uint32_t getnext_stream_id) const {
  MS_EXCEPTION_IF_NULL(exec_order);
  MS_EXCEPTION_IF_NULL(order_index);
  for (; *order_index < orders.size(); (*order_index)++) {
    auto node = orders[*order_index];
    (*exec_order).push_back(node);
    AnfAlgo::SetStreamId(getnext_stream_id, (*exec_order)[(*exec_order).size() - 1].get());
    if (common::AnfAlgo::GetCNodeName(node) == kGetNextOpName) {
      *getnext_cnode = node;
      break;
    }
  }
}

void KernelAdjust::InsertGetNextLoopFpBpStartSend(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                                  std::vector<CNodePtr> *exec_order, uint32_t *fpbp_start_event_id,
                                                  uint32_t getnext_stream_id) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(exec_order);
  MS_EXCEPTION_IF_NULL(fpbp_start_event_id);
  device::ascend::AscendStreamMng &resource_manager = device::ascend::AscendStreamMng::GetInstance();
  *fpbp_start_event_id = resource_manager.ApplyNewEvent();
  CNodePtr fpbp_start_send = CreateSendApplyKernel(kernel_graph_ptr, *fpbp_start_event_id);
  AnfAlgo::SetStreamId(getnext_stream_id, fpbp_start_send.get());
  (*exec_order).push_back(fpbp_start_send);
  MS_LOG(INFO) << "GetNext loop insert FpBp start Send " << fpbp_start_send->fullname_with_scope();
}

void KernelAdjust::InsertGetNextLoopEosStartSend(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                                 std::vector<CNodePtr> *exec_order, uint32_t *eos_start_event_id,
                                                 uint32_t getnext_stream_id) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(exec_order);
  MS_EXCEPTION_IF_NULL(eos_start_event_id);
  device::ascend::AscendStreamMng &resource_manager = device::ascend::AscendStreamMng::GetInstance();
  *eos_start_event_id = resource_manager.ApplyNewEvent();
  CNodePtr eos_start_send = CreateSendApplyKernel(kernel_graph_ptr, *eos_start_event_id);
  AnfAlgo::SetStreamId(getnext_stream_id, eos_start_send.get());
  (*exec_order).push_back(eos_start_send);
  MS_LOG(INFO) << "GetNext loop insert EoS start Send " << eos_start_send->fullname_with_scope();
}

void KernelAdjust::InsertEosStreamSwitch(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                         const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input,
                                         std::vector<CNodePtr> *exec_order, uint32_t *eos_switch_stream_id,
                                         uint32_t *eos_stream_id) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(exec_order);
  MS_EXCEPTION_IF_NULL(eos_switch_stream_id);
  MS_EXCEPTION_IF_NULL(eos_stream_id);
  device::ascend::AscendStreamMng &resource_manager = device::ascend::AscendStreamMng::GetInstance();
  *eos_switch_stream_id = resource_manager.ApplyNewStream();
  *eos_stream_id = resource_manager.ApplyNewStream();
  CNodePtr eos_switch_app = CreateStreamSwitchOp(kernel_graph_ptr, switch_loop_input, kEosStreamSwitch);
  MS_EXCEPTION_IF_NULL(eos_switch_app);
  AnfAlgo::SetStreamId(*eos_switch_stream_id, eos_switch_app.get());
  common::AnfAlgo::SetNodeAttr(kStreamNeedActivedFirst, MakeValue<bool>(true), eos_switch_app);
  // update eos loop stream switch true_branch_stream attr
  common::AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, MakeValue<uint32_t>(*eos_stream_id), eos_switch_app);
  common::AnfAlgo::SetNodeAttr(kAttrStreamSwitchKind, MakeValue<uint32_t>(kEosStreamSwitch), eos_switch_app);
  (*exec_order).push_back(eos_switch_app);
  MS_LOG(INFO) << "EoS loop insert Stream Switch " << eos_switch_app->fullname_with_scope();
}

void KernelAdjust::InsertGetNextLoopEosStartRecv(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                                 std::vector<CNodePtr> *exec_order, uint32_t eos_start_event_id,
                                                 uint32_t eos_stream_id) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(exec_order);
  CNodePtr eos_start_recv = CreateRecvApplyKernel(kernel_graph_ptr, eos_start_event_id);
  AnfAlgo::SetStreamId(eos_stream_id, eos_start_recv.get());
  (*exec_order).push_back(eos_start_recv);
  MS_LOG(INFO) << "EoS loop insert EoS Recv " << eos_start_recv->fullname_with_scope();
}

void KernelAdjust::InsertEosOp(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                               std::vector<CNodePtr> *exec_order, const CNodePtr &getnext_cnode,
                               uint32_t eos_stream_id) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(exec_order);
  MS_EXCEPTION_IF_NULL(getnext_cnode);
  CNodePtr end_of_sequence_op = CreateEndOfSequenceOP(kernel_graph_ptr, getnext_cnode);
  MS_EXCEPTION_IF_NULL(end_of_sequence_op);
  AnfAlgo::SetStreamId(eos_stream_id, end_of_sequence_op.get());
  (*exec_order).push_back(end_of_sequence_op);
  MS_LOG(INFO) << "EoS loop insert Eos Op " << end_of_sequence_op->fullname_with_scope();
}

void KernelAdjust::InsertEosDoneSend(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                     std::vector<CNodePtr> *exec_order, uint32_t *eos_done_event_id,
                                     uint32_t eos_stream_id) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(exec_order);
  MS_EXCEPTION_IF_NULL(eos_done_event_id);
  device::ascend::AscendStreamMng &resource_manager = device::ascend::AscendStreamMng::GetInstance();
  *eos_done_event_id = resource_manager.ApplyNewEvent();
  CNodePtr eos_done_send = CreateSendApplyKernel(kernel_graph_ptr, *eos_done_event_id);
  AnfAlgo::SetStreamId(eos_stream_id, eos_done_send.get());
  (*exec_order).push_back(eos_done_send);
  MS_LOG(INFO) << "EoS loop insert EoS done Send " << eos_done_send->fullname_with_scope();
}

void KernelAdjust::ProcessLoopSink(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  device::ascend::AscendStreamMng &resource_manager = device::ascend::AscendStreamMng::GetInstance();
  resource_manager.ResetResource();
  if (!NeedLoopSink()) {
    InsertEndGraphTaskSink(kernel_graph_ptr);
    return;
  }
  if (kernel_graph_ptr->is_dynamic_shape()) {
    MS_LOG(INFO) << "KernelGraph:" << kernel_graph_ptr->graph_id() << " is dynamic shape, skip ProcessLoopSink";
    return;
  }
  bool exist_getnext = ExistGetNext(kernel_graph_ptr);
  bool eos_mode = ConfigManager::GetInstance().iter_num() == INT32_MAX && exist_getnext;
  MS_LOG(INFO) << "GetNext exist:" << exist_getnext << " End of Sequence mode:" << eos_mode
               << " iter num:" << ConfigManager::GetInstance().iter_num();
  if (exist_getnext) {
    ReorderGetNext(kernel_graph_ptr);
  }
  auto switch_loop_input = kernel_graph_ptr->device_loop_control_params();

  const std::vector<CNodePtr> &orders = kernel_graph_ptr->execution_order();
  if (orders.empty()) {
    MS_LOG(WARNING) << "graph " << kernel_graph_ptr->graph_id() << " execution order is empty";
    return;
  }

  std::vector<CNodePtr> exec_order;
  CNodePtr getnext_cnode;
  uint32_t getnext_switch_stream_id = UINT32_MAX;
  uint32_t fpbp_start_event_id = UINT32_MAX;
  uint32_t eos_start_event_id = UINT32_MAX;
  uint32_t getnext_stream_id = UINT32_MAX;
  size_t order_index = 0;

  if (exist_getnext) {
    InsertGetNextLoopStreamSwitch(kernel_graph_ptr, &exec_order, &getnext_switch_stream_id, &getnext_stream_id,
                                  switch_loop_input);
    SetBeforeGetNextStreamID(&exec_order, orders, &order_index, &getnext_cnode, getnext_stream_id);
    InsertGetNextLoopFpBpStartSend(kernel_graph_ptr, &exec_order, &fpbp_start_event_id, getnext_stream_id);
    if (eos_mode) {
      InsertGetNextLoopEosStartSend(kernel_graph_ptr, &exec_order, &eos_start_event_id, getnext_stream_id);
    }
  }

  uint32_t eos_switch_stream_id = UINT32_MAX;
  uint32_t eos_stream_id = UINT32_MAX;
  uint32_t eos_done_event_id = UINT32_MAX;
  std::vector<uint32_t> fpbp_active_streams;
  if (eos_mode) {
    InsertEosStreamSwitch(kernel_graph_ptr, switch_loop_input, &exec_order, &eos_switch_stream_id, &eos_stream_id);
    InsertGetNextLoopEosStartRecv(kernel_graph_ptr, &exec_order, eos_start_event_id, eos_stream_id);
    InsertEosOp(kernel_graph_ptr, &exec_order, getnext_cnode, eos_stream_id);
    InsertEosDoneSend(kernel_graph_ptr, &exec_order, &eos_done_event_id, eos_stream_id);
    fpbp_active_streams.push_back(eos_switch_stream_id);
  }

  bool exist_independent = ExistIndependent(kernel_graph_ptr);
  if (exist_independent) {
    InsertIndepentParallel(kernel_graph_ptr, switch_loop_input, &exec_order);
  }

  uint32_t fpbp_stream_id = UINT32_MAX;
  uint32_t fpbp_switch_stream_id = UINT32_MAX;
  InsertFpBpLoopStreamSwitch(kernel_graph_ptr, switch_loop_input, &exec_order, &fpbp_stream_id, &fpbp_switch_stream_id);
  InsertEndGraph(kernel_graph_ptr, &exec_order, fpbp_switch_stream_id);
  if (exist_getnext) {
    InsertFpBpStartRecv(kernel_graph_ptr, &exec_order, fpbp_start_event_id, fpbp_stream_id);
  }
  InsertNextLoopAssignAdd(kernel_graph_ptr, &exec_order, switch_loop_input, fpbp_stream_id);

  std::vector<CNodePtr> memcpy_list;
  std::vector<CNodePtr> other_list;
  if (exist_getnext) {
    CopyMemcpyList(kernel_graph_ptr, orders, order_index, &memcpy_list, &other_list);
    (void)std::copy(memcpy_list.begin(), memcpy_list.end(), std::back_inserter(exec_order));
  } else {
    other_list = orders;
  }

  if (eos_mode) {
    InsertEosDoneRecv(kernel_graph_ptr, &exec_order, eos_done_event_id, fpbp_stream_id);
  }
  std::vector<uint32_t> getnext_active_streams;
  if (exist_getnext) {
    // small loop active
    getnext_active_streams.push_back(getnext_switch_stream_id);
    InsertGetNextLoopStreamActive(kernel_graph_ptr, &exec_order, getnext_active_streams);
  }

  (void)std::copy(other_list.begin(), other_list.end(), std::back_inserter(exec_order));
  InsertCurrentLoopAssignAdd(kernel_graph_ptr, &exec_order, switch_loop_input);
  // big loop active
  fpbp_active_streams.push_back(fpbp_switch_stream_id);
  InsertFpBpAndEosLoopStreamActive(kernel_graph_ptr, &exec_order, fpbp_active_streams);
  kernel_graph_ptr->set_execution_order(exec_order);
}

kernel::KernelBuildInfo::KernelBuildInfoBuilder KernelAdjust::CreateMngKernelBuilder(
  const std::vector<std::string> &formats, const std::vector<TypeId> &type_ids) const {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder;
  selected_kernel_builder.SetInputsFormat(formats);
  selected_kernel_builder.SetInputsDeviceType(type_ids);

  selected_kernel_builder.SetFusionType(kernel::kPatternOpaque);
  selected_kernel_builder.SetProcessor(kernel::Processor::AICORE);
  selected_kernel_builder.SetKernelType(KernelType::RT_KERNEL);
  return selected_kernel_builder;
}

CNodePtr KernelAdjust::CreateStreamSwitchOp(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                            const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input,
                                            StreamSwitchKind kind) const {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder = CreateMngKernelBuilder(
    {kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt32});
  auto typeNone_abstract = std::make_shared<abstract::AbstractNone>();
  auto stream_switch = std::make_shared<Primitive>(kStreamSwitchOpName);
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(stream_switch));
  if (kind == kFpBpStreamSwitch || kind == kEosStreamSwitch) {
    inputs.push_back(switch_loop_input.at(kNextLoopCountName));
  } else if (kind == kGetNextStreamSwitch || kind == kIndependentStreamSwitch) {
    inputs.push_back(switch_loop_input.at(kNextLoopCountName));
  } else {
    MS_LOG(ERROR) << "unknown stream switch kind: " << kind;
  }

  inputs.push_back(switch_loop_input.at(kConstLoopNumInEpochName));
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  CNodePtr stream_switch_app = kernel_graph_ptr->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(stream_switch_app);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), stream_switch_app.get());
  stream_switch_app->set_abstract(typeNone_abstract);
  // set attr: cond_ RT_LESS
  int condition = static_cast<int>(RT_LESS_OR_EQUAL);
  ValuePtr cond = MakeValue(condition);
  common::AnfAlgo::SetNodeAttr(kAttrSwitchCondition, cond, stream_switch_app);
  // set attr:data_type
  int data_type = static_cast<int>(RT_SWITCH_INT32);
  ValuePtr dt = MakeValue(data_type);
  common::AnfAlgo::SetNodeAttr(kAttrDataType, dt, stream_switch_app);
  // set distinction label and graph id
  return stream_switch_app;
}

CNodePtr KernelAdjust::CreateEndGraphOp(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder = CreateMngKernelBuilder(
    {kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt32});
  auto abstract = std::make_shared<abstract::AbstractNone>();
  auto end_graph = std::make_shared<Primitive>(kEndGraph);
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(end_graph));
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  CNodePtr end_graph_node = kernel_graph_ptr->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(end_graph_node);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), end_graph_node.get());
  end_graph_node->set_abstract(abstract);
  return end_graph_node;
}

CNodePtr KernelAdjust::CreateStreamActiveOp(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder = CreateMngKernelBuilder(
    {kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt32});
  abstract::AbstractBasePtr typeNone_abstract = std::make_shared<abstract::AbstractNone>();
  auto stream_active_others = std::make_shared<Primitive>(kStreamActiveOpName);
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(stream_active_others));
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  CNodePtr stream_active_others_app = kernel_graph_ptr->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(stream_active_others_app);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), stream_active_others_app.get());
  stream_active_others_app->set_abstract(typeNone_abstract);
  return stream_active_others_app;
}

CNodePtr KernelAdjust::CreatTupleGetItemNode(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                             const CNodePtr &node, size_t output_idx) const {
  auto idx = NewValueNode(SizeToLong(output_idx));
  MS_EXCEPTION_IF_NULL(idx);
  auto imm = std::make_shared<Int64Imm>(SizeToInt(output_idx));
  auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
  idx->set_abstract(abstract_scalar);
  CNodePtr tuple_getitem = kernel_graph_ptr->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  tuple_getitem->set_scope(node->scope());
  auto origin_shape = common::AnfAlgo::GetOutputInferShape(node, output_idx);
  TypeId origin_type = common::AnfAlgo::GetOutputInferDataType(node, output_idx);
  common::AnfAlgo::SetOutputInferTypeAndShape({origin_type}, {origin_shape}, tuple_getitem.get());
  return tuple_getitem;
}

CNodePtr KernelAdjust::CreateEndOfSequenceOP(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                             const CNodePtr &getnext_cnode) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder;
  selected_kernel_builder.SetInputsFormat({kOpFormat_DEFAULT});
  selected_kernel_builder.SetInputsDeviceType({kNumberTypeUInt8});

  selected_kernel_builder.SetFusionType(kernel::kPatternOpaque);
  selected_kernel_builder.SetProcessor(kernel::Processor::AICPU);
  selected_kernel_builder.SetKernelType(KernelType::AICPU_KERNEL);

  selected_kernel_builder.SetOutputsFormat({kOpFormat_DEFAULT});
  selected_kernel_builder.SetOutputsDeviceType({kNumberTypeUInt8});
  // EndOfSequence
  auto end_of_sequence = std::make_shared<Primitive>(kEndOfSequence);
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(end_of_sequence));
  // GetNext output 0 is EndOfSequence's input
  auto tuple_get_item = CreatTupleGetItemNode(kernel_graph_ptr, getnext_cnode, 0);
  inputs.push_back(tuple_get_item);
  CNodePtr end_of_sequence_node = kernel_graph_ptr->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(end_of_sequence_node);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), end_of_sequence_node.get());
  std::vector<std::string> input_names = {"x"};
  ValuePtr input_names_v = MakeValue(input_names);
  common::AnfAlgo::SetNodeAttr("input_names", input_names_v, end_of_sequence_node);
  std::vector<std::string> output_names = {"y"};
  ValuePtr output_names_v = MakeValue(output_names);
  common::AnfAlgo::SetNodeAttr("output_names", output_names_v, end_of_sequence_node);
  end_of_sequence_node->set_abstract(tuple_get_item->abstract());
  return end_of_sequence_node;
}

CNodePtr KernelAdjust::CreateStreamAssignAddnOP(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                                const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input,
                                                bool cur_loop) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder = CreateMngKernelBuilder(
    {kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt32});
  selected_kernel_builder.SetOutputsFormat({kOpFormat_DEFAULT});
  selected_kernel_builder.SetOutputsDeviceType({kNumberTypeInt32});
  // AssignAdd
  auto assign_add = std::make_shared<Primitive>(kAssignAddOpName);
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(assign_add));
  if (cur_loop) {
    inputs.push_back(switch_loop_input.at(kCurLoopCountName));
  } else {
    inputs.push_back(switch_loop_input.at(kNextLoopCountName));
  }

  inputs.push_back(switch_loop_input.at(kConstOneName));
  CNodePtr assign_add_one = kernel_graph_ptr->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(assign_add_one);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), assign_add_one.get());
  std::vector<std::string> input_names = {"ref", "value"};
  std::vector<std::string> output_names = {"output"};
  ValuePtr input_names_v = MakeValue(input_names);
  ValuePtr output_names_v = MakeValue(output_names);
  common::AnfAlgo::SetNodeAttr("input_names", input_names_v, assign_add_one);
  common::AnfAlgo::SetNodeAttr("output_names", output_names_v, assign_add_one);
  selected_kernel_builder.SetKernelType(KernelType::TBE_KERNEL);
  MS_EXCEPTION_IF_NULL(switch_loop_input.at(kCurLoopCountName));
  assign_add_one->set_abstract(switch_loop_input.at(kCurLoopCountName)->abstract());
  // add AssignAdd op to kernel ref node map
  session::AnfWithOutIndex final_pair = std::make_pair(assign_add_one, 0);
  session::KernelWithIndex kernel_with_index =
    common::AnfAlgo::VisitKernel(common::AnfAlgo::GetInputNode(assign_add_one, 0), 0);
  kernel_graph_ptr->AddRefCorrespondPairs(final_pair, kernel_with_index);
  return assign_add_one;
}

#ifndef ENABLE_SECURITY
void KernelAdjust::Profiling(NotNull<session::KernelGraph *> kernel_graph_ptr) {
  if (!ascend::ProfilingManager::GetInstance().IsProfilingInitialized() ||
      ascend::ProfilingManager::GetInstance().IsMsprofiling()) {
    MS_LOG(INFO) << "No need to profiling";
    return;
  }
  ProfilingTraceInfo profiling_trace_info = ProfilingUtils::GenerateProfilingTrace(*kernel_graph_ptr);
  if (!profiling_trace_info.IsValid()) {
    MS_LOG(INFO) << "[profiling] no profiling node found!";
    return;
  }
  InsertProfilingKernel(profiling_trace_info, kernel_graph_ptr);
}

void KernelAdjust::InsertProfilingKernel(const ProfilingTraceInfo &profiling_trace_info,
                                         NotNull<session::KernelGraph *> kernel_graph_ptr) {
  MS_LOG(INFO) << "[profiling] Insert profiling kernel start";
  if (!profiling_trace_info.IsValid()) {
    MS_LOG(WARNING) << "Profiling trace point not found";
    return;
  }
  std::vector<CNodePtr> new_cnode_list;
  std::vector<CNodePtr> cnode_ptr_list = kernel_graph_ptr->execution_order();
  if (cnode_ptr_list.empty()) {
    MS_LOG(ERROR) << "No CNode in graph " << kernel_graph_ptr->graph_id();
    return;
  }
  for (const auto &cnode_ptr : cnode_ptr_list) {
    ProfilingUtils::InsertProfilingTraceFp(cnode_ptr, profiling_trace_info, kernel_graph_ptr,
                                           NOT_NULL(&new_cnode_list));
    (void)new_cnode_list.emplace_back(cnode_ptr);
    ProfilingUtils::InsertProfilingCustomOp(cnode_ptr, profiling_trace_info, kernel_graph_ptr,
                                            NOT_NULL(&new_cnode_list));
    ProfilingUtils::InsertProfilingTraceBpEnd(cnode_ptr, profiling_trace_info, kernel_graph_ptr,
                                              NOT_NULL(&new_cnode_list));
    ProfilingUtils::InsertProfilingTraceIterEnd(cnode_ptr, profiling_trace_info, kernel_graph_ptr,
                                                NOT_NULL(&new_cnode_list));
  }
  kernel_graph_ptr->set_execution_order(new_cnode_list);
}
#endif

CNodePtr KernelAdjust::CreateNPUGetFloatStatus(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                               const CNodePtr &npu_alloc_cnode) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(npu_alloc_cnode);
  auto npu_get_primitive = std::make_shared<Primitive>(kNPUGetFloatStatusOpName);
  std::vector<AnfNodePtr> npu_get_inputs = {NewValueNode(npu_get_primitive), npu_alloc_cnode};
  auto npu_get_cnode = kernel_graph_ptr->NewCNode(npu_get_inputs);
  MS_EXCEPTION_IF_NULL(npu_get_cnode);
  npu_alloc_cnode->set_scope(kDefaultScope);
  npu_get_cnode->set_abstract(npu_alloc_cnode->abstract());

  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder;
  selected_kernel_builder.SetInputsFormat({kOpFormat_DEFAULT});
  selected_kernel_builder.SetInputsDeviceType({kNumberTypeFloat32});
  selected_kernel_builder.SetFusionType(kernel::kPatternOpaque);
  selected_kernel_builder.SetProcessor(kernel::Processor::AICORE);
  selected_kernel_builder.SetKernelType(KernelType::TBE_KERNEL);
  selected_kernel_builder.SetOutputsFormat({kOpFormat_DEFAULT});
  selected_kernel_builder.SetOutputsDeviceType({kNumberTypeFloat32});
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), npu_get_cnode.get());
  return npu_get_cnode;
}

CNodePtr KernelAdjust::CreateNPUClearStatus(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                            const CNodePtr &npu_alloc_cnode) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(npu_alloc_cnode);
  auto npu_clear_primitive = std::make_shared<Primitive>(kNPUClearFloatStatusOpName);
  std::vector<AnfNodePtr> npu_clear_inputs = {NewValueNode(npu_clear_primitive), npu_alloc_cnode};
  auto npu_clear_cnode = kernel_graph_ptr->NewCNode(npu_clear_inputs);
  MS_EXCEPTION_IF_NULL(npu_clear_cnode);
  npu_alloc_cnode->set_scope(kDefaultScope);
  npu_clear_cnode->set_abstract(npu_alloc_cnode->abstract());

  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder;
  selected_kernel_builder.SetInputsFormat({kOpFormat_DEFAULT});
  selected_kernel_builder.SetInputsDeviceType({kNumberTypeFloat32});
  selected_kernel_builder.SetFusionType(kernel::kPatternOpaque);
  selected_kernel_builder.SetProcessor(kernel::Processor::AICORE);
  selected_kernel_builder.SetKernelType(KernelType::TBE_KERNEL);
  selected_kernel_builder.SetOutputsFormat({kOpFormat_DEFAULT});
  selected_kernel_builder.SetOutputsDeviceType({kNumberTypeFloat32});
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), npu_clear_cnode.get());

  return npu_clear_cnode;
}

CNodePtr KernelAdjust::CreateNPUAllocStatus(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  // create npu_alloc_cnode
  auto npu_alloc_primitive = std::make_shared<Primitive>(kNPUAllocFloatStatusOpName);
  std::vector<AnfNodePtr> npu_alloc_inputs = {NewValueNode(npu_alloc_primitive)};
  auto npu_alloc_cnode = kernel_graph_ptr->NewCNode(npu_alloc_inputs);
  MS_EXCEPTION_IF_NULL(npu_alloc_cnode);
  npu_alloc_cnode->set_scope(kDefaultScope);
  ShapeVector npu_output_shape = {kNPUShape};
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {npu_output_shape}, npu_alloc_cnode.get());

  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder;
  selected_kernel_builder.SetFusionType(kernel::kPatternOpaque);
  selected_kernel_builder.SetProcessor(kernel::Processor::AICORE);
  selected_kernel_builder.SetKernelType(KernelType::TBE_KERNEL);
  selected_kernel_builder.SetOutputsFormat({kOpFormat_DEFAULT});
  selected_kernel_builder.SetOutputsDeviceType({kNumberTypeFloat32});
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), npu_alloc_cnode.get());
  return npu_alloc_cnode;
}

CNodePtr KernelAdjust::CreateAssignAdd(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                       const CNodePtr &npu_alloc_cnode, const AnfNodePtr &specify_para) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(npu_alloc_cnode);
  MS_EXCEPTION_IF_NULL(specify_para);
  auto assign_add_primitive = std::make_shared<Primitive>(kAssignAddOpName);
  std::vector<AnfNodePtr> assign_add_inputs = {NewValueNode(assign_add_primitive), specify_para, npu_alloc_cnode};
  auto assign_add_cnode = kernel_graph_ptr->NewCNode(assign_add_inputs);
  MS_EXCEPTION_IF_NULL(assign_add_cnode);
  assign_add_cnode->set_scope(kDefaultScope);
  assign_add_cnode->set_abstract(specify_para->abstract());

  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder = CreateMngKernelBuilder(
    {kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {TypeId::kNumberTypeFloat32, TypeId::kNumberTypeFloat32});
  selected_kernel_builder.SetOutputsFormat({kOpFormat_DEFAULT});
  selected_kernel_builder.SetOutputsDeviceType({kNumberTypeFloat32});

  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), assign_add_cnode.get());
  std::vector<std::string> input_names = {"ref", "value"};
  std::vector<std::string> output_names = {"output"};
  ValuePtr input_names_v = MakeValue(input_names);
  ValuePtr output_names_v = MakeValue(output_names);
  common::AnfAlgo::SetNodeAttr("input_names", input_names_v, assign_add_cnode);
  common::AnfAlgo::SetNodeAttr("output_names", output_names_v, assign_add_cnode);
  selected_kernel_builder.SetKernelType(KernelType::TBE_KERNEL);

  session::AnfWithOutIndex final_pair = std::make_pair(assign_add_cnode, 0);
  session::KernelWithIndex kernel_with_index =
    common::AnfAlgo::VisitKernel(common::AnfAlgo::GetInputNode(assign_add_cnode, 0), 0);
  kernel_graph_ptr->AddRefCorrespondPairs(final_pair, kernel_with_index);
  return assign_add_cnode;
}

CNodePtr KernelAdjust::CreateAssign(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                    const AnfNodePtr &specify_para) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(specify_para);

  std::vector<float> reset(kNPUShape, 0.0);
  ShapeVector reset_shape({kNPUShape});
  auto shp_buf_size = sizeof(float) * reset.size();
  auto reset_tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, reset_shape, reset.data(), shp_buf_size);
  auto reset_value_node = std::make_shared<ValueNode>(reset_tensor);
  MS_EXCEPTION_IF_NULL(reset_value_node);
  reset_value_node->set_abstract(specify_para->abstract());
  kernel_graph_ptr->AddValueNodeToGraph(reset_value_node);

  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  reset_value_node->set_kernel_info(kernel_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetOutputsFormat({kOpFormat_DEFAULT});
  builder1.SetOutputsDeviceType({kNumberTypeFloat32});
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), reset_value_node.get());

  auto assign_primitive = std::make_shared<Primitive>(kAssignOpName);
  std::vector<AnfNodePtr> assign_inputs = {NewValueNode(assign_primitive), specify_para, reset_value_node};
  auto assign_cnode = kernel_graph_ptr->NewCNode(assign_inputs);
  MS_EXCEPTION_IF_NULL(assign_cnode);
  assign_cnode->set_scope(kDefaultScope);
  assign_cnode->set_abstract(specify_para->abstract());

  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder = CreateMngKernelBuilder(
    {kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {TypeId::kNumberTypeFloat32, TypeId::kNumberTypeFloat32});
  selected_kernel_builder.SetOutputsFormat({kOpFormat_DEFAULT});
  selected_kernel_builder.SetOutputsDeviceType({kNumberTypeFloat32});

  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), assign_cnode.get());
  std::vector<std::string> input_names = {"ref", "value"};
  std::vector<std::string> output_names = {"output"};
  ValuePtr input_names_v = MakeValue(input_names);
  ValuePtr output_names_v = MakeValue(output_names);
  common::AnfAlgo::SetNodeAttr("input_names", input_names_v, assign_cnode);
  common::AnfAlgo::SetNodeAttr("output_names", output_names_v, assign_cnode);
  selected_kernel_builder.SetKernelType(KernelType::TBE_KERNEL);

  session::AnfWithOutIndex final_pair = std::make_pair(assign_cnode, 0);
  session::KernelWithIndex kernel_with_index =
    common::AnfAlgo::VisitKernel(common::AnfAlgo::GetInputNode(assign_cnode, 0), 0);
  kernel_graph_ptr->AddRefCorrespondPairs(final_pair, kernel_with_index);
  return assign_cnode;
}

void KernelAdjust::InsertOverflowCheckOperations(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const {
  MS_LOG(INFO) << "Start Insert Overflow Check Operations.";

  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  std::set<std::shared_ptr<session::KernelGraph>> child_graph_list;
  std::queue<std::shared_ptr<session::KernelGraph>> graph_queue;
  graph_queue.push(kernel_graph_ptr);
  while (!graph_queue.empty()) {
    auto graph = graph_queue.front();
    (void)child_graph_list.insert(graph);
    graph_queue.pop();
    for (auto child_graph : graph->child_graph_order()) {
      if (child_graph_list.count(child_graph.lock()) == 0) {
        graph_queue.push(child_graph.lock());
      }
    }
  }

  AnfNodePtr specify_param = nullptr;
  std::vector<AnfNodePtr> dynamic_loss_scale_param_list;
  // find parameter in all child graph.
  for (const auto child_graph : child_graph_list) {
    auto parameters = child_graph->parameters();
    for (auto param : parameters) {
      auto param_fullname = param->fullname_with_scope();
      if (param_fullname.find(kSpecifyParameter) != std::string::npos) {
        specify_param = param;
        continue;
      }
      if (param_fullname.find(kMixLayerStatusParameter) != std::string::npos) {
        (void)dynamic_loss_scale_param_list.emplace_back(param);
      }
    }
  }

  for (const auto &param : dynamic_loss_scale_param_list) {
    MS_LOG(DEBUG) << "dynamic_loss_scale_param_list:" << param->DebugString();
  }

  if (specify_param != nullptr) {
    InsertGradientOverflowCheckOperations(specify_param, kernel_graph_ptr);
  }
  if (!dynamic_loss_scale_param_list.empty()) {
    InsertDynamicLossScaleCheckOperations(kernel_graph_ptr, &dynamic_loss_scale_param_list);
  }

  for (const auto &node : kernel_graph_ptr->execution_order()) {
    MS_LOG(DEBUG) << "After insert Overflow Status Order:" << node->DebugString();
  }
  MS_LOG(INFO) << "End Insert Overflow Check Operations.";
}

void KernelAdjust::InsertGradientOverflowCheckOperations(
  const AnfNodePtr &specify_para, const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const {
  MS_LOG(INFO) << "Start Insert Gradient Overflow Check Operations.";
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);

  bool first_grad_op = true;
  CNodePtr npu_alloc_cnode;
  std::vector<CNodePtr> new_execution_order;
  auto execution_order = kernel_graph_ptr->execution_order();
  for (size_t i = 0; i < execution_order.size() - 1; i++) {
    new_execution_order.push_back(execution_order[i]);
    auto cur_full_name = execution_order[i]->fullname_with_scope();
    auto next_full_name = execution_order[i + 1]->fullname_with_scope();
    auto cur_stream_id = AnfAlgo::GetStreamId(execution_order[i]);
    auto next_stream_id = AnfAlgo::GetStreamId(execution_order[i + 1]);

    if (cur_full_name.find(kGradients) == std::string::npos && next_full_name.find(kGradients) != std::string::npos) {
      if (first_grad_op) {
        npu_alloc_cnode = CreateNPUAllocStatus(kernel_graph_ptr);
        auto npu_clear_cnode = CreateNPUClearStatus(kernel_graph_ptr, npu_alloc_cnode);
        auto assign_cnode = CreateAssign(kernel_graph_ptr, specify_para);
        AnfAlgo::SetStreamId(next_stream_id, npu_alloc_cnode.get());
        AnfAlgo::SetStreamId(next_stream_id, npu_clear_cnode.get());
        AnfAlgo::SetStreamId(next_stream_id, assign_cnode.get());
        new_execution_order.push_back(npu_alloc_cnode);
        new_execution_order.push_back(npu_clear_cnode);
        new_execution_order.push_back(assign_cnode);
        first_grad_op = false;
      } else {
        auto npu_clear_cnode = CreateNPUClearStatus(kernel_graph_ptr, npu_alloc_cnode);
        AnfAlgo::SetStreamId(next_stream_id, npu_clear_cnode.get());
        new_execution_order.push_back(npu_clear_cnode);
      }
    }
    if (cur_full_name.find(kGradients) != std::string::npos && next_full_name.find(kGradients) == std::string::npos) {
      auto npu_get_cnode = CreateNPUGetFloatStatus(kernel_graph_ptr, npu_alloc_cnode);
      auto assign_add_cnode = CreateAssignAdd(kernel_graph_ptr, npu_alloc_cnode, specify_para);
      AnfAlgo::SetStreamId(cur_stream_id, npu_get_cnode.get());
      AnfAlgo::SetStreamId(cur_stream_id, assign_add_cnode.get());
      new_execution_order.push_back(npu_get_cnode);
      new_execution_order.push_back(assign_add_cnode);
    }
    if (i == execution_order.size() - kLastHandleDiff) {
      new_execution_order.push_back(execution_order[i + 1]);
      if (next_full_name.find(kGradients) != std::string::npos) {
        auto npu_get_cnode = CreateNPUGetFloatStatus(kernel_graph_ptr, npu_alloc_cnode);
        auto assign_add_cnode = CreateAssignAdd(kernel_graph_ptr, npu_alloc_cnode, specify_para);
        AnfAlgo::SetStreamId(cur_stream_id, npu_get_cnode.get());
        AnfAlgo::SetStreamId(cur_stream_id, assign_add_cnode.get());
        new_execution_order.push_back(npu_get_cnode);
        new_execution_order.push_back(assign_add_cnode);
      }
    }
  }
  kernel_graph_ptr->set_execution_order(new_execution_order);
  MS_LOG(INFO) << "End Insert Gradient Overflow Check Operations.";
}

void KernelAdjust::InsertDynamicLossScaleCheckOperations(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                                         std::vector<AnfNodePtr> *dynamic_loss_scale_param_list) const {
  MS_LOG(INFO) << "Start Insert Dynamic Loss Scale Operations.";
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  auto execution_order = kernel_graph_ptr->execution_order();
  size_t end_gradient_index = 0;
  for (size_t i = 0; i < execution_order.size(); ++i) {
    if (execution_order[i]->fullname_with_scope().find(kGradients) != std::string::npos) {
      end_gradient_index = i;
    }
  }

  std::sort(dynamic_loss_scale_param_list->begin(), dynamic_loss_scale_param_list->end(),
            [](const AnfNodePtr &node_a, const AnfNodePtr &node_b) {
              const auto &param_name_a = node_a->fullname_with_scope();
              const auto &param_name_b = node_b->fullname_with_scope();
              int value_a = -1;
              int value_b = -1;
              try {
                value_a = std::stoi(param_name_a.substr(param_name_a.rfind("_") + 1, param_name_a.size()).data());
                value_b = std::stoi(param_name_b.substr(param_name_b.rfind("_") + 1, param_name_b.size()).data());
              } catch (std::invalid_argument &) {
                MS_LOG(EXCEPTION) << "Invalid param name:" << param_name_a << " and " << param_name_b;
              }
              return value_a < value_b;
            });

  // Manual module
  // If this mul operator has kSplitOverFlow attr, insert npuallocstatus and npuclearstatus.
  // If this mul operator has kLayerOverFLow attr, compare it with current dynamic parameter.
  // insert npugetstatus ans assign value to this param.
  bool first_layer_op = true;
  std::vector<CNodePtr> new_execution_order;
  int64_t cur_param = static_cast<int64_t>(dynamic_loss_scale_param_list->size()) - 1;
  CNodePtr npu_alloc_cnode;
  std::set<int64_t> viewed_id;
  for (size_t i = 0; i < execution_order.size(); ++i) {
    auto cur_node = execution_order[i];
    auto cur_stream_id = AnfAlgo::GetStreamId(cur_node);
    if (common::AnfAlgo::HasNodeAttr(kSplitOverFlow, cur_node) || (i == end_gradient_index)) {
      if (first_layer_op) {
        npu_alloc_cnode = CreateNPUAllocStatus(kernel_graph_ptr);
        AnfAlgo::SetStreamId(cur_stream_id, npu_alloc_cnode.get());
        (void)new_execution_order.emplace_back(npu_alloc_cnode);
        for (const auto &param : *dynamic_loss_scale_param_list) {
          auto assign_cnode = CreateAssign(kernel_graph_ptr, param);
          AnfAlgo::SetStreamId(cur_stream_id, assign_cnode.get());
          (void)new_execution_order.emplace_back(assign_cnode);
        }
        first_layer_op = false;
      } else {
        if (common::AnfAlgo::HasNodeAttr(kLayerOverFlow, cur_node)) {
          cur_param = common::AnfAlgo::GetNodeAttr<int64_t>(cur_node, kLayerOverFlow);
        }
        if (cur_param < 0 || cur_param >= static_cast<int64_t>(dynamic_loss_scale_param_list->size())) {
          MS_LOG(WARNING) << "Overflow check index is invalid, value is " << cur_param;
          (void)new_execution_order.emplace_back(cur_node);
          continue;
        }
        if (viewed_id.count(cur_param) != 0) {
          auto assign_cnode = CreateAssign(kernel_graph_ptr, dynamic_loss_scale_param_list->at(cur_param));
          AnfAlgo::SetStreamId(cur_stream_id, assign_cnode.get());
          (void)new_execution_order.emplace_back(assign_cnode);
        }
        auto npu_get_cnode = CreateNPUGetFloatStatus(kernel_graph_ptr, npu_alloc_cnode);
        AnfAlgo::SetStreamId(cur_stream_id, npu_get_cnode.get());
        (void)new_execution_order.emplace_back(npu_get_cnode);
        auto assign_add_cnode =
          CreateAssignAdd(kernel_graph_ptr, npu_alloc_cnode, dynamic_loss_scale_param_list->at(cur_param));
        AnfAlgo::SetStreamId(cur_stream_id, assign_add_cnode.get());
        (void)new_execution_order.emplace_back(assign_add_cnode);
        (void)viewed_id.insert(cur_param);
        cur_param--;
      }
      auto npu_clear_cnode = CreateNPUClearStatus(kernel_graph_ptr, npu_alloc_cnode);
      AnfAlgo::SetStreamId(cur_stream_id, npu_clear_cnode.get());
      (void)new_execution_order.emplace_back(npu_clear_cnode);
    }
    (void)new_execution_order.emplace_back(cur_node);
  }
  kernel_graph_ptr->set_execution_order(new_execution_order);
  MS_LOG(INFO) << "End Insert Dynamic Loss Scale Operations.";
}

// device loop control
std::shared_ptr<Tensor> KernelAdjust::CreateTensor(int32_t initial_value) const {
  ShapeVector shp = {1};
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(tensor);
  auto val = static_cast<int32_t *>(tensor->data_c());
  MS_EXCEPTION_IF_NULL(val);
  *val = initial_value;
  return tensor;
}

std::shared_ptr<Parameter> KernelAdjust::CreateParameter(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                                         const string parameter_name) const {
  ShapeVector shp = {1};
  tensor::TensorPtr tensor_ptr = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  mindspore::abstract::AbstractBasePtr parameter_abstract_ptr = tensor_ptr->ToAbstract();
  if (parameter_abstract_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Create abstract for device loop control failed!";
  }

  ParameterPtr param = std::make_shared<Parameter>(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(param);
  param->set_name(parameter_name);
  param->set_abstract(parameter_abstract_ptr);
  ParameterPtr graph_parameter = kernel_graph_ptr->NewParameter(param);
  return graph_parameter;
}

void KernelAdjust::InsertDeviceLoopCtrl(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  std::map<std::string, tensor::TensorPtr> device_loop_ctrl_tensors;
  std::map<std::string, mindspore::ParameterPtr> device_loop_ctrl_params;

  // current loop count
  device_loop_ctrl_tensors[kCurLoopCountName] = CreateTensor(0);
  device_loop_ctrl_params[kCurLoopCountName] = CreateParameter(kernel_graph_ptr, kCurLoopCountName);

  // next loop count tensor
  device_loop_ctrl_tensors[kNextLoopCountName] = CreateTensor(0);
  device_loop_ctrl_params[kNextLoopCountName] = CreateParameter(kernel_graph_ptr, kNextLoopCountName);

  // current epoch count tensor
  device_loop_ctrl_tensors[kCurEpochCountName] = CreateTensor(0);
  device_loop_ctrl_params[kCurEpochCountName] = CreateParameter(kernel_graph_ptr, kCurEpochCountName);

  // constant one tensor
  device_loop_ctrl_tensors[kConstOneName] = CreateTensor(1);
  device_loop_ctrl_params[kConstOneName] = CreateParameter(kernel_graph_ptr, kConstOneName);

  // constant loop num in epoch tensor
  int32_t initial_value = 0;
  if (NeedLoopSink()) {
    // iter_num minus one because the device side counts from 0
    initial_value = SizeToInt(LongToSize(ConfigManager::GetInstance().iter_num() - 1));
  } else {
    MS_LOG(INFO) << "Tensor const_loop_num_in_epoch only used in loop sink mode.";
    initial_value = 0;
  }
  MS_LOG(INFO) << "Loop num in epoch is " << initial_value;
  device_loop_ctrl_tensors[kConstLoopNumInEpochName] = CreateTensor(initial_value);
  device_loop_ctrl_params[kConstLoopNumInEpochName] = CreateParameter(kernel_graph_ptr, kConstLoopNumInEpochName);

  kernel_graph_ptr->set_device_loop_ctrl_tensors(device_loop_ctrl_tensors);
  kernel_graph_ptr->set_device_loop_ctrl_params(device_loop_ctrl_params);
}

void KernelAdjust::AssignLoopCtrlTensorMem(const session::KernelGraph &kernel_graph, KernelRuntime *runtime_instance,
                                           const string name) const {
  MS_EXCEPTION_IF_NULL(runtime_instance);
  auto device_loop_control_params = kernel_graph.device_loop_control_params();
  if (device_loop_control_params.count(name) == 0) {
    MS_LOG(WARNING) << "Can't find Device Loop Control Parameter " << name;
    return;
  }
  auto param = device_loop_control_params.at(name);
  MS_EXCEPTION_IF_NULL(param);

  DeviceAddressPtr device_address = nullptr;
  if (AnfAlgo::OutputAddrExist(param, 0)) {
    device_address = AnfAlgo::GetMutableOutputAddr(param, 0);
    MS_EXCEPTION_IF_NULL(device_address);
  } else {
    MS_LOG(INFO) << "Device Loop Control Parameter " << name << " have no address, allocating...";
    auto size = AnfAlgo::GetOutputTensorMemSize(param, 0);
    auto format = AnfAlgo::GetOutputFormat(param, 0);
    auto type_id = AnfAlgo::GetOutputDeviceDataType(param, 0);

    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device_address =
      std::make_shared<device::ascend::AscendDeviceAddress>(nullptr, size, format, type_id, kAscendDevice, device_id);
    device_address->set_is_ptr_persisted(true);

    if (runtime_instance->MallocMem(kStaticMem, size, device_address) == nullptr) {
      MS_LOG(EXCEPTION) << "Cannot alloc static memory for device loop control parameter " << name
                        << " , tensor size is : " << size;
    }
    MS_EXCEPTION_IF_NULL(device_address);
    AnfAlgo::SetOutputAddr(device_address, 0, param.get());
  }

  auto device_loop_control_tensors = kernel_graph.device_loop_control_tensors();
  auto tensor = device_loop_control_tensors.at(name);
  MS_EXCEPTION_IF_NULL(tensor);
  tensor->set_device_address(device_address);
  if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(param, 0), LongToSize(tensor->data().nbytes()),
                                        tensor->data_type(), tensor->data_c(), tensor->device_info().host_format_)) {
    MS_LOG(EXCEPTION) << "SyncHostToDevice failed for device loop control parameter " << name;
  }
}

void KernelAdjust::AssignLoopCtrlMemory(const session::KernelGraph &kernel_graph_ptr) const {
  auto device_loop_control_tensors = kernel_graph_ptr.device_loop_control_tensors();
  if (device_loop_control_tensors.empty()) {
    return;
  }
  MS_LOG(INFO) << "Assign device loop control memory";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto runtime_instance = KernelRuntimeManager::Instance().GetSingleKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  AssignLoopCtrlTensorMem(kernel_graph_ptr, runtime_instance, kCurLoopCountName);
  AssignLoopCtrlTensorMem(kernel_graph_ptr, runtime_instance, kNextLoopCountName);
  AssignLoopCtrlTensorMem(kernel_graph_ptr, runtime_instance, kCurEpochCountName);
  AssignLoopCtrlTensorMem(kernel_graph_ptr, runtime_instance, kConstOneName);
  AssignLoopCtrlTensorMem(kernel_graph_ptr, runtime_instance, kConstLoopNumInEpochName);
}

void KernelAdjust::SetDeviceLoopCtrlTensor(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                           const std::string name, int32_t value) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  auto device_loop_control_tensors = kernel_graph_ptr->device_loop_control_tensors();
  if (device_loop_control_tensors.count(name) == 0) {
    MS_LOG(WARNING) << "Can't find Device Loop Control Tensor " << name;
    return;
  }
  auto tensor = device_loop_control_tensors.at(name);
  MS_EXCEPTION_IF_NULL(tensor);
  auto *cur_val = static_cast<int32_t *>(tensor->data_c());
  MS_EXCEPTION_IF_NULL(cur_val);
  *cur_val = value;
  tensor->set_sync_status(kNeedSyncHostToDevice);
  auto device_address = tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  if (!device_address->SyncHostToDevice(tensor->shape(), LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                        tensor->data_c(), tensor->device_info().host_format_)) {
    MS_LOG(EXCEPTION) << "SyncHostToDevice failed for device loop control parameter " << name;
  }
}

void KernelAdjust::LoadDeviceLoopCtrlParameters(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  auto device_loop_control_tensors = kernel_graph_ptr->device_loop_control_tensors();
  if (device_loop_control_tensors.empty()) {
    return;
  }
  MS_LOG(INFO) << "Load device loop control data";
  SetDeviceLoopCtrlTensor(kernel_graph_ptr, kCurLoopCountName, 0);
  SetDeviceLoopCtrlTensor(kernel_graph_ptr, kNextLoopCountName, 0);
#ifndef ENABLE_SECURITY
  SetDeviceLoopCtrlTensor(kernel_graph_ptr, kCurEpochCountName,
                          SizeToInt(DumpJsonParser::GetInstance().cur_dump_iter()));
#else
  SetDeviceLoopCtrlTensor(kernel_graph_ptr, kCurEpochCountName, 0);
#endif

  kernel_graph_ptr->set_current_epoch(kernel_graph_ptr->current_epoch() + 1);
}
}  // namespace device
}  // namespace mindspore
