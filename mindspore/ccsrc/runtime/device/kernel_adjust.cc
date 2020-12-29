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

#include "runtime/device/kernel_adjust.h"

#include <map>
#include <algorithm>
#include <string>
#include <vector>
#include <utility>

#include "backend/session/anf_runtime_algorithm.h"
#include "utils/ms_context.h"
#include "common/trans.h"
#include "utils/config_manager.h"
#include "utils/ms_utils.h"
#include "backend/kernel_compiler/kernel_build_info.h"
#include "utils/utils.h"
#include "runtime/device/ascend/profiling/profiling_manager.h"
#include "runtime/base.h"
#include "runtime/device/ascend/ascend_stream_assign.h"
#include "utils/shape_utils.h"

namespace {
constexpr auto kProfilingGraphId = "PROFILING_GRAPH_ID";
}  // namespace
namespace mindspore {
namespace device {
using device::ascend::ProfilingUtils;
void KernelAdjust::ReorderGetNext(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  const std::vector<CNodePtr> &origin_cnode_list = kernel_graph_ptr->execution_order();
  std::vector<CNodePtr> getnext_list;
  std::vector<CNodePtr> other_list;
  for (const auto &cnode : origin_cnode_list) {
    if (AnfAlgo::GetCNodeName(cnode) == kGetNextOpName) {
      getnext_list.emplace_back(cnode);
    } else {
      other_list.emplace_back(cnode);
    }
  }
  std::vector<CNodePtr> new_order_list;
  new_order_list.insert(new_order_list.end(), getnext_list.begin(), getnext_list.end());
  new_order_list.insert(new_order_list.end(), other_list.begin(), other_list.end());
  kernel_graph_ptr->set_execution_order(new_order_list);
}

bool KernelAdjust::NeedInsertSwitch() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  return (context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK) &&
          context_ptr->get_param<bool>(MS_CTX_ENABLE_LOOP_SINK) && ConfigManager::GetInstance().iter_num() > 1);
}

CNodePtr KernelAdjust::CreateSendApplyKernel(const std::shared_ptr<session::KernelGraph> &graph_ptr,
                                             uint32_t event_id) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
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
  return send_node_ptr;
}

CNodePtr KernelAdjust::CreateRecvApplyKernel(const std::shared_ptr<session::KernelGraph> &graph_ptr,
                                             uint32_t event_id) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
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
  auto abstract_none = std::make_shared<abstract::AbstractNone>();
  MS_EXCEPTION_IF_NULL(abstract_none);
  recv_node_ptr->set_abstract(abstract_none);
  return recv_node_ptr;
}

bool KernelAdjust::ExistGetNext(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  const std::vector<CNodePtr> &cnode_list = kernel_graph_ptr->execution_order();
  for (const auto &cnode : cnode_list) {
    if (AnfAlgo::GetCNodeName(cnode) == kGetNextOpName) {
      return true;
    }
  }
  return false;
}

bool KernelAdjust::ExistIndependent(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  const auto &exe_orders = kernel_graph_ptr->execution_order();
  for (const auto &node : exe_orders) {
    if (AnfAlgo::IsIndependentNode(node)) {
      MS_LOG(INFO) << "graph exit independent node";
      return true;
    }
  }

  return false;
}

void KernelAdjust::InsertSwitchLoop(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
  device::ascend::AscendResourceMng &resource_manager = device::ascend::AscendResourceMng::GetInstance();
  resource_manager.ResetResource();
  if (!NeedInsertSwitch()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  if (kernel_graph_ptr->is_dynamic_shape()) {
    MS_LOG(INFO) << "KernelGraph:" << kernel_graph_ptr->graph_id() << " is dynamic shape, skip InsertSwitchLoop";
    return;
  }
  bool exist_getnext = ExistGetNext(kernel_graph_ptr);
  bool eos_mode = ConfigManager::GetInstance().iter_num() == INT32_MAX && exist_getnext;
  MS_LOG(INFO) << "GetNext exist:" << exist_getnext << " End of Sequence mode:" << eos_mode
               << " iter num:" << ConfigManager::GetInstance().iter_num();
  if (exist_getnext) {
    ReorderGetNext(kernel_graph_ptr);
  }
  std::map<std::string, mindspore::ParameterPtr> switch_loop_input;
  CreateSwitchOpParameters(kernel_graph_ptr, &switch_loop_input);

  std::vector<AnfNodePtr> *mute_inputs = kernel_graph_ptr->MutableInputs();
  MS_EXCEPTION_IF_NULL(mute_inputs);
  mute_inputs->push_back(switch_loop_input[kCurLoopCountParamName]);
  mute_inputs->push_back(switch_loop_input[kNextLoopCountParamName]);
  mute_inputs->push_back(switch_loop_input[kEpochParamName]);
  mute_inputs->push_back(switch_loop_input[kIterLoopParamName]);
  mute_inputs->push_back(switch_loop_input[kOneParamName]);
  for (const auto &input : kernel_graph_ptr->inputs()) {
    MS_EXCEPTION_IF_NULL(input);
    if (input->isa<Parameter>()) {
      ParameterPtr param_ptr = input->cast<ParameterPtr>();
      if (param_ptr == nullptr) {
        MS_EXCEPTION(NotSupportError) << "Cast to parameter point failed !";
      }
    }
  }

  const std::vector<CNodePtr> &orders = kernel_graph_ptr->execution_order();
  if (orders.empty()) {
    MS_LOG(EXCEPTION) << "graph execution order is empty";
  }

  std::vector<CNodePtr> exec_order;
  std::vector<uint32_t> getnext_active_streams;
  std::vector<uint32_t> fpbp_active_streams;
  CNodePtr getnext_cnode;
  uint32_t getnext_switch_stream_id = UINT32_MAX;
  uint32_t fpbp_start_event_id = UINT32_MAX;
  uint32_t eos_start_event_id = UINT32_MAX;
  uint32_t eos_done_event_id = UINT32_MAX;
  size_t i = 0;

  // getnext loop process
  if (exist_getnext) {
    // getnext loop stream switch op
    getnext_switch_stream_id = resource_manager.ApplyNewStream();
    uint32_t getnext_stream_id = resource_manager.ApplyNewStream();
    CNodePtr getnext_switch_app = CreateStreamSwitchOp(kernel_graph_ptr, switch_loop_input, kGetNextStreamSwitch);
    MS_EXCEPTION_IF_NULL(getnext_switch_app);
    AnfAlgo::SetStreamId(getnext_switch_stream_id, getnext_switch_app.get());
    // update getnext loop stream switch true_branch_stream attr
    AnfAlgo::SetNodeAttr(kStreamNeedActivedFirst, MakeValue<bool>(true), getnext_switch_app);
    AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, MakeValue<uint32_t>(getnext_stream_id), getnext_switch_app);
    AnfAlgo::SetNodeAttr(kAttrStreamSwitchKind, MakeValue<uint32_t>(kGetNextStreamSwitch), getnext_switch_app);
    exec_order.push_back(getnext_switch_app);
    MS_LOG(INFO) << "GetNext loop insert Stream Switch " << getnext_switch_app->fullname_with_scope();

    // getnext op
    for (; i < orders.size(); i++) {
      auto node = orders[i];
      exec_order.push_back(node);
      AnfAlgo::SetStreamId(getnext_stream_id, exec_order[exec_order.size() - 1].get());
      if (AnfAlgo::GetCNodeName(node) == kGetNextOpName) {
        getnext_cnode = node;
        break;
      }
    }

    // getnext loop fpbp start send
    fpbp_start_event_id = resource_manager.ApplyNewEvent();
    CNodePtr fpbp_start_send = CreateSendApplyKernel(kernel_graph_ptr, fpbp_start_event_id);
    AnfAlgo::SetStreamId(getnext_stream_id, fpbp_start_send.get());
    exec_order.push_back(fpbp_start_send);
    MS_LOG(INFO) << "GetNext loop insert FpBp start Send " << fpbp_start_send->fullname_with_scope();

    if (eos_mode) {
      // getnext loop eos start send
      eos_start_event_id = resource_manager.ApplyNewEvent();
      CNodePtr eos_start_send = CreateSendApplyKernel(kernel_graph_ptr, eos_start_event_id);
      AnfAlgo::SetStreamId(getnext_stream_id, eos_start_send.get());
      exec_order.push_back(eos_start_send);
      MS_LOG(INFO) << "GetNext loop insert EoS start Send " << eos_start_send->fullname_with_scope();
    }
  }

  // End Of Sequence loop process
  if (eos_mode) {
    // eos loop stream switch
    uint32_t eos_switch_stream_id = resource_manager.ApplyNewStream();
    uint32_t eos_stream_id = resource_manager.ApplyNewStream();
    CNodePtr eos_switch_app = CreateStreamSwitchOp(kernel_graph_ptr, switch_loop_input, kEosStreamSwitch);
    MS_EXCEPTION_IF_NULL(eos_switch_app);
    AnfAlgo::SetStreamId(eos_switch_stream_id, eos_switch_app.get());
    AnfAlgo::SetNodeAttr(kStreamNeedActivedFirst, MakeValue<bool>(true), eos_switch_app);
    // update eos loop stream switch true_branch_stream attr
    AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, MakeValue<uint32_t>(eos_stream_id), eos_switch_app);
    AnfAlgo::SetNodeAttr(kAttrStreamSwitchKind, MakeValue<uint32_t>(kEosStreamSwitch), eos_switch_app);
    exec_order.push_back(eos_switch_app);
    MS_LOG(INFO) << "EoS loop insert Stream Switch " << eos_switch_app->fullname_with_scope();

    // eos loop eos start recv
    CNodePtr eos_start_recv = CreateRecvApplyKernel(kernel_graph_ptr, eos_start_event_id);
    AnfAlgo::SetStreamId(eos_stream_id, eos_start_recv.get());
    exec_order.push_back(eos_start_recv);
    MS_LOG(INFO) << "EoS loop insert EoS Recv " << eos_start_recv->fullname_with_scope();

    // EndOfSequence op
    CNodePtr end_of_sequence_op = CreateEndOfSequenceOP(kernel_graph_ptr, getnext_cnode);
    MS_EXCEPTION_IF_NULL(end_of_sequence_op);
    AnfAlgo::SetStreamId(eos_stream_id, end_of_sequence_op.get());
    exec_order.push_back(end_of_sequence_op);
    MS_LOG(INFO) << "EoS loop insert Eos Op " << end_of_sequence_op->fullname_with_scope();

    // eos loop eos done send
    eos_done_event_id = resource_manager.ApplyNewEvent();
    CNodePtr eos_done_send = CreateSendApplyKernel(kernel_graph_ptr, eos_done_event_id);
    AnfAlgo::SetStreamId(eos_stream_id, eos_done_send.get());
    exec_order.push_back(eos_done_send);
    MS_LOG(INFO) << "EoS loop insert EoS done Send " << eos_done_send->fullname_with_scope();

    // eos loop stream active
    fpbp_active_streams.push_back(eos_switch_stream_id);
  }

  bool exist_independent = ExistIndependent(kernel_graph_ptr);
  if (exist_independent) {
    // Independet parallel
    CNodePtr independent_switch_app =
      CreateStreamSwitchOp(kernel_graph_ptr, switch_loop_input, kIndependentStreamSwitch);
    MS_EXCEPTION_IF_NULL(independent_switch_app);
    uint32_t independent_switch_stream_id = resource_manager.ApplyNewStream();
    AnfAlgo::SetStreamId(independent_switch_stream_id, independent_switch_app.get());
    AnfAlgo::SetNodeAttr(kStreamNeedActivedFirst, MakeValue<bool>(true), independent_switch_app);
    AnfAlgo::SetNodeAttr(kAttrStreamSwitchKind, MakeValue<uint32_t>(kIndependentStreamSwitch), independent_switch_app);
    exec_order.push_back(independent_switch_app);
    MS_LOG(INFO) << "Independent op loop insert Stream Switch " << independent_switch_app->fullname_with_scope();
  }

  // fpbp loop process
  // fpbp loop stream switch
  uint32_t fpbp_switch_stream_id = resource_manager.ApplyNewStream();
  uint32_t fpbp_stream_id = resource_manager.ApplyNewStream();
  CNodePtr fpbp_switch_app = CreateStreamSwitchOp(kernel_graph_ptr, switch_loop_input, kFpBpStreamSwitch);
  MS_EXCEPTION_IF_NULL(fpbp_switch_app);
  AnfAlgo::SetStreamId(fpbp_switch_stream_id, fpbp_switch_app.get());
  AnfAlgo::SetNodeAttr(kStreamNeedActivedFirst, MakeValue<bool>(true), fpbp_switch_app);
  // update fpbp loop stream switch true_branch_stream attr
  AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, MakeValue<uint32_t>(fpbp_stream_id), fpbp_switch_app);
  AnfAlgo::SetNodeAttr(kAttrStreamSwitchKind, MakeValue<uint32_t>(kFpBpStreamSwitch), fpbp_switch_app);
  exec_order.push_back(fpbp_switch_app);
  MS_LOG(INFO) << "FpBp loop insert Stream Switch " << fpbp_switch_app->fullname_with_scope();

  if (exist_getnext) {
    // fpbp loop fpbp start recv
    CNodePtr fpbp_start_recv = CreateRecvApplyKernel(kernel_graph_ptr, fpbp_start_event_id);
    AnfAlgo::SetStreamId(fpbp_stream_id, fpbp_start_recv.get());
    exec_order.push_back(fpbp_start_recv);
    MS_LOG(INFO) << "FpBp loop insert FpBp start Recv " << fpbp_start_recv->fullname_with_scope();
  }

  // next loop AssignAdd
  CNodePtr assign_add_one = CreateStreamAssignAddnOP(kernel_graph_ptr, switch_loop_input, false);
  MS_EXCEPTION_IF_NULL(assign_add_one);
  AnfAlgo::SetStreamId(fpbp_stream_id, assign_add_one.get());
  exec_order.push_back(assign_add_one);
  MS_LOG(INFO) << "FpBp loop insert next loop AssignAdd " << assign_add_one->fullname_with_scope();

  // fpbp getnext output memcpy
  std::vector<CNodePtr> memcpy_list;
  std::vector<CNodePtr> other_list;
  if (exist_getnext) {
    CNodePtr cur_cnode = nullptr;
    for (size_t idx = i + 1; idx < orders.size(); idx++) {
      cur_cnode = orders[idx];
      if (AnfAlgo::HasNodeAttr(kAttrLabelForInsertStreamActive, cur_cnode)) {
        auto pre_node = orders[idx - 1];
        auto pre_kernel_name = AnfAlgo::GetCNodeName(pre_node);
        if (pre_kernel_name == kAtomicAddrCleanOpName) {
          other_list.pop_back();
          memcpy_list.push_back(pre_node);
        }
        memcpy_list.emplace_back(cur_cnode);
      } else {
        other_list.emplace_back(cur_cnode);
      }
    }
    (void)std::copy(memcpy_list.begin(), memcpy_list.end(), std::back_inserter(exec_order));
  } else {
    other_list = orders;
  }

  // fpbp loop eos done recv
  if (eos_mode) {
    CNodePtr eos_done_recv = CreateRecvApplyKernel(kernel_graph_ptr, eos_done_event_id);
    AnfAlgo::SetStreamId(fpbp_stream_id, eos_done_recv.get());
    exec_order.push_back(eos_done_recv);
    MS_LOG(INFO) << "FpBp loop insert EoS done Recv " << eos_done_recv->fullname_with_scope();
  }

  // stream active to activate getnext loop
  if (exist_getnext) {
    CNodePtr getnext_active_app = CreateStreamActiveOp(kernel_graph_ptr);
    MS_EXCEPTION_IF_NULL(getnext_active_app);
    getnext_active_streams.push_back(getnext_switch_stream_id);
    AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(getnext_active_streams),
                         getnext_active_app);
    exec_order.push_back(getnext_active_app);
    MS_LOG(INFO) << "FpBp loop insert GetNext loop Stream Active " << getnext_active_app->fullname_with_scope();
  }

  // fpbp loop other ops
  (void)std::copy(other_list.begin(), other_list.end(), std::back_inserter(exec_order));

  // current assign add op
  CNodePtr cur_assign_add = CreateStreamAssignAddnOP(kernel_graph_ptr, switch_loop_input, true);
  MS_EXCEPTION_IF_NULL(cur_assign_add);
  AnfAlgo::SetNodeAttr(kAttrFpBpEnd, MakeValue<bool>(true), cur_assign_add);
  exec_order.push_back(cur_assign_add);
  MS_LOG(INFO) << "FpBp loop insert current loop AssignAdd " << cur_assign_add->fullname_with_scope();

  // stream active to activate fpbp loop and eos loop
  CNodePtr fpbp_active_app = CreateStreamActiveOp(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(fpbp_active_app);
  fpbp_active_streams.push_back(fpbp_switch_stream_id);
  AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(fpbp_active_streams), fpbp_active_app);
  exec_order.push_back(fpbp_active_app);
  MS_LOG(INFO) << "FpBp loop insert FpBp loop and Eos loop Stream Active " << fpbp_active_app->fullname_with_scope();

  kernel_graph_ptr->set_execution_order(exec_order);
}

void KernelAdjust::CreateSwitchOpParameters(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                            std::map<std::string, mindspore::ParameterPtr> *switch_loop_input) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(switch_loop_input);
  ShapeVector shp = {1};
  tensor::TensorPtr tensor_ptr = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  mindspore::abstract::AbstractBasePtr paremeter_abstract_ptr = tensor_ptr->ToAbstract();
  if (paremeter_abstract_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "create abstract before insert switch op failed!";
  }

  ParameterPtr cur_loop_count = std::make_shared<Parameter>(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(cur_loop_count);
  cur_loop_count->set_name(kCurLoopCountParamName);
  cur_loop_count->set_abstract(paremeter_abstract_ptr);
  ParameterPtr loop_count_cur = kernel_graph_ptr->NewParameter(cur_loop_count);
  (*switch_loop_input)[kCurLoopCountParamName] = loop_count_cur;

  ParameterPtr next_loop_count = std::make_shared<Parameter>(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(next_loop_count);
  next_loop_count->set_name(kNextLoopCountParamName);
  next_loop_count->set_abstract(paremeter_abstract_ptr);
  ParameterPtr loop_count_next = kernel_graph_ptr->NewParameter(next_loop_count);
  (*switch_loop_input)[kNextLoopCountParamName] = loop_count_next;

  ParameterPtr iter_loop = std::make_shared<Parameter>(kernel_graph_ptr);
  iter_loop->set_name(kIterLoopParamName);
  iter_loop->set_abstract(paremeter_abstract_ptr);
  ParameterPtr iter_loop_new = kernel_graph_ptr->NewParameter(iter_loop);
  (*switch_loop_input)[kIterLoopParamName] = iter_loop_new;

  ParameterPtr one = std::make_shared<Parameter>(kernel_graph_ptr);
  one->set_name(kOneParamName);
  one->set_abstract(paremeter_abstract_ptr);
  ParameterPtr one_new = kernel_graph_ptr->NewParameter(one);
  (*switch_loop_input)[kOneParamName] = one_new;

  ParameterPtr epoch = std::make_shared<Parameter>(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(epoch);
  epoch->set_name(kEpochParamName);
  epoch->set_abstract(paremeter_abstract_ptr);
  ParameterPtr epoch_new = kernel_graph_ptr->NewParameter(epoch);
  (*switch_loop_input)[kEpochParamName] = epoch_new;
}

kernel::KernelBuildInfo::KernelBuildInfoBuilder KernelAdjust::CreateMngKernelBuilder(
  const std::vector<std::string> &formats, const std::vector<TypeId> &type_ids) {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder;
  selected_kernel_builder.SetInputsFormat(formats);
  selected_kernel_builder.SetInputsDeviceType(type_ids);

  selected_kernel_builder.SetFusionType(kernel::FusionType::OPAQUE);
  selected_kernel_builder.SetProcessor(kernel::Processor::AICORE);
  selected_kernel_builder.SetKernelType(KernelType::RT_KERNEL);
  return selected_kernel_builder;
}

CNodePtr KernelAdjust::CreateStreamSwitchOp(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                            const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input,
                                            StreamSwitchKind kind) {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder = CreateMngKernelBuilder(
    {kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt32});
  auto typeNone_abstract = std::make_shared<abstract::AbstractNone>();
  auto stream_switch = std::make_shared<Primitive>(kStreamSwitchOpName);
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(stream_switch));
  if (kind == kFpBpStreamSwitch || kind == kEosStreamSwitch) {
    inputs.push_back(switch_loop_input.at(kNextLoopCountParamName));
  } else if (kind == kGetNextStreamSwitch || kind == kIndependentStreamSwitch) {
    inputs.push_back(switch_loop_input.at(kNextLoopCountParamName));
  } else {
    MS_LOG(ERROR) << "unknown stream switch kind";
  }

  inputs.push_back(switch_loop_input.at(kIterLoopParamName));
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  CNodePtr stream_switch_app = kernel_graph_ptr->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(stream_switch_app);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), stream_switch_app.get());
  stream_switch_app->set_abstract(typeNone_abstract);
  // set attr: cond_ RT_LESS
  int condition = static_cast<int>(RT_LESS);
  ValuePtr cond = MakeValue(condition);
  AnfAlgo::SetNodeAttr(kAttrSwitchCondition, cond, stream_switch_app);
  // set attr:data_type
  int data_type = static_cast<int>(RT_SWITCH_INT64);
  ValuePtr dt = MakeValue(data_type);
  AnfAlgo::SetNodeAttr(kAttrDataType, dt, stream_switch_app);
  // set distinction label and graph id
  return stream_switch_app;
}

CNodePtr KernelAdjust::CreateStreamActiveOp(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
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
                                             const CNodePtr &node, size_t output_idx) {
  auto idx = NewValueNode(SizeToLong(output_idx));
  MS_EXCEPTION_IF_NULL(idx);
  auto imm = std::make_shared<Int64Imm>(SizeToInt(output_idx));
  auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
  idx->set_abstract(abstract_scalar);
  CNodePtr tuple_getitem = kernel_graph_ptr->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  tuple_getitem->set_scope(node->scope());
  std::vector<size_t> origin_shape = AnfAlgo::GetOutputInferShape(node, output_idx);
  TypeId origin_type = AnfAlgo::GetOutputInferDataType(node, output_idx);
  AnfAlgo::SetOutputInferTypeAndShape({origin_type}, {origin_shape}, tuple_getitem.get());
  return tuple_getitem;
}

CNodePtr KernelAdjust::CreateEndOfSequenceOP(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                             const CNodePtr &getnext_cnode) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder;
  selected_kernel_builder.SetInputsFormat({kOpFormat_DEFAULT});
  selected_kernel_builder.SetInputsDeviceType({kNumberTypeUInt8});

  selected_kernel_builder.SetFusionType(kernel::FusionType::OPAQUE);
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
  AnfAlgo::SetNodeAttr("input_names", input_names_v, end_of_sequence_node);
  std::vector<std::string> output_names = {"y"};
  ValuePtr output_names_v = MakeValue(output_names);
  AnfAlgo::SetNodeAttr("output_names", output_names_v, end_of_sequence_node);
  end_of_sequence_node->set_abstract(tuple_get_item->abstract());
  return end_of_sequence_node;
}

CNodePtr KernelAdjust::CreateStreamAssignAddnOP(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                                const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input,
                                                bool cur_loop) {
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
    inputs.push_back(switch_loop_input.at(kCurLoopCountParamName));
  } else {
    inputs.push_back(switch_loop_input.at(kNextLoopCountParamName));
  }

  inputs.push_back(switch_loop_input.at(kOneParamName));
  CNodePtr assign_add_one = kernel_graph_ptr->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(assign_add_one);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), assign_add_one.get());
  std::vector<std::string> input_names = {"ref", "value"};
  std::vector<std::string> output_names = {"output"};
  ValuePtr input_names_v = MakeValue(input_names);
  ValuePtr output_names_v = MakeValue(output_names);
  AnfAlgo::SetNodeAttr("input_names", input_names_v, assign_add_one);
  AnfAlgo::SetNodeAttr("output_names", output_names_v, assign_add_one);
  selected_kernel_builder.SetKernelType(KernelType::TBE_KERNEL);
  MS_EXCEPTION_IF_NULL(switch_loop_input.at(kCurLoopCountParamName));
  assign_add_one->set_abstract(switch_loop_input.at(kCurLoopCountParamName)->abstract());
  // add AssignAdd op to kernel ref node map
  session::AnfWithOutIndex final_pair = std::make_pair(assign_add_one, 0);
  session::KernelWithIndex kernel_with_index = AnfAlgo::VisitKernel(AnfAlgo::GetInputNode(assign_add_one, 0), 0);
  kernel_graph_ptr->AddRefCorrespondPairs(final_pair, kernel_with_index);
  return assign_add_one;
}

bool KernelAdjust::StepLoadCtrlInputs(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
  if (!NeedInsertSwitch()) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  if (kernel_graph_ptr->is_dynamic_shape()) {
    MS_LOG(INFO) << "Skip StepLoadCtrlInputs";
    return true;
  }
  auto input_nodes = kernel_graph_ptr->inputs();
  std::vector<tensor::TensorPtr> inputs;
  LoadSwitchInputs(&inputs);
  std::shared_ptr<std::vector<tensor::TensorPtr>> inputsPtr = std::make_shared<std::vector<tensor::TensorPtr>>(inputs);
  kernel_graph_ptr->set_input_ctrl_tensors(inputsPtr);
  size_t input_ctrl_size = inputs.size();
  // inputs_node:include four ctrl nodes in the back. such as:conv,loop_cnt, ites_loop, zero, one.
  // deal four ctrl nodes.
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto tensor = inputs[i];
    size_t deal_index = input_nodes.size() - input_ctrl_size + i;
    if (deal_index >= input_nodes.size()) {
      MS_LOG(EXCEPTION) << "deal_index[" << deal_index << "] out of range";
    }
    auto input_node = input_nodes[deal_index];
    bool need_sync = false;
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_node->isa<Parameter>()) {
      auto pk_node = input_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      MS_EXCEPTION_IF_NULL(pk_node);
      if (tensor->NeedSyncHostToDevice() || !pk_node->has_default()) {
        need_sync = true;
      }
    }
    if (need_sync) {
      auto pk_node = input_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(pk_node);
      auto device_address = AnfAlgo::GetMutableOutputAddr(pk_node, 0);
      MS_EXCEPTION_IF_NULL(device_address);
      tensor->set_device_address(device_address);
      if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(pk_node, 0),
                                            LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                            tensor->data_c())) {
        MS_LOG(INFO) << "SyncHostToDevice failed.";
        return false;
      }
    }
    tensor->set_sync_status(kNoNeedSync);
  }
  return true;
}

void KernelAdjust::LoadSwitchInputs(std::vector<tensor::TensorPtr> *inputs) {
  MS_LOG(INFO) << "---------------- LoadSwitchInputs---";
  MS_EXCEPTION_IF_NULL(inputs);
  // current loop count
  ShapeVector shp = {1};
  tensor::TensorPtr cur_loop_count = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(cur_loop_count);
  int32_t *val = nullptr;
  val = static_cast<int32_t *>(cur_loop_count->data_c());
  MS_EXCEPTION_IF_NULL(val);
  *val = 0;
  inputs->push_back(cur_loop_count);

  // next loop count
  tensor::TensorPtr next_loop_count = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(next_loop_count);
  val = static_cast<int32_t *>(next_loop_count->data_c());
  MS_EXCEPTION_IF_NULL(val);
  *val = 0;
  inputs->push_back(next_loop_count);

  // Epoch in device
  tensor::TensorPtr epoch_tensor = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(epoch_tensor);
  val = static_cast<int32_t *>(epoch_tensor->data_c());
  MS_EXCEPTION_IF_NULL(val);
  *val = 0;
  inputs->push_back(epoch_tensor);

  // total loop count per iter
  tensor::TensorPtr iter_loop_tensor = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(iter_loop_tensor);
  val = static_cast<int32_t *>(iter_loop_tensor->data_c());
  MS_EXCEPTION_IF_NULL(val);
  *val = SizeToInt(LongToSize(ConfigManager::GetInstance().iter_num()));
  MS_LOG(INFO) << "iter_loop_tensor = " << *val;
  inputs->push_back(iter_loop_tensor);

  tensor::TensorPtr one_tensor = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(one_tensor);
  val = static_cast<int32_t *>(one_tensor->data_c());
  MS_EXCEPTION_IF_NULL(val);
  *val = 1;
  inputs->push_back(one_tensor);

  MS_LOG(INFO) << "---------------- LoadSwitchInputs End--";
}

void KernelAdjust::Profiling(NotNull<session::KernelGraph *> kernel_graph_ptr) {
  if (!ascend::ProfilingManager::GetInstance().IsProfiling()) {
    MS_LOG(INFO) << "No need to profiling";
    return;
  }
  auto graph_id_env = std::getenv(kProfilingGraphId);
  if (graph_id_env != nullptr) {
    auto graph_id = std::stoul(graph_id_env);
    if (graph_id != kernel_graph_ptr->graph_id()) {
      MS_LOG(WARNING) << "Get PROFILING_GRAPH_ID " << graph_id
                      << " Not Match Current Graph Id:" << kernel_graph_ptr->graph_id();
      return;
    }
  }
  ProfilingTraceInfo profiling_trace_info = ProfilingUtils::GetProfilingTraceFromEnv(kernel_graph_ptr);
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
    MS_LOG(ERROR) << "No CNode in graph";
    return;
  }
  for (const auto &cnode_ptr : cnode_ptr_list) {
    ProfilingUtils::ProfilingTraceFpStart(cnode_ptr, profiling_trace_info, kernel_graph_ptr, NOT_NULL(&new_cnode_list));
    new_cnode_list.emplace_back(cnode_ptr);
    ProfilingUtils::ProfilingCustomOp(cnode_ptr, profiling_trace_info, kernel_graph_ptr, NOT_NULL(&new_cnode_list));
    ProfilingUtils::ProfilingTraceBpEnd(cnode_ptr, profiling_trace_info, kernel_graph_ptr, NOT_NULL(&new_cnode_list));
    ProfilingUtils::ProfilingTraceEnd(cnode_ptr, profiling_trace_info, kernel_graph_ptr, NOT_NULL(&new_cnode_list));
  }
  kernel_graph_ptr->set_execution_order(new_cnode_list);
}
}  // namespace device
}  // namespace mindspore
