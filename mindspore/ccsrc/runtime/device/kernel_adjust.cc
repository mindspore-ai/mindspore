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
  return (context_ptr->enable_task_sink() && context_ptr->loop_sink_flag() &&
          ConfigManager::GetInstance().iter_num() > 1);
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

void KernelAdjust::InsertSwitchLoop(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
  device::ascend::AscendResourceMng &resource_manager = device::ascend::AscendResourceMng::GetInstance();
  resource_manager.ResetResource();
  if (!NeedInsertSwitch()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  bool eos_mode = ConfigManager::GetInstance().iter_num() == INT32_MAX;
  ReorderGetNext(kernel_graph_ptr);
  std::map<std::string, mindspore::ParameterPtr> switch_loop_input;
  CreateSwitchOpParameters(kernel_graph_ptr, &switch_loop_input);

  std::vector<AnfNodePtr> *mute_inputs = kernel_graph_ptr->MutableInputs();
  MS_EXCEPTION_IF_NULL(mute_inputs);
  mute_inputs->push_back(switch_loop_input[kLoopCountParamName]);
  mute_inputs->push_back(switch_loop_input[kEpochParamName]);
  mute_inputs->push_back(switch_loop_input[kIterLoopParamName]);
  mute_inputs->push_back(switch_loop_input[kZeroParamName]);
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
  uint32_t eos_done_event_id = UINT32_MAX;

  // getnext loop process
  // getnext loop stream switch op
  CNodePtr getnext_switch_app = CreateStreamSwitchOp(kernel_graph_ptr, switch_loop_input);
  MS_EXCEPTION_IF_NULL(getnext_switch_app);
  uint32_t getnext_switch_stream_id = resource_manager.ApplyNewStream();
  AnfAlgo::SetStreamId(getnext_switch_stream_id, getnext_switch_app.get());
  exec_order.push_back(getnext_switch_app);

  // getnext op
  uint32_t getnext_stream_id = resource_manager.ApplyNewStream();
  size_t i = 0;
  for (; i < orders.size(); i++) {
    auto node = orders[i];
    exec_order.push_back(node);
    AnfAlgo::SetStreamId(getnext_stream_id, exec_order[exec_order.size() - 1].get());
    if (AnfAlgo::GetCNodeName(node) == kGetNextOpName) {
      getnext_cnode = node;
      break;
    }
  }

  // update getnext loop stream switch true_branch_stream attr
  AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, MakeValue<uint32_t>(getnext_stream_id), getnext_switch_app);

  // getnext loop fpbp start send
  uint32_t fpbp_start_event_id = resource_manager.ApplyNewEvent();
  CNodePtr fpbp_start_send = CreateSendApplyKernel(kernel_graph_ptr, fpbp_start_event_id);
  AnfAlgo::SetStreamId(getnext_stream_id, fpbp_start_send.get());
  exec_order.push_back(fpbp_start_send);

  if (eos_mode) {
    // getnext loop eos start send
    uint32_t eos_start_event_id = resource_manager.ApplyNewEvent();
    CNodePtr eos_start_send = CreateSendApplyKernel(kernel_graph_ptr, eos_start_event_id);
    AnfAlgo::SetStreamId(getnext_stream_id, eos_start_send.get());
    exec_order.push_back(eos_start_send);

    // End Of Sequence loop process
    // eos loop stream switch
    CNodePtr eos_switch_app = CreateStreamSwitchOp(kernel_graph_ptr, switch_loop_input);
    MS_EXCEPTION_IF_NULL(eos_switch_app);
    uint32_t eos_switch_stream_id = resource_manager.ApplyNewStream();
    AnfAlgo::SetStreamId(eos_switch_stream_id, eos_switch_app.get());
    AnfAlgo::SetNodeAttr(kStreamNeedActivedFirst, MakeValue<bool>(true), eos_switch_app);
    exec_order.push_back(eos_switch_app);

    // eos loop eos start recv
    CNodePtr eos_start_recv = CreateRecvApplyKernel(kernel_graph_ptr, eos_start_event_id);
    uint32_t eos_stream_id = resource_manager.ApplyNewStream();
    AnfAlgo::SetStreamId(eos_stream_id, eos_start_recv.get());
    exec_order.push_back(eos_start_recv);

    // update eos loop stream switch true_branch_stream attr
    AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, MakeValue<uint32_t>(eos_stream_id), eos_switch_app);

    // EndOfSequence op
    CNodePtr end_of_sequence_op = CreateEndOfSequenceOP(kernel_graph_ptr, getnext_cnode);
    MS_EXCEPTION_IF_NULL(end_of_sequence_op);
    AnfAlgo::SetStreamId(eos_stream_id, end_of_sequence_op.get());
    exec_order.push_back(end_of_sequence_op);

    // eos loop eos done send
    eos_done_event_id = resource_manager.ApplyNewEvent();
    CNodePtr eos_done_send = CreateSendApplyKernel(kernel_graph_ptr, eos_done_event_id);
    AnfAlgo::SetStreamId(eos_stream_id, eos_done_send.get());
    exec_order.push_back(eos_done_send);

    // eos loop stream active
    fpbp_active_streams.push_back(eos_switch_stream_id);
  }

  // fpbp loop process
  // fpbp loop stream switch
  CNodePtr fpbp_switch_app = CreateStreamSwitchOp(kernel_graph_ptr, switch_loop_input);
  MS_EXCEPTION_IF_NULL(fpbp_switch_app);
  uint32_t fpbp_switch_stream_id = resource_manager.ApplyNewStream();
  AnfAlgo::SetStreamId(fpbp_switch_stream_id, fpbp_switch_app.get());
  AnfAlgo::SetNodeAttr(kStreamNeedActivedFirst, MakeValue<bool>(true), fpbp_switch_app);
  exec_order.push_back(fpbp_switch_app);

  // fpbp loop fpbp start recv
  CNodePtr fpbp_start_recv = CreateRecvApplyKernel(kernel_graph_ptr, fpbp_start_event_id);
  uint32_t fpbp_stream_id = resource_manager.ApplyNewStream();
  AnfAlgo::SetStreamId(fpbp_stream_id, fpbp_start_recv.get());
  exec_order.push_back(fpbp_start_recv);

  // update fpbp loop stream switch true_branch_stream attr
  AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, MakeValue<uint32_t>(fpbp_stream_id), fpbp_switch_app);

  // fpbp loop AssignAdd
  CNodePtr assign_add_one = CreateStreamAssignAddnOP(kernel_graph_ptr, switch_loop_input);
  MS_EXCEPTION_IF_NULL(assign_add_one);
  AnfAlgo::SetStreamId(fpbp_stream_id, assign_add_one.get());
  exec_order.push_back(assign_add_one);

  // fpbp memcpy
  std::vector<CNodePtr> memcpy_list;
  std::vector<CNodePtr> other_list;
  CNodePtr cur_cnode = nullptr;
  for (size_t idx = i + 1; idx < orders.size(); idx++) {
    cur_cnode = orders[idx];
    if (AnfAlgo::HasNodeAttr(kAttrLabelForInsertStreamActive, cur_cnode)) {
      memcpy_list.emplace_back(cur_cnode);
    } else {
      other_list.emplace_back(cur_cnode);
    }
  }

  (void)std::copy(memcpy_list.begin(), memcpy_list.end(), std::back_inserter(exec_order));

  // fpbp loop eos done recv
  if (eos_mode) {
    CNodePtr eos_done_recv = CreateRecvApplyKernel(kernel_graph_ptr, eos_done_event_id);
    AnfAlgo::SetStreamId(fpbp_stream_id, eos_done_recv.get());
    exec_order.push_back(eos_done_recv);
  }

  // stream active to activate getnext loop
  CNodePtr getnext_active_app = CreateStreamActiveOp(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(getnext_active_app);
  getnext_active_streams.push_back(getnext_switch_stream_id);
  AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(getnext_active_streams),
                       getnext_active_app);
  exec_order.push_back(getnext_active_app);

  // fpbp loop other ops
  (void)std::copy(other_list.begin(), other_list.end(), std::back_inserter(exec_order));

  // stream active to activate fpbp loop and eos loop
  CNodePtr fpbp_active_app = CreateStreamActiveOp(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(fpbp_active_app);
  fpbp_active_streams.push_back(fpbp_switch_stream_id);
  AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(fpbp_active_streams), fpbp_active_app);
  exec_order.push_back(fpbp_active_app);

  kernel_graph_ptr->set_execution_order(exec_order);
}

void KernelAdjust::CreateSwitchOpParameters(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                            std::map<std::string, mindspore::ParameterPtr> *switch_loop_input) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(switch_loop_input);
  std::vector<int> shp = {1};
  tensor::TensorPtr tensor_ptr = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  mindspore::abstract::AbstractBasePtr paremeter_abstract_ptr = tensor_ptr->ToAbstract();
  if (paremeter_abstract_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "create abstract before insert switch op failed!";
  }

  ParameterPtr loop_count = std::make_shared<Parameter>(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(loop_count);
  loop_count->set_name(kLoopCountParamName);
  loop_count->set_abstract(paremeter_abstract_ptr);
  ParameterPtr loop_count_new = kernel_graph_ptr->NewParameter(loop_count);

  (*switch_loop_input)[kLoopCountParamName] = loop_count_new;

  ParameterPtr iter_loop = std::make_shared<Parameter>(kernel_graph_ptr);
  iter_loop->set_name(kIterLoopParamName);
  iter_loop->set_abstract(paremeter_abstract_ptr);
  ParameterPtr iter_loop_new = kernel_graph_ptr->NewParameter(iter_loop);
  (*switch_loop_input)[kIterLoopParamName] = iter_loop_new;

  ParameterPtr zero = std::make_shared<Parameter>(kernel_graph_ptr);
  zero->set_name(kZeroParamName);
  zero->set_abstract(paremeter_abstract_ptr);
  ParameterPtr zero_new = kernel_graph_ptr->NewParameter(zero);
  (*switch_loop_input)[kZeroParamName] = zero_new;

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
                                            const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input) {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder = CreateMngKernelBuilder(
    {kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt32});
  auto typeNone_abstract = std::make_shared<abstract::AbstractNone>();
  auto stream_switch = std::make_shared<Primitive>(kStreamSwitchOpName);
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(stream_switch));
  inputs.push_back(switch_loop_input.at(kLoopCountParamName));
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
  auto idx = NewValueNode(SizeToInt(output_idx));
  MS_EXCEPTION_IF_NULL(idx);
  auto imm = std::make_shared<Int32Imm>(SizeToInt(output_idx));
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

CNodePtr KernelAdjust::CreateStreamAssignAddnOP(
  const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
  const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder = CreateMngKernelBuilder(
    {kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt32});
  selected_kernel_builder.SetOutputsFormat({kOpFormat_DEFAULT});
  selected_kernel_builder.SetOutputsDeviceType({kNumberTypeInt32});
  // AssignAdd
  auto assign_add = std::make_shared<Primitive>(kAssignAddOpName);
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(assign_add));
  inputs.push_back(switch_loop_input.at(kLoopCountParamName));
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
  MS_EXCEPTION_IF_NULL(switch_loop_input.at(kLoopCountParamName));
  assign_add_one->set_abstract(switch_loop_input.at(kLoopCountParamName)->abstract());
  return assign_add_one;
}

bool KernelAdjust::StepLoadCtrlInputs(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
  if (!NeedInsertSwitch()) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
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
      if (tensor->is_dirty() || !pk_node->has_default()) {
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
    tensor->set_dirty(false);
  }
  return true;
}

void KernelAdjust::LoadSwitchInputs(std::vector<tensor::TensorPtr> *inputs) {
  MS_LOG(INFO) << "---------------- LoadSwitchInputs---";
  MS_EXCEPTION_IF_NULL(inputs);
  std::vector<int> shp = {1};
  tensor::TensorPtr loop_count_tensor = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(loop_count_tensor);
  int32_t *val = nullptr;
  val = static_cast<int32_t *>(loop_count_tensor->data_c());
  MS_EXCEPTION_IF_NULL(val);
  *val = 0;
  inputs->push_back(loop_count_tensor);

  // Epoch in device
  tensor::TensorPtr epoch_tensor = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(epoch_tensor);
  val = static_cast<int32_t *>(epoch_tensor->data_c());
  MS_EXCEPTION_IF_NULL(val);
  *val = 0;
  inputs->push_back(epoch_tensor);

  tensor::TensorPtr iter_loop_tensor = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(iter_loop_tensor);
  val = static_cast<int32_t *>(iter_loop_tensor->data_c());
  MS_EXCEPTION_IF_NULL(val);
  *val = SizeToInt(LongToSize(ConfigManager::GetInstance().iter_num()));
  MS_LOG(INFO) << "iter_loop_tensor = " << *val;
  inputs->push_back(iter_loop_tensor);

  tensor::TensorPtr zero_tensor = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(zero_tensor);
  val = static_cast<int32_t *>(zero_tensor->data_c());
  MS_EXCEPTION_IF_NULL(val);
  *val = 0;
  inputs->push_back(zero_tensor);

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
    MS_LOG(WARNING) << "[profiling] no profiling node found!";
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
