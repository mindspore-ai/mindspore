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

#include "device/kernel_adjust.h"

#include <map>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "session/anf_runtime_algorithm.h"
#include "utils/context/ms_context.h"
#include "utils/config_manager.h"
#include "common/utils.h"
#include "kernel/kernel_build_info.h"
#include "utils/utils.h"
#include "device/ascend/profiling/profiling_manager.h"
#include "device/ascend/kernel_select_ascend.h"
#include "device/kernel_info.h"

constexpr auto kLoopCountParamName = "loop_count";
constexpr auto kIterLoopParamName = "iter_loop";
constexpr auto kZeroParamName = "zero";
constexpr auto kOneParamName = "one";
constexpr auto kStreamSwitch = "StreamSwitch";
constexpr auto kStreamActive = "StreamActive";
constexpr auto kAssignAdd = "AssignAdd";
namespace mindspore {
namespace device {
using device::ascend::ProfilingUtils;
void KernelAdjust::Reorder(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  const std::vector<CNodePtr> &origin_cnode_list = kernel_graph_ptr->execution_order();
  std::vector<CNodePtr> momentum_list;
  std::vector<CNodePtr> other_list;
  for (const auto &cnode : origin_cnode_list) {
    if (kOptOpeatorSet.find(AnfAlgo::GetCNodeName(cnode)) != kOptOpeatorSet.end()) {
      momentum_list.emplace_back(cnode);
    } else {
      other_list.emplace_back(cnode);
    }
  }
  std::vector<CNodePtr> new_order_list;
  new_order_list.insert(new_order_list.end(), other_list.begin(), other_list.end());
  new_order_list.insert(new_order_list.end(), momentum_list.begin(), momentum_list.end());
  kernel_graph_ptr->set_execution_order(new_order_list);
}

bool KernelAdjust::NeedInsertSwitch() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  return (context_ptr->enable_task_sink() && context_ptr->loop_sink_flag() &&
          ConfigManager::GetInstance().iter_num() > 1);
}

void KernelAdjust::InsertSwitchLoop(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
  if (!NeedInsertSwitch()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  std::map<std::string, mindspore::ParameterPtr> switch_loop_input;
  CreateSwitchOpParameters(kernel_graph_ptr, &switch_loop_input);

  std::vector<AnfNodePtr> *mute_inputs = kernel_graph_ptr->MutableInputs();
  MS_EXCEPTION_IF_NULL(mute_inputs);
  mute_inputs->push_back(switch_loop_input[kLoopCountParamName]);
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
  std::vector<CNodePtr> exec_order;
  CNodePtr stream_switch_app = CreateStreamSwitchOp(kernel_graph_ptr, switch_loop_input);
  MS_EXCEPTION_IF_NULL(stream_switch_app);
  exec_order.push_back(stream_switch_app);

  CNodePtr stream_active_switch_app = CreateStreamActiveSwitchOp(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(stream_active_switch_app);

  CNodePtr assign_add_one = CreateStreamAssignAddnOP(kernel_graph_ptr, switch_loop_input);
  MS_EXCEPTION_IF_NULL(assign_add_one);
  exec_order.push_back(assign_add_one);

  auto original_exec_order = kernel_graph_ptr->execution_order();
  (void)std::copy(original_exec_order.begin(), original_exec_order.end(), std::back_inserter(exec_order));
  exec_order.push_back(stream_active_switch_app);
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
    MS_LOG(EXCEPTION) << "create abstract brfore insert switch op failed!";
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
  auto stream_switch = std::make_shared<Primitive>(kStreamSwitch);
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
  // set attr:true branch graph id ,which is same to stream distinction label
  if (kernel_graph_ptr->execution_order().empty()) {
    MS_LOG(EXCEPTION) << "empty execution order";
  }
  auto first_node = kernel_graph_ptr->execution_order()[0];
  auto first_stream = AnfAlgo::GetStreamDistinctionLabel(first_node.get());
  AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, MakeValue<uint32_t>(first_stream), stream_switch_app);
  // set attr:data_type
  int data_type = static_cast<int>(RT_SWITCH_INT64);
  ValuePtr dt = MakeValue(data_type);
  AnfAlgo::SetNodeAttr(kAttrDataType, dt, stream_switch_app);
  // set distinction label and graph id
  AnfAlgo::SetGraphId(kInvalidGraphId - 1, stream_switch_app.get());
  AnfAlgo::SetStreamDistinctionLabel(kInvalidDistincLabel - 1, stream_switch_app.get());
  return stream_switch_app;
}

CNodePtr KernelAdjust::CreateSteamActiveOp(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder = CreateMngKernelBuilder(
    {kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt32});
  abstract::AbstractBasePtr typeNone_abstract = std::make_shared<abstract::AbstractNone>();
  auto stream_active_others = std::make_shared<Primitive>(kStreamActive);
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(stream_active_others));
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  CNodePtr stream_active_others_app = kernel_graph_ptr->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(stream_active_others_app);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), stream_active_others_app.get());
  stream_active_others_app->set_abstract(typeNone_abstract);
  return stream_active_others_app;
}

CNodePtr KernelAdjust::CreateStreamActiveSwitchOp(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder = CreateMngKernelBuilder(
    {kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt32});
  abstract::AbstractBasePtr typeNone_abstract = std::make_shared<abstract::AbstractNone>();
  auto stream_active_switch = std::make_shared<Primitive>(kStreamActive);
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(stream_active_switch));
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  CNodePtr stream_active_switch_app = kernel_graph_ptr->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(stream_active_switch_app);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), stream_active_switch_app.get());
  stream_active_switch_app->set_abstract(typeNone_abstract);
  // set attr,which stream to active
  std::vector<uint32_t> active_index_value = {kInvalidDistincLabel - 1};
  auto value = MakeValue<std::vector<uint32_t>>(active_index_value);
  AnfAlgo::SetNodeAttr(kAttrActiveStreamList, value, stream_active_switch_app);
  // set the distinction label of stream active
  if (kernel_graph_ptr->execution_order().empty()) {
    MS_LOG(EXCEPTION) << "empty execution order";
  }
  auto first_node = kernel_graph_ptr->execution_order()[0];
  auto label = AnfAlgo::GetStreamDistinctionLabel(first_node.get());
  // find the first switch's distinction label
  for (auto node : kernel_graph_ptr->execution_order()) {
    if (AnfAlgo::GetCNodeName(node) == "StreamSwitch") {
      label = AnfAlgo::GetStreamDistinctionLabel(node.get());
      break;
    }
  }
  AnfAlgo::SetStreamDistinctionLabel(label, stream_active_switch_app.get());
  return stream_active_switch_app;
}

CNodePtr KernelAdjust::CreateStreamActiveOtherOp(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder = CreateMngKernelBuilder(
    {kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt32});
  abstract::AbstractBasePtr typeNone_abstract = std::make_shared<abstract::AbstractNone>();
  auto stream_active_others = std::make_shared<Primitive>(kStreamActive);
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(stream_active_others));
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  CNodePtr stream_active_others_app = kernel_graph_ptr->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(stream_active_others_app);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), stream_active_others_app.get());
  stream_active_others_app->set_abstract(typeNone_abstract);
  // set attr
  ValuePtr active_target = MakeValue(kValueTargetOther);
  AnfAlgo::SetNodeAttr(kAttrActiveTarget, active_target, stream_active_others_app);
  return stream_active_others_app;
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
  auto assign_add = std::make_shared<Primitive>(kAssignAdd);
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
  // set the distinction label of assign add
  if (kernel_graph_ptr->execution_order().empty()) {
    MS_LOG(EXCEPTION) << "empty execution order";
  }
  auto first_node = kernel_graph_ptr->execution_order()[0];
  auto label = AnfAlgo::GetStreamDistinctionLabel(first_node.get());
  AnfAlgo::SetStreamDistinctionLabel(label, assign_add_one.get());
  return assign_add_one;
}

void KernelAdjust::SetStreamActiveOPs(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                      const std::unordered_set<uint32_t> &ctrl_stream_list,
                                      const std::unordered_set<uint32_t> &comm_stream_list,
                                      const std::unordered_set<uint32_t> &momentum_stream_list) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  for (const auto &cnode_ptr : kernel_graph_ptr->execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode_ptr);
    if (AnfAlgo::GetCNodeName(cnode_ptr) == kStreamActive) {
      auto primitive = AnfAlgo::GetCNodePrimitive(cnode_ptr);
      ValuePtr active_target = primitive->GetAttr(kAttrActiveTarget);
      std::vector<uint32_t> index_list;
      index_list.clear();
      if (GetValue<string>(active_target) == kValueTargetSwitch) {
        index_list.insert(index_list.end(), ctrl_stream_list.begin(), ctrl_stream_list.end());
      } else if (GetValue<string>(active_target) == kValueTargetOther) {
        for (uint32_t index : comm_stream_list) {
          if (AnfAlgo::GetStreamId(cnode_ptr) == index) {
            continue;
          }
          index_list.emplace_back(index);
        }
        index_list.insert(index_list.end(), momentum_stream_list.begin(), momentum_stream_list.end());
      }
      ValuePtr index_list_value = MakeValue(index_list);
      AnfAlgo::SetNodeAttr(kAttrActiveStreamList, index_list_value, cnode_ptr);
    }
  }
}

void KernelAdjust::SetStreamSwitchOps(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  CNodePtr switch_cnode_ptr = nullptr;
  uint32_t target_stream_id = 0;
  for (const auto &cnode_ptr : kernel_graph_ptr->execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode_ptr);
    if (AnfAlgo::GetCNodeName(cnode_ptr) == kStreamSwitch) {
      switch_cnode_ptr = cnode_ptr;
    }
    if (AnfAlgo::GetCNodeName(cnode_ptr) == kStreamActive) {
      auto primitive = AnfAlgo::GetCNodePrimitive(cnode_ptr);
      ValuePtr active_target = primitive->GetAttr(kAttrActiveTarget);
      if (GetValue<string>(active_target) == kValueTargetOther) {
        target_stream_id = AnfAlgo::GetStreamId(cnode_ptr);
      }
    }
  }
  if (switch_cnode_ptr != nullptr) {
    // set attr:true stream
    ValuePtr true_index = MakeValue(target_stream_id);
    AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, true_index, switch_cnode_ptr);
    MS_LOG(INFO) << "switch to true_index:" << target_stream_id;
  }
}

bool KernelAdjust::StepLoadCtrlInputs(const std::shared_ptr<session::Context> &context,
                                      const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
  if (!NeedInsertSwitch()) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  auto input_nodes = kernel_graph_ptr->inputs();
  std::vector<tensor::TensorPtr> inputs;
  LoadSwitchInputs(&inputs);
  std::shared_ptr<std::vector<tensor::TensorPtr>> inputsPtr = std::make_shared<std::vector<tensor::TensorPtr>>(inputs);
  context->SetResult(session::kInputCtrlTensors, inputsPtr);
  size_t input_ctrl_size = inputs.size();
  // inputs_node:include four ctrl nodes in the back. such as:conv,loop_cnt, ites_loop, zero, one.
  // deal four ctrl nodes.
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto tensor = inputs[i];
    size_t deal_index = input_nodes.size() - input_ctrl_size + i;
    if (deal_index >= input_nodes.size()) {
      MS_LOG(EXCEPTION) << "deak_index[" << deal_index << "] outof range";
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
      if (!device_address->SyncHostToDevice(tensor->shape(), LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                            tensor->data_c(false))) {
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
  val = static_cast<int32_t *>(loop_count_tensor->data_c(true));
  MS_EXCEPTION_IF_NULL(val);
  *val = 0;
  inputs->push_back(loop_count_tensor);

  tensor::TensorPtr iter_loop_tensor = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(iter_loop_tensor);
  val = static_cast<int32_t *>(iter_loop_tensor->data_c(true));
  MS_EXCEPTION_IF_NULL(val);
  *val = SizeToInt(LongToSize(ConfigManager::GetInstance().iter_num()));
  MS_LOG(INFO) << "iter_loop_tensor = " << *val;
  inputs->push_back(iter_loop_tensor);

  tensor::TensorPtr zero_tensor = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(zero_tensor);
  val = static_cast<int32_t *>(zero_tensor->data_c(true));
  MS_EXCEPTION_IF_NULL(val);
  *val = 0;
  inputs->push_back(zero_tensor);

  tensor::TensorPtr one_tensor = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(one_tensor);
  val = static_cast<int32_t *>(one_tensor->data_c(true));
  MS_EXCEPTION_IF_NULL(val);
  *val = 1;
  inputs->push_back(one_tensor);
  MS_LOG(INFO) << "---------------- LoadSwitchInputs End--";
}

void KernelAdjust::Profiling(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
  if (!ascend::ProfilingManager::GetInstance().IsProfiling()) {
    MS_LOG(INFO) << "no need to profiling";
    return;
  }
  ProfilingTraceInfo profiling_trace_info;
  if (ProfilingUtils::GetProfilingTraceInfo(kernel_graph_ptr, &profiling_trace_info)) {
    InsertProfilingKernel(kernel_graph_ptr, profiling_trace_info);
  } else {
    MS_LOG(WARNING) << "[profiling] GetProfilingTraceInfo failed";
  }
}

void KernelAdjust::InsertProfilingKernel(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                         const ProfilingTraceInfo &profiling_trace_info) {
  MS_LOG(INFO) << "[profiling] insert profiling kernel start";
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  if (!profiling_trace_info.IsValid()) {
    MS_LOG(WARNING) << "profiling trace point not found";
    return;
  }
  std::vector<CNodePtr> new_cnode_list;
  std::vector<CNodePtr> cnode_ptr_list = kernel_graph_ptr->execution_order();
  for (const auto &cnode_ptr : cnode_ptr_list) {
    ProfilingUtils::ProfilingTraceFpStart(kernel_graph_ptr, cnode_ptr, profiling_trace_info, &new_cnode_list);
    ProfilingUtils::ProfilingAllReduce(kernel_graph_ptr, cnode_ptr, ascend::kProfilingAllReduce1Start,
                                       profiling_trace_info.profiling_allreduce1_start, &new_cnode_list);
    ProfilingUtils::ProfilingAllReduce(kernel_graph_ptr, cnode_ptr, ascend::kProfilingAllReduce2Start,
                                       profiling_trace_info.profiling_allreduce2_start, &new_cnode_list);
    new_cnode_list.emplace_back(cnode_ptr);

    ProfilingUtils::ProfilingAllReduce(kernel_graph_ptr, cnode_ptr, ascend::kProfilingAllReduce1End,
                                       profiling_trace_info.profiling_allreduce1_end, &new_cnode_list);
    ProfilingUtils::ProfilingAllReduce(kernel_graph_ptr, cnode_ptr, ascend::kProfilingAllReduce2End,
                                       profiling_trace_info.profiling_allreduce2_end, &new_cnode_list);
    ProfilingUtils::ProfilingTraceEnd(kernel_graph_ptr, cnode_ptr, profiling_trace_info, &new_cnode_list);
  }
  kernel_graph_ptr->set_execution_order(new_cnode_list);
}
}  // namespace device
}  // namespace mindspore
