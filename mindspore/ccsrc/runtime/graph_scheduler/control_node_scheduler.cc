/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/control_node_scheduler.h"
#include "runtime/graph_scheduler/control_node_parser.h"
#include "runtime/graph_scheduler/scheduler_helper.h"

namespace mindspore {
namespace runtime {
namespace {
std::string GetActorName(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto debug_name = node->DebugString();
  auto index = debug_name.find('{');
  if ((index != std::string::npos) && (index > 0)) {
    debug_name = debug_name.substr(0, index);
  }

  if (common::AnfAlgo::IsCallNode(node)) {
    return "Call_" + node->UniqueName() + "_" + debug_name;
  } else {
    return node->UniqueName() + "_" + debug_name;
  }
}

std::string GetStackActorNameByExitName(const std::string &exit_name) {
  size_t pos = exit_name.find(kExitActorNameSuffix);
  if (pos == std::string::npos) {
    MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid exit actor name:" << exit_name;
  }

  return exit_name.substr(0, pos) + kStackActorNameSuffix;
}

// Parameter and ref node can not copy the device tensor.
bool is_need_copy_device_tensor(const AnfNodePtr &backend_node, size_t index) {
  MS_EXCEPTION_IF_NULL(backend_node);
  // Skip the parameter and Load node.
  const auto &real_backend_node = common::AnfAlgo::VisitKernelWithReturnType(backend_node, index, false).first;
  if (real_backend_node != nullptr && (!real_backend_node->isa<CNode>())) {
    return false;
  }

  if (common::AnfAlgo::HasAbstractRef(backend_node)) {
    return false;
  }

  auto kernel_graph = AnfAlgo::FetchKernelGraph(backend_node.get());
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if ((!kernel_graph->is_graph_run_mode()) && kernel_graph->IsInRefOutputMap({backend_node, index})) {
    return false;
  }

  return true;
}

// Check whether the exit actor corresponding to the call node to the to actor already exists control arrow.
bool IsControlArrowExistForCallNode(const AnfNodePtr &node, const AbstractActor *const to_actor,
                                    const ControlNodeParserPtr &parser) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(parser);
  if (!common::AnfAlgo::IsCallNode(node)) {
    MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid call node:" << node->DebugString();
  }
  int branch_id = parser->FetchBranchIDByCallNode(node);

  const auto &func_graphs = parser->FetchFuncGraphbyCallNode(node);
  if (func_graphs.empty()) {
    MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Failed to get funcgraph by call node:" << node->DebugString();
  }
  MS_EXCEPTION_IF_NULL(*(func_graphs.begin()));
  auto actor_name = (*(func_graphs.begin()))->ToString() + kExitActorNameSuffix;
  const auto &actor = FetchActor(actor_name);
  MS_EXCEPTION_IF_NULL(actor);
  const auto &exit_actor = dynamic_cast<ExitActor *>(actor);
  MS_EXCEPTION_IF_NULL(exit_actor);

  const auto &branch_arrows = exit_actor->output_branch_control_arrows();
  const auto &arrow_iter = branch_arrows.find(branch_id);
  if (arrow_iter == branch_arrows.end()) {
    return false;
  }
  const auto &arrows = arrow_iter->second;
  return std::find(arrows.begin(), arrows.end(), to_actor->GetAID()) != arrows.end();
}
}  // namespace

ControlActorSetPtr ControlNodeScheduler::Build(const GraphCompilerInfo &graph_compiler_info,
                                               const AID &memory_manager_aid) {
  const auto &control_nodes = graph_compiler_info.control_nodes_;
  if (control_nodes.size() <= kSingleControlNode) {
    return nullptr;
  }

  memory_manager_aid_ = memory_manager_aid;
  ControlActorSetPtr control_actors = std::make_shared<ControlActorSet>();
  MS_EXCEPTION_IF_NULL(control_actors);
  control_actors->switch_actors_ = BuildSwitchActor(graph_compiler_info);
  control_actors->gather_actors_ = BuildGatherActor(graph_compiler_info);
  control_actors->entrance_actors_ = BuildEntranceActor(graph_compiler_info);
  control_actors->exit_actors_ = BuildExitActor(graph_compiler_info);
  control_actors->stack_actors_ = BuildStackActor(graph_compiler_info);
  return control_actors;
}

std::vector<SwitchActorPtr> ControlNodeScheduler::BuildSwitchActor(const GraphCompilerInfo &graph_compiler_info) const {
  std::vector<SwitchActorPtr> switch_actors;
  const auto &control_nodes = graph_compiler_info.control_nodes_;

  for (const auto &control_node : control_nodes) {
    // Switch node and switch layer node will be converted to switch actor.
    if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitch) ||
        common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitchLayer)) {
      const auto &actor_name = GetActorName(control_node);
      const auto &parameters = FetchInputNodeByCNode(control_node);
      const auto &switch_actor =
        std::make_shared<SwitchActor>(actor_name, memory_manager_aid_, parameters, control_node);
      (void)switch_actors.emplace_back(switch_actor);
      for (const auto &parameter : parameters) {
        MS_EXCEPTION_IF_NULL(parameter.first);
        MS_LOG(DEBUG) << "Print formal parameter for actor:" << actor_name
                      << " parameter:" << parameter.first->DebugString() << " index:" << parameter.second;
      }
      InsertActor(switch_actor.get());
    }
  }
  return switch_actors;
}

void ControlNodeScheduler::BuildDataSourceActorForControlNode(
  const GraphCompilerInfo &graph_compiler_info, const HostTensorQueuePtr &host_queue,
  const HostQueueDSActorPtr &host_queue_ds_actor, const AID &memory_manager_aid,
  std::vector<DataSourceActorPtr> *data_source_actors) const {
  HostQueueDSActorPtr control_node_ds_actor = host_queue_ds_actor;
  const auto parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  MS_EXCEPTION_IF_NULL(data_source_actors);

  // Initialize the parameter in the control node, first get all the front parameters in the control node, then find
  // the corresponding backend parameter from the map, and insert it into the host data source actor.
  const auto &control_node_parameters = parser->control_node_parameters();
  for (const auto &parameter_with_index : control_node_parameters) {
    MS_EXCEPTION_IF_NULL(parameter_with_index.first);
    if (IsPersistentDeviceTensor(parameter_with_index.first)) {
      continue;
    }
    if (control_node_ds_actor == nullptr) {
      auto actor_name = graph_compiler_info.name_ + kHostDSActorNameSuffix;
      MS_LOG(INFO) << "Create host queue data source actor: " << actor_name;
      control_node_ds_actor =
        std::make_shared<HostQueueDataSourceActor>(actor_name, 1, memory_manager_aid, nullptr, nullptr, host_queue);
      MS_EXCEPTION_IF_NULL(control_node_ds_actor);
      InsertActor(control_node_ds_actor.get());
      (void)data_source_actors->emplace_back(control_node_ds_actor);
    }

    auto &node_map = control_node_ds_actor->data_node_position_map_;
    if (node_map.find(parameter_with_index) != node_map.end()) {
      continue;
    }
    const auto &node_with_index_with_context =
      parser->FetchBackendParameterWithContextByFrontParameter(parameter_with_index);
    const auto &node_with_index = node_with_index_with_context.first;
    const auto &device_context = node_with_index_with_context.second;
    MS_EXCEPTION_IF_NULL(node_with_index.first);
    MS_EXCEPTION_IF_NULL(device_context);
    auto iter = find(control_node_ds_actor->data_node_with_indexs_.begin(),
                     control_node_ds_actor->data_node_with_indexs_.end(), node_with_index);
    if (iter != control_node_ds_actor->data_node_with_indexs_.end()) {
      (void)node_map.emplace(parameter_with_index, iter - control_node_ds_actor->data_node_with_indexs_.begin());
      MS_LOG(DEBUG) << "Insert front node:" << parameter_with_index.first->DebugString()
                    << " index:" << parameter_with_index.second << " to host queue data source actor.";
    } else {
      if (parameter_with_index.first->kernel_info() == nullptr) {
        // Create kernel info for control node parameters.
        const auto &backend_kernel_info = static_cast<device::KernelInfo *>(node_with_index.first->kernel_info());
        MS_EXCEPTION_IF_NULL(backend_kernel_info);
        const auto &backend_build_info = backend_kernel_info->GetMutableSelectKernelBuildInfo();
        MS_EXCEPTION_IF_NULL(backend_build_info);

        std::shared_ptr<KernelBuildInfoBuilder> builder = std::make_shared<KernelBuildInfoBuilder>();
        MS_EXCEPTION_IF_NULL(builder);
        builder->SetOutputsFormat(backend_build_info->GetAllOutputFormats());
        builder->SetOutputsDeviceType(backend_build_info->GetAllOutputDeviceTypes());

        auto kernel_info = std::make_shared<device::KernelInfo>();
        MS_EXCEPTION_IF_NULL(kernel_info);
        kernel_info->set_select_kernel_build_info(builder->Build());
        parameter_with_index.first->set_kernel_info(kernel_info);
      }

      // Create device tensor.
      const auto &device_address = AnfAlgo::GetMutableOutputAddr(node_with_index.first, node_with_index.second, false);
      MS_EXCEPTION_IF_NULL(device_address);
      auto new_address = device_context->device_res_manager_->CreateDeviceAddress(
        nullptr, device_address->GetSize(), device_address->format(), device_address->type_id(),
        device_address->host_shape());
      MS_EXCEPTION_IF_NULL(new_address);
      MS_LOG(DEBUG) << "Create new address for node that has no corresponding backend node:"
                    << parameter_with_index.first->DebugString() << " index:" << parameter_with_index.second
                    << " addr:" << new_address << " size:" << device_address->GetSize()
                    << ", type id:" << device_address->type_id();
      AnfAlgo::SetOutputAddr(new_address, parameter_with_index.second, parameter_with_index.first.get());

      (void)node_map.emplace(parameter_with_index, control_node_ds_actor->data_node_with_indexs_.size());
      (void)control_node_ds_actor->data_node_with_indexs_.emplace_back(parameter_with_index);
      (void)control_node_ds_actor->device_contexts_.emplace_back(device_context);
    }
  }
}

std::vector<GatherActorPtr> ControlNodeScheduler::BuildGatherActor(const GraphCompilerInfo &graph_compiler_info) const {
  std::vector<GatherActorPtr> gather_actors;
  const auto &control_nodes = graph_compiler_info.control_nodes_;
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    // Partial node and call node will be converted to gather actor.
    if ((common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimPartial) && (!IsInvalidPartial(control_node))) ||
        common::AnfAlgo::IsCallNode(control_node)) {
      const auto &actor_name = GetActorName(control_node);
      const auto &parameters = FetchInputNodeByCNode(control_node);
      const auto &gather_actor =
        std::make_shared<GatherActor>(actor_name, memory_manager_aid_, parameters, control_node);
      MS_EXCEPTION_IF_NULL(gather_actor);
      (void)gather_actors.emplace_back(gather_actor);
      for (const auto &parameter : parameters) {
        MS_EXCEPTION_IF_NULL(parameter.first);
        MS_LOG(DEBUG) << "Print formal parameter for actor:" << actor_name
                      << " parameter:" << parameter.first->DebugString() << " index:" << parameter.second;
      }
      InsertActor(gather_actor.get());

      // The gather actor corresponding to a call node needs to set the branch id.
      if (common::AnfAlgo::IsCallNode(control_node)) {
        gather_actor->output_branch_id_ = parser->FetchBranchIDByCallNode(control_node);
      }

      // Fetch device contexts for gather actor.
      const auto &iter = parser->control_node_to_device_contexts_.find(control_node);
      if (iter == parser->control_node_to_device_contexts_.end()) {
        MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Failed to get device contexts for node:"
                          << control_node->DebugString();
      }
      gather_actor->device_contexts_ = iter->second;
    }
  }
  return gather_actors;
}

std::vector<EntranceActorPtr> ControlNodeScheduler::BuildEntranceActor(
  const GraphCompilerInfo &graph_compiler_info) const {
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  const auto &call_node_to_func_graphs = parser->call_node_to_func_graphs_;
  std::unordered_map<FuncGraphPtr, std::set<KernelWithIndex>> func_graph_to_call_nodes;
  for (const auto &call_node_to_func_graph : call_node_to_func_graphs) {
    const auto &node = call_node_to_func_graph.first;
    for (const auto &func_graph : call_node_to_func_graph.second) {
      (void)func_graph_to_call_nodes[func_graph].emplace(node, 0);
    }
  }

  std::vector<EntranceActorPtr> entrance_actors;
  const auto &control_nodes = graph_compiler_info.control_nodes_;
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      const auto &func_graph = control_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
      std::vector<KernelWithIndex> formal_parameters;

      // The entrance actor has two parts of node members :
      // 1. The formal parameters of the subgraph are used to connect the actor's output arrows.
      for (const auto &parameter : func_graph->parameters()) {
        MS_EXCEPTION_IF_NULL(parameter);
        const auto &abstract = parameter->abstract();
        MS_EXCEPTION_IF_NULL(abstract);
        size_t output_num = common::AnfAlgo::GetOutputNumByAbstract(abstract);
        for (size_t i = 0; i < output_num; ++i) {
          (void)formal_parameters.emplace_back(parameter, i);
        }
      }

      // 2. The caller of the subgraph, namely call nodes, is used to connect the input arrows.
      std::set<KernelWithIndex> call_nodes;
      const auto &iter = func_graph_to_call_nodes.find(func_graph);
      if (iter != func_graph_to_call_nodes.end()) {
        call_nodes = iter->second;
      }
      for (const auto &formal_parameter : formal_parameters) {
        MS_EXCEPTION_IF_NULL(formal_parameter.first);
        MS_LOG(DEBUG) << "Print formal parameter for actor:" << actor_name
                      << " parameter:" << formal_parameter.first->DebugString() << " index:" << formal_parameter.second;
      }
      const auto &entrance_actor =
        std::make_shared<EntranceActor>(actor_name, memory_manager_aid_, formal_parameters, call_nodes, control_node);
      MS_EXCEPTION_IF_NULL(entrance_actor);
      auto context_iter = parser->func_graph_to_device_contexts_.find(func_graph);
      if (context_iter == parser->func_graph_to_device_contexts_.end() ||
          context_iter->second.size() < formal_parameters.size()) {
        MS_LOG(EXCEPTION)
          << "#dmsg#Runtime error info:#dmsg#Invalid device contexts for funcgraph:" << func_graph->ToString()
          << " parameter num:" << formal_parameters.size() << " device contexts num:"
          << (context_iter == parser->func_graph_to_device_contexts_.end() ? 0 : context_iter->second.size());
      }
      entrance_actor->device_contexts_.clear();
      (void)entrance_actor->device_contexts_.insert(
        entrance_actor->device_contexts_.begin(), context_iter->second.begin(),
        context_iter->second.begin() + SizeToLong(formal_parameters.size()));
      (void)entrance_actors.emplace_back(entrance_actor);
      InsertActor(entrance_actor.get());
    }
  }

  return entrance_actors;
}

std::vector<ExitActorPtr> ControlNodeScheduler::BuildExitActor(const GraphCompilerInfo &graph_compiler_info) const {
  std::vector<ExitActorPtr> exit_actors;
  const auto &control_nodes = graph_compiler_info.control_nodes_;
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  // The exit actor is used in 2 places:
  // 1.funcgraph output, that is the output of the return node.
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      const auto &func_graph = control_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &actor_name = func_graph->ToString() + kExitActorNameSuffix;
      const auto &parameters = FetchInputNodeByCNode(control_node);
      const auto &exit_actor = std::make_shared<ExitActor>(actor_name, memory_manager_aid_, parameters, control_node);
      MS_EXCEPTION_IF_NULL(exit_actor);
      for (const auto &parameter : parameters) {
        MS_EXCEPTION_IF_NULL(parameter.first);
        MS_LOG(DEBUG) << "Print formal parameter for actor:" << actor_name
                      << " parameter:" << parameter.first->DebugString() << " index:" << parameter.second;
      }
      auto context_iter = parser->control_node_to_device_contexts_.find(control_node);
      if (context_iter == parser->control_node_to_device_contexts_.end() ||
          context_iter->second.size() != parameters.size()) {
        MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Failed to get device contexts for funcgraph:"
                          << func_graph->ToString();
      }
      exit_actor->device_contexts_ = context_iter->second;
      (void)exit_actors.emplace_back(exit_actor);
      InsertActor(exit_actor.get());
    }
  }

  if (graph_compiler_info.graphs_.size() != graph_compiler_info.device_contexts_.size()) {
    MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid graphs num:" << graph_compiler_info.graphs_.size()
                      << " and contexts num:" << graph_compiler_info.device_contexts_.size();
  }

  // 2. Replace the device address in the kernel actor when calling funcgraph, that is to say in the data exchange
  // between kernel graph and the control node, in fact, it is the output of the kernel graph.
  for (const auto &kernel_graph_group_info : parser->kernel_graph_group_infos_) {
    MS_EXCEPTION_IF_NULL(kernel_graph_group_info);
    if (kernel_graph_group_info->graphs_.empty()) {
      continue;
    }

    std::vector<bool> is_need_copy_device_tensors;
    std::vector<KernelWithIndex> formal_parameters;
    std::vector<const DeviceContext *> device_contexts;

    for (const auto &node_with_context : kernel_graph_group_info->front_output_nodes_) {
      if (HasAbstractMonad(node_with_context.first.first) || IsCsrNode(node_with_context.first.first)) {
        continue;
      }
      // Collect inputs of exit actor.
      (void)formal_parameters.emplace_back(node_with_context.first);
      // Get the device contexts of the exit actor's cnode inputs.
      const AnfNodePtr &backend_node = node_with_context.second.first.first;
      MS_EXCEPTION_IF_NULL(backend_node);
      (void)is_need_copy_device_tensors.emplace_back(
        is_need_copy_device_tensor(backend_node, node_with_context.second.first.second));
      (void)device_contexts.emplace_back(node_with_context.second.second);
    }

    const auto &actor_name = kernel_graph_group_info->group_name_ + kExitActorNameSuffix;
    const auto &exit_actor = std::make_shared<ExitActor>(actor_name, memory_manager_aid_, formal_parameters, nullptr);
    MS_EXCEPTION_IF_NULL(exit_actor);
    exit_actor->is_need_copy_device_tensors_.swap(is_need_copy_device_tensors);
    exit_actor->device_contexts_.swap(device_contexts);
    (void)exit_actors.emplace_back(exit_actor);
    InsertActor(exit_actor.get());
  }

  return exit_actors;
}

std::vector<StackActorPtr> ControlNodeScheduler::BuildStackActor(const GraphCompilerInfo &graph_compiler_info) const {
  std::vector<StackActorPtr> stack_actors;
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  // Create a corresponding stack actor for each kernel graph that has a call node as input.
  for (const auto &kernel_graph_group_info : parser->kernel_graph_group_infos_) {
    MS_EXCEPTION_IF_NULL(kernel_graph_group_info);
    if (!kernel_graph_group_info->need_stack_) {
      continue;
    }
    const auto &actor_name = kernel_graph_group_info->group_name_ + kStackActorNameSuffix;
    size_t input_parameter_data_num = 0;
    std::vector<const DeviceContext *> device_contexts;
    std::vector<KernelWithIndex> formal_parameters;
    // Collect inputs of stack actor.
    for (const auto &node_with_context : kernel_graph_group_info->front_input_nodes_) {
      // If the input comes from inside funcgraph, put it at the front of the vector, otherwise put it at the end.
      const auto &from_node = node_with_context.first.first;
      MS_EXCEPTION_IF_NULL(from_node);
      auto iter = parser->node_to_level_.find(from_node);
      if (iter == parser->node_to_level_.end()) {
        MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Failed to get level by from node:"
                          << from_node->DebugString() << " in graph:" << kernel_graph_group_info->group_name_;
      }
      if (iter->second == kernel_graph_group_info->level_ && (!parser->IsRootGraphPersistentDeviceTensor(from_node))) {
        (void)formal_parameters.emplace_back(node_with_context.first);
        (void)device_contexts.emplace_back(node_with_context.second);
        MS_LOG(DEBUG) << "Add normal parameter for actor:" << actor_name << " node:" << from_node->DebugString()
                      << " index:" << node_with_context.first.second;
      } else {
        (void)formal_parameters.insert(formal_parameters.begin(), node_with_context.first);
        (void)device_contexts.insert(device_contexts.begin(), node_with_context.second);
        MS_LOG(DEBUG) << "Add stack parameter for actor:" << actor_name << " node:" << from_node->DebugString()
                      << " index:" << node_with_context.first.second;
        input_parameter_data_num++;
      }
    }
    const auto &stack_actor = std::make_shared<StackActor>(actor_name, memory_manager_aid_, formal_parameters);
    MS_EXCEPTION_IF_NULL(stack_actor);
    (void)stack_actors.emplace_back(stack_actor);
    stack_actor->device_contexts_.swap(device_contexts);
    stack_actor->input_stack_data_num_ = input_parameter_data_num;
    InsertActor(stack_actor.get());
  }
  // Create stack actors for control nodes.
  BuildStackActorForControlNode(graph_compiler_info, &stack_actors);

  return stack_actors;
}

void ControlNodeScheduler::BuildStackActorForControlNode(const GraphCompilerInfo &graph_compiler_info,
                                                         std::vector<StackActorPtr> *const stack_actors) const {
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  for (const auto &need_stack_control_node : parser->need_stack_control_nodes_) {
    MS_EXCEPTION_IF_NULL(need_stack_control_node);
    MS_LOG(DEBUG) << "Build stack actor for control node:" << need_stack_control_node->DebugString();
    const auto &stack_actor_name = GetActorName(need_stack_control_node) + kStackActorNameSuffix;
    std::vector<KernelWithIndex> formal_parameters;
    std::vector<const DeviceContext *> device_contexts;
    size_t input_parameter_data_num = 0;
    size_t input_parameter_partials_num = 0;

    // Fetch the control actor of control node.
    std::string control_actor_name = "";
    if (common::AnfAlgo::CheckPrimitiveType(need_stack_control_node, prim::kPrimReturn)) {
      const auto &func_graph = need_stack_control_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      control_actor_name = func_graph->ToString() + kExitActorNameSuffix;
    } else if (common::AnfAlgo::CheckPrimitiveType(need_stack_control_node, prim::kPrimPartial) ||
               common::AnfAlgo::CheckPrimitiveType(need_stack_control_node, prim::kPrimSwitch) ||
               common::AnfAlgo::CheckPrimitiveType(need_stack_control_node, prim::kPrimSwitchLayer) ||
               common::AnfAlgo::IsCallNode(need_stack_control_node)) {
      control_actor_name = GetActorName(need_stack_control_node);
    } else {
      MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid control node:"
                        << need_stack_control_node->DebugString();
    }

    auto iter = parser->node_to_level_.find(need_stack_control_node);
    if (iter == parser->node_to_level_.end()) {
      MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Failed to get level for need stack control node:"
                        << need_stack_control_node->DebugString();
    }
    size_t control_node_level = iter->second;

    auto actor = FetchActor(control_actor_name);
    if (actor == nullptr) {
      MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid actor name:" << control_actor_name;
    }
    auto control_actor = dynamic_cast<ControlActor *>(actor);
    MS_EXCEPTION_IF_NULL(control_actor);
    if (control_actor->formal_parameters_.size() > control_actor->device_contexts_.size()) {
      MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid device context size:"
                        << control_actor->device_contexts_.size()
                        << " and formal parameter size:" << control_actor->formal_parameters_.size()
                        << " for actor:" << control_actor->GetAID();
    }

    // Collect formal parameters and device contexts, skip the value nodes.
    for (size_t i = 0; i < control_actor->formal_parameters_.size(); ++i) {
      const auto &parameter = control_actor->formal_parameters_[i];
      auto device_context = control_actor->device_contexts_[i];
      MS_EXCEPTION_IF_NULL(parameter.first);
      if (parameter.first->isa<ValueNode>()) {
        continue;
      }

      iter = parser->node_to_level_.find(parameter.first);
      if (iter == parser->node_to_level_.end()) {
        MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Failed to get level for formal parameter:"
                          << parameter.first->DebugString()
                          << " for need stack control node:" << need_stack_control_node->DebugString();
      }

      if (control_node_level == iter->second && (!parser->IsRootGraphPersistentDeviceTensor(parameter.first))) {
        MS_LOG(DEBUG) << "Add normal parameter:" << parameter.first->DebugString()
                      << " for stack actor:" << stack_actor_name;
        (void)formal_parameters.emplace_back(parameter);
        (void)device_contexts.emplace_back(device_context);
      } else {
        MS_LOG(DEBUG) << "Add stack parameter:" << parameter.first->DebugString()
                      << " for stack actor:" << stack_actor_name;
        (void)formal_parameters.insert(formal_parameters.begin(), parameter);
        (void)device_contexts.insert(device_contexts.begin(), device_context);

        const auto &abstract = parameter.first->abstract();
        MS_EXCEPTION_IF_NULL(abstract);
        const auto &real_abstract = common::AnfAlgo::FetchAbstractByIndex(abstract, parameter.second);
        MS_EXCEPTION_IF_NULL(real_abstract);
        if (real_abstract->isa<abstract::AbstractFunction>()) {
          input_parameter_partials_num++;
        } else {
          input_parameter_data_num++;
        }
      }
    }
    // Create stack actor.
    const auto &stack_actor = std::make_shared<StackActor>(stack_actor_name, memory_manager_aid_, formal_parameters);
    MS_EXCEPTION_IF_NULL(stack_actor);
    stack_actor->device_contexts_ = device_contexts;
    stack_actor->input_stack_data_num_ = input_parameter_data_num;
    stack_actor->input_stack_partials_num_ = input_parameter_partials_num;

    InsertActor(stack_actor.get());
    (void)stack_actors->emplace_back(stack_actor);
  }
}

void ControlNodeScheduler::Link(ActorSet *const actor_set, const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(actor_set->control_actors_);
  MS_LOG(DEBUG) << "Control node scheduler link start.";
  // Link data arrows and partial arrows between control actors.
  LinkArrowForControlActor(actor_set->control_actors_.get(), graph_compiler_info);

  // Link arrows from host data source actor or data prepare actor to entrance actor of root graph.
  LinkArrowForRootGraphEntranceActor(graph_compiler_info);

  // Link output data arrows from control actors to output actor.
  LinkDataArrowForOutputActor(actor_set, graph_compiler_info);

  // Link data arrows from entrance actors to kernel actors.
  LinkDataArrowForKernelActor(graph_compiler_info);

  // Link branch id arrows between control actors.
  LinkBranchIDArrowForControlActor(actor_set->control_actors_.get());

  // Link all control arrows between control actors.
  LinkControlArrowForControlActor(actor_set, graph_compiler_info);

  // Link control arrows for no input and no output kernel actor.
  LinkControlArrowForKernelActor(actor_set, graph_compiler_info);

  LinkControlArrowForLoopCountActor(actor_set, graph_compiler_info);

  LinkDataArrowForCustomActor(actor_set, graph_compiler_info);

  LinkControlArrowForCustomActor(actor_set, graph_compiler_info);

  SetTimeSummaryForControlActor(graph_compiler_info);
  MS_LOG(DEBUG) << "Control node scheduler link end.";
}

namespace {
AnfNodePtr FetchInternalParameterInput(const AnfNodePtr &node, const ControlNodeParserPtr &parser,
                                       const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(WARNING) << "Node:" << node->DebugString() << " is not a cnode.";
    return nullptr;
  }
  const auto &kernel = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(kernel);
  for (size_t i = 0; i < common::AnfAlgo::GetInputNum(kernel); ++i) {
    auto input_node = common::AnfAlgo::GetInputNode(kernel, i);
    MS_EXCEPTION_IF_NULL(input_node);
    auto input_with_index = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, false);
    auto input = input_with_index.first;
    MS_EXCEPTION_IF_NULL(input);
    if (HasAbstractMonad(input) || (!parser->IsControlFlowDataArrow(graph, input))) {
      continue;
    }

    auto from_node_with_index = GetFrontNodeByKernelGraph(input, graph.get());
    MS_EXCEPTION_IF_NULL(from_node_with_index.first);
    const auto &from_node = from_node_with_index.first;
    if (from_node->isa<CNode>()) {
      return from_node;
    }
  }
  return nullptr;
}
}  // namespace

void ControlNodeScheduler::LinkControlArrowForCustomActor(ActorSet *const actor_set,
                                                          const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  for (auto &custom_actor : actor_set->custom_actors_) {
    MS_EXCEPTION_IF_NULL(custom_actor);
    const auto &kernel = custom_actor->kernel().lock();
    MS_EXCEPTION_IF_NULL(kernel);
    const auto &graph = kernel->func_graph();
    MS_EXCEPTION_IF_NULL(graph);
    if (custom_actor->output_data_arrows().empty() && custom_actor->output_control_arrows().empty()) {
      const auto &actor_name = graph->ToString() + kExitActorNameSuffix;
      auto actor = FetchActor(actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      SchedulerHelper::AddControlArrow(custom_actor.get(), actor);
    }
    if (custom_actor->input_control_arrow_aids().empty() && custom_actor->input_data_arrow_aids().empty()) {
      const auto &kernel_graph = std::dynamic_pointer_cast<KernelGraph>(graph);
      MS_EXCEPTION_IF_NULL(kernel_graph);
      auto base_node = AnfUtils::GetCustomActorBaseNode(kernel);
      AnfNodePtr internal_parameter = nullptr;
      if (base_node != nullptr) {
        internal_parameter = FetchInternalParameterInput(base_node, parser, kernel_graph);
      }
      AbstractActor *from_actor = nullptr;
      if (parser->IsCallInputKernelGraph(kernel_graph.get())) {
        auto kernel_graph_ptr = std::dynamic_pointer_cast<KernelGraph>(kernel->func_graph());
        const auto &actor_name = parser->FetchGroupNameByKernelGraph(kernel_graph_ptr) + kStackActorNameSuffix;
        from_actor = FetchActor(actor_name);
      } else if (internal_parameter != nullptr) {
        const auto &from_graph = parser->FetchKernelGraphByFrontNode(internal_parameter);
        MS_EXCEPTION_IF_NULL(from_graph);
        from_actor = FetchActor(parser->FetchGroupNameByKernelGraph(from_graph) + kExitActorNameSuffix);
      } else {
        const auto &func_graph = parser->FetchFuncGraphByKernelGraph(kernel_graph.get());
        MS_EXCEPTION_IF_NULL(func_graph);
        const auto &actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
        from_actor = FetchActor(actor_name);
      }
      MS_EXCEPTION_IF_NULL(from_actor);
      SchedulerHelper::AddControlArrow(from_actor, custom_actor.get());
    }
  }
}

void ControlNodeScheduler::ClearActorData(const ControlActorSet *control_actor_set) const {
  if (control_actor_set == nullptr) {
    return;
  }

  for (auto &switch_actor : control_actor_set->switch_actors_) {
    MS_EXCEPTION_IF_NULL(switch_actor);
    switch_actor->memory_free_lists_ = std::queue<std::vector<DeviceTensor *>>();
  }

  for (auto &gather_actor : control_actor_set->gather_actors_) {
    MS_EXCEPTION_IF_NULL(gather_actor);
    gather_actor->memory_free_lists_ = std::queue<std::vector<DeviceTensor *>>();
  }

  for (auto &entrance_actor : control_actor_set->entrance_actors_) {
    MS_EXCEPTION_IF_NULL(entrance_actor);
    entrance_actor->memory_free_lists_ = std::queue<std::vector<DeviceTensor *>>();
  }

  for (auto &stack_actor : control_actor_set->stack_actors_) {
    MS_EXCEPTION_IF_NULL(stack_actor);
    stack_actor->memory_free_lists_ = std::queue<std::vector<DeviceTensor *>>();
  }

  for (auto &exit_actor : control_actor_set->exit_actors_) {
    MS_EXCEPTION_IF_NULL(exit_actor);
    exit_actor->memory_free_lists_ = std::queue<std::vector<DeviceTensor *>>();
    exit_actor->created_device_tensors_.clear();
  }
}

void ControlNodeScheduler::LinkArrowForControlActor(ControlActorSet *const control_actor_set,
                                                    const GraphCompilerInfo &graph_compiler_info) const {
  if (control_actor_set == nullptr) {
    return;
  }

  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
  const auto &parser = graph_compiler_info.control_node_parser_;
  for (auto &switch_actor : control_actor_set->switch_actors_) {
    MS_EXCEPTION_IF_NULL(switch_actor);
    if (!parser->IsNeedStackControlNode(switch_actor->node_)) {
      for (size_t i = 0; i < switch_actor->formal_parameters_.size(); ++i) {
        LinkArrowbyFormalParameter(switch_actor.get(), switch_actor->formal_parameters_[i], {switch_actor->node_, i},
                                   graph_compiler_info);
      }
    } else {
      // If the control actor has a corresponding stack actor, the input should be linked to the stack actor.
      auto stack_actor_name = GetActorName(switch_actor->node_) + kStackActorNameSuffix;
      auto actor = FetchActor(stack_actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto stack_actor = dynamic_cast<StackActor *>(actor);
      MS_EXCEPTION_IF_NULL(stack_actor);
      LinkArrowFromStackActor(stack_actor, switch_actor.get(), graph_compiler_info);
    }
  }

  for (auto &gather_actor : control_actor_set->gather_actors_) {
    MS_EXCEPTION_IF_NULL(gather_actor);
    MS_EXCEPTION_IF_NULL(gather_actor->node_);
    if (!parser->IsNeedStackControlNode(gather_actor->node_)) {
      for (size_t i = 0; i < gather_actor->formal_parameters_.size(); ++i) {
        LinkArrowbyFormalParameter(gather_actor.get(), gather_actor->formal_parameters_[i], {gather_actor->node_, i},
                                   graph_compiler_info);
      }
    } else {
      // If the control actor has a corresponding stack actor, the input should be linked to the stack actor.
      auto stack_actor_name = GetActorName(gather_actor->node_) + kStackActorNameSuffix;
      auto actor = FetchActor(stack_actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto stack_actor = dynamic_cast<StackActor *>(actor);
      MS_EXCEPTION_IF_NULL(stack_actor);
      LinkArrowFromStackActor(stack_actor, gather_actor.get(), graph_compiler_info);
    }
  }

  for (auto &entrance_actor : control_actor_set->entrance_actors_) {
    MS_EXCEPTION_IF_NULL(entrance_actor);
    for (const auto &call_node : entrance_actor->call_nodes_) {
      LinkArrowbyFormalParameter(entrance_actor.get(), call_node, {entrance_actor->node_, 0}, graph_compiler_info);
    }
  }

  for (auto &exit_actor : control_actor_set->exit_actors_) {
    MS_EXCEPTION_IF_NULL(exit_actor);

    auto stack_actor_name = (exit_actor->node_ == nullptr ? GetStackActorNameByExitName(exit_actor->GetAID().Name())
                                                          : GetActorName(exit_actor->node_) + kStackActorNameSuffix);
    auto actor = FetchActor(stack_actor_name);
    if (actor == nullptr) {
      for (size_t i = 0; i < exit_actor->formal_parameters_.size(); ++i) {
        LinkArrowbyFormalParameter(exit_actor.get(), exit_actor->formal_parameters_[i], {exit_actor->node_, i},
                                   graph_compiler_info);
      }
    } else {
      // If the control actor has a corresponding stack actor, the input should be linked to the stack actor.
      auto stack_actor = dynamic_cast<StackActor *>(actor);
      MS_EXCEPTION_IF_NULL(stack_actor);
      LinkArrowFromStackActor(stack_actor, exit_actor.get(), graph_compiler_info);
    }
  }

  for (auto &stack_actor : control_actor_set->stack_actors_) {
    MS_EXCEPTION_IF_NULL(stack_actor);
    for (size_t i = 0; i < stack_actor->formal_parameters_.size(); ++i) {
      LinkArrowbyFormalParameter(stack_actor.get(), stack_actor->formal_parameters_[i], {stack_actor->node_, i},
                                 graph_compiler_info);
    }
  }
}

void ControlNodeScheduler::LinkArrowFromStackActor(StackActor *const stack_actor, ControlActor *const to_actor,
                                                   const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(stack_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  for (size_t to_index = 0; to_index < to_actor->formal_parameters_.size(); ++to_index) {
    const auto &formal_parameter =
      common::AnfAlgo::FetchRealNodeSkipMonadControl(to_actor->formal_parameters_[to_index]);
    const auto &from_node = formal_parameter.first;
    MS_EXCEPTION_IF_NULL(from_node);
    if (from_node->isa<ValueNode>()) {
      LinkArrowByValueNode(from_node, to_actor, formal_parameter.second, to_index);
      continue;
    }

    // Fetch the arrow type of input.
    if (to_actor->type_ == KernelTransformType::kExitActor && to_actor->node_ == nullptr && from_node->isa<CNode>() &&
        (!common::AnfAlgo::IsCallNode(from_node)) &&
        (!common::AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimPartial)) &&
        to_actor->GetAID().Name().find(
          parser->FetchGroupNameByKernelGraph(parser->FetchKernelGraphByFrontNode(from_node))) != std::string::npos) {
      LinkArrowByKernel(from_node, to_actor, formal_parameter, {to_actor->node_, to_index}, graph_compiler_info);
      continue;
    }

    size_t from_index = stack_actor->FetchNodePosition(formal_parameter);
    const auto &abstract = from_node->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    const auto &real_abstract = common::AnfAlgo::FetchAbstractByIndex(abstract, formal_parameter.second);
    MS_EXCEPTION_IF_NULL(real_abstract);

    // Link arrow according to abstract.
    if (real_abstract->isa<abstract::AbstractFunction>()) {
      SchedulerHelper::AddPartialArrow(stack_actor, to_actor, from_index, to_index);
    } else {
      SchedulerHelper::AddDataArrow(stack_actor, to_actor, from_index, to_index);
    }
  }
}

void ControlNodeScheduler::LinkArrowbyFormalParameter(ControlActor *const to_actor,
                                                      const KernelWithIndex &from_node_with_index,
                                                      const KernelWithIndex &to_node_with_index,
                                                      const GraphCompilerInfo &graph_compiler_info) const {
  const auto &real_from_node_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(from_node_with_index);
  const auto &from_node = real_from_node_with_index.first;
  MS_EXCEPTION_IF_NULL(from_node);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_LOG(DEBUG) << "Link arrow by formal parameter, from node:" << from_node->DebugString()
                << " from index:" << real_from_node_with_index.second << " to actor:" << to_actor->GetAID()
                << " to index:" << to_node_with_index.second;
  if (from_node->isa<ValueNode>()) {
    LinkArrowByValueNode(from_node, to_actor, real_from_node_with_index.second, to_node_with_index.second);
  } else if (from_node->isa<Parameter>()) {
    LinkArrowByParameter(from_node, to_actor, real_from_node_with_index, to_node_with_index,
                         graph_compiler_info.control_node_parser_);
  } else if (common::AnfAlgo::IsCallNode(from_node)) {
    // Link arrow by call node.
    LinkArrowByCallNode(from_node, to_actor, real_from_node_with_index, to_node_with_index,
                        graph_compiler_info.control_node_parser_);
  } else if (common::AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimSwitch) ||
             common::AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimSwitchLayer)) {
    // Link arrow from switch actor.
    const auto &actor_name = GetActorName(from_node);
    const auto &actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    const auto &switch_actor = dynamic_cast<SwitchActor *>(actor);
    MS_EXCEPTION_IF_NULL(switch_actor);
    if (IsPartialInput(from_node)) {
      SchedulerHelper::AddPartialArrow(switch_actor, to_actor, real_from_node_with_index.second,
                                       to_node_with_index.second);
    } else {
      SchedulerHelper::AddDataArrow(switch_actor, to_actor, real_from_node_with_index.second,
                                    to_node_with_index.second);
    }
  } else if (common::AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimPartial)) {
    // If the funcgraph of the partial node is a deadnode, in order to ensure the correspondence between formal
    // parameters and real parameters, we need to create an empty partial for it.
    if (IsInvalidPartial(from_node)) {
      MS_LOG(DEBUG) << "Invalid partial node:" << from_node->DebugString();
      to_actor->local_partials_[to_node_with_index.second] = std::make_shared<OpPartial>();
      return;
    }
    // Link arrow from gather actor
    const auto &actor_name = GetActorName(from_node);
    const auto &actor = FetchActor(actor_name);
    if (actor == nullptr) {
      MS_LOG(DEBUG) << "No actor of " << actor_name;
      return;
    }
    const auto &gather_actor = dynamic_cast<GatherActor *>(actor);
    MS_EXCEPTION_IF_NULL(gather_actor);
    SchedulerHelper::AddPartialArrow(gather_actor, to_actor, real_from_node_with_index.second,
                                     to_node_with_index.second);
  } else if (from_node->isa<CNode>()) {
    // Link arrow by kernel.
    LinkArrowByKernel(from_node, to_actor, real_from_node_with_index, to_node_with_index, graph_compiler_info);
  }
}

void ControlNodeScheduler::LinkArrowByValueNode(const AnfNodePtr &value_node, ControlActor *const to_actor,
                                                size_t from_index, size_t to_index) const {
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(to_actor);

  if (IsValueNode<FuncGraph>(value_node)) {
    // Link local partial.
    const auto &func_graph = GetValueNode<FuncGraphPtr>(value_node);
    MS_EXCEPTION_IF_NULL(func_graph);
    MS_LOG(DEBUG) << "Add local partial, graph:" << func_graph->ToString() << " for actor:" << to_actor->GetAID();
    to_actor->local_partials_[to_index] = std::make_shared<OpPartial>();
    *(to_actor->local_partials_[to_index]) = {func_graph.get(), {}, {}};
  } else {
    // Link device store value node.
    if (!AnfAlgo::OutputAddrExist(value_node, from_index)) {
      auto node = value_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(node);
      auto value = node->value();
      MS_EXCEPTION_IF_NULL(value);
      // If the from index exceeds the size of the value node, we need to change the from index to 0.
      if (!value->isa<ValueTuple>() && from_index > 0) {
        from_index = 0;
      } else {
        MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid output address index:" << from_index
                          << " for value node:" << value_node->DebugString() << " to actor:" << to_actor->GetAID();
      }
    }
    to_actor->local_device_tensors_[to_index] = AnfAlgo::GetMutableOutputAddr(value_node, from_index, false).get();
    to_actor->local_device_tensors_[to_index]->SetNodeIndex(value_node, from_index);
    MS_LOG(DEBUG) << "Add local device tensor:" << to_actor->local_device_tensors_[to_index] << " index:" << to_index
                  << " for actor:" << to_actor->GetAID() << " from index:" << from_index;
  }
}

void ControlNodeScheduler::LinkArrowByParameter(const AnfNodePtr &parameter, ControlActor *const to_actor,
                                                const KernelWithIndex &from_node_with_index,
                                                const KernelWithIndex &to_node_with_index,
                                                const ControlNodeParserPtr &parser) const {
  MS_EXCEPTION_IF_NULL(parameter);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(parser);
  MS_LOG(DEBUG) << "Link arrow by parameter:" << parameter->DebugString() << " indx:" << from_node_with_index.second
                << " for actor:" << to_actor->GetAID();
  if (parser->IsRootGraphPersistentDeviceTensor(parameter)) {
    (void)to_actor->device_tensor_store_keys_.emplace_back(to_node_with_index.second, parameter);
    return;
  }
  // Link arrow from entrance actor.
  const auto &func_graph = parameter->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
  auto actor = FetchActor(actor_name);
  MS_EXCEPTION_IF_NULL(actor);

  // If the input of the exit actor of the kernel graph is a parameter node, and there is a corresponding stack actor,
  // it should be linked to the stack actor.
  if (to_actor->type() == KernelTransformType::kExitActor) {
    auto stack_actor_name = (to_actor->node_ == nullptr ? GetStackActorNameByExitName(to_actor->GetAID().Name())
                                                        : GetActorName(to_actor->node_) + kStackActorNameSuffix);
    auto stack_actor = FetchActor(stack_actor_name);
    actor = (stack_actor == nullptr ? actor : stack_actor);
  }

  auto from_actor = dynamic_cast<ControlActor *>(actor);
  MS_EXCEPTION_IF_NULL(from_actor);

  auto abstract = parameter->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  auto dst_abstract = common::AnfAlgo::FetchAbstractByIndex(abstract, from_node_with_index.second);
  MS_EXCEPTION_IF_NULL(dst_abstract);
  if (dst_abstract->isa<abstract::AbstractFunction>()) {
    SchedulerHelper::AddPartialArrow(from_actor, to_actor, from_actor->FetchNodePosition(from_node_with_index),
                                     to_node_with_index.second);
  } else {
    SchedulerHelper::AddDataArrow(from_actor, to_actor, from_actor->FetchNodePosition(from_node_with_index),
                                  to_node_with_index.second);
  }
}

void ControlNodeScheduler::LinkArrowByCallNode(const AnfNodePtr &call_node, ControlActor *const to_actor,
                                               const KernelWithIndex &from_node_with_index,
                                               const KernelWithIndex &to_node_with_index,
                                               const ControlNodeParserPtr &parser) const {
  MS_EXCEPTION_IF_NULL(parser);
  MS_EXCEPTION_IF_NULL(call_node);
  MS_EXCEPTION_IF_NULL(to_actor);
  const auto &from_node = from_node_with_index.first;
  MS_EXCEPTION_IF_NULL(from_node);

  if (to_actor->type_ != KernelTransformType::kEntranceActor) {
    // Link arrow from exit actor to control actor.
    const auto &abstract = call_node->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    const auto &real_abstract = common::AnfAlgo::FetchAbstractByIndex(abstract, from_node_with_index.second);
    MS_EXCEPTION_IF_NULL(real_abstract);

    const auto &func_graphs = parser->FetchFuncGraphbyCallNode(from_node);
    for (const auto &func_graph : func_graphs) {
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &actor_name = func_graph->ToString() + kExitActorNameSuffix;
      auto actor = FetchActor(actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto exit_actor = dynamic_cast<ExitActor *>(actor);
      MS_EXCEPTION_IF_NULL(exit_actor);
      auto branch_id = parser->FetchBranchIDByCallNode(from_node);
      if (real_abstract->isa<abstract::AbstractFunction>()) {
        SchedulerHelper::AddPartialArrowForExitActor(exit_actor, to_actor, from_node_with_index.second,
                                                     to_node_with_index.second, branch_id);
      } else {
        SchedulerHelper::AddDataArrowForExitActor(exit_actor, to_actor, from_node_with_index.second,
                                                  to_node_with_index.second, branch_id);
      }
      MS_LOG(DEBUG) << "Link data arrow from:" << exit_actor->GetAID() << " index:" << from_node_with_index.second
                    << " to:" << to_actor->GetAID() << " index" << to_node_with_index.second;
    }
    if (real_abstract->isa<abstract::AbstractFunction>()) {
      to_actor->input_partials_num_++;
    } else {
      MS_LOG(DEBUG) << "Actor:" << to_actor->GetAID() << " add input num:" << to_actor->input_datas_num_;
      to_actor->input_datas_num_++;
    }
  } else {
    // Link arrow from gather actor to entrance actor.
    const auto &actor_name = GetActorName(from_node);
    const auto &actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    const auto &gather_actor = dynamic_cast<GatherActor *>(actor);
    MS_EXCEPTION_IF_NULL(gather_actor);
    const auto &to_node = to_node_with_index.first;
    MS_EXCEPTION_IF_NULL(to_node);
    const auto &func_graph = to_node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    SchedulerHelper::AddDataWithBranchIDArrow(gather_actor, dynamic_cast<EntranceActor *>(to_actor), func_graph);
  }
}

void ControlNodeScheduler::LinkArrowByKernel(const AnfNodePtr &kernel, ControlActor *const to_actor,
                                             const KernelWithIndex &from_node_with_index,
                                             const KernelWithIndex &to_node_with_index,
                                             const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(to_actor);
  const auto &from_node = from_node_with_index.first;
  MS_EXCEPTION_IF_NULL(from_node);
  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  const auto &graph = parser->FetchKernelGraphByFrontNode(from_node);
  MS_LOG(DEBUG) << "Link arrow by kernel, from mode:" << from_node->DebugString() << " to actor:" << to_actor->GetAID();
  MS_EXCEPTION_IF_NULL(graph);
  const auto &group_name = parser->FetchGroupNameByKernelGraph(graph);

  if (to_actor->type_ == KernelTransformType::kExitActor && to_actor->node_ == nullptr &&
      to_actor->GetAID().Name().find(group_name) != std::string::npos) {
    // Link arrow from actor of output node to exit actor of kernel graph.
    const auto &kernel_with_index = parser->FetchBackendNodeByFrontNode(from_node_with_index);
    MS_EXCEPTION_IF_NULL(kernel_with_index.first);
    auto type = FetchKernelTransformType(kernel_with_index.first, graph, {});
    auto from_actor = FetchActor(type, graph_compiler_info.name_, kernel_with_index.first, graph);
    MS_EXCEPTION_IF_NULL(from_actor);
    SchedulerHelper::AddDataArrow(from_actor, to_actor, kernel_with_index.second, to_node_with_index.second,
                                  kernel_with_index.first);
  } else {
    // Link arrow from exit actor of kernel graph to exit actor of function graph.
    const auto &actor_name = parser->FetchGroupNameByKernelGraph(graph) + kExitActorNameSuffix;
    MS_LOG(DEBUG) << "Actor name:" << actor_name << " from node:" << from_node->DebugString();
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto exit_actor = dynamic_cast<ExitActor *>(actor);
    MS_EXCEPTION_IF_NULL(exit_actor);
    size_t from_index = exit_actor->FetchNodePosition(from_node_with_index);
    SchedulerHelper::AddDataArrow(exit_actor, to_actor, from_index, to_node_with_index.second);
  }
}

void ControlNodeScheduler::LinkControlArrowForControlActor(ActorSet *const actor_set,
                                                           const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto control_actor_set = actor_set->control_actors_.get();
  MS_EXCEPTION_IF_NULL(control_actor_set);
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  LinkControlArrowForEntranceActor(actor_set, graph_compiler_info);

  // When the switch actor and gather actor have no input, need to link a control arrow from entrance actor.
  std::vector<ControlActor *> need_check_control_actors;
  (void)std::transform(control_actor_set->switch_actors_.begin(), control_actor_set->switch_actors_.end(),
                       std::back_inserter(need_check_control_actors),
                       [](const auto &switch_actor) { return switch_actor.get(); });
  (void)std::transform(control_actor_set->gather_actors_.begin(), control_actor_set->gather_actors_.end(),
                       std::back_inserter(need_check_control_actors),
                       [](const auto &gather_actor) { return gather_actor.get(); });

  for (auto control_actor : need_check_control_actors) {
    MS_EXCEPTION_IF_NULL(control_actor);
    if (IsNoInputActor(control_actor)) {
      MS_EXCEPTION_IF_NULL(control_actor->node_);
      const FuncGraphPtr &func_graph = control_actor->node_->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
      const auto &entrance_actor = dynamic_cast<EntranceActor *>(FetchActor(actor_name));
      MS_EXCEPTION_IF_NULL(entrance_actor);
      SchedulerHelper::AddControlArrow(entrance_actor, control_actor);
    }
  }

  // Link auto monad control arrow for control actor.
  std::vector<ControlActor *> control_actors;
  (void)std::transform(control_actor_set->switch_actors_.begin(), control_actor_set->switch_actors_.end(),
                       std::back_inserter(control_actors), [](auto &switch_actor) { return switch_actor.get(); });
  (void)std::transform(control_actor_set->gather_actors_.begin(), control_actor_set->gather_actors_.end(),
                       std::back_inserter(control_actors), [](auto &gather_actor) { return gather_actor.get(); });
  (void)std::transform(control_actor_set->exit_actors_.begin(), control_actor_set->exit_actors_.end(),
                       std::back_inserter(control_actors), [](auto &exit_actor) { return exit_actor.get(); });
  for (auto control_actor : control_actors) {
    MS_EXCEPTION_IF_NULL(control_actor);
    const auto &node = control_actor->node_;
    if (node == nullptr) {
      continue;
    }

    auto to_actor = control_actor;
    if (parser->IsNeedStackControlNode(node)) {
      const auto &stack_actor_name = GetActorName(node) + kStackActorNameSuffix;
      auto actor = FetchActor(stack_actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      to_actor = dynamic_cast<ControlActor *>(actor);
      MS_EXCEPTION_IF_NULL(to_actor);
    }

    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &inputs = cnode->inputs();
    for (const auto &input : inputs) {
      MS_EXCEPTION_IF_NULL(input);
      std::vector<AnfNodePtr> monad_nodes = FetchAllMonadNodeByNode(input);
      for (const auto &monad_node : monad_nodes) {
        MS_EXCEPTION_IF_NULL(monad_node);
        LinkControlArrowByAutoMonad(to_actor, monad_node, parser);
      }
    }
  }

  // Link copy actor to exit actor.
  for (const auto &copy_actor : actor_set->copy_actors_) {
    MS_EXCEPTION_IF_NULL(copy_actor);
    if ((!copy_actor->output_data_arrows_.empty()) || (!copy_actor->output_control_arrows_.empty())) {
      continue;
    }
    if (copy_actor->from_kernel_ == nullptr) {
      MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid copy actor:" << copy_actor->GetAID().Name();
    }
    MS_EXCEPTION_IF_NULL(copy_actor->from_kernel_);
    auto graph = copy_actor->from_kernel_->func_graph();
    MS_EXCEPTION_IF_NULL(graph);
    auto kernel_graph = std::dynamic_pointer_cast<KernelGraph>(graph);
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto exit_actor_name = parser->FetchGroupNameByKernelGraph(kernel_graph) + kExitActorNameSuffix;
    auto exit_actor = FetchActor(exit_actor_name);
    MS_EXCEPTION_IF_NULL(exit_actor);
    SchedulerHelper::AddControlArrow(copy_actor.get(), exit_actor);
  }

  LinkControlArrowByKernelGraphGroup(graph_compiler_info);
}

void ControlNodeScheduler::LinkControlArrowForEntranceActor(ActorSet *const actor_set,
                                                            const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto control_actor_set = actor_set->control_actors_.get();
  MS_EXCEPTION_IF_NULL(control_actor_set);
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  // Since only one set of real parameters are allowed to be executed in funcgraph at the same time, when the funcgraph
  // stops running, it is necessary to send the control arrow to the corresponding entrance actor at the exit of the
  // graph to run the next set of real parameters. The corresponding nodes of the actors that need to send the control
  // arrow have been parsed in the control node parser.
  for (const auto &graph_to_nodes : parser->func_graph_to_first_control_nodes_) {
    // Fetch the entrance actor.
    const auto &func_graph = graph_to_nodes.first;
    MS_EXCEPTION_IF_NULL(func_graph);
    auto actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
    auto entrance_actor = dynamic_cast<EntranceActor *>(FetchActor(actor_name));
    MS_EXCEPTION_IF_NULL(entrance_actor);

    const auto &nodes = graph_to_nodes.second;
    for (const auto &node : nodes) {
      // Fetch the source actor of control arrow.
      MS_EXCEPTION_IF_NULL(node);
      if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
        actor_name = func_graph->ToString() + kExitActorNameSuffix;
      } else {
        actor_name = GetActorName(node);
      }
      auto from_actor = dynamic_cast<ControlActor *>(FetchActor(actor_name));
      MS_EXCEPTION_IF_NULL(from_actor);
      SchedulerHelper::AddLoopBodyControlArrow(from_actor, entrance_actor);
    }
  }

  // In the recursive scene, some kernel graph needs to be completed before the next set of data is sent by the
  // entrance actor. At this time, it is necessary to connect a control arrow from the exit actor of the graph
  // to the entrance actor.
  for (const auto &func_graph_to_group_info : parser->func_graph_to_first_kernel_graphs_) {
    const auto &func_graph = func_graph_to_group_info.first;
    MS_EXCEPTION_IF_NULL(func_graph);
    auto actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto entrance_actor = dynamic_cast<EntranceActor *>(actor);
    MS_EXCEPTION_IF_NULL(entrance_actor);
    for (const auto &group_info : func_graph_to_group_info.second) {
      MS_EXCEPTION_IF_NULL(group_info);
      actor_name = group_info->group_name_ + kExitActorNameSuffix;
      auto from_actor = FetchActor(actor_name);
      MS_EXCEPTION_IF_NULL(from_actor);
      SchedulerHelper::AddLoopBodyControlArrow(from_actor, entrance_actor);
    }
  }
}

void ControlNodeScheduler::LinkControlArrowForLoopCountActor(const ActorSet *actor_set,
                                                             const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto loop_count_actor = actor_set->loop_count_actor_;
  MS_EXCEPTION_IF_NULL(loop_count_actor);

  // The final output is always sent by the exit of the root graph in control flow.
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  const auto &root_graph = parser->root_func_graph_;
  MS_EXCEPTION_IF_NULL(root_graph);
  auto exit_actor_name = root_graph->ToString() + kExitActorNameSuffix;
  auto root_exit_actor = dynamic_cast<ExitActor *>(FetchActor(exit_actor_name));
  MS_EXCEPTION_IF_NULL(root_exit_actor);
  // link control arrow from root exit actor to loop count actor.
  SchedulerHelper::AddControlArrowForExitActor(root_exit_actor, loop_count_actor.get(), kMainBranchID);

  // The entrance actor will generate some data in the loop body execution, so need clear on the end of step.
  MS_EXCEPTION_IF_NULL(actor_set->control_actors_);
  for (auto &entrance_actor : actor_set->control_actors_->entrance_actors_) {
    MS_EXCEPTION_IF_NULL(entrance_actor);
    (void)loop_count_actor->entrance_aids_.emplace_back(entrance_actor->GetAID());
  }
}

void ControlNodeScheduler::LinkControlArrowForKernelActor(ActorSet *const actor_set,
                                                          const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  // Link control arrow from entrance actors or stack actors to no input kernel actors.
  for (const auto &no_input_kernel_actor : actor_set->no_input_kernel_actors_) {
    // In control flow, when the input of the kernel actor is a parameter, this input needs to be linked to the
    // control actor, so the no-input kernel actor collected in the graph scheduler will also collect this actor,
    // and it needs to be skipped here.
    MS_EXCEPTION_IF_NULL(no_input_kernel_actor);
    if ((no_input_kernel_actor->input_datas_num_ != 0) || (no_input_kernel_actor->input_controls_num_ != 0)) {
      continue;
    }

    KernelGraphPtr kernel_graph = nullptr;
    if (no_input_kernel_actor->type_ == KernelTransformType::kSuperKernelActor) {
      const auto &super_kernel_actor = dynamic_cast<SuperKernelActor *>(no_input_kernel_actor.get());
      MS_EXCEPTION_IF_NULL(super_kernel_actor);
      kernel_graph = super_kernel_actor->graph();
    } else if (no_input_kernel_actor->type_ == KernelTransformType::kKernelActor) {
      const auto &kernel_actor = dynamic_cast<KernelActor *>(no_input_kernel_actor.get());
      MS_EXCEPTION_IF_NULL(kernel_actor);
      kernel_graph = AnfAlgo::FetchKernelGraph(kernel_actor->kernel().get());
    } else {
      continue;
    }
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto actor_name = parser->FetchGroupNameByKernelGraph(kernel_graph) + kStackActorNameSuffix;
    if (!parser->IsCallInputKernelGraph(kernel_graph.get())) {
      const auto &func_graph = parser->FetchFuncGraphByKernelGraph(kernel_graph.get());
      MS_EXCEPTION_IF_NULL(func_graph);
      actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
    }

    auto from_actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(from_actor);
    SchedulerHelper::AddControlArrow(from_actor, no_input_kernel_actor.get());
  }

  // Link control arrows from no output kernel actor to the corresponding exit actor.
  for (auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    if ((kernel_actor->output_data_arrows_.size() == 0) && (kernel_actor->output_control_arrows_.size() == 0)) {
      auto kernel_graph = AnfAlgo::FetchKernelGraph(kernel_actor->kernel().get());
      MS_EXCEPTION_IF_NULL(kernel_graph);
      auto to_actor_name = parser->FetchGroupNameByKernelGraph(kernel_graph) + kExitActorNameSuffix;
      auto to_actor = FetchActor(to_actor_name);
      MS_EXCEPTION_IF_NULL(to_actor);
      SchedulerHelper::AddControlArrow(kernel_actor.get(), to_actor);
    }
  }

  // Link control arrows from no super kernel actor to the corresponding exit actor.
  for (auto &super_actor : actor_set->super_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(super_actor);
    if ((super_actor->output_data_arrows_.size() == 0) && (super_actor->output_control_arrows_.size() == 0)) {
      auto kernel_graph = super_actor->graph();
      MS_EXCEPTION_IF_NULL(kernel_graph);
      auto to_actor_name = parser->FetchGroupNameByKernelGraph(kernel_graph) + kExitActorNameSuffix;
      auto to_actor = FetchActor(to_actor_name);
      MS_EXCEPTION_IF_NULL(to_actor);
      SchedulerHelper::AddControlArrow(super_actor.get(), to_actor);
    }
  }
}

void ControlNodeScheduler::LinkControlArrowByAutoMonad(ControlActor *to_actor, const AnfNodePtr &from_node,
                                                       const ControlNodeParserPtr &parser) const {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(from_node);
  MS_EXCEPTION_IF_NULL(parser);
  MS_LOG(DEBUG) << "Link auto monad control arrow from node:" << from_node->DebugString()
                << " to actor:" << to_actor->GetAID();

  std::set<AnfNodePtr> depend_nodes;
  FetchRealDependNodeByAutoMonad(from_node, &depend_nodes);

  for (const auto &depend_node : depend_nodes) {
    MS_EXCEPTION_IF_NULL(depend_node);
    MS_LOG(DEBUG) << "Add depend node:" << depend_node->DebugString() << " for actor:" << to_actor->GetAID();
    auto from_actor = FetchActor(GetActorName(depend_node));
    auto graph = parser->FetchKernelGraphByFrontNode(depend_node);

    std::vector<AbstractActor *> from_actors;
    if (common::AnfAlgo::IsCallNode(depend_node)) {
      // If the actor already exists with control arrow, skip it.
      if (IsControlArrowExistForCallNode(depend_node, to_actor, parser)) {
        MS_LOG(DEBUG) << "Control arrow from call node:" << depend_node << " to actor:" << to_actor->GetAID()
                      << "is exist, skip it";
        continue;
      }
      int branch_id = parser->FetchBranchIDByCallNode(depend_node);
      const auto &func_graphs = parser->FetchFuncGraphbyCallNode(depend_node);
      if (func_graphs.empty()) {
        MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Failed to get funcgraph by call node:"
                          << depend_node->DebugString();
      }
      for (const auto &func_graph : func_graphs) {
        MS_EXCEPTION_IF_NULL(func_graph);
        auto exit_actor_name = func_graph->ToString() + kExitActorNameSuffix;
        from_actor = FetchActor(exit_actor_name);
        MS_EXCEPTION_IF_NULL(from_actor);
        (void)from_actors.emplace_back(from_actor);
        auto exit_actor = dynamic_cast<ExitActor *>(from_actor);
        MS_EXCEPTION_IF_NULL(exit_actor);
        SchedulerHelper::AddControlArrowForExitActor(exit_actor, to_actor, branch_id);
      }
      to_actor->input_controls_num_ -= (func_graphs.size() - 1);
    } else if (from_actor != nullptr) {
      (void)from_actors.emplace_back(from_actor);
      SchedulerHelper::AddControlArrow(from_actor, to_actor);
    } else {
      if (graph == nullptr) {
        MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Failed to find actor for node:"
                          << depend_node->DebugString();
      }
      from_actor = FetchActor(parser->FetchGroupNameByKernelGraph(graph) + kExitActorNameSuffix);
      MS_EXCEPTION_IF_NULL(from_actor);
      if (std::find_if(from_actor->output_control_arrows_.begin(), from_actor->output_control_arrows_.end(),
                       [&to_actor](auto &output_control_arrow) {
                         MS_EXCEPTION_IF_NULL(output_control_arrow);
                         return output_control_arrow->to_op_id_.Name() == to_actor->GetAID().Name();
                       }) != from_actor->output_control_arrows_.end()) {
        MS_LOG(DEBUG) << "Link auto monad control from actor:" << from_actor->GetAID()
                      << " to actor:" << to_actor->GetAID() << " is already exist.";
        continue;
      }
      (void)from_actors.emplace_back(from_actor);
      SchedulerHelper::AddControlArrow(from_actor, to_actor);
    }
    if (to_actor->type_ != KernelTransformType::kStackActor || parser->IsNeedStackControlNode(depend_node) ||
        parser->IsRecursionCallNode(depend_node) || (graph != nullptr && parser->IsRecursionKernelGraph(graph))) {
      continue;
    }
    // If the control arrow comes from a recursive call node or a recursive kernel graph, these control edges will be
    // directly linked to the stack actor, otherwise, they need to be cached in the stack of the stack actor.
    auto stack_actor = dynamic_cast<StackActor *>(to_actor);
    MS_EXCEPTION_IF_NULL(stack_actor);
    stack_actor->input_controls_num_--;
    stack_actor->input_stack_controls_num_++;
    for (const auto &actor : from_actors) {
      MS_EXCEPTION_IF_NULL(actor);
      MS_LOG(DEBUG) << "Add stack control aid:" << actor->GetAID() << " for actor:" << stack_actor->GetAID();
      (void)stack_actor->stack_control_aids_.emplace(actor->GetAID());
      stack_actor->control_aid_to_indexs_[actor->GetAID()] = stack_actor->input_stack_controls_num_;
    }
  }
  MS_LOG(DEBUG) << "Link auto monad control arrow from node:" << from_node->DebugString()
                << " to actor:" << to_actor->GetAID() << " end";
}

void ControlNodeScheduler::LinkControlArrowByKernelGraphGroup(const GraphCompilerInfo &graph_compiler_info) const {
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  for (const auto &graph_group : parser->kernel_graph_group_infos_) {
    MS_EXCEPTION_IF_NULL(graph_group);
    if (!graph_group->need_stack_) {
      continue;
    }
    auto stack_actor = FetchActor(graph_group->group_name_ + kStackActorNameSuffix);
    MS_EXCEPTION_IF_NULL(stack_actor);
    auto to_actor = dynamic_cast<ControlActor *>(stack_actor);
    MS_EXCEPTION_IF_NULL(to_actor);
    for (const auto &monad_input : graph_group->monad_inputs_) {
      MS_EXCEPTION_IF_NULL(monad_input);
      MS_LOG(DEBUG) << "Add monad control arrow for group:" << graph_group->group_name_
                    << " to actor:" << to_actor->GetAID() << " by monad input:" << monad_input->DebugString();
      LinkControlArrowByAutoMonad(to_actor, monad_input, parser);
    }
  }
}

void ControlNodeScheduler::LinkBranchIDArrowForControlActor(ControlActorSet *const control_actor_set) const {
  MS_EXCEPTION_IF_NULL(control_actor_set);

  // Connect the branch id arrows from the entrance actor to the exit actor for each funcgraph.
  for (auto exit_actor : control_actor_set->exit_actors_) {
    MS_EXCEPTION_IF_NULL(exit_actor);

    // If the node in the exit actor is empty, it means that it is between the kernel actor and the control actor,
    // and no need to send the branch id.
    const auto &node = exit_actor->node_;
    if (node == nullptr) {
      continue;
    }

    const auto &func_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    const auto &actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto entrance_actor = dynamic_cast<EntranceActor *>(actor);
    MS_EXCEPTION_IF_NULL(entrance_actor);
    SchedulerHelper::AddBranchIDArrow(entrance_actor, exit_actor.get());
  }

  // Connect the branch id arrows from the entrance actor to the stack actor.
  for (auto stack_actor : control_actor_set->stack_actors_) {
    MS_EXCEPTION_IF_NULL(stack_actor);
    if (stack_actor->formal_parameters_.empty()) {
      MS_LOG(ERROR) << "Invalid stack actor:" << stack_actor->GetAID();
    }
    const auto &node = stack_actor->formal_parameters_.back().first;
    MS_EXCEPTION_IF_NULL(node);
    const auto &func_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    const auto &actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto entrance_actor = dynamic_cast<EntranceActor *>(actor);
    MS_EXCEPTION_IF_NULL(entrance_actor);
    SchedulerHelper::AddBranchIDArrow(entrance_actor, stack_actor.get());
  }
}

void ControlNodeScheduler::LinkDataArrowForKernelActor(const GraphCompilerInfo &graph_compiler_info) const {
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  // Link data arrows from entrance actors and stack actors to kernel actors.
  for (const auto &func_graph_to_kernel_graphs : parser->func_graph_to_kernel_graph_groups_) {
    // Fetch the source entrance actor.
    const auto &func_graph = func_graph_to_kernel_graphs.first;
    MS_EXCEPTION_IF_NULL(func_graph);
    auto actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto entrance_actor = dynamic_cast<ControlActor *>(actor);
    MS_EXCEPTION_IF_NULL(entrance_actor);

    for (const auto &kernel_graph_group : func_graph_to_kernel_graphs.second) {
      for (const auto &kernel_graph : kernel_graph_group) {
        MS_EXCEPTION_IF_NULL(kernel_graph);
        if (kernel_graph->execution_order().empty()) {
          continue;
        }
        LinkDataArrowByKernelGraph(kernel_graph, entrance_actor, parser);
      }
    }
  }
}

void ControlNodeScheduler::LinkDataArrowForCustomActor(const ActorSet *actor_set,
                                                       const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  for (const auto &custom_actor : actor_set->custom_actors_) {
    MS_EXCEPTION_IF_NULL(custom_actor);
    auto kernel = custom_actor->kernel().lock();
    MS_EXCEPTION_IF_NULL(kernel);
    if (AnfUtils::GetCustomActorType(kernel) != kInfer) {
      continue;
    }
    // Kernel in depends form map should link data arrow for infer shape.
    auto base_node = AnfUtils::GetCustomActorBaseNode(kernel);
    MS_EXCEPTION_IF_NULL(base_node);
    auto dynamic_shape_depends = abstract::GetValueDependArgIndices(base_node);
    for (auto iter = dynamic_shape_depends.begin(); iter != dynamic_shape_depends.end(); ++iter) {
      auto input_node = common::AnfAlgo::GetInputNode(base_node, LongToSize(*iter));
      MS_EXCEPTION_IF_NULL(input_node);
      KernelWithIndex from_kernel_with_index = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, false);
      const AnfNodePtr real_input_node = from_kernel_with_index.first;
      MS_EXCEPTION_IF_NULL(real_input_node);
      if (real_input_node->isa<ValueNode>()) {
        continue;
      }
      auto graph = AnfAlgo::FetchKernelGraph(real_input_node.get());
      MS_EXCEPTION_IF_NULL(graph);
      if (!parser->IsControlFlowDataArrow(graph, real_input_node)) {
        continue;
      }

      // Link data arrow from entrance actor or stack actor to infer shape custom actor.
      const auto &front_node_with_index = GetFrontNodeByKernelGraph(real_input_node, graph.get());
      MS_EXCEPTION_IF_NULL(front_node_with_index.first);
      AbstractActor *from_base_actor = nullptr;
      if (parser->IsCallInputKernelGraph(graph.get())) {
        from_base_actor = FetchActor(parser->FetchGroupNameByKernelGraph(graph) + kStackActorNameSuffix);
        MS_EXCEPTION_IF_NULL(from_base_actor);
      } else if (!front_node_with_index.first->isa<Parameter>()) {
        MS_LOG(INFO) << "Internal front node:" << front_node_with_index.first->DebugString()
                     << " index:" << front_node_with_index.second << " for custom actor:" << custom_actor->GetAID()
                     << " kernel:" << kernel->fullname_with_scope() << " input index:" << *iter;
        const auto &from_graph = parser->FetchKernelGraphByFrontNode(front_node_with_index.first);
        MS_EXCEPTION_IF_NULL(from_graph);
        from_base_actor = FetchActor(parser->FetchGroupNameByKernelGraph(from_graph) + kExitActorNameSuffix);
        MS_EXCEPTION_IF_NULL(from_base_actor);
      } else {
        const auto &func_graph = front_node_with_index.first->func_graph();
        MS_EXCEPTION_IF_NULL(func_graph);
        from_base_actor = FetchActor(func_graph->ToString() + kEntranceActorNameSuffix);
        MS_EXCEPTION_IF_NULL(from_base_actor);
      }
      const auto &from_actor = dynamic_cast<ControlActor *>(from_base_actor);
      MS_EXCEPTION_IF_NULL(from_actor);
      size_t from_index = from_actor->FetchNodePosition(front_node_with_index);
      MS_LOG(DEBUG) << "Link data arrow from actor:" << from_actor->GetAID()
                    << " to custom actor:" << custom_actor->GetAID();
      SchedulerHelper::AddDataArrow(from_actor, custom_actor.get(), from_index, LongToSize(*iter));
    }
  }
}

void ControlNodeScheduler::LinkDataArrowByKernelGraphInSinkMode(const KernelGraphPtr &graph,
                                                                ControlActor *const from_actor,
                                                                const ControlNodeParserPtr &parser) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(parser);
  MS_LOG(DEBUG) << "Link data arrow in sink mode by kernel graph:" << graph->ToString();
  auto to_actor = FetchActor(KernelTransformType::kSuperKernelActor, "", nullptr, graph);
  MS_EXCEPTION_IF_NULL(to_actor);
  auto super_kernel_actor = dynamic_cast<SuperKernelActor *>(to_actor);
  MS_EXCEPTION_IF_NULL(super_kernel_actor);

  auto &input_nodes = graph->input_nodes();
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    const auto &input_node = input_nodes[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (HasAbstractMonad(input_node) || (!parser->IsControlFlowDataArrow(graph, input_node))) {
      continue;
    }
    size_t to_index = super_kernel_actor->FetchInputNodePosition(input_node);
    const auto &front_node_with_index = GetFrontNodeByKernelGraph(input_node, graph.get());
    MS_EXCEPTION_IF_NULL(front_node_with_index.first);
    if (front_node_with_index.first->isa<ValueNode>()) {
      continue;
    }
    size_t from_index = from_actor->FetchNodePosition(front_node_with_index);
    SchedulerHelper::AddFormalParameterDeviceTensor(from_actor, from_index, input_node, graph);
    SchedulerHelper::AddDataArrow(from_actor, to_actor, from_index, to_index);
  }
  return;
}

void ControlNodeScheduler::LinkDataArrowByKernelGraph(const KernelGraphPtr &graph, ControlActor *const entrance_actor,
                                                      const ControlNodeParserPtr &parser) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parser);
  MS_LOG(DEBUG) << "Link data arrow by kernel graph:" << graph->ToString();
  auto from_actor = entrance_actor;
  // If there is a call node in the input of the graph, the parameter of the graph needs to be sent by the
  // corresponding stack actor, otherwise it is sent by the entrance actor.
  if (parser->IsCallInputKernelGraph(graph.get())) {
    auto actor = FetchActor(parser->FetchGroupNameByKernelGraph(graph) + kStackActorNameSuffix);
    MS_EXCEPTION_IF_NULL(actor);
    from_actor = dynamic_cast<ControlActor *>(actor);
  }

  if (graph->is_graph_run_mode()) {
    // Link data arrow in graph mode.
    LinkDataArrowByKernelGraphInSinkMode(graph, from_actor, parser);
    return;
  }

  auto &execution_order = graph->execution_order();
  for (const auto &kernel : execution_order) {
    MS_EXCEPTION_IF_NULL(kernel);
    if ((!graph->is_graph_run_mode()) && (IsSkippedKernelActor(kernel) || !IsKernelActor(kernel))) {
      continue;
    }
    for (size_t i = 0; i < common::AnfAlgo::GetInputNum(kernel); ++i) {
      auto input_node = common::AnfAlgo::GetInputNode(kernel, i);
      MS_EXCEPTION_IF_NULL(input_node);
      auto input_with_index = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, false);
      auto input = input_with_index.first;
      MS_EXCEPTION_IF_NULL(input);
      if (HasAbstractMonad(input) || (!parser->IsControlFlowDataArrow(graph, input))) {
        continue;
      }

      auto from_node_with_index = GetFrontNodeByKernelGraph(input, graph.get());
      MS_EXCEPTION_IF_NULL(from_node_with_index.first);
      if (common::AnfAlgo::CheckPrimitiveType(from_node_with_index.first, prim::kPrimTupleGetItem)) {
        MS_LOG(WARNING) << "Input node:" << from_node_with_index.first->DebugString()
                        << " for graph:" << graph->ToString() << " is a tuple get item";
        from_node_with_index = FetchRealNodeByGetItem(from_node_with_index);
      }

      // If the formal parameter is a tuple type, the parameter of the kernel graph will not directly correspond
      // to the front parameter, but the node in the internal parameter.
      const auto &from_node = from_node_with_index.first;
      MS_LOG(DEBUG) << "Graph:" << graph->ToString() << " from node:" << from_node_with_index.first->DebugString()
                    << " index:" << from_node_with_index.second;

      // Fetch actor and link.
      auto type = FetchKernelTransformType(kernel, graph, {});
      auto to_actor = FetchActor(type, "", kernel, graph);
      MS_EXCEPTION_IF_NULL(to_actor);
      size_t from_index = 0;
      // If the input is a switch node and the graph does not need a stack, then the data arrow needs to be connected
      // from the switch actor.
      if ((common::AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimSwitch) ||
           common::AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimSwitchLayer)) &&
          (from_actor->type() != KernelTransformType::kStackActor)) {
        const auto &actor_name = GetActorName(from_node);
        auto actor = FetchActor(actor_name);
        MS_EXCEPTION_IF_NULL(actor);
        from_actor = dynamic_cast<ControlActor *>(actor);
      } else if (from_node->isa<CNode>() && (from_actor->type() != KernelTransformType::kStackActor)) {
        // If the input is an internal parameter, the input arrow should be linked to the exit actor of the kernel
        // graph which the internal parameter belong.
        MS_LOG(INFO) << "Internal parameter in control flow, backend input:" << input->DebugString()
                     << " front node:" << from_node->DebugString();
        const auto &from_graph = parser->FetchKernelGraphByFrontNode(from_node);
        MS_EXCEPTION_IF_NULL(from_graph);
        auto actor = FetchActor(parser->FetchGroupNameByKernelGraph(from_graph) + kExitActorNameSuffix);
        MS_EXCEPTION_IF_NULL(actor);
        auto exit_actor = dynamic_cast<ControlActor *>(actor);
        from_index = exit_actor->FetchNodePosition(from_node_with_index);
        SchedulerHelper::AddFormalParameterDeviceTensor(exit_actor, from_index, input, graph);
        SchedulerHelper::AddDataArrow(exit_actor, to_actor, from_index, i);
        continue;
      } else {
        from_index = from_actor->FetchNodePosition(from_node_with_index);
      }

      MS_EXCEPTION_IF_NULL(from_actor);
      SchedulerHelper::AddFormalParameterDeviceTensor(from_actor, from_index, input, graph);
      SchedulerHelper::AddDataArrow(from_actor, to_actor, from_index, i);
    }
  }
}

void ControlNodeScheduler::LinkDataArrowForOutputActor(ActorSet *const actor_set,
                                                       const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto &to_actor = actor_set->output_actor_;
  MS_EXCEPTION_IF_NULL(to_actor);
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  const auto &root_graph = parser->root_func_graph_;
  MS_EXCEPTION_IF_NULL(root_graph);
  const auto &return_node = root_graph->return_node();
  MS_EXCEPTION_IF_NULL(return_node);

  const auto &exit_actor_name = root_graph->ToString() + kExitActorNameSuffix;
  auto actor = FetchActor(exit_actor_name);
  MS_EXCEPTION_IF_NULL(actor);
  auto exit_actor = dynamic_cast<ExitActor *>(actor);
  MS_EXCEPTION_IF_NULL(exit_actor);
  for (size_t i = 0; i < exit_actor->formal_parameters_.size(); ++i) {
    SchedulerHelper::AddDataArrowForExitActor(exit_actor, to_actor.get(), i, i, 0);
    to_actor->input_datas_num_++;
  }

  auto control_node_to_device_contexts = parser->control_node_to_device_contexts_;
  auto iter = control_node_to_device_contexts.find(return_node);
  if (iter == control_node_to_device_contexts.end()) {
    MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Failed to find device contexts for node:"
                      << return_node->DebugString();
  }
  if (iter->second.size() != to_actor->device_contexts().size()) {
    MS_LOG(EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid context size, need:"
                      << to_actor->device_contexts().size() << " current:" << iter->second.size();
  }
  to_actor->device_contexts_ = iter->second;
}

void ControlNodeScheduler::LinkArrowForRootGraphEntranceActor(const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
  const auto &root_graph = graph_compiler_info.control_node_parser_->root_func_graph_;
  MS_EXCEPTION_IF_NULL(root_graph);
  const auto &entrance_actor_name = root_graph->ToString() + kEntranceActorNameSuffix;
  auto to_actor = dynamic_cast<EntranceActor *>(FetchActor(entrance_actor_name));
  MS_EXCEPTION_IF_NULL(to_actor);

  const auto &host_ds_actor_name = graph_compiler_info.name_ + kHostDSActorNameSuffix;
  auto host_ds_actor = dynamic_cast<HostQueueDataSourceActor *>(FetchActor(host_ds_actor_name));
  // No host data source actor scenario.
  if (host_ds_actor == nullptr) {
    const auto &data_prepare_actor_name = graph_compiler_info.name_ + kDataPrepareActorNameSuffix;
    auto data_prepare_actor = FetchActor(data_prepare_actor_name);
    MS_EXCEPTION_IF_NULL(data_prepare_actor);
    SchedulerHelper::AddControlArrow(data_prepare_actor, to_actor);
    return;
  }

  // The host data source actor sends all the input to the entrance actor of the root graph.
  for (size_t i = 0; i < to_actor->formal_parameters_.size(); ++i) {
    const auto &formal_parameter = to_actor->formal_parameters_[i];
    MS_EXCEPTION_IF_NULL(formal_parameter.first);
    MS_LOG(DEBUG) << "Formal parameter:" << formal_parameter.first->DebugString()
                  << " index:" << formal_parameter.second;
    const auto &iter = host_ds_actor->data_node_position_map_.find(formal_parameter);
    if (iter != host_ds_actor->data_node_position_map_.end()) {
      const auto &parameter_with_index = host_ds_actor->data_nodes()[iter->second];
      SchedulerHelper::AddDataArrow(host_ds_actor, to_actor, parameter_with_index.second, i,
                                    parameter_with_index.first);
    } else {
      MS_LOG(INFO) << "Invalid formal parameter:" << formal_parameter.first->DebugString()
                   << " index:" << formal_parameter.second << " for actor:" << to_actor->GetAID();
    }
  }
}

void ControlNodeScheduler::SetTimeSummaryForControlActor(const GraphCompilerInfo &graph_compiler_info) const {
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  for (const auto &kernel_graph_group_info : parser->kernel_graph_group_infos_) {
    MS_EXCEPTION_IF_NULL(kernel_graph_group_info);
    const auto &exit_actor_name = kernel_graph_group_info->group_name_ + kExitActorNameSuffix;
    const auto &exit_base_actor = FetchActor(exit_actor_name);
    if (exit_base_actor == nullptr) {
      continue;
    }
    const auto &exit_actor = dynamic_cast<ControlActor *>(exit_base_actor);
    MS_EXCEPTION_IF_NULL(exit_actor);

    // Set the exit actor of kernel graph to its entrance actor or stack actor.
    if (kernel_graph_group_info->need_stack_ == false) {
      if (kernel_graph_group_info->graphs_.empty()) {
        continue;
      }
      const auto &graph = *(kernel_graph_group_info->graphs_.begin());
      const auto &func_graph = parser->FetchFuncGraphByKernelGraph(graph.get());
      MS_EXCEPTION_IF_NULL(func_graph);
      auto entrance_base_actor = FetchActor(func_graph->ToString() + kEntranceActorNameSuffix);
      if (entrance_base_actor != nullptr) {
        const auto &entrance_actor = dynamic_cast<ControlActor *>(entrance_base_actor);
        MS_EXCEPTION_IF_NULL(entrance_actor);
        entrance_actor->end_actors_.emplace(exit_actor);
        MS_LOG(DEBUG) << "Add time summart for exit actor:" << exit_actor->GetAID()
                      << " to actor:" << entrance_actor->GetAID();
      }
      continue;
    }

    auto stack_base_actor = FetchActor(kernel_graph_group_info->group_name_ + kStackActorNameSuffix);
    if (stack_base_actor != nullptr) {
      const auto &stack_actor = dynamic_cast<ControlActor *>(stack_base_actor);
      MS_EXCEPTION_IF_NULL(stack_actor);
      stack_actor->end_actors_.emplace(exit_actor);
      MS_LOG(DEBUG) << "Add time summart for exit actor:" << exit_actor->GetAID()
                    << " to actor:" << stack_actor->GetAID();
    }
  }
}

bool ControlNodeScheduler::IsNoInputActor(const ControlActor *control_actor) const {
  MS_EXCEPTION_IF_NULL(control_actor);
  return (control_actor->input_datas_num_ == 0 && control_actor->input_controls_num_ == 0 &&
          control_actor->input_partials_num_ == 0 && control_actor->input_branch_ids_num_ == 0);
}
}  // namespace runtime
}  // namespace mindspore
