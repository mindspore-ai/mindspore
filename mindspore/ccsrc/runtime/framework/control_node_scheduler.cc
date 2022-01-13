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

#include "runtime/framework/control_node_scheduler.h"
#include "runtime/framework/control_node_parser.h"

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

  if (AnfAlgo::IsCallNode(node)) {
    return "Call_" + node->UniqueName() + "_" + debug_name;
  } else {
    return node->UniqueName() + "_" + debug_name;
  }
}

std::string GetStackActorNameByExitName(const std::string &exit_name) {
  size_t pos = exit_name.find(kExitActorNameSuffix);
  if (pos == std::string::npos) {
    MS_LOG(EXCEPTION) << "Invalid exit actor name:" << exit_name;
  }

  return exit_name.substr(0, pos) + kStackActorNameSuffix;
}

// Fetch the depend nodes according to the monad node.
void FetchRealDependNodeByAutoMonad(const AnfNodePtr &node, std::set<AnfNodePtr> *const depend_nodes) {
  // Find the real input node, include the monad node and make tuple node.
  const std::vector<PrimitivePtr> return_types = {prim::kPrimDepend, prim::kPrimUpdateState, prim::kPrimLoad,
                                                  prim::kPrimMakeTuple};
  const auto &node_with_index = AnfAlgo::VisitKernelWithReturnType(node, 0, false, return_types);
  auto real_node = node_with_index.first;
  MS_EXCEPTION_IF_NULL(real_node);
  if (!real_node->isa<CNode>()) {
    return;
  }

  const auto &real_cnode = real_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(real_cnode);
  const auto &real_inputs = real_cnode->inputs();

  // Make tuple node needs to be expanded.
  if (AnfAlgo::CheckPrimitiveType(real_node, prim::kPrimMakeTuple)) {
    for (size_t i = 1; i < real_inputs.size(); ++i) {
      MS_EXCEPTION_IF_NULL(real_inputs[i]);
      FetchRealDependNodeByAutoMonad(real_inputs[i], depend_nodes);
    }
    return;
  }

  if (AnfAlgo::CheckPrimitiveType(real_node, prim::kPrimDepend) ||
      AnfAlgo::CheckPrimitiveType(real_node, prim::kPrimLoad)) {
    FetchRealDependNodeByAutoMonad(real_inputs[kDependAttachNodeIndex], depend_nodes);
  } else if (AnfAlgo::CheckPrimitiveType(real_node, prim::kPrimUpdateState)) {
    for (size_t i = kUpdateStateRealInput; i < real_inputs.size(); ++i) {
      FetchRealDependNodeByAutoMonad(real_inputs[i], depend_nodes);
    }
  } else {
    (void)depend_nodes->emplace(real_node);
  }
}

// Parameter and ref node can not copy the device tensor.
bool is_need_copy_device_tensor(const AnfNodePtr &backend_node, size_t index) {
  MS_EXCEPTION_IF_NULL(backend_node);
  if (!backend_node->isa<CNode>()) {
    return false;
  }

  if (HasAbstractRef(backend_node)) {
    return false;
  }

  auto kernel_graph = FetchKernelGraph(backend_node);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (kernel_graph->IsInRefOutputMap({backend_node, index})) {
    return false;
  }

  return true;
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
  control_actors->switch_actors_ = BuildSwitchActor(graph_compiler_info);
  control_actors->gather_actors_ = BuildGatherActor(graph_compiler_info);
  control_actors->entrance_actors_ = BuildEntranceActor(graph_compiler_info);
  control_actors->exit_actors_ = BuildExitActor(graph_compiler_info);
  control_actors->stack_actors_ = BuildStackActor(graph_compiler_info);
  return control_actors;
}

std::vector<SwitchActorPtr> ControlNodeScheduler::BuildSwitchActor(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<SwitchActorPtr> switch_actors;
  const auto &control_nodes = graph_compiler_info.control_nodes_;

  for (const auto &control_node : control_nodes) {
    // Switch node and switch layer node will be converted to switch actor.
    if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitch) ||
        AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitchLayer)) {
      const auto &actor_name = GetActorName(control_node);
      const auto &parameters = FetchInputNodeByCNode(control_node);
      const auto &switch_actor =
        std::make_shared<SwitchActor>(actor_name, memory_manager_aid_, parameters, control_node);
      (void)switch_actors.emplace_back(switch_actor);
      InsertActor(switch_actor.get());
    }
  }
  return switch_actors;
}

std::vector<GatherActorPtr> ControlNodeScheduler::BuildGatherActor(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<GatherActorPtr> gather_actors;
  const auto &control_nodes = graph_compiler_info.control_nodes_;
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  for (const auto &control_node : control_nodes) {
    // Partial node and call node will be converted to gather actor.
    if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimPartial) || AnfAlgo::IsCallNode(control_node)) {
      const auto &actor_name = GetActorName(control_node);
      const auto &parameters = FetchInputNodeByCNode(control_node);
      const auto &gather_actor =
        std::make_shared<GatherActor>(actor_name, memory_manager_aid_, parameters, control_node);
      (void)gather_actors.emplace_back(gather_actor);
      InsertActor(gather_actor.get());

      // The gather actor corresponding to a call node needs to set the branch id.
      if (AnfAlgo::IsCallNode(control_node)) {
        gather_actor->output_branch_id_ = parser->FetchBranchIDByCallNode(control_node);
      }

      // Fetch device contexts for gather actor.
      const auto &iter = parser->control_node_to_device_contexts_.find(control_node);
      if (iter == parser->control_node_to_device_contexts_.end()) {
        MS_LOG(EXCEPTION) << "Failed to get device contexts for node:" << control_node->DebugString();
      }
      gather_actor->device_contexts_ = iter->second;
    }
  }
  return gather_actors;
}

std::vector<EntranceActorPtr> ControlNodeScheduler::BuildEntranceActor(const GraphCompilerInfo &graph_compiler_info) {
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
    if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      const auto &func_graph = control_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
      std::vector<KernelWithIndex> formal_parameters;

      // The entrance actor has two parts of node members :
      // 1. The formal parameters of the subgraph are used to connect the actor's output arrows.
      for (const auto &parameter : func_graph->parameters()) {
        if (HasAbstractMonad(parameter)) {
          continue;
        }
        const auto &abstract = parameter->abstract();
        MS_EXCEPTION_IF_NULL(abstract);
        size_t output_num = AnfAlgo::GetOutputNumByAbstract(abstract);
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
      const auto &entrance_actor =
        std::make_shared<EntranceActor>(actor_name, memory_manager_aid_, formal_parameters, call_nodes, control_node);
      auto context_iter = parser->func_graph_to_device_contexts_.find(func_graph);
      if (context_iter == parser->func_graph_to_device_contexts_.end() ||
          context_iter->second.size() < formal_parameters.size()) {
        MS_LOG(EXCEPTION) << "Invalid device contexts for funcgraph:" << func_graph->ToString()
                          << " parameter num:" << formal_parameters.size() << " device contexts num:"
                          << (context_iter == parser->func_graph_to_device_contexts_.end()
                                ? 0
                                : context_iter->second.size());
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

std::vector<ExitActorPtr> ControlNodeScheduler::BuildExitActor(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<ExitActorPtr> exit_actors;
  const auto &control_nodes = graph_compiler_info.control_nodes_;
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  // The exit actor is used in 2 places:
  // 1.funcgraph output, that is the output of the return node.
  for (const auto &control_node : control_nodes) {
    if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      const auto &func_graph = control_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &actor_name = func_graph->ToString() + kExitActorNameSuffix;
      const auto &parameters = FetchInputNodeByCNode(control_node);
      const auto &exit_actor = std::make_shared<ExitActor>(actor_name, memory_manager_aid_, parameters, control_node);
      auto context_iter = parser->control_node_to_device_contexts_.find(control_node);
      if (context_iter == parser->control_node_to_device_contexts_.end() ||
          context_iter->second.size() != parameters.size()) {
        MS_LOG(EXCEPTION) << "Failed to get device contexts for funcgraph:" << func_graph->ToString();
      }
      exit_actor->device_contexts_ = context_iter->second;
      (void)exit_actors.emplace_back(exit_actor);
      InsertActor(exit_actor.get());
    }
  }

  if (graph_compiler_info.graphs_.size() != graph_compiler_info.device_contexts_.size()) {
    MS_LOG(EXCEPTION) << "Invalid graphs num:" << graph_compiler_info.graphs_.size()
                      << " and contexts num:" << graph_compiler_info.device_contexts_.size();
  }

  // 2. Replace the device address in the kernel actor when calling funcgraph, that is to say in the data exchange
  // between kernel graph and the control node, in fact, it is the output of the kernel graph.
  for (const auto &kernel_graph_group_info : parser->kernel_graph_group_infos_) {
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
    exit_actor->is_need_copy_device_tensors_.swap(is_need_copy_device_tensors);
    exit_actor->device_contexts_.swap(device_contexts);
    (void)exit_actors.emplace_back(exit_actor);
    InsertActor(exit_actor.get());
  }

  return exit_actors;
}

std::vector<StackActorPtr> ControlNodeScheduler::BuildStackActor(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<StackActorPtr> stack_actors;
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  // Create a corresponding stack actor for each kernel graph that has a call node as input.
  for (const auto &kernel_graph_group_info : parser->kernel_graph_group_infos_) {
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
        MS_LOG(EXCEPTION) << "Failed to get level by from node:" << from_node->DebugString()
                          << " in graph:" << kernel_graph_group_info->group_name_;
      }
      if (iter->second == kernel_graph_group_info->level_ &&
          (!(parser->IsRootGraphParameter(from_node) && IsPersistentDeviceTensor(from_node)))) {
        (void)formal_parameters.emplace_back(node_with_context.first);
        (void)device_contexts.emplace_back(node_with_context.second);
      } else {
        (void)formal_parameters.insert(formal_parameters.begin(), node_with_context.first);
        (void)device_contexts.insert(device_contexts.begin(), node_with_context.second);
        input_parameter_data_num++;
      }
    }
    const auto &stack_actor = std::make_shared<StackActor>(actor_name, memory_manager_aid_, formal_parameters);
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
                                                         std::vector<StackActorPtr> *const stack_actors) {
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  for (const auto &need_stack_control_node : parser->need_stack_control_nodes_) {
    MS_EXCEPTION_IF_NULL(need_stack_control_node);

    const auto &stack_actor_name = GetActorName(need_stack_control_node) + kStackActorNameSuffix;
    std::vector<KernelWithIndex> formal_parameters;
    std::vector<const DeviceContext *> device_contexts;
    size_t input_parameter_data_num = 0;
    size_t input_parameter_partials_num = 0;

    // Fetch the control actor of control node.
    std::string control_actor_name = "";
    if (AnfAlgo::CheckPrimitiveType(need_stack_control_node, prim::kPrimReturn)) {
      const auto &func_graph = need_stack_control_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      control_actor_name = func_graph->ToString() + kExitActorNameSuffix;
    } else if (AnfAlgo::CheckPrimitiveType(need_stack_control_node, prim::kPrimPartial) ||
               AnfAlgo::CheckPrimitiveType(need_stack_control_node, prim::kPrimSwitch) ||
               AnfAlgo::CheckPrimitiveType(need_stack_control_node, prim::kPrimSwitchLayer) ||
               AnfAlgo::IsCallNode(need_stack_control_node)) {
      control_actor_name = GetActorName(need_stack_control_node);
    } else {
      MS_LOG(EXCEPTION) << "Invalid control node:" << need_stack_control_node->DebugString();
    }

    auto iter = parser->node_to_level_.find(need_stack_control_node);
    if (iter == parser->node_to_level_.end()) {
      MS_LOG(EXCEPTION) << "Failed to get level for need stack control node:" << need_stack_control_node->DebugString();
    }
    size_t control_node_level = iter->second;

    auto actor = FetchActor(control_actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto control_actor = dynamic_cast<ControlActor *>(actor);
    MS_EXCEPTION_IF_NULL(control_actor);
    if (control_actor->formal_parameters_.size() > control_actor->device_contexts_.size()) {
      MS_LOG(EXCEPTION) << "Invalid device context size:" << control_actor->device_contexts_.size()
                        << " and formal parameter size:" << control_actor->formal_parameters_.size()
                        << " for actor:" << control_actor->GetAID();
    }

    // Collect formal parameters and device contexts, skip the value nodes.
    for (size_t i = 0; i < control_actor->formal_parameters_.size(); ++i) {
      const auto &parameter = control_actor->formal_parameters_[i];
      auto device_context = control_actor->device_contexts_[i];
      if (parameter.first->isa<ValueNode>()) {
        continue;
      }

      iter = parser->node_to_level_.find(parameter.first);
      if (iter == parser->node_to_level_.end()) {
        MS_LOG(EXCEPTION) << "Failed to get level for formal parameter:" << parameter.first->DebugString()
                          << " for need stack control node:" << need_stack_control_node->DebugString();
      }

      if (control_node_level == iter->second &&
          (!(parser->IsRootGraphParameter(parameter.first) && IsPersistentDeviceTensor(parameter.first)))) {
        (void)formal_parameters.emplace_back(parameter);
        (void)device_contexts.emplace_back(device_context);
      } else {
        (void)formal_parameters.insert(formal_parameters.begin(), parameter);
        (void)device_contexts.insert(device_contexts.begin(), device_context);

        const auto &abstract = parameter.first->abstract();
        MS_EXCEPTION_IF_NULL(abstract);
        const auto &real_abstract = FetchAbstractByIndex(abstract, parameter.second);
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
    stack_actor->device_contexts_ = device_contexts;
    stack_actor->input_stack_data_num_ = input_parameter_data_num;
    stack_actor->input_stack_partials_num_ = input_parameter_partials_num;

    InsertActor(stack_actor.get());
    (void)stack_actors->emplace_back(stack_actor);
  }
}

void ControlNodeScheduler::Link(ActorSet *const actor_set, const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(actor_set->control_actors_);

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
}

void ControlNodeScheduler::ClearActorData(const ControlActorSet *control_actor_set) {
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
                                                    const GraphCompilerInfo &graph_compiler_info) {
  if (control_actor_set == nullptr) {
    return;
  }

  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
  const auto &parser = graph_compiler_info.control_node_parser_;
  for (auto &switch_actor : control_actor_set->switch_actors_) {
    MS_EXCEPTION_IF_NULL(switch_actor);
    if (parser->need_stack_control_nodes_.find(switch_actor->node_) == parser->need_stack_control_nodes_.end()) {
      for (size_t i = 0; i < switch_actor->formal_parameters_.size(); ++i) {
        LinkArrowbyFormalParameter(switch_actor.get(), switch_actor->formal_parameters_[i], {switch_actor->node_, i},
                                   parser);
      }
    } else {
      // If the control actor has a corresponding stack actor, the input should be linked to the stack actor.
      auto stack_actor_name = GetActorName(switch_actor->node_) + kStackActorNameSuffix;
      auto actor = FetchActor(stack_actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto stack_actor = dynamic_cast<StackActor *>(actor);
      MS_EXCEPTION_IF_NULL(stack_actor);
      LinkArrowFromStackActor(stack_actor, switch_actor.get());
    }
  }

  for (auto &gather_actor : control_actor_set->gather_actors_) {
    MS_EXCEPTION_IF_NULL(gather_actor->node_);
    if (parser->need_stack_control_nodes_.find(gather_actor->node_) == parser->need_stack_control_nodes_.end()) {
      for (size_t i = 0; i < gather_actor->formal_parameters_.size(); ++i) {
        LinkArrowbyFormalParameter(gather_actor.get(), gather_actor->formal_parameters_[i], {gather_actor->node_, i},
                                   parser);
      }
    } else {
      // If the control actor has a corresponding stack actor, the input should be linked to the stack actor.
      auto stack_actor_name = GetActorName(gather_actor->node_) + kStackActorNameSuffix;
      auto actor = FetchActor(stack_actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto stack_actor = dynamic_cast<StackActor *>(actor);
      MS_EXCEPTION_IF_NULL(stack_actor);
      LinkArrowFromStackActor(stack_actor, gather_actor.get());
    }
  }

  for (auto &entrance_actor : control_actor_set->entrance_actors_) {
    for (const auto &call_node : entrance_actor->call_nodes_) {
      LinkArrowbyFormalParameter(entrance_actor.get(), call_node, {entrance_actor->node_, 0}, parser);
    }
  }

  for (auto &exit_actor : control_actor_set->exit_actors_) {
    MS_EXCEPTION_IF_NULL(exit_actor);
    if (exit_actor->node_ == nullptr ||
        (parser->need_stack_control_nodes_.find(exit_actor->node_) == parser->need_stack_control_nodes_.end())) {
      for (size_t i = 0; i < exit_actor->formal_parameters_.size(); ++i) {
        LinkArrowbyFormalParameter(exit_actor.get(), exit_actor->formal_parameters_[i], {exit_actor->node_, i}, parser);
      }
    } else {
      // If the control actor has a corresponding stack actor, the input should be linked to the stack actor.
      auto stack_actor_name = (exit_actor->node_ == nullptr ? GetStackActorNameByExitName(exit_actor->GetAID().Name())
                                                            : GetActorName(exit_actor->node_) + kStackActorNameSuffix);
      auto actor = FetchActor(stack_actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto stack_actor = dynamic_cast<StackActor *>(actor);
      MS_EXCEPTION_IF_NULL(stack_actor);
      LinkArrowFromStackActor(stack_actor, exit_actor.get());
    }
  }

  for (auto &stack_actor : control_actor_set->stack_actors_) {
    for (size_t i = 0; i < stack_actor->formal_parameters_.size(); ++i) {
      LinkArrowbyFormalParameter(stack_actor.get(), stack_actor->formal_parameters_[i], {stack_actor->node_, i},
                                 parser);
    }
  }
}

void ControlNodeScheduler::LinkArrowFromStackActor(StackActor *const stack_actor, ControlActor *const to_actor) {
  MS_EXCEPTION_IF_NULL(stack_actor);
  MS_EXCEPTION_IF_NULL(to_actor);

  for (size_t to_index = 0; to_index < to_actor->formal_parameters_.size(); ++to_index) {
    const auto &formal_parameter = to_actor->formal_parameters_[to_index];
    const auto &from_node = formal_parameter.first;
    if (from_node->isa<ValueNode>()) {
      LinkArrowByValueNode(from_node, to_actor, formal_parameter.second, to_index);
      continue;
    }

    // Fetch the arrow type of input.
    size_t from_index = stack_actor->FetchNodePosition(formal_parameter);
    const auto &abstract = formal_parameter.first->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    const auto &real_abstract = FetchAbstractByIndex(abstract, formal_parameter.second);
    MS_EXCEPTION_IF_NULL(real_abstract);

    // Link arrow according to abstract.
    if (real_abstract->isa<abstract::AbstractFunction>()) {
      LinkPartialArrow(stack_actor, to_actor, from_index, to_index);
    } else {
      LinkDataArrow(stack_actor, to_actor, from_index, to_index);
    }
  }
}

void ControlNodeScheduler::LinkArrowbyFormalParameter(ControlActor *const to_actor,
                                                      const KernelWithIndex &from_node_with_index,
                                                      const KernelWithIndex &to_node_with_index,
                                                      const ControlNodeParserPtr &parser) {
  const auto &from_node = from_node_with_index.first;
  MS_EXCEPTION_IF_NULL(from_node);
  if (from_node->isa<ValueNode>()) {
    LinkArrowByValueNode(from_node, to_actor, from_node_with_index.second, to_node_with_index.second);
  } else if (from_node->isa<Parameter>()) {
    LinkArrowByParameter(from_node, to_actor, from_node_with_index, to_node_with_index, parser);
  } else if (AnfAlgo::IsCallNode(from_node)) {
    // Link arrow by call node.
    LinkArrowByCallNode(from_node, to_actor, from_node_with_index, to_node_with_index, parser);
  } else if (AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimSwitch) ||
             AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimSwitchLayer)) {
    // Link arrow from switch actor.
    const auto &actor_name = GetActorName(from_node);
    const auto &actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    const auto &switch_actor = dynamic_cast<SwitchActor *>(actor);
    MS_EXCEPTION_IF_NULL(switch_actor);

    const auto &abstract = from_node->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    if (abstract->isa<abstract::AbstractFunction>()) {
      LinkPartialArrow(switch_actor, to_actor, from_node_with_index.second, to_node_with_index.second);
    } else {
      LinkDataArrow(switch_actor, to_actor, from_node_with_index.second, to_node_with_index.second);
    }
  } else if (AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimPartial)) {
    // Link arrow from gather actor
    const auto &actor_name = GetActorName(from_node);
    const auto &actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    const auto &gather_actor = dynamic_cast<GatherActor *>(actor);
    MS_EXCEPTION_IF_NULL(gather_actor);
    LinkPartialArrow(gather_actor, to_actor, from_node_with_index.second, to_node_with_index.second);
  } else if (from_node->isa<CNode>()) {
    // Link arrow by kernel.
    LinkArrowByKernel(from_node, to_actor, from_node_with_index, to_node_with_index, parser);
  }
}

void ControlNodeScheduler::LinkArrowByValueNode(const AnfNodePtr &value_node, ControlActor *const to_actor,
                                                size_t from_index, size_t to_index) {
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(to_actor);

  if (IsValueNode<FuncGraph>(value_node)) {
    // Link local partial.
    const auto &func_graph = GetValueNode<FuncGraphPtr>(value_node);
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
        MS_LOG(EXCEPTION) << "Invalid output address index:" << from_index
                          << " for value node:" << value_node->DebugString() << " to actor:" << to_actor->GetAID();
      }
    }
    to_actor->local_device_tensors_[to_index] = AnfAlgo::GetMutableOutputAddr(value_node, from_index, false).get();
    to_actor->local_device_tensors_[to_index]->SetNodeIndex(value_node, from_index);
  }
}

void ControlNodeScheduler::LinkArrowByParameter(const AnfNodePtr &parameter, ControlActor *const to_actor,
                                                const KernelWithIndex &from_node_with_index,
                                                const KernelWithIndex &to_node_with_index,
                                                const ControlNodeParserPtr &parser) {
  if (parser->IsRootGraphParameter(parameter) && IsPersistentDeviceTensor(parameter)) {
    (void)to_actor->device_tensor_store_keys_.emplace_back(to_node_with_index.second, parameter);
    return;
  }
  // Link arrow from entrance actor.
  const auto &func_graph = parameter->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
  auto actor = FetchActor(actor_name);
  MS_EXCEPTION_IF_NULL(actor);
  auto entrance_actor = dynamic_cast<EntranceActor *>(actor);
  MS_EXCEPTION_IF_NULL(entrance_actor);

  auto abstract = parameter->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  auto dst_abstract = FetchAbstractByIndex(abstract, from_node_with_index.second);
  if (dst_abstract->isa<abstract::AbstractFunction>()) {
    LinkPartialArrow(entrance_actor, to_actor, entrance_actor->FetchNodePosition(from_node_with_index),
                     to_node_with_index.second);
  } else {
    LinkDataArrow(entrance_actor, to_actor, entrance_actor->FetchNodePosition(from_node_with_index),
                  to_node_with_index.second);
  }
}

void ControlNodeScheduler::LinkArrowByCallNode(const AnfNodePtr &call_node, ControlActor *const to_actor,
                                               const KernelWithIndex &from_node_with_index,
                                               const KernelWithIndex &to_node_with_index,
                                               const ControlNodeParserPtr &parser) {
  MS_EXCEPTION_IF_NULL(call_node);
  MS_EXCEPTION_IF_NULL(to_actor);
  const auto &from_node = from_node_with_index.first;
  MS_EXCEPTION_IF_NULL(from_node);

  if (to_actor->type_ != KernelTransformType::kEntranceActor) {
    // Link arrow from exit actor to control actor.
    const auto &abstract = call_node->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    const auto &real_abstract = FetchAbstractByIndex(abstract, from_node_with_index.second);
    MS_EXCEPTION_IF_NULL(real_abstract);

    const auto &func_graphs = parser->FetchFuncGraphbyCallNode(from_node);
    for (const auto &func_graph : func_graphs) {
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &actor_name = func_graph->ToString() + kExitActorNameSuffix;
      auto actor = FetchActor(actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto exit_actor = dynamic_cast<ExitActor *>(actor);
      auto branch_id = parser->FetchBranchIDByCallNode(from_node);
      if (real_abstract->isa<abstract::AbstractFunction>()) {
        LinkPartialArrowForExitActor(exit_actor, to_actor, from_node_with_index.second, to_node_with_index.second,
                                     branch_id);
      } else {
        LinkDataArrowForExitActor(exit_actor, to_actor, from_node_with_index.second, to_node_with_index.second,
                                  branch_id);
      }
    }
    if (real_abstract->isa<abstract::AbstractFunction>()) {
      to_actor->input_partials_num_++;
    } else {
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
    LinkDataWithBranchIDArrow(gather_actor, dynamic_cast<EntranceActor *>(to_actor), func_graph);
  }
}

void ControlNodeScheduler::LinkArrowByKernel(const AnfNodePtr &kernel, ControlActor *const to_actor,
                                             const KernelWithIndex &from_node_with_index,
                                             const KernelWithIndex &to_node_with_index,
                                             const ControlNodeParserPtr &parser) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(to_actor);
  const auto &from_node = from_node_with_index.first;
  MS_EXCEPTION_IF_NULL(from_node);
  const auto &graph = parser->FetchKernelGraphByFrontNode(from_node);
  MS_EXCEPTION_IF_NULL(graph);

  if (to_actor->type_ == KernelTransformType::kExitActor && to_actor->node_ == nullptr) {
    // Link arrow from actor of output node to exit actor of kernel graph.
    const auto &kernel_with_index = parser->FetchBackendNodeByFrontNode(from_node_with_index);
    MS_EXCEPTION_IF_NULL(kernel_with_index.first);
    auto type = FetchKernelTransformType(kernel_with_index.first, graph, {});
    auto from_actor = FetchActor(type, "", kernel_with_index.first, graph);
    MS_EXCEPTION_IF_NULL(from_actor);
    if (!AnfAlgo::OutputAddrExist(kernel_with_index.first, kernel_with_index.second, false)) {
      MS_LOG(EXCEPTION) << "Invalid output index:" << kernel_with_index.second
                        << " for parameter:" << kernel_with_index.first->DebugString();
    }
    auto device_tensor = AnfAlgo::GetMutableOutputAddr(kernel_with_index.first, kernel_with_index.second, false);
    UpdateRefCount(device_tensor.get(), true);
    device_tensor->SetNodeIndex(kernel_with_index.first, kernel_with_index.second);
    LinkDataArrow(from_actor, to_actor, kernel_with_index.second, to_node_with_index.second, kernel_with_index.first);
  } else {
    // Link arrow from exit actor of kernel graph to exit actor of function graph.
    const auto &actor_name = parser->FetchGroupNameByKernelGraph(graph) + kExitActorNameSuffix;
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto exit_actor = dynamic_cast<ExitActor *>(actor);
    MS_EXCEPTION_IF_NULL(exit_actor);
    size_t from_index = exit_actor->FetchNodePosition(from_node_with_index);
    LinkDataArrow(exit_actor, to_actor, from_index, to_node_with_index.second);
  }
}

void ControlNodeScheduler::LinkControlArrowForControlActor(ActorSet *const actor_set,
                                                           const GraphCompilerInfo &graph_compiler_info) {
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
      if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
        actor_name = func_graph->ToString() + kExitActorNameSuffix;
      } else {
        actor_name = GetActorName(node);
      }
      auto from_actor = dynamic_cast<ControlActor *>(FetchActor(actor_name));
      MS_EXCEPTION_IF_NULL(from_actor);
      LinkLoopBodyControlArrow(from_actor, entrance_actor);
    }
  }

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
      LinkControlArrow(entrance_actor, control_actor);
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

    auto from_actor = control_actor;
    if (parser->need_stack_control_nodes_.find(node) != parser->need_stack_control_nodes_.end()) {
      const auto &stack_actor_name = GetActorName(node) + kStackActorNameSuffix;
      auto actor = FetchActor(stack_actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      from_actor = dynamic_cast<ControlActor *>(actor);
      MS_EXCEPTION_IF_NULL(from_actor);
    }

    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &inputs = cnode->inputs();
    for (const auto &input : inputs) {
      MS_EXCEPTION_IF_NULL(input);
      if (AnfAlgo::CheckPrimitiveType(input, prim::kPrimUpdateState) ||
          AnfAlgo::CheckPrimitiveType(input, prim::kPrimDepend) ||
          AnfAlgo::CheckPrimitiveType(input, prim::kPrimLoad)) {
        LinkControlArrowByAutoMonad(from_actor, input, parser);
      }
    }
  }

  for (const auto &copy_actor : actor_set->copy_actors_) {
    if (copy_actor->output_data_arrows_.size() != 0 || copy_actor->output_control_arrows_.size() != 0) {
      continue;
    }
    if (copy_actor->input_control_arrow_aids_.empty()) {
      MS_LOG(EXCEPTION) << "Invalid copy actor:" << copy_actor->GetAID();
    }
    auto from_actor = FetchActor(copy_actor->input_control_arrow_aids_[0].Name());
    MS_EXCEPTION_IF_NULL(from_actor);
    auto kernel_actor = dynamic_cast<KernelActor *>(from_actor);
    MS_EXCEPTION_IF_NULL(kernel_actor);
    MS_EXCEPTION_IF_NULL(kernel_actor->kernel_);
    auto graph = kernel_actor->kernel_->func_graph();
    MS_EXCEPTION_IF_NULL(graph);
    auto kernel_graph = std::dynamic_pointer_cast<KernelGraph>(graph);
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto exit_actor_name = parser->FetchGroupNameByKernelGraph(kernel_graph) + kExitActorNameSuffix;
    auto exit_actor = FetchActor(exit_actor_name);
    MS_EXCEPTION_IF_NULL(exit_actor);
    LinkControlArrow(copy_actor.get(), exit_actor);
  }
}

void ControlNodeScheduler::LinkControlArrowForLoopCountActor(const ActorSet *actor_set,
                                                             const GraphCompilerInfo &graph_compiler_info) {
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
  LinkControlArrowForExitActor(root_exit_actor, loop_count_actor.get(), kMainBranchID);

  // The entrance actor will generate some data in the loop body execution, so need clear on the end of step.
  MS_EXCEPTION_IF_NULL(actor_set->control_actors_);
  for (auto &entrance_actor : actor_set->control_actors_->entrance_actors_) {
    MS_EXCEPTION_IF_NULL(entrance_actor);
    (void)loop_count_actor->entrance_aids_.emplace_back(entrance_actor->GetAID());
  }
}

void ControlNodeScheduler::LinkControlArrowForKernelActor(ActorSet *const actor_set,
                                                          const GraphCompilerInfo &graph_compiler_info) {
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  // Link control arrow from entrance actors or stack actors to no input kernel actors.
  for (const auto &no_input_kernel_actor : actor_set->no_input_kernel_actors_) {
    // In control flow, when the input of the kernel actor is a parameter, this input needs to be linked to the
    // control actor, so the no-input kernel actor collected in the graph scheduler will also collect this actor,
    // and it needs to be skipped here.
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
      kernel_graph = FetchKernelGraph(kernel_actor->kernel());
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
    LinkControlArrow(from_actor, no_input_kernel_actor.get());
  }

  // Link control arrows from no output kernel actor to the corresponding exit actor.
  for (auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    if ((kernel_actor->output_data_arrows_.size() == 0) && (kernel_actor->output_control_arrows_.size() == 0)) {
      auto kernel_graph = FetchKernelGraph(kernel_actor->kernel());
      MS_EXCEPTION_IF_NULL(kernel_graph);
      auto to_actor_name = parser->FetchGroupNameByKernelGraph(kernel_graph) + kExitActorNameSuffix;
      auto to_actor = FetchActor(to_actor_name);
      MS_EXCEPTION_IF_NULL(to_actor);
      LinkControlArrow(kernel_actor.get(), to_actor);
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
      LinkControlArrow(super_actor.get(), to_actor);
    }
  }
}

void ControlNodeScheduler::LinkControlArrowByAutoMonad(ControlActor *to_actor, const AnfNodePtr &from_node,
                                                       const ControlNodeParserPtr &parser) {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(from_node);
  MS_EXCEPTION_IF_NULL(parser);

  std::set<AnfNodePtr> depend_nodes;
  FetchRealDependNodeByAutoMonad(from_node, &depend_nodes);

  for (const auto &depend_node : depend_nodes) {
    MS_EXCEPTION_IF_NULL(depend_node);
    auto from_actor = FetchActor(depend_node->DebugString());
    auto graph = parser->FetchKernelGraphByFrontNode(depend_node);

    std::vector<AbstractActor *> from_actors;
    if (AnfAlgo::IsCallNode(depend_node)) {
      int branch_id = parser->FetchBranchIDByCallNode(depend_node);
      const auto &func_graphs = parser->FetchFuncGraphbyCallNode(depend_node);
      if (func_graphs.empty()) {
        MS_LOG(EXCEPTION) << "Failed to get funcgraph by call node:" << depend_node->DebugString();
      }
      for (const auto func_graph : func_graphs) {
        auto exit_actor_name = func_graph->ToString() + kExitActorNameSuffix;
        from_actor = FetchActor(exit_actor_name);
        MS_EXCEPTION_IF_NULL(from_actor);
        (void)from_actors.emplace_back(from_actor);
        auto exit_actor = dynamic_cast<ExitActor *>(from_actor);
        MS_EXCEPTION_IF_NULL(exit_actor);
        LinkControlArrowForExitActor(exit_actor, to_actor, branch_id);
      }
      to_actor->input_controls_num_ -= (func_graphs.size() - 1);
    } else if (from_actor != nullptr) {
      (void)from_actors.emplace_back(from_actor);
      LinkControlArrow(from_actor, to_actor);
    } else {
      if (graph == nullptr) {
        MS_LOG(EXCEPTION) << "Failed to find actor for node:" << depend_node->DebugString();
      }
      from_actor = FetchActor(parser->FetchGroupNameByKernelGraph(graph) + kExitActorNameSuffix);
      MS_EXCEPTION_IF_NULL(from_actor);
      if (find(from_actor->output_control_arrows_.begin(), from_actor->output_control_arrows_.end(),
               to_actor->GetAID()) != from_actor->output_control_arrows_.end()) {
        MS_LOG(DEBUG) << "Link auto monad control from actor:" << from_actor->GetAID()
                      << " to actor:" << to_actor->GetAID() << " is already exist.";
        continue;
      }
      (void)from_actors.emplace_back(from_actor);
      LinkControlArrow(from_actor, to_actor);
    }
    if (to_actor->type_ != KernelTransformType::kStackActor || parser->IsRecursionCallNode(depend_node) ||
        (graph != nullptr && parser->IsRecursionKernelGraph(graph))) {
      continue;
    }
    // If the control arrow comes from a recursive call node or a recursive kernel graph, these control edges will be
    // directly linked to the stack actor, otherwise, they need to be cached in the stack of the stack actor.
    auto stack_actor = dynamic_cast<StackActor *>(to_actor);
    MS_EXCEPTION_IF_NULL(stack_actor);
    stack_actor->input_controls_num_--;
    stack_actor->input_stack_controls_num_++;
    for (const auto &actor : from_actors) {
      (void)stack_actor->stack_control_aids_.emplace(actor->GetAID());
    }
  }
}

void ControlNodeScheduler::LinkBranchIDArrowForControlActor(ControlActorSet *const control_actor_set) {
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
    LinkBranchIDArrow(entrance_actor, exit_actor.get());
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
    LinkBranchIDArrow(entrance_actor, stack_actor.get());
  }
}

void ControlNodeScheduler::LinkDataArrowForKernelActor(const GraphCompilerInfo &graph_compiler_info) {
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

void ControlNodeScheduler::LinkDataArrowByKernelGraph(const KernelGraphPtr &graph, ControlActor *const entrance_actor,
                                                      const ControlNodeParserPtr &parser) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parser);

  auto from_actor = entrance_actor;
  // If there is a call node in the input of the graph, the parameter of the graph needs to be sent by the
  // corresponding stack actor, otherwise it is sent by the entrance actor.
  if (parser->IsCallInputKernelGraph(graph.get())) {
    auto actor = FetchActor(parser->FetchGroupNameByKernelGraph(graph) + kStackActorNameSuffix);
    MS_EXCEPTION_IF_NULL(actor);
    from_actor = dynamic_cast<ControlActor *>(actor);
  }

  std::set<AnfNodePtr> sink_input_node_linked;
  auto &execution_order = graph->execution_order();
  for (const auto &kernel : execution_order) {
    MS_EXCEPTION_IF_NULL(kernel);
    if ((!graph->is_executing_sink()) && (IsSkippedKernelActor(kernel) || !IsKernelActor(kernel))) {
      continue;
    }
    for (size_t i = 0; i < AnfAlgo::GetInputNum(kernel); ++i) {
      auto input_node = AnfAlgo::GetInputNode(kernel, i);
      MS_EXCEPTION_IF_NULL(input_node);
      auto input_with_index = AnfAlgo::VisitKernelWithReturnType(input_node, 0, false);
      auto input = input_with_index.first;
      MS_EXCEPTION_IF_NULL(input);
      if (sink_input_node_linked.count(input) > 0 || HasAbstractMonad(input) || parser == nullptr ||
          (!parser->IsControlFlowDataArrow(graph, input))) {
        continue;
      }
      auto front_node = graph->GetFrontAnfByBackendAnf(input);
      auto internal_node_with_index = graph->GetFrontNodeByInternalParameter(input);
      auto tuple_node_with_index = graph->GetElementInTupleBackendFrontIndexMap(input);
      KernelWithIndex from_node_with_index =
        (front_node == nullptr) ? internal_node_with_index : KernelWithIndex(front_node, 0);
      if (from_node_with_index.first == nullptr) {
        from_node_with_index = tuple_node_with_index;
      }

      if (AnfAlgo::CheckPrimitiveType(from_node_with_index.first, prim::kPrimTupleGetItem)) {
        MS_LOG(WARNING) << "Input node:" << from_node_with_index.first->DebugString()
                        << " for graph:" << graph->ToString() << " is a tuple get item";
        from_node_with_index = FetchRealNodeByGetItem(from_node_with_index);
      }

      // If the formal parameter is a tuple type, the parameter of the kernel graph will not directly correspond
      // to the front parameter, but the node in the internal parameter.
      const auto &from_node = from_node_with_index.first;

      // Fetch actor and link.
      auto type = FetchKernelTransformType(kernel, graph, {});
      auto to_actor = FetchActor(type, "", kernel, graph);
      MS_EXCEPTION_IF_NULL(to_actor);
      size_t from_index = 0;
      if (AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimSwitch) ||
          AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimSwitchLayer)) {
        const auto &actor_name = GetActorName(from_node);
        auto actor = FetchActor(actor_name);
        MS_EXCEPTION_IF_NULL(actor);
        from_actor = dynamic_cast<ControlActor *>(actor);
      } else {
        from_index = from_actor->FetchNodePosition(from_node_with_index);
      }

      MS_EXCEPTION_IF_NULL(from_actor);
      auto to_index = i;
      if (type == KernelTransformType::kSuperKernelActor) {
        auto super_kernel_actor = dynamic_cast<SuperKernelActor *>(to_actor);
        MS_EXCEPTION_IF_NULL(super_kernel_actor);
        to_index = super_kernel_actor->FetchInputNodePosition(input);
        (void)sink_input_node_linked.insert(input);
      }
      AddFormalParameterDeviceTensor(from_actor, from_index, input);
      LinkDataArrow(from_actor, to_actor, from_index, to_index);
    }
  }
}

void ControlNodeScheduler::LinkDataArrowForOutputActor(ActorSet *const actor_set,
                                                       const GraphCompilerInfo &graph_compiler_info) {
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
    LinkDataArrowForExitActor(exit_actor, to_actor.get(), i, i, 0);
    to_actor->input_datas_num_++;
  }

  auto control_node_to_device_contexts = parser->control_node_to_device_contexts_;
  auto iter = control_node_to_device_contexts.find(return_node);
  if (iter == control_node_to_device_contexts.end()) {
    MS_LOG(EXCEPTION) << "Failed to find device contexts for node:" << return_node->DebugString();
  }
  if (iter->second.size() != actor_set->output_actor_->device_contexts().size()) {
    MS_LOG(EXCEPTION) << "Invalid context size, need:" << actor_set->output_actor_->device_contexts().size()
                      << " current:" << iter->second.size();
  }
  actor_set->output_actor_->device_contexts_ = iter->second;
}

void ControlNodeScheduler::LinkArrowForRootGraphEntranceActor(const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
  const auto &root_graph = graph_compiler_info.control_node_parser_->root_func_graph_;
  MS_EXCEPTION_IF_NULL(root_graph);
  const auto &entrance_actor_name = root_graph->ToString() + kEntranceActorNameSuffix;
  auto to_actor = dynamic_cast<EntranceActor *>(FetchActor(entrance_actor_name));
  MS_EXCEPTION_IF_NULL(to_actor);

  const auto &host_ds_actor_name = graph_compiler_info.name_ + "_HostDSActor";
  auto host_ds_actor = dynamic_cast<HostQueueDataSourceActor *>(FetchActor(host_ds_actor_name));
  // No host data source actor scenario.
  if (host_ds_actor == nullptr) {
    const auto &data_prepare_actor_name = graph_compiler_info.name_ + "_DataPrepareActor";
    auto data_prepare_actor = FetchActor(data_prepare_actor_name);
    MS_EXCEPTION_IF_NULL(data_prepare_actor);
    LinkControlArrow(data_prepare_actor, to_actor);
    return;
  }

  // The host data source actor sends all the input to the entrance actor of the root graph.
  for (size_t i = 0; i < to_actor->formal_parameters_.size(); ++i) {
    const auto &formal_parameter = to_actor->formal_parameters_[i];
    MS_EXCEPTION_IF_NULL(formal_parameter.first);
    const auto &iter = host_ds_actor->data_node_position_map_.find(formal_parameter.first);
    if (iter != host_ds_actor->data_node_position_map_.end()) {
      const auto &parameter = host_ds_actor->data_nodes()[iter->second];
      LinkDataArrow(host_ds_actor, to_actor, iter->second, i, parameter);
      if (!AnfAlgo::OutputAddrExist(parameter, 0, false)) {
        MS_LOG(EXCEPTION) << "Invalid output index:" << 0 << " for parameter:" << parameter->DebugString();
      }
      auto device_tensor = AnfAlgo::GetMutableOutputAddr(parameter, 0, false);
      UpdateRefCount(device_tensor.get(), true);
      device_tensor->SetNodeIndex(parameter, 0);
    }
  }
}

void ControlNodeScheduler::AddFormalParameterDeviceTensor(ControlActor *const from_actor, size_t from_index,
                                                          const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(input_node);
  if (!HasAbstractRef(input_node)) {
    return;
  }

  auto device_tensor = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
  MS_EXCEPTION_IF_NULL(device_tensor);
  (void)from_actor->ref_formal_parameter_device_tensors_[from_index].insert(device_tensor);

  UpdateRefCount(device_tensor.get(), true);
  device_tensor->SetNodeIndex(input_node, 0);
}

void ControlNodeScheduler::LinkDataArrow(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                         size_t from_index, size_t to_index, const AnfNodePtr &from_kernel) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  if (from_actor->type() == KernelTransformType::kKernelActor && to_actor->type() != KernelTransformType::kExitActor) {
    MS_LOG(WARNING) << "Kernel actor:" << from_actor->GetAID() << " link data arrow to actor:" << to_actor->GetAID()
                    << " is not an exit actor.";
  }
  auto data_arrow = std::make_shared<DataArrow>(from_index, to_actor->GetAID(), to_index);
  (void)from_actor->output_data_arrows_.emplace_back(data_arrow);
  (void)from_actor->output_data_nodes_.emplace_back(from_kernel);
  to_actor->input_datas_num_++;
  (void)to_actor->input_data_arrow_aids_.emplace_back(from_actor->GetAID());
}

void ControlNodeScheduler::LinkControlArrow(AbstractActor *const from_actor, AbstractActor *to_actor) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  (void)from_actor->output_control_arrows_.emplace_back(to_actor->GetAID());
  to_actor->input_controls_num_++;
  (void)to_actor->input_control_arrow_aids_.emplace_back(from_actor->GetAID());
}

void ControlNodeScheduler::LinkLoopBodyControlArrow(AbstractActor *from_actor, EntranceActor *to_actor) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  (void)from_actor->output_control_arrows_.emplace_back(to_actor->GetAID());
  to_actor->loop_body_input_controls_nums_++;
  (void)to_actor->loop_body_input_control_arrow_aids_.emplace_back(from_actor->GetAID());
}

void ControlNodeScheduler::LinkDataArrowForExitActor(ExitActor *const exit_actor, AbstractActor *const to_actor,
                                                     size_t from_index, size_t to_index, int branch_id) {
  MS_EXCEPTION_IF_NULL(exit_actor);
  MS_EXCEPTION_IF_NULL(to_actor);

  auto data_arrow = std::make_shared<DataArrow>(from_index, to_actor->GetAID(), to_index);
  (void)exit_actor->output_branch_data_arrows_[branch_id].emplace_back(data_arrow);
  (void)to_actor->input_data_arrow_aids_.emplace_back(exit_actor->GetAID());
}

void ControlNodeScheduler::LinkPartialArrowForExitActor(ExitActor *const exit_actor, ControlActor *const to_actor,
                                                        size_t from_index, size_t to_index, int branch_id) {
  MS_EXCEPTION_IF_NULL(exit_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  auto partial_arrow = std::make_shared<DataArrow>(from_index, to_actor->GetAID(), to_index);
  (void)exit_actor->output_branch_partial_arrows_[branch_id].emplace_back(partial_arrow);
  (void)to_actor->input_partial_arrow_aids_.emplace_back(exit_actor->GetAID());
}

void ControlNodeScheduler::LinkControlArrowForExitActor(ExitActor *from_actor, AbstractActor *to_actor, int branch_id) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);

  (void)from_actor->output_branch_control_arrows_[branch_id].emplace_back(to_actor->GetAID());
  to_actor->input_controls_num_++;
  (void)to_actor->input_control_arrow_aids_.emplace_back(from_actor->GetAID());
}

void ControlNodeScheduler::LinkDataWithBranchIDArrow(GatherActor *const gather_actor,
                                                     EntranceActor *const entrance_actor,
                                                     const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(gather_actor);
  MS_EXCEPTION_IF_NULL(entrance_actor);
  (void)gather_actor->output_data_with_branch_id_arrows_[func_graph.get()].emplace_back(entrance_actor->GetAID());
}

void ControlNodeScheduler::LinkPartialArrow(ControlActor *const from_actor, ControlActor *const to_actor,
                                            size_t from_index, size_t to_index) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  auto op_arrow = std::make_shared<DataArrow>(from_index, to_actor->GetAID(), to_index);
  (void)from_actor->output_partial_arrows_.emplace_back(op_arrow);
  to_actor->input_partials_num_++;
  (void)to_actor->input_partial_arrow_aids_.emplace_back(from_actor->GetAID());
}

void ControlNodeScheduler::LinkBranchIDArrow(ControlActor *const from_actor, ControlActor *const to_actor) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  (void)from_actor->output_branch_id_arrows_.emplace_back(to_actor->GetAID());
  (void)to_actor->input_branch_id_arrow_aids_.emplace_back(from_actor->GetAID());
  to_actor->input_branch_ids_num_++;
}

bool ControlNodeScheduler::CheckActorValid(const ControlActorSetPtr &control_actor_set) {
  MS_EXCEPTION_IF_NULL(control_actor_set);
  for (const auto &gather_actor : control_actor_set->gather_actors_) {
    if (gather_actor->input_partials_num_ != 1) {
      MS_LOG(EXCEPTION) << "Invalid partial num:" << gather_actor->input_partials_num_
                        << " for actor:" << gather_actor->GetAID();
    }
  }
  return true;
}

bool ControlNodeScheduler::IsNoInputActor(const ControlActor *control_actor) const {
  return (control_actor->input_datas_num_ == 0 && control_actor->input_controls_num_ == 0 &&
          control_actor->input_partials_num_ == 0 && control_actor->input_branch_ids_num_ == 0);
}
}  // namespace runtime
}  // namespace mindspore
