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
// Get all the real input of the frontend node, skip the virtual node like maketuple, tuplegetitem.
std::vector<KernelWithIndex> FetchInputNodeByCNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return {};
  }

  std::vector<KernelWithIndex> results;
  // The first input of normal cnode is the primitive of node, and the real input starts from the second input,
  // but in control flow, the call node has no primitive, and the 0th input is funcgraph or partial.
  size_t input_start_pos = kCNodeInputStartPos;
  if (AnfAlgo::IsCallNode(node)) {
    input_start_pos = 0;
  }
  const auto &cnode = node->cast<CNodePtr>();
  const auto inputs = cnode->inputs();

  // The first branch of the input of the switch node is the true branch, and the second is the false branch.
  // But in switch actor, since the false value is 0, it corresponds to the first branch. Therefore, the input
  // of the switch node needs to exchange the positions of the two branches. So deal separately.
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch)) {
    if (inputs.size() != kSwitchInputNum) {
      MS_LOG(EXCEPTION) << "Invalid switch node:" << node->DebugString();
    }
    results.emplace_back(AnfAlgo::VisitKernelWithReturnType(inputs[kSwitchCondPos], 0));
    results.emplace_back(AnfAlgo::VisitKernelWithReturnType(inputs[kSwitchFalseBranchPos], 0));
    results.emplace_back(AnfAlgo::VisitKernelWithReturnType(inputs[kSwitchTrueBranchPos], 0));
    return results;
  }

  for (size_t i = input_start_pos; i < inputs.size(); ++i) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    // skip monad node.
    if (HasAbstractMonad(inputs[i])) {
      continue;
    }

    const auto &node_with_index =
      AnfAlgo::VisitKernelWithReturnType(inputs[i], 0, false, {prim::kPrimTupleGetItem, prim::kPrimMakeTuple});
    MS_EXCEPTION_IF_NULL(node_with_index.first);
    size_t output_num = AnfAlgo::GetOutputTensorNum(node_with_index.first);
    for (size_t j = 0; j < output_num; ++j) {
      if (AnfAlgo::CheckPrimitiveType(node_with_index.first, prim::kPrimMakeTuple)) {
        const auto &make_tuple_cnode = node_with_index.first->cast<CNodePtr>();
        const auto &make_tuple_inputs = make_tuple_cnode->inputs();
        if (make_tuple_inputs.size() <= j + kMakeTupleInputStartPos) {
          MS_LOG(EXCEPTION) << "Invalid input:" << j + kMakeTupleInputStartPos
                            << " for make tuple node:" << make_tuple_cnode->DebugString();
        }
        results.emplace_back(AnfAlgo::VisitKernelWithReturnType(make_tuple_inputs[j + kMakeTupleInputStartPos], 0));
      } else {
        results.emplace_back(AnfAlgo::VisitKernelWithReturnType(node_with_index.first, j));
      }
    }
  }
  return results;
}
}  // namespace

ControlActorSetPtr ControlNodeScheduler::Build(const GraphCompilerInfo &graph_compiler_info) {
  const auto &control_nodes = graph_compiler_info.control_nodes_;
  if (control_nodes.size() <= kSingleControlNode) {
    return nullptr;
  }

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
      const auto &actor_name = control_node->DebugString();
      const auto &parameters = FetchInputNodeByCNode(control_node);
      const auto &switch_actor = std::make_shared<SwitchActor>(actor_name, parameters, control_node);
      switch_actors.emplace_back(switch_actor);
      InsertActor(switch_actor.get());
    }
  }
  return switch_actors;
}

std::vector<GatherActorPtr> ControlNodeScheduler::BuildGatherActor(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<GatherActorPtr> gather_actors;
  const auto &control_nodes = graph_compiler_info.control_nodes_;
  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);

  for (const auto &control_node : control_nodes) {
    // Partial node and call node will be converted to gather actor.
    if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimPartial) || AnfAlgo::IsCallNode(control_node)) {
      const auto &actor_name = control_node->DebugString();
      const auto &parameters = FetchInputNodeByCNode(control_node);
      const auto &gather_actor = std::make_shared<GatherActor>(actor_name, parameters, control_node);
      gather_actors.emplace_back(gather_actor);
      InsertActor(gather_actor.get());

      // The gather actor corresponding to a call node needs to set the branch id.
      if (AnfAlgo::IsCallNode(control_node)) {
        gather_actor->output_branch_id_ = graph_compiler_info.control_node_parser_->GetBranchIDByCallNode(control_node);
      }
    }
  }
  return gather_actors;
}

std::vector<EntranceActorPtr> ControlNodeScheduler::BuildEntranceActor(const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
  const auto &call_node_to_func_graphs = graph_compiler_info.control_node_parser_->call_node_to_func_graphs_;
  std::unordered_map<FuncGraphPtr, std::set<KernelWithIndex>> func_graph_to_call_nodes;
  for (const auto &call_node_to_func_graph : call_node_to_func_graphs) {
    const auto &node = call_node_to_func_graph.first;
    for (const auto &func_graph : call_node_to_func_graph.second) {
      func_graph_to_call_nodes[func_graph].emplace(node, 0);
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
        if (!HasAbstractMonad(parameter)) {
          formal_parameters.emplace_back(parameter, 0);
        }
      }

      // 2. The caller of the subgraph, namely call nodes, is used to connect the input arrows.
      std::set<KernelWithIndex> call_nodes;
      const auto &iter = func_graph_to_call_nodes.find(func_graph);
      if (iter != func_graph_to_call_nodes.end()) {
        call_nodes = iter->second;
      }
      const auto &entrance_actor =
        std::make_shared<EntranceActor>(actor_name, formal_parameters, call_nodes, control_node);
      entrance_actors.emplace_back(entrance_actor);
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
      const auto &actor_name = control_node->DebugString();
      const auto &parameters = FetchInputNodeByCNode(control_node);
      const auto &exit_actor = std::make_shared<ExitActor>(actor_name, parameters, control_node);
      exit_actors.emplace_back(exit_actor);
      InsertActor(exit_actor.get());
    }
  }

  // 2. Replace the device address in the kernel actor when calling funcgraph, that is to say in the data exchange
  // between kernel graph and the control node, in fact, it is the output of the kernel graph.
  for (const auto func_graph_to_kernel_graphs : parser->func_graph_to_kernel_graphs_) {
    for (const auto &kernel_graph : func_graph_to_kernel_graphs.second) {
      MS_EXCEPTION_IF_NULL(kernel_graph);
      std::vector<KernelWithIndex> formal_parameters;
      const auto &graph_outputs = kernel_graph->graph_output_map();
      std::vector<const DeviceContext *> device_contexts;

      for (const auto &backend_to_front : graph_outputs) {
        // Collect inputs of exit actor.
        formal_parameters.emplace_back(backend_to_front.second);

        // Get the device contexts of the exit actor's cnode inputs.
        const AnfNodePtr &backend_node = backend_to_front.first.first;
        MS_EXCEPTION_IF_NULL(backend_node);
        if ((!backend_node->isa<CNode>())) {
          device_contexts.emplace_back(nullptr);
          continue;
        }

        const auto &actor_name = backend_node->fullname_with_scope();
        const auto &actor = FetchActor(actor_name);
        MS_EXCEPTION_IF_NULL(actor);
        const auto &kernel_actor = dynamic_cast<KernelActor *>(actor);
        MS_EXCEPTION_IF_NULL(kernel_actor);
        if (kernel_actor->device_contexts_.empty() || kernel_actor->device_contexts_[0] == nullptr) {
          MS_LOG(EXCEPTION) << "Failed to get device context for kernel:" << backend_node->DebugString();
        }
        device_contexts.emplace_back(kernel_actor->device_contexts_[0]);
      }

      const auto &actor_name = kernel_graph->ToString();
      const auto &exit_actor = std::make_shared<ExitActor>(actor_name, formal_parameters, nullptr);
      exit_actors.emplace_back(exit_actor);
      exit_actor->device_contexts_.swap(device_contexts);
      InsertActor(exit_actor.get());
    }
  }
  return exit_actors;
}

std::vector<StackActorPtr> ControlNodeScheduler::BuildStackActor(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<StackActorPtr> stack_actors;
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  std::vector<KernelWithIndex> formal_parameters;

  // Create a corresponding stack actor for each kernel graph that has a call node as input.
  for (const auto &graph_with_context : parser->call_input_kernel_graphs_) {
    const auto &graph = graph_with_context.first;
    const auto &device_context = graph_with_context.second;
    MS_EXCEPTION_IF_NULL(graph);
    MS_EXCEPTION_IF_NULL(device_context);
    const auto &real_parameters = graph->input_nodes();

    // Collect inputs of stack actor.
    for (const auto &parameter : real_parameters) {
      const auto &front_node_with_index = GetFrontNodeByKernelGraph(parameter, graph);
      MS_EXCEPTION_IF_NULL(front_node_with_index.first);
      formal_parameters.emplace_back(front_node_with_index);
    }

    const auto &actor_name = graph->ToString() + kStackActorNameSuffix;
    const auto &stack_actor = std::make_shared<StackActor>(actor_name, formal_parameters);
    stack_actors.emplace_back(stack_actor);
    stack_actor->device_contexts_.insert(stack_actor->device_contexts_.begin(), formal_parameters.size(),
                                         device_context);
    InsertActor(stack_actor.get());
  }
  return stack_actors;
}

void ControlNodeScheduler::Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(actor_set->control_actors_);

  // Link data arrows and partial arrows between control actors.
  LinkArrowForControlActor(actor_set->control_actors_.get(), graph_compiler_info);

  // Link output data arrows from control actors to output actor.
  LinkDataArrowForOutputActor(actor_set, graph_compiler_info);

  // Link data arrows from host data source actor to control actors.
  LinkDataArrowForHostDSActor(graph_compiler_info);

  // Link data arrows from entrance actors to kernel actors.
  LinkDataArrowForKernelActor(graph_compiler_info);

  // Link branch id arrows between control actors.
  LinkBranchIDArrowForControlActor(actor_set->control_actors_.get());

  // Link all control arrows between control actors.
  LinkControlArrowForControlActor(actor_set, graph_compiler_info);
}

void ControlNodeScheduler::LinkArrowForControlActor(ControlActorSet *const control_actor_set,
                                                    const GraphCompilerInfo &graph_compiler_info) {
  if (control_actor_set == nullptr) {
    return;
  }

  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
  const auto &parser = graph_compiler_info.control_node_parser_;
  for (auto &switch_actor : control_actor_set->switch_actors_) {
    for (size_t i = 0; i < switch_actor->formal_parameters_.size(); ++i) {
      LinkArrowbyFormalParameter(switch_actor.get(), switch_actor->formal_parameters_[i], {switch_actor->node_, i},
                                 parser);
    }
  }

  for (auto &gather_actor : control_actor_set->gather_actors_) {
    for (size_t i = 0; i < gather_actor->formal_parameters_.size(); ++i) {
      LinkArrowbyFormalParameter(gather_actor.get(), gather_actor->formal_parameters_[i], {gather_actor->node_, i},
                                 parser);
    }
  }
  for (auto &entrance_actor : control_actor_set->entrance_actors_) {
    for (const auto &call_node : entrance_actor->call_nodes_) {
      LinkArrowbyFormalParameter(entrance_actor.get(), call_node, {entrance_actor->node_, 0}, parser);
    }
  }

  for (auto &exit_actor : control_actor_set->exit_actors_) {
    for (size_t i = 0; i < exit_actor->formal_parameters_.size(); ++i) {
      LinkArrowbyFormalParameter(exit_actor.get(), exit_actor->formal_parameters_[i], {exit_actor->node_, i}, parser);
    }
  }

  for (auto &stack_actor : control_actor_set->stack_actors_) {
    for (size_t i = 0; i < stack_actor->formal_parameters_.size(); ++i) {
      LinkArrowbyFormalParameter(stack_actor.get(), stack_actor->formal_parameters_[i], {stack_actor->node_, i},
                                 parser);
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
    if (IsValueNode<FuncGraph>(from_node)) {
      // Link local partial.
      const auto &func_graph = GetValueNode<FuncGraphPtr>(from_node);
      to_actor->local_partials_[to_node_with_index.second] = OpPartial(func_graph.get(), {});
    } else {
      // Link device store value node.
      to_actor->device_tensor_store_keys_.emplace_back(to_node_with_index.second, from_node.get());
    }
  } else if (from_node->isa<Parameter>()) {
    // Link arrow from entrance actor.
    const auto &func_graph = from_node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    const auto &actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto entrance_actor = dynamic_cast<EntranceActor *>(actor);
    MS_EXCEPTION_IF_NULL(entrance_actor);
    LinkDataArrow(entrance_actor, to_actor, entrance_actor->FetchNodePosition(from_node_with_index),
                  to_node_with_index.second);
  } else if (AnfAlgo::IsCallNode(from_node)) {
    // Link arrow by call node.
    LinkArrowByCallNode(from_node, to_actor, from_node_with_index, to_node_with_index, parser);
  } else if (AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimSwitch) ||
             AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimSwitchLayer)) {
    // Link arrow from switch actor.
    const auto &actor_name = from_node->DebugString();
    const auto &actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    const auto &switch_actor = dynamic_cast<SwitchActor *>(actor);
    MS_EXCEPTION_IF_NULL(switch_actor);
    LinkPartialArrow(switch_actor, to_actor, from_node_with_index.second, to_node_with_index.second);
  } else if (AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimPartial)) {
    // Link arrow from gather actor
    const auto &actor_name = from_node->DebugString();
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
    const auto &func_graphs = AnfAlgo::GetFuncGraphbyCallNode(from_node);
    for (const auto &func_graph : func_graphs) {
      const auto &return_node = func_graph->return_node();
      MS_EXCEPTION_IF_NULL(return_node);
      const auto &actor_name = return_node->DebugString();
      auto actor = FetchActor(actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto exit_actor = dynamic_cast<ExitActor *>(actor);
      size_t branch_id = parser->GetBranchIDByCallNode(from_node);
      LinkDataArrowForExitActor(exit_actor, to_actor, from_node_with_index.second, to_node_with_index.second,
                                branch_id);
    }
    to_actor->input_datas_num_++;
  } else {
    // Link arrow from gather actor to entrance actor.
    const auto &actor_name = from_node->DebugString();
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

  if (to_actor->type_ == KernelTransformType::kExitActor && to_actor->node_ == nullptr) {
    // Link arrow from kernel actor to exit actor.
    const auto &kernel_with_index = parser->FetchBackendNodeByFrontNode(from_node_with_index);
    MS_EXCEPTION_IF_NULL(kernel_with_index.first);
    const auto &actor_name = kernel_with_index.first->fullname_with_scope();
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto kernel_actor = dynamic_cast<KernelActor *>(actor);
    MS_EXCEPTION_IF_NULL(kernel_actor);

    if (!AnfAlgo::OutputAddrExist(kernel_with_index.first, kernel_with_index.second, false)) {
      MS_LOG(EXCEPTION) << "Invalid output index:" << kernel_with_index.second
                        << " for parameter:" << kernel_with_index.first->DebugString();
    }
    auto device_tensor = AnfAlgo::GetMutableOutputAddr(kernel_with_index.first, kernel_with_index.second, false);
    UpdateRefCount(device_tensor.get(), true);
    device_tensor->SetNodeIndex(kernel_with_index.first, kernel_with_index.second);

    kernel_actor->output_data_nodes_.emplace_back(kernel_with_index.first);
    LinkDataArrow(kernel_actor, to_actor, kernel_with_index.second, to_node_with_index.second);
  } else {
    // Link arrow from exit actor.
    const auto &graph = parser->FetchKernelGraphByFrontNode(from_node);
    const auto &actor_name = graph->ToString();
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
  // Get the exit actor of root graph, In control flow, the final output is always sent by the exit of the root graph.
  MS_EXCEPTION_IF_NULL(actor_set);
  auto control_actor_set = actor_set->control_actors_.get();
  MS_EXCEPTION_IF_NULL(control_actor_set);
  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
  const auto &root_graph = graph_compiler_info.control_node_parser_->root_func_graph_;
  MS_EXCEPTION_IF_NULL(root_graph);
  const auto &return_node = root_graph->return_node();
  MS_EXCEPTION_IF_NULL(return_node);
  const auto &exit_actor_name = return_node->DebugString();
  auto actor = FetchActor(exit_actor_name);
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(actor_set->loop_count_actor_);
  auto root_exit_actor = dynamic_cast<ExitActor *>(actor);
  // link control arrow from root exit actor to loop count actor.
  LinkControlArrowForExitActor(root_exit_actor, actor_set->loop_count_actor_.get(), kMainBranchID);

  // Since only one set of real parameters are allowed to be executed in funcgraph at the same time, when the funcgraph
  // stops running, it is necessary to send the control arrow to the corresponding entrance actor at the exit of the
  // graph to run the next set of real parameters. The corresponding nodes of the actors that need to send the control
  // arrow have been parsed in the control node parser.
  for (const auto &graph_to_nodes : graph_compiler_info.control_node_parser_->func_graph_to_first_control_nodes_) {
    // Fetch the entrance actor.
    const auto &func_graph = graph_to_nodes.first;
    MS_EXCEPTION_IF_NULL(func_graph);
    auto actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
    actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto entrance_actor = dynamic_cast<EntranceActor *>(actor);
    MS_EXCEPTION_IF_NULL(entrance_actor);

    const auto &nodes = graph_to_nodes.second;
    for (const auto &node : nodes) {
      // Fetch the source actor of control arrow.
      MS_EXCEPTION_IF_NULL(node);
      actor_name = node->DebugString();
      actor = FetchActor(actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto from_actor = dynamic_cast<ControlActor *>(actor);
      MS_EXCEPTION_IF_NULL(from_actor);
      LinkControlArrow(from_actor, entrance_actor);
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
    entrance_actor->output_branch_id_arrows_.emplace_back(exit_actor->GetAID());
  }
}

void ControlNodeScheduler::LinkDataArrowForKernelActor(const GraphCompilerInfo &graph_compiler_info) {
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  // Link data arrows from entrance actors and stack actors to kernel actors.
  for (const auto &func_graph_to_kernel_graphs : parser->func_graph_to_kernel_graphs_) {
    // Fetch the source entrance actor.
    const auto &func_graph = func_graph_to_kernel_graphs.first;
    MS_EXCEPTION_IF_NULL(func_graph);
    auto actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto entrance_actor = dynamic_cast<ControlActor *>(actor);
    MS_EXCEPTION_IF_NULL(entrance_actor);

    for (const auto &kernel_graph : func_graph_to_kernel_graphs.second) {
      MS_EXCEPTION_IF_NULL(kernel_graph);
      LinkDataArrowByKernelGraph(kernel_graph, parser->IsCallInputKernelGraph(kernel_graph), entrance_actor);
    }
  }
}

void ControlNodeScheduler::LinkDataArrowByKernelGraph(const KernelGraphPtr &graph, bool is_call_input_graph,
                                                      ControlActor *const entrance_actor) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &execution_order = graph->execution_order();

  for (const auto &kernel : execution_order) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (AnfAlgo::IsCommunicationOp(kernel) || IsSkippedKernelActor(kernel) || (!IsKernelActor(kernel))) {
      continue;
    }

    for (size_t i = 0; i < AnfAlgo::GetInputNum(kernel); ++i) {
      auto input = AnfAlgo::GetInputNode(kernel, i);
      if (!input->isa<Parameter>()) {
        continue;
      }

      auto front_node = graph->GetFrontAnfByBackendAnf(input);
      auto internal_node_with_index = graph->GetFrontNodeByInternalParameter(input);
      auto from_actor = entrance_actor;
      KernelWithIndex from_node_with_index =
        (front_node == nullptr ? internal_node_with_index : KernelWithIndex(front_node, 0));

      // If there is a call node in the input of the graph, the parameter of the graph needs to be sent by the
      // corresponding stack actor, otherwise it is sent by the entrance actor.
      if (is_call_input_graph) {
        auto actor = FetchActor(graph->ToString());
        MS_EXCEPTION_IF_NULL(actor);
        from_actor = dynamic_cast<ControlActor *>(actor);
        MS_EXCEPTION_IF_NULL(from_actor);
      } else if (front_node == nullptr) {
        continue;
      }

      // fetch the destine kernel actor.
      auto actor = FetchActor(kernel->fullname_with_scope());
      MS_EXCEPTION_IF_NULL(actor);
      auto kernel_actor = dynamic_cast<KernelActor *>(actor);
      MS_EXCEPTION_IF_NULL(kernel_actor);
      auto position = from_actor->FetchNodePosition(from_node_with_index);
      LinkDataArrow(from_actor, kernel_actor, position, i);
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
  const auto &return_node = root_graph->return_node();
  MS_EXCEPTION_IF_NULL(return_node);

  const auto &exit_actor_name = return_node->DebugString();
  auto actor = FetchActor(exit_actor_name);
  MS_EXCEPTION_IF_NULL(actor);
  auto exit_actor = dynamic_cast<ExitActor *>(actor);
  MS_EXCEPTION_IF_NULL(exit_actor);
  for (size_t i = 0; i < exit_actor->formal_parameters_.size(); ++i) {
    LinkDataArrow(exit_actor, to_actor.get(), i, i);
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

void ControlNodeScheduler::LinkDataArrowForHostDSActor(const GraphCompilerInfo &graph_compiler_info) {
  // In control flow, the host data source actor sends all the input to the entrance actor of the root graph.
  const auto &host_ds_actor_name = graph_compiler_info.name_ + "_HostDSActor";
  auto actor = FetchActor(host_ds_actor_name);
  MS_EXCEPTION_IF_NULL(actor);
  const auto host_ds_actor = dynamic_cast<HostQueueDataSourceActor *>(actor);
  MS_EXCEPTION_IF_NULL(host_ds_actor);

  const auto &root_graph = graph_compiler_info.control_node_parser_->root_func_graph_;
  MS_EXCEPTION_IF_NULL(root_graph);
  const auto &entrance_actor_name = root_graph->ToString() + kEntranceActorNameSuffix;
  actor = FetchActor(entrance_actor_name);
  MS_EXCEPTION_IF_NULL(actor);
  auto to_actor = dynamic_cast<EntranceActor *>(actor);

  for (size_t i = 0; i < to_actor->formal_parameters_.size(); ++i) {
    const auto &formal_parameter = to_actor->formal_parameters_[i];
    MS_EXCEPTION_IF_NULL(formal_parameter.first);
    const auto &iter = host_ds_actor->data_node_position_map_.find(formal_parameter.first);
    if (iter != host_ds_actor->data_node_position_map_.end()) {
      const auto &parameter = host_ds_actor->data_nodes()[iter->second];
      LinkDataArrow(host_ds_actor, to_actor, iter->second, i);

      // Set the source node to the device address.
      host_ds_actor->output_data_nodes_.emplace_back(parameter);
      if (!AnfAlgo::OutputAddrExist(parameter, 0, false)) {
        MS_LOG(EXCEPTION) << "Invalid output index:" << 0 << " for parameter:" << parameter->DebugString();
      }
      auto device_tensor = AnfAlgo::GetMutableOutputAddr(parameter, 0, false);
      UpdateRefCount(device_tensor.get(), true);
      device_tensor->SetNodeIndex(parameter, 0);
    }
  }
}

void ControlNodeScheduler::LinkDataArrow(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                         size_t from_index, size_t to_index) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);

  auto data_arrow = std::make_shared<DataArrow>(from_index, to_actor->GetAID(), to_index);
  (void)from_actor->output_data_arrows_.emplace_back(data_arrow);
  to_actor->input_datas_num_++;
  (void)to_actor->input_data_arrow_aids_.emplace_back(from_actor->GetAID());
}

void ControlNodeScheduler::LinkControlArrow(AbstractActor *from_actor, AbstractActor *to_actor) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  (void)from_actor->output_control_arrows_.emplace_back(to_actor->GetAID());
  to_actor->input_controls_num_++;
  (void)to_actor->input_control_arrow_aids_.emplace_back(from_actor->GetAID());
}

void ControlNodeScheduler::LinkDataArrowForExitActor(ExitActor *const exit_actor, ControlActor *const to_actor,
                                                     size_t from_index, size_t to_index, int branch_id) {
  MS_EXCEPTION_IF_NULL(exit_actor);
  MS_EXCEPTION_IF_NULL(to_actor);

  auto data_arrow = std::make_shared<DataArrow>(from_index, to_actor->GetAID(), to_index);
  (void)exit_actor->output_branch_data_arrows_[branch_id].emplace_back(data_arrow);
  (void)to_actor->input_data_arrow_aids_.emplace_back(exit_actor->GetAID());
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
  gather_actor->output_data_with_branch_id_arrows_[func_graph.get()].emplace_back(entrance_actor->GetAID());
}

void ControlNodeScheduler::LinkPartialArrow(ControlActor *const from_actor, ControlActor *const to_actor,
                                            size_t from_index, size_t to_index) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  auto op_arrow = std::make_shared<DataArrow>(from_index, to_actor->GetAID(), to_index);
  from_actor->output_partial_arrows_.emplace_back(op_arrow);
  to_actor->input_partials_num_++;
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
}  // namespace runtime
}  // namespace mindspore
