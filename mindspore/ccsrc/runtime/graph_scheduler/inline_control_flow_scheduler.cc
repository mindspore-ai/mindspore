/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/inline_control_flow_scheduler.h"
#include <vector>
#include "runtime/graph_scheduler/scheduler_helper.h"
#include "ops/framework_ops.h"

namespace mindspore {
namespace runtime {
bool IsInlineKernelActor(const AbstractActorPtr &actor) {
  MS_EXCEPTION_IF_NULL(actor);
  if (actor->type() != KernelTransformType::kKernelActor &&
      actor->type() != KernelTransformType::kConditionGatherActor &&
      actor->type() != KernelTransformType::kConditionSwitchActor) {
    return false;
  }
  const auto &kernel_actor = dynamic_cast<KernelActor *>(actor.get());
  MS_EXCEPTION_IF_NULL(kernel_actor);
  MS_EXCEPTION_IF_NULL(kernel_actor->kernel());
  const auto &func_graph = kernel_actor->kernel()->func_graph();
  if (func_graph == nullptr || (!func_graph->isa<KernelGraph>())) {
    return false;
  }
  const auto &kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  return kernel_graph->inline_sub_graph_kernels().find(kernel_actor->kernel()) !=
         kernel_graph->inline_sub_graph_kernels().end();
}

namespace {
std::string GetBranchNameByKernelActor(const KernelActor *const kernel_actor) {
  MS_EXCEPTION_IF_NULL(kernel_actor);
  MS_EXCEPTION_IF_NULL(kernel_actor->kernel());
  const auto &func_graph = kernel_actor->kernel()->func_graph();
  if (func_graph == nullptr || (!func_graph->isa<KernelGraph>())) {
    MS_LOG(EXCEPTION) << "Invalid funcgraph in kernel:" << kernel_actor->kernel()->fullname_with_scope();
  }
  const auto &kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  return kernel_graph->inline_sub_graph_kernels().at(kernel_actor->kernel());
}

void GetBranchNameToCondtionActor(const KernelGraphPtr &graph,
                                  mindspore::HashMap<std::string, AbstractActor *> *branch_name_to_switch_actor,
                                  mindspore::HashMap<std::string, AbstractActor *> *branch_name_to_gather_actor) {
  MS_EXCEPTION_IF_NULL(branch_name_to_gather_actor);
  MS_EXCEPTION_IF_NULL(branch_name_to_switch_actor);
  for (const auto &gather_to_switch : graph->condition_gather_to_switch()) {
    MS_EXCEPTION_IF_NULL(gather_to_switch.first);
    if (!common::AnfAlgo::CheckPrimitiveType(gather_to_switch.first, prim::kPrimConditionGather) ||
        !common::AnfAlgo::CheckPrimitiveType(gather_to_switch.second, prim::kPrimConditionSwitch)) {
      MS_LOG_WITH_NODE(EXCEPTION, gather_to_switch.first)
        << "Invalid condition gather node:" << gather_to_switch.first->DebugString()
        << " or condition switch node:" << gather_to_switch.second->DebugString();
    }
    const auto &gather_cnode = gather_to_switch.first->cast<CNodePtr>();
    const auto &switch_cnode = gather_to_switch.second->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(gather_cnode);
    MS_EXCEPTION_IF_NULL(switch_cnode);
    if (!gather_cnode->HasAttr(kAttrBranchGraphName)) {
      MS_LOG_WITH_NODE(EXCEPTION, gather_cnode)
        << "Failed to get inline graph name by node:" << gather_cnode->fullname_with_scope();
    }
    const auto &branch_graph_names = gather_cnode->GetAttr(kAttrBranchGraphName);
    MS_EXCEPTION_IF_NULL(branch_graph_names);
    MS_LOG(DEBUG) << "Branch graph name:" << branch_graph_names->ToString()
                  << " for node:" << gather_cnode->fullname_with_scope();
    if (!branch_graph_names->isa<ValueTuple>()) {
      MS_LOG_WITH_NODE(EXCEPTION, gather_cnode) << "Invalid branch group name:" << branch_graph_names->ToString()
                                                << " for node:" << gather_cnode->fullname_with_scope();
    }
    const auto &tuple_name = branch_graph_names->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_name);
    const auto &gather_actor = FetchActor(GetActorIdByKernel(gather_cnode));
    const auto &switch_actor = FetchActor(GetActorIdByKernel(switch_cnode));
    MS_EXCEPTION_IF_NULL(gather_actor);
    MS_EXCEPTION_IF_NULL(switch_actor);
    for (const auto &value : tuple_name->value()) {
      const auto &branch_name = GetValue<std::string>(value);
      (*branch_name_to_gather_actor)[branch_name] = gather_actor;
      (*branch_name_to_switch_actor)[branch_name] = switch_actor;
    }
  }
}
}  // namespace

void InlineControlFlowScheduler::LinkControlArrowByExecutionOrder(const KernelGraphPtr &graph,
                                                                  const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &inline_sub_graph_kernels = graph->inline_sub_graph_kernels();
  if (graph->is_graph_run_mode() || graph->is_any_type_input() || inline_sub_graph_kernels.empty()) {
    return;
  }

  mindspore::HashMap<std::string, AbstractActor *> branch_name_to_switch_actor;
  mindspore::HashMap<std::string, AbstractActor *> branch_name_to_gather_actor;
  GetBranchNameToCondtionActor(graph, &branch_name_to_switch_actor, &branch_name_to_gather_actor);

  MS_LOG(DEBUG) << "Link control arrow for graph:" << graph->ToString();
  // Only link control arrow between kernels in the same graph.
  mindspore::HashMap<std::string, AbstractActor *> branch_last_actor;
  for (size_t i = 0; i < graph->execution_order().size(); ++i) {
    const auto &to_kernel = graph->execution_order()[i];
    if (IsRpcActor(to_kernel)) {
      MS_LOG(INFO) << "Rpc op is not available in the execution order, from kernel: "
                   << graph->execution_order()[i - 1]->fullname_with_scope()
                   << ", to kernel:" << graph->execution_order()[i]->fullname_with_scope();
      continue;
    }
    const auto &iter = inline_sub_graph_kernels.find(to_kernel);
    std::string current_branch = graph->ToString();
    if (iter != inline_sub_graph_kernels.end()) {
      current_branch = iter->second;
      MS_LOG(DEBUG) << "Kernel:" << to_kernel->fullname_with_scope() << " branch:" << current_branch;
    }

    const auto to_kernel_type = FetchKernelTransformType(to_kernel, graph, {}, GraphExecutionStrategy::kPipeline);
    auto to_actor = FetchActor(to_kernel_type, graph_compiler_info.name_, to_kernel, graph);
    const auto &actor_iter = branch_last_actor.find(current_branch);
    if (actor_iter == branch_last_actor.end()) {
      if (!common::AnfAlgo::CheckPrimitiveType(to_kernel, prim::kPrimConditionSwitch)) {
        branch_last_actor[current_branch] = to_actor;
        MS_LOG(DEBUG) << "For branch:" << current_branch << " start actor:" << to_actor->GetAID();
      }
      continue;
    }
    MS_LOG(DEBUG) << "Add control arrow between " << actor_iter->second->GetAID() << " and " << to_actor->GetAID();
    SchedulerHelper::AddControlArrow(actor_iter->second, to_actor);
    if (common::AnfAlgo::CheckPrimitiveType(to_kernel, prim::kPrimConditionSwitch)) {
      // The control relation end after the condition switch node in graph.
      branch_last_actor.erase(current_branch);
      MS_LOG(DEBUG) << "For branch:" << current_branch << " end actor:" << to_actor->GetAID();
    } else {
      // The control relation start first kernel in graph.
      branch_last_actor[current_branch] = to_actor;
      MS_LOG(DEBUG) << "For branch:" << current_branch << " start actor:" << to_actor->GetAID();
    }
  }

  for (const auto &pair : branch_last_actor) {
    const auto &branch_name = pair.first;
    if (pair.second == nullptr || pair.second->type() != KernelTransformType::kKernelActor) {
      continue;
    }
    const auto &iter = branch_name_to_gather_actor.find(branch_name);
    if (iter == branch_name_to_gather_actor.end()) {
      MS_LOG(INFO) << "Invalid branch name:" << branch_name << " in graph:" << graph->ToString();
      continue;
    }
    SchedulerHelper::AddControlArrow(pair.second, iter->second);
    MS_LOG(DEBUG) << "Add control arrow between:" << pair.second->GetAID() << " to:" << iter->second->GetAID();
  }
}

// Get the branch name by input data arrow.
std::string InlineControlFlowScheduler::GetBranchNameByConditionGatherActor(KernelActor *condition_switch_actor,
                                                                            KernelActor *condition_gather_actor,
                                                                            DataArrow *data_arrow,
                                                                            const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(condition_switch_actor);
  MS_EXCEPTION_IF_NULL(condition_gather_actor);
  MS_EXCEPTION_IF_NULL(data_arrow);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const auto &condition_gather_kernel = condition_gather_actor->kernel();
  MS_EXCEPTION_IF_NULL(condition_gather_kernel);
  auto gather_to_switch = kernel_graph->condition_gather_to_switch();
  const auto &condition_pair_iter = gather_to_switch.find(condition_gather_kernel);
  if (condition_pair_iter == gather_to_switch.end() ||
      condition_pair_iter->second != condition_switch_actor->kernel()) {
    MS_LOG(EXCEPTION) << "Condition switch actor:" << condition_switch_actor->GetAID()
                      << " and gather actor:" << condition_gather_actor << " is not match.";
  }
  if (!condition_gather_kernel->HasAttr(kAttrBranchOutputNum)) {
    MS_LOG(EXCEPTION) << "Failed to get branch output num by actor:" << condition_gather_actor->GetAID();
  }
  // Get the output branch index in condition gather actor.
  const auto &output_value = condition_gather_kernel->GetAttr(kAttrBranchOutputNum);
  MS_EXCEPTION_IF_NULL(output_value);
  size_t branch_index = IntToSize(data_arrow->to_input_index_) / GetValue<size_t>(output_value);
  if (!condition_gather_kernel->HasAttr(kAttrBranchGraphName)) {
    MS_LOG(EXCEPTION) << "Failed to get branch graph name by actor:" << condition_gather_actor->GetAID();
  }

  // Get output branch name by branch index.
  const auto &branch_graph_names = condition_gather_kernel->GetAttr(kAttrBranchGraphName);
  MS_EXCEPTION_IF_NULL(branch_graph_names);
  MS_LOG(DEBUG) << "Branch graph name:" << branch_graph_names->ToString()
                << " for actor:" << condition_gather_actor->GetAID();
  if (!branch_graph_names->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "Invalid branch group name:" << branch_graph_names->ToString()
                      << " for actor:" << condition_gather_actor->GetAID();
  }
  const auto &tuple_name = branch_graph_names->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_name);
  if (branch_index >= tuple_name->size()) {
    MS_LOG(EXCEPTION) << "Invalid to index:" << data_arrow->to_input_index_
                      << " output num:" << GetValue<size_t>(output_value)
                      << " branch graph name:" << tuple_name->ToString()
                      << " from actor:" << condition_switch_actor->GetAID()
                      << " to actor:" << condition_gather_actor->GetAID();
  }
  MS_EXCEPTION_IF_NULL(tuple_name->value()[branch_index]);
  return GetValue<std::string>(tuple_name->value()[branch_index]);
}

void InlineControlFlowScheduler::InitOutputDataBranchInfoForConditionSwitchActor(
  ConditionSwitchActor *const condition_switch_actor, const KernelGraphPtr &kernel_graph) {
  const auto &inline_sub_graph_kernels = kernel_graph->inline_sub_graph_kernels();
  size_t output_num = AnfAlgo::GetOutputTensorNum(condition_switch_actor->kernel());
  condition_switch_actor->output_data_branch_indexes_.resize(condition_switch_actor->output_data_arrows().size());
  // Get the index for each output data arrow.
  for (size_t i = 0; i < condition_switch_actor->output_data_arrows().size(); ++i) {
    const auto &output_node = condition_switch_actor->output_data_nodes()[i];
    const auto &data_arrow = condition_switch_actor->output_data_arrows()[i];
    MS_EXCEPTION_IF_NULL(output_node);
    MS_EXCEPTION_IF_NULL(data_arrow);
    const auto &to_actor = FetchActor(data_arrow->to_op_id_.Name());
    if (to_actor == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to get actor:" << data_arrow->to_op_id_.Name()
                        << " from actor:" << condition_switch_actor->GetAID();
    }
    if (to_actor->type() != KernelTransformType::kConditionSwitchActor &&
        to_actor->type() != KernelTransformType::kConditionGatherActor &&
        to_actor->type() != KernelTransformType::kKernelActor) {
      MS_LOG(EXCEPTION) << "Invalid to actor:" << to_actor->GetAID()
                        << " from actor:" << condition_switch_actor->GetAID();
    }

    const auto &to_kernel_actor = dynamic_cast<KernelActor *>(to_actor);
    MS_EXCEPTION_IF_NULL(to_kernel_actor);
    MS_EXCEPTION_IF_NULL(to_kernel_actor->kernel());
    std::string current_branch_name;
    if (to_actor->type() == KernelTransformType::kConditionGatherActor) {
      current_branch_name =
        GetBranchNameByConditionGatherActor(condition_switch_actor, to_kernel_actor, data_arrow.get(), kernel_graph);
    } else {
      if (inline_sub_graph_kernels.find(to_kernel_actor->kernel()) == inline_sub_graph_kernels.end()) {
        MS_LOG(EXCEPTION) << "Failed to get inline sub graph name by data user node:"
                          << to_kernel_actor->kernel()->fullname_with_scope()
                          << " in actor:" << condition_switch_actor->GetAID();
      }
      MS_LOG(DEBUG) << "Sub graph kernel:" << to_kernel_actor->kernel()->fullname_with_scope()
                    << " belong graph:" << inline_sub_graph_kernels.at(to_kernel_actor->kernel())
                    << " in actor:" << condition_switch_actor->GetAID()
                    << " from index:" << data_arrow->from_output_index_ << " to actor:" << data_arrow->to_op_id_
                    << " to index:" << data_arrow->to_input_index_;
      current_branch_name = inline_sub_graph_kernels.at(to_kernel_actor->kernel());
    }
    // Get branch index for output data arrow.
    const auto &iter = std::find(condition_switch_actor->branch_names_.begin(),
                                 condition_switch_actor->branch_names_.end(), current_branch_name);
    if (iter == condition_switch_actor->branch_names_.end()) {
      MS_LOG(EXCEPTION) << "Invalid branch name:" << current_branch_name
                        << " total branch name:" << condition_switch_actor->branch_names_
                        << " from actor:" << condition_switch_actor->GetAID() << " to actor:" << to_actor->GetAID();
    }
    size_t branch_index = LongToSize(iter - condition_switch_actor->branch_names_.begin());
    if (IntToSize(data_arrow->from_output_index_) >= output_num ||
        branch_index >= condition_switch_actor->branch_names_.size()) {
      MS_LOG(EXCEPTION) << "Invalid output index:" << data_arrow->from_output_index_ << " total:" << output_num
                        << " and branch index:" << branch_index
                        << " total:" << condition_switch_actor->branch_names_.size()
                        << " for actor:" << condition_switch_actor->GetAID();
    }
    condition_switch_actor->output_data_branch_indexes_[i] = branch_index;
    condition_switch_actor->branch_origin_ref_count_[branch_index][data_arrow->from_output_index_]++;
  }
}

void InlineControlFlowScheduler::InitOutputControlBranchInfoForConditionSwitchActor(
  ConditionSwitchActor *const condition_switch_actor, const KernelGraphPtr &kernel_graph) {
  const auto &inline_sub_graph_kernels = kernel_graph->inline_sub_graph_kernels();
  condition_switch_actor->output_control_branch_indexes_.resize(condition_switch_actor->output_control_arrows().size());
  // Get the index for each output control arrow.
  for (size_t i = 0; i < condition_switch_actor->output_control_arrows().size(); ++i) {
    const auto &arrow = condition_switch_actor->output_control_arrows()[i];
    MS_EXCEPTION_IF_NULL(arrow);
    const auto &to_actor = FetchActor(arrow->to_op_id_.Name());
    if (to_actor == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to get actor:" << arrow->to_op_id_.Name()
                        << " from actor:" << condition_switch_actor->GetAID();
    }
    if (to_actor->type() == KernelTransformType::kConditionGatherActor) {
      condition_switch_actor->output_control_branch_indexes_[i] = SIZE_MAX;
      continue;
    }
    if (to_actor->type() != KernelTransformType::kKernelActor &&
        to_actor->type() != KernelTransformType::kConditionSwitchActor) {
      MS_LOG(EXCEPTION) << "Invalid to actor:" << to_actor->GetAID()
                        << " from actor:" << condition_switch_actor->GetAID();
    }
    const auto &to_kernel_actor = dynamic_cast<KernelActor *>(to_actor);
    MS_EXCEPTION_IF_NULL(to_kernel_actor);
    MS_EXCEPTION_IF_NULL(to_kernel_actor->kernel());
    if (inline_sub_graph_kernels.find(to_kernel_actor->kernel()) == inline_sub_graph_kernels.end()) {
      MS_LOG(EXCEPTION) << "Failed to get inline sub graph name by control user node:"
                        << to_kernel_actor->kernel()->fullname_with_scope()
                        << " in actor:" << condition_switch_actor->GetAID();
    }
    MS_LOG(DEBUG) << "Sub graph kernel:" << to_kernel_actor->kernel()->fullname_with_scope()
                  << " belong graph:" << inline_sub_graph_kernels.at(to_kernel_actor->kernel())
                  << " in actor:" << condition_switch_actor->GetAID() << " to actor:" << arrow->to_op_id_;
    const auto &current_branch_name = inline_sub_graph_kernels.at(to_kernel_actor->kernel());
    const auto &iter = std::find(condition_switch_actor->branch_names_.begin(),
                                 condition_switch_actor->branch_names_.end(), current_branch_name);
    if (iter == condition_switch_actor->branch_names_.end()) {
      MS_LOG(EXCEPTION) << "Invalid branch name:" << current_branch_name
                        << " total branch name:" << condition_switch_actor->branch_names_
                        << " for actor:" << condition_switch_actor->GetAID();
    }
    size_t branch_index = LongToSize(iter - condition_switch_actor->branch_names_.begin());
    condition_switch_actor->output_control_branch_indexes_[i] = branch_index;
  }
}

void InlineControlFlowScheduler::InitOutputBranchInfoForConditionSwitchActor(
  ConditionSwitchActor *const condition_switch_actor, const KernelGraphPtr &kernel_graph) {
  if (condition_switch_actor->output_data_nodes().size() != condition_switch_actor->output_data_arrows().size()) {
    MS_LOG(EXCEPTION) << "Invalid data node size:" << condition_switch_actor->output_data_nodes().size()
                      << " and arrow size:" << condition_switch_actor->output_data_arrows().size()
                      << " for actor:" << condition_switch_actor->GetAID();
  }
  InitOutputDataBranchInfoForConditionSwitchActor(condition_switch_actor, kernel_graph);
  InitOutputControlBranchInfoForConditionSwitchActor(condition_switch_actor, kernel_graph);
  MS_LOG(DEBUG) << "Branch origin ref count:" << condition_switch_actor->branch_origin_ref_count_
                << " output data branch index:" << condition_switch_actor->output_data_branch_indexes_
                << " output control branch index:" << condition_switch_actor->output_control_branch_indexes_
                << " for actor:" << condition_switch_actor->GetAID();
}

void InlineControlFlowScheduler::HandleConditionSwitchActor(const KernelActorPtr &kernel_actor) {
  MS_EXCEPTION_IF_NULL(kernel_actor);
  const auto &condition_switch_actor = dynamic_cast<ConditionSwitchActor *>(kernel_actor.get());
  MS_EXCEPTION_IF_NULL(condition_switch_actor);
  MS_EXCEPTION_IF_NULL(condition_switch_actor->kernel());
  const auto &graph = condition_switch_actor->kernel()->func_graph();
  if (graph == nullptr || !graph->isa<KernelGraph>()) {
    MS_LOG(EXCEPTION) << "Failed to get kernel graph by actor:" << condition_switch_actor->GetAID();
  }
  const auto &kernel_graph = graph->cast<KernelGraphPtr>();
  MS_LOG(DEBUG) << "Fetch kernel graph:" << kernel_graph->ToString()
                << " by actor:" << condition_switch_actor->GetAID();
  if (!condition_switch_actor->kernel()->HasAttr(kInlineSubGraphName)) {
    MS_LOG(EXCEPTION) << "Failed to get inline graph name by actor:" << condition_switch_actor->GetAID();
  }
  const auto &inline_sub_graph_names = condition_switch_actor->kernel()->GetAttr(kInlineSubGraphName);
  MS_EXCEPTION_IF_NULL(inline_sub_graph_names);
  MS_LOG(DEBUG) << "inline sub graph name:" << inline_sub_graph_names->ToString()
                << " for actor:" << condition_switch_actor->GetAID();
  if (!inline_sub_graph_names->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "Invalid input subgraph name:" << inline_sub_graph_names->ToString()
                      << " for actor:" << condition_switch_actor->GetAID();
  }
  const auto &tuple_name = inline_sub_graph_names->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_name);
  std::vector<std::string> branch_names;
  for_each(tuple_name->value().begin(), tuple_name->value().end(),
           [&branch_names](const auto &value) { branch_names.emplace_back(GetValue<std::string>(value)); });
  condition_switch_actor->branch_names_ = branch_names;
  // Fix ref count.
  size_t output_num = AnfAlgo::GetOutputTensorNum(condition_switch_actor->kernel());
  condition_switch_actor->branch_origin_ref_count_ =
    std::vector<std::vector<size_t>>(tuple_name->size(), vector<size_t>(output_num, 0));

  InitOutputBranchInfoForConditionSwitchActor(condition_switch_actor, kernel_graph);
}

void InlineControlFlowScheduler::AddRefCountForConditionSwitchActor(ConditionSwitchActor *const switch_actor,
                                                                    const std::string &branch_name, size_t output_index,
                                                                    size_t ref_count) {
  const auto &iter = std::find(switch_actor->branch_names_.begin(), switch_actor->branch_names_.end(), branch_name);
  if (iter == switch_actor->branch_names_.end()) {
    MS_LOG(EXCEPTION) << "Failed to get branch name:" << branch_name << " total:" << switch_actor->branch_names_
                      << " in actor:" << switch_actor->GetAID();
  }
  size_t index = LongToSize(iter - switch_actor->branch_names_.begin());
  if (index >= switch_actor->branch_origin_ref_count_.size()) {
    MS_LOG(EXCEPTION) << " Invalid index:" << index
                      << " for branch origin ref count:" << switch_actor->branch_origin_ref_count_
                      << " for actor:" << switch_actor->GetAID();
  }
  if (output_index >= switch_actor->branch_origin_ref_count_[index].size()) {
    MS_LOG(EXCEPTION) << " Invalid output index:" << output_index << " branch index:" << index
                      << " for branch origin ref count:" << switch_actor->branch_origin_ref_count_
                      << " for actor:" << switch_actor->GetAID();
  }
  MS_LOG(DEBUG) << "Add ref count:" << ref_count << " for branch index:" << index << " index:" << output_index
                << " origin ref count:" << switch_actor->branch_origin_ref_count_
                << " for actor:" << switch_actor->GetAID();
  switch_actor->branch_origin_ref_count_[index][output_index] += ref_count;
}

void InlineControlFlowScheduler::FixRefCountForRefNode(const KernelWithIndex &input_with_index, size_t ref_count,
                                                       const std::string &branch_name,
                                                       const KernelGraph *const kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(input_with_index.first);
  auto new_branch_name = branch_name;
  if (common::AnfAlgo::CheckPrimitiveType(input_with_index.first, prim::kPrimConditionSwitch)) {
    MS_LOG(DEBUG) << "Check switch node:" << input_with_index.first->fullname_with_scope()
                  << " index:" << input_with_index.second << " ref count:" << ref_count
                  << " branch name:" << branch_name;
    const auto &actor = FetchActor(GetActorIdByKernel(input_with_index.first));
    MS_EXCEPTION_IF_NULL(actor);
    const auto &switch_actor = dynamic_cast<ConditionSwitchActor *>(actor);
    MS_EXCEPTION_IF_NULL(switch_actor);
    AddRefCountForConditionSwitchActor(switch_actor, branch_name, input_with_index.second, ref_count);
    const auto &iter = kernel_graph->inline_sub_graph_kernels().find(input_with_index.first);
    new_branch_name =
      (iter == kernel_graph->inline_sub_graph_kernels().end() ? kernel_graph->ToString() : iter->second);
    MS_LOG(DEBUG) << "Switch branch name from:" << branch_name << " to:" << new_branch_name
                  << " by switch node:" << input_with_index.first->fullname_with_scope()
                  << " in kernel graph:" << kernel_graph->ToString() << " ref count:" << ref_count;
  } else if (common::AnfAlgo::CheckPrimitiveType(input_with_index.first, prim::kPrimConditionGather)) {
    const auto &actor = FetchActor(GetActorIdByKernel(input_with_index.first));
    MS_EXCEPTION_IF_NULL(actor);
    const auto &gather_actor = dynamic_cast<ConditionGatherActor *>(actor);
    MS_EXCEPTION_IF_NULL(gather_actor);
    const auto &gather_cnode = input_with_index.first->cast<CNodePtr>();
    size_t input_num = common::AnfAlgo::GetInputNum(gather_cnode);
    if (input_num == 0 || input_num != gather_actor->branch_names_.size() * gather_actor->branch_output_num_) {
      MS_LOG_WITH_NODE(EXCEPTION, gather_cnode)
        << "Invalid input num:" << input_num << " branch output num:" << gather_actor->branch_output_num_
        << " branch num:" << gather_actor->branch_names_.size() << " for node:" << gather_cnode->fullname_with_scope();
    }
    for (size_t i = input_with_index.second; i < input_num; i = i + gather_actor->branch_output_num_) {
      FixRefCountForInputNode(common::AnfAlgo::VisitKernelWithReturnType(gather_cnode->input(i + 1), 0), ref_count,
                              gather_actor->branch_names_[i / gather_actor->branch_output_num_]);
    }
    return;
  }

  if (kernel_graph->IsInRefOutputMap(input_with_index)) {
    const auto &ref_value = kernel_graph->GetRefCorrespondOutput(input_with_index);
    if (ref_value.first == nullptr) {
      return;
    }
    MS_LOG(DEBUG) << "Check input node:" << ref_value.first->fullname_with_scope() << " index:" << ref_value.second
                  << " output node:" << input_with_index.first->fullname_with_scope()
                  << " index:" << input_with_index.second;
    FixRefCountForRefNode(ref_value, ref_count, new_branch_name, kernel_graph);
  }
}

void InlineControlFlowScheduler::FixRefCountForInputNode(const KernelWithIndex &input_with_index, size_t ref_count,
                                                         const std::string &branch_name) {
  const auto &node = input_with_index.first;
  MS_EXCEPTION_IF_NULL(node);
  const auto &device_address = AnfAlgo::GetMutableOutputAddr(node, input_with_index.second, false);
  MS_EXCEPTION_IF_NULL(device_address);
  if (ref_count == SIZE_MAX) {
    MS_LOG(DEBUG) << "set ref count to max for device address:" << device_address;
    device_address->set_original_ref_count(ref_count);
  } else {
    MS_LOG(DEBUG) << "set ref count from:" << device_address->original_ref_count()
                  << " to:" << device_address->original_ref_count() + ref_count
                  << " for device address:" << device_address;
    device_address->set_original_ref_count(device_address->original_ref_count() + ref_count);
  }
  device_address->ResetRefCount();
  if (node->isa<CNode>()) {
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &graph = cnode->func_graph();
    if (graph != nullptr && graph->isa<KernelGraph>()) {
      const auto &kernel_graph = dynamic_cast<KernelGraph *>(graph.get());
      MS_EXCEPTION_IF_NULL(kernel_graph);
      if (kernel_graph->IsInRefOutputMap(input_with_index)) {
        FixRefCountForRefNode(input_with_index, ref_count, branch_name, kernel_graph);
        return;
      }
    }
  }

  if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimConditionGather)) {
    const auto &gather_cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(gather_cnode);
    const auto &actor = FetchActor(GetActorIdByKernel(gather_cnode));
    MS_EXCEPTION_IF_NULL(actor);
    const auto &gather_actor = dynamic_cast<ConditionGatherActor *>(actor);
    MS_EXCEPTION_IF_NULL(gather_actor);
    size_t input_num = common::AnfAlgo::GetInputNum(gather_cnode);
    if (input_num == 0 || input_num != gather_actor->branch_names_.size() * gather_actor->branch_output_num_) {
      MS_LOG_WITH_NODE(EXCEPTION, gather_cnode)
        << "Invalid input num:" << input_num << " branch output num:" << gather_actor->branch_output_num_
        << " branch num:" << gather_actor->branch_names_.size() << " for node:" << gather_cnode->fullname_with_scope();
    }
    for (size_t i = input_with_index.second; i < input_num; i = i + gather_actor->branch_output_num_) {
      FixRefCountForInputNode(common::AnfAlgo::VisitKernelWithReturnType(gather_cnode->input(i + 1), 0), ref_count,
                              gather_actor->branch_names_[i / gather_actor->branch_output_num_]);
    }
  }
}

void InlineControlFlowScheduler::FixRefCountByConditionGatherActor(ConditionGatherActor *const condition_gather_actor,
                                                                   const KernelGraphPtr &kernel_graph) {
  std::vector<size_t> need_add_ref_count;
  size_t output_num = AnfAlgo::GetOutputTensorNum(condition_gather_actor->kernel());
  for (size_t i = 0; i < output_num; ++i) {
    const auto &device_address = AnfAlgo::GetMutableOutputAddr(condition_gather_actor->kernel(), i, false);
    MS_EXCEPTION_IF_NULL(device_address);
    need_add_ref_count.emplace_back(
      device_address->original_ref_count() == SIZE_MAX ? SIZE_MAX : device_address->original_ref_count() - 1);
    MS_LOG(DEBUG) << "For actor:" << condition_gather_actor->GetAID() << " output device address:" << device_address
                  << " output index:" << i << " ref_count:" << device_address->original_ref_count()
                  << " need add:" << need_add_ref_count.back();
  }
  size_t input_num = common::AnfAlgo::GetInputNum(condition_gather_actor->kernel());
  if (input_num == 0 ||
      input_num != condition_gather_actor->branch_output_num_ * condition_gather_actor->branch_names_.size()) {
    MS_LOG(EXCEPTION) << "Invalid input num:" << input_num
                      << " branch output num:" << condition_gather_actor->branch_output_num_
                      << " for actor:" << condition_gather_actor->GetAID();
  }
  for (size_t i = 0; i < input_num; ++i) {
    const auto &device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(condition_gather_actor->kernel(), i, false);
    MS_EXCEPTION_IF_NULL(device_address);
    MS_LOG(DEBUG) << "For actor::" << condition_gather_actor->GetAID() << " input device address:" << device_address
                  << " input index:" << i << " ref_count:" << device_address->original_ref_count();
    if (device_address->original_ref_count() == SIZE_MAX) {
      continue;
    }
    const auto &input_with_index =
      common::AnfAlgo::VisitKernelWithReturnType(condition_gather_actor->kernel()->input(i + 1), 0);
    FixRefCountForInputNode(input_with_index, need_add_ref_count[i % condition_gather_actor->branch_output_num_],
                            condition_gather_actor->branch_names_[i / condition_gather_actor->branch_output_num_]);
    MS_LOG(DEBUG) << "Condition gather actor:" << condition_gather_actor->GetAID() << " input index:" << i
                  << " input node:" << input_with_index.first->DebugString()
                  << " with index:" << input_with_index.second
                  << " need add ref count:" << need_add_ref_count[i % condition_gather_actor->branch_output_num_];
  }
}

void InlineControlFlowScheduler::InitInputDataBranchInfoForConditionGatherActor(
  ConditionGatherActor *const condition_gather_actor, const KernelGraphPtr &kernel_graph) {
  const auto &inline_sub_graph_kernels = kernel_graph->inline_sub_graph_kernels();
  MS_LOG(DEBUG) << "Fetch kernel graph:" << kernel_graph->ToString()
                << " by actor:" << condition_gather_actor->GetAID();
  for (const auto &pair : condition_gather_actor->input_data_arrow_aids_) {
    const auto &from_aid = pair.first;
    const auto &data_arrow = pair.second;
    MS_EXCEPTION_IF_NULL(data_arrow);
    const auto &from_actor = FetchActor(from_aid.Name());
    if (from_actor == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to get from actor:" << from_aid << " to actor:" << condition_gather_actor->GetAID();
    }
    if (from_actor->type() != KernelTransformType::kKernelActor &&
        from_actor->type() != KernelTransformType::kConditionSwitchActor &&
        from_actor->type() != KernelTransformType::kConditionGatherActor) {
      MS_LOG(EXCEPTION) << "Invalid to actor:" << from_actor->GetAID()
                        << " from actor:" << condition_gather_actor->GetAID();
    }
    const auto &from_kernel_actor = dynamic_cast<KernelActor *>(from_actor);
    MS_EXCEPTION_IF_NULL(from_kernel_actor);
    MS_EXCEPTION_IF_NULL(from_kernel_actor->kernel());
    std::string current_branch_name;
    if (from_actor->type() == KernelTransformType::kConditionSwitchActor) {
      current_branch_name =
        GetBranchNameByConditionGatherActor(from_kernel_actor, condition_gather_actor, data_arrow, kernel_graph);
    } else {
      if (inline_sub_graph_kernels.find(from_kernel_actor->kernel()) == inline_sub_graph_kernels.end()) {
        MS_LOG(EXCEPTION) << "Failed to get inline sub graph name by data user node:"
                          << from_kernel_actor->kernel()->fullname_with_scope()
                          << " in actor:" << condition_gather_actor->GetAID();
      }
      MS_LOG(DEBUG) << "Sub graph kernel:" << from_kernel_actor->kernel()->fullname_with_scope()
                    << " belong graph:" << inline_sub_graph_kernels.at(from_kernel_actor->kernel())
                    << " in actor:" << condition_gather_actor->GetAID();
      current_branch_name = inline_sub_graph_kernels.at(from_kernel_actor->kernel());
    }
    const auto &iter = condition_gather_actor->branch_name_to_id_.find(current_branch_name);
    if (iter == condition_gather_actor->branch_name_to_id_.end()) {
      condition_gather_actor->branch_name_to_id_[current_branch_name] =
        condition_gather_actor->branch_name_to_id_.size();
      MS_LOG(DEBUG) << "Add branch index:" << condition_gather_actor->branch_name_to_id_[current_branch_name]
                    << " branch name:" << current_branch_name << " for actor:" << condition_gather_actor->GetAID();
    }
    // Get the input data num of each branch.
    if (condition_gather_actor->branch_name_to_input_data_num_.find(current_branch_name) ==
        condition_gather_actor->branch_name_to_input_data_num_.end()) {
      condition_gather_actor->branch_name_to_input_data_num_[current_branch_name] = 1;
    } else {
      condition_gather_actor->branch_name_to_input_data_num_[current_branch_name]++;
    }
  }
}

void InlineControlFlowScheduler::InitInputControlBranchInfoForConditionGatherActor(
  ConditionGatherActor *const condition_gather_actor, const KernelGraphPtr &kernel_graph) {
  const auto &inline_sub_graph_kernels = kernel_graph->inline_sub_graph_kernels();
  MS_LOG(DEBUG) << "Fetch kernel graph:" << kernel_graph->ToString()
                << " by actor:" << condition_gather_actor->GetAID();

  for (const auto &pair : condition_gather_actor->input_control_arrow_aids_) {
    const auto &from_aid = pair.first;
    const auto &from_actor = FetchActor(from_aid.Name());
    if (from_actor == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to get from actor:" << from_aid << " to actor:" << condition_gather_actor->GetAID();
    }
    if (from_actor->type() == KernelTransformType::kConditionSwitchActor) {
      continue;
    }
    if (from_actor->type() != KernelTransformType::kKernelActor &&
        from_actor->type() != KernelTransformType::kConditionGatherActor) {
      MS_LOG(EXCEPTION) << "Invalid from actor:" << from_actor->GetAID()
                        << " to actor:" << condition_gather_actor->GetAID();
    }
    const auto &from_kernel_actor = dynamic_cast<KernelActor *>(from_actor);
    MS_EXCEPTION_IF_NULL(from_kernel_actor);
    MS_EXCEPTION_IF_NULL(from_kernel_actor->kernel());
    if (inline_sub_graph_kernels.find(from_kernel_actor->kernel()) == inline_sub_graph_kernels.end()) {
      MS_LOG(EXCEPTION) << "Failed to get inline sub graph name by control user node:"
                        << from_kernel_actor->kernel()->fullname_with_scope()
                        << " in actor:" << condition_gather_actor->GetAID();
    }
    MS_LOG(DEBUG) << "Sub graph kernel:" << from_kernel_actor->kernel()->fullname_with_scope()
                  << " belong graph:" << inline_sub_graph_kernels.at(from_kernel_actor->kernel())
                  << " in actor:" << condition_gather_actor->GetAID();
    const auto &current_branch_name = inline_sub_graph_kernels.at(from_kernel_actor->kernel());
    // Get input op control num of each branch.
    if (condition_gather_actor->branch_name_to_input_control_num_.find(current_branch_name) ==
        condition_gather_actor->branch_name_to_input_control_num_.end()) {
      condition_gather_actor->branch_name_to_input_control_num_[current_branch_name] = 1;
    } else {
      condition_gather_actor->branch_name_to_input_control_num_[current_branch_name]++;
    }
  }
}

void InlineControlFlowScheduler::InitInputBranchInfoForConditionGatherActor(
  ConditionGatherActor *const condition_gather_actor, const KernelGraphPtr &kernel_graph) {
  InitInputDataBranchInfoForConditionGatherActor(condition_gather_actor, kernel_graph);
  InitInputControlBranchInfoForConditionGatherActor(condition_gather_actor, kernel_graph);
}

void InlineControlFlowScheduler::HandleConditionGatherActor(const KernelActorPtr &kernel_actor) {
  const auto &condition_gather_actor = dynamic_cast<ConditionGatherActor *>(kernel_actor.get());
  MS_EXCEPTION_IF_NULL(condition_gather_actor);
  const auto &gather_node = condition_gather_actor->kernel();
  MS_EXCEPTION_IF_NULL(gather_node);
  const auto &graph = gather_node->func_graph();
  if (graph == nullptr || !graph->isa<KernelGraph>()) {
    MS_LOG(EXCEPTION) << "Failed to get kernel graph by actor:" << condition_gather_actor->GetAID();
  }
  const auto &kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const auto &gather_switch_map = kernel_graph->condition_gather_to_switch();
  const auto &gather_switch_iter = gather_switch_map.find(gather_node);
  if (gather_switch_iter == gather_switch_map.end()) {
    MS_LOG_WITH_NODE(EXCEPTION, gather_node)
      << "Failed to get switch node by gather node:" << gather_node->fullname_with_scope();
  }
  if (gather_switch_iter->second == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, gather_node)
      << "Failed to get switch node by gather node:" << gather_node->fullname_with_scope()
      << " in kernel graph:" << kernel_graph->ToString();
  }
  const auto &actor = FetchActor(GetActorIdByKernel(gather_switch_iter->second));
  MS_EXCEPTION_IF_NULL(actor);
  const auto &condition_switch_actor = dynamic_cast<ConditionSwitchActor *>(actor);
  MS_EXCEPTION_IF_NULL(condition_switch_actor);
  condition_switch_actor->gather_aid_ = const_cast<AID *>(&condition_gather_actor->GetAID());

  if (!gather_node->HasAttr(kAttrBranchOutputNum)) {
    MS_LOG(EXCEPTION) << "Failed to get branch output num by actor:" << condition_gather_actor->GetAID();
  }
  const auto &output_value = gather_node->GetAttr(kAttrBranchOutputNum);
  MS_EXCEPTION_IF_NULL(output_value);
  condition_gather_actor->branch_output_num_ = GetValue<size_t>(output_value);

  if (!gather_node->HasAttr(kAttrBranchGraphName)) {
    MS_LOG(EXCEPTION) << "Failed to get inline graph name by actor:" << condition_gather_actor->GetAID();
  }
  const auto &branch_graph_names = gather_node->GetAttr(kAttrBranchGraphName);
  MS_EXCEPTION_IF_NULL(branch_graph_names);
  MS_LOG(DEBUG) << "Branch graph name:" << branch_graph_names->ToString()
                << " for actor:" << condition_gather_actor->GetAID();
  if (!branch_graph_names->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "Invalid branch group name:" << branch_graph_names->ToString()
                      << " for actor:" << condition_gather_actor->GetAID();
  }
  const auto &tuple_name = branch_graph_names->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_name);
  std::vector<std::string> branch_names;
  std::for_each(tuple_name->value().begin(), tuple_name->value().end(),
                [&branch_names](const auto &value) { branch_names.emplace_back(GetValue<std::string>(value)); });
  condition_gather_actor->branch_names_ = branch_names;
  // Fix ref count.
  FixRefCountByConditionGatherActor(condition_gather_actor, kernel_graph);
  InitInputBranchInfoForConditionGatherActor(condition_gather_actor, kernel_graph);
}

void InlineControlFlowScheduler::LinkControlArrowForNoInputOrOutputActor(
  ActorSet *actor_set, const mindspore::HashMap<std::string, AbstractActor *> &branch_name_to_switch_actor,
  const mindspore::HashMap<std::string, AbstractActor *> &branch_name_to_gather_actor) {
  MS_EXCEPTION_IF_NULL(actor_set);
  for (const auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    if ((kernel_actor->input_datas_num_ == 0) && (kernel_actor->input_controls_num_ == 0) &&
        IsInlineKernelActor(kernel_actor)) {
      const auto &branch_name = GetBranchNameByKernelActor(kernel_actor.get());
      const auto &iter = branch_name_to_switch_actor.find(branch_name);
      if (iter == branch_name_to_switch_actor.end()) {
        MS_LOG(EXCEPTION) << "Failed to get condition switch actor by branch name:" << branch_name;
      }
      MS_LOG(DEBUG) << "Inline control flow scheduler add control flow from switch actor:" << iter->second->GetAID()
                    << " to kernel actor:" << kernel_actor->GetAID();
      SchedulerHelper::AddControlArrow(iter->second, kernel_actor.get());
    }
    if (kernel_actor->output_data_arrows_.size() == 0 && kernel_actor->output_control_arrows_.size() == 0 &&
        IsInlineKernelActor(kernel_actor)) {
      const auto &branch_name = GetBranchNameByKernelActor(kernel_actor.get());
      const auto &iter = branch_name_to_gather_actor.find(branch_name);
      if (iter == branch_name_to_gather_actor.end()) {
        MS_LOG(EXCEPTION) << "Failed to get condition gather actor by branch name:" << branch_name;
      }
      MS_LOG(DEBUG) << "Inline control flow scheduler add control flow from kernel actor:" << kernel_actor->GetAID()
                    << " to gather actor:" << iter->second->GetAID();
      SchedulerHelper::AddControlArrow(kernel_actor.get(), iter->second);
    }
  }
}

void InlineControlFlowScheduler::Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info,
                                      bool execution_order_running) {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  mindspore::HashMap<std::string, AbstractActor *> branch_name_to_switch_actor;
  mindspore::HashMap<std::string, AbstractActor *> branch_name_to_gather_actor;
  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    GetBranchNameToCondtionActor(graph, &branch_name_to_switch_actor, &branch_name_to_gather_actor);
  }
  LinkControlArrowForNoInputOrOutputActor(actor_set, branch_name_to_switch_actor, branch_name_to_gather_actor);
  for (const auto &kernel_actor : actor_set->kernel_actors_) {
    if (kernel_actor->type() == KernelTransformType::kConditionSwitchActor) {
      HandleConditionSwitchActor(kernel_actor);
    } else if (kernel_actor->type() == KernelTransformType::kConditionGatherActor) {
      HandleConditionGatherActor(kernel_actor);
    }
  }
  for (const auto &kernel_graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(kernel_graph);
    if (kernel_graph->inline_sub_graph_kernels().empty()) {
      continue;
    }
    for (const auto &ref_pair : kernel_graph->GetRefMap()) {
      const auto &output_pair = ref_pair.first;
      const auto &input_pair = ref_pair.second;
      MS_EXCEPTION_IF_NULL(output_pair.first);
      MS_EXCEPTION_IF_NULL(input_pair.first);
      MS_LOG(DEBUG) << "output node:" << output_pair.first->fullname_with_scope()
                    << " input node:" << input_pair.first->fullname_with_scope();
      const auto &actor = FetchActor(GetActorIdByKernel(output_pair.first));
      if (actor == nullptr) {
        MS_LOG_WITH_NODE(EXCEPTION, output_pair.first)
          << "Failed to get actor by ref node:" << output_pair.first->fullname_with_scope()
          << " index:" << output_pair.second << " origin node:" << input_pair.first->fullname_with_scope()
          << " index:" << input_pair.second << " in graph:" << kernel_graph->ToString();
      }
      size_t ref_count = 1;
      std::for_each(actor->output_data_arrows().begin(), actor->output_data_arrows().end(),
                    [&ref_count, &output_pair](const auto &data_arrow) {
                      MS_EXCEPTION_IF_NULL(data_arrow);
                      if (IntToSize(data_arrow->from_output_index_) == output_pair.second) {
                        ++ref_count;
                      }
                    });
      FixRefCountRecursively(output_pair, input_pair, kernel_graph, ref_count);
    }
  }
}

void InlineControlFlowScheduler::FixRefCountRecursively(const KernelWithIndex &output_pair,
                                                        const KernelWithIndex &input_pair,
                                                        const KernelGraphPtr &kernel_graph, size_t ref_count) {
  MS_EXCEPTION_IF_NULL(output_pair.first);
  MS_EXCEPTION_IF_NULL(input_pair.first);
  if (common::AnfAlgo::CheckPrimitiveType(input_pair.first, prim::kPrimConditionGather)) {
    return;
  }
  if (common::AnfAlgo::CheckPrimitiveType(input_pair.first, prim::kPrimConditionSwitch)) {
    const auto &iter = kernel_graph->inline_sub_graph_kernels().find(output_pair.first);
    if (iter == kernel_graph->inline_sub_graph_kernels().end()) {
      MS_LOG_WITH_NODE(EXCEPTION, input_pair.first)
        << "Invalid ref node pair, input node:" << input_pair.first->fullname_with_scope()
        << " index:" << input_pair.second << " output node:" << output_pair.first->fullname_with_scope()
        << " index:" << output_pair.second << " in kernel graph:" << kernel_graph->ToString();
    }
    const auto &branch_name = iter->second;
    const auto &actor = FetchActor(GetActorIdByKernel(input_pair.first));
    MS_EXCEPTION_IF_NULL(actor);
    const auto &switch_actor = dynamic_cast<ConditionSwitchActor *>(actor);
    MS_EXCEPTION_IF_NULL(switch_actor);
    AddRefCountForConditionSwitchActor(switch_actor, branch_name, input_pair.second, ref_count);
  }
  if (kernel_graph->IsInRefOutputMap(input_pair)) {
    FixRefCountRecursively(input_pair, kernel_graph->GetRefCorrespondOutput(input_pair), kernel_graph, ref_count);
  }
}
}  // namespace runtime
}  // namespace mindspore
