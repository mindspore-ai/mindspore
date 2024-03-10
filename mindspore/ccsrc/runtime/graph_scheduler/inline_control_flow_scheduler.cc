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
#include "runtime/graph_scheduler/scheduler_helper.h"
#include "ops/framework_ops.h"

namespace mindspore {
namespace runtime {
void InlineControlFlowScheduler::LinkControlArrowByExecutionOrder(const KernelGraphPtr &graph,
                                                                  const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &inline_sub_graph_kernels = graph->inline_sub_graph_kernels();
  if (inline_sub_graph_kernels.empty()) {
    return;
  }
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
  MS_EXCEPTION_IF_NULL(condition_gather_actor->kernel());
  const auto &condition_pair_iter = kernel_graph->condition_gather_to_switch().find(condition_gather_actor->kernel());
  if (condition_pair_iter == kernel_graph->condition_gather_to_switch().end() ||
      condition_pair_iter->second != condition_switch_actor->kernel()) {
    MS_LOG(EXCEPTION) << "Condition switch actor:" << condition_switch_actor->GetAID()
                      << " and gather actor:" << condition_gather_actor << " is not match.";
  }
  if (!condition_gather_actor->kernel()->HasAttr(kAttrBranchOutputNum)) {
    MS_LOG(EXCEPTION) << "Failed to get branch output num by actor:" << condition_gather_actor->GetAID();
  }
  // Get the output branch index in condition gather actor.
  const auto &output_value = condition_gather_actor->kernel()->GetAttr(kAttrBranchOutputNum);
  MS_EXCEPTION_IF_NULL(output_value);
  size_t branch_index = data_arrow->to_input_index_ / GetValue<size_t>(output_value);
  if (!condition_gather_actor->kernel()->HasAttr(kAttrBranchGraphName)) {
    MS_LOG(EXCEPTION) << "Failed to get branch graph name by actor:" << condition_gather_actor->GetAID();
  }

  // Get output branch name by branch index.
  const auto &branch_graph_names = condition_gather_actor->kernel()->GetAttr(kAttrBranchGraphName);
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

void InlineControlFlowScheduler::FixRefCountByKernelGraphRefMap(ConditionSwitchActor *const condition_switch_actor,
                                                                const KernelGraphPtr &kernel_graph) {
  const auto &inline_sub_graph_kernels = kernel_graph->inline_sub_graph_kernels();
  size_t output_num = AnfAlgo::GetOutputTensorNum(condition_switch_actor->kernel());
  for (const auto &ref_pair : kernel_graph->GetRefMap()) {
    const auto &output_pair = ref_pair.first;
    const auto &origin_pair = ref_pair.second;
    MS_LOG(DEBUG) << "output node:" << output_pair.first->fullname_with_scope()
                  << " origin node:" << origin_pair.first->fullname_with_scope();
    const auto &recursive_origin_pair = kernel_graph->GetRefNodeRecursive(output_pair);
    // If the input node of ref node pair is a condition switch node , the ref count of corresponding switch node input
    // should add 1.
    if (recursive_origin_pair.first == condition_switch_actor->kernel() && output_pair.first != nullptr) {
      MS_LOG(DEBUG) << "Condtion switch node is an input of ref node:" << output_pair.first->fullname_with_scope();
      if (inline_sub_graph_kernels.find(output_pair.first) == inline_sub_graph_kernels.end()) {
        MS_LOG(EXCEPTION) << "Failed to get inline subgraph name by ref node:"
                          << output_pair.first->fullname_with_scope();
      }
      // Get the branch index for ref output.
      const auto &current_branch_name = inline_sub_graph_kernels.at(output_pair.first);
      const auto &iter = std::find(condition_switch_actor->branch_names_.begin(),
                                   condition_switch_actor->branch_names_.end(), current_branch_name);
      if (iter == condition_switch_actor->branch_names_.end()) {
        MS_LOG(EXCEPTION) << "Invalid branch name:" << current_branch_name
                          << " total branch name:" << condition_switch_actor->branch_names_
                          << " for actor:" << condition_switch_actor->GetAID();
      }
      size_t branch_index = iter - condition_switch_actor->branch_names_.begin();

      if (recursive_origin_pair.second >= output_num || branch_index >= condition_switch_actor->branch_names_.size()) {
        MS_LOG(EXCEPTION) << "Invalid output index:" << recursive_origin_pair.second << " total:" << output_num
                          << " and branch index:" << branch_index
                          << " total:" << condition_switch_actor->branch_names_.size()
                          << " for actor:" << condition_switch_actor->GetAID();
      }
      // The ref count of the corresponding branch add 1.
      condition_switch_actor->branch_origin_ref_count_[branch_index][recursive_origin_pair.second]++;
      MS_LOG(DEBUG) << "Add ref count for current branch:" << current_branch_name << " branch index:" << branch_index
                    << " output index:" << recursive_origin_pair.second
                    << " of actor:" << condition_switch_actor->GetAID();
    }
  }
}

void InlineControlFlowScheduler::FixRefCountByConditionSwitchActor(ConditionSwitchActor *const condition_switch_actor,
                                                                   const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(condition_switch_actor);
  // Collect all the output ref count of condition switch actor.
  std::vector<size_t> total_ref_count;
  size_t output_num = AnfAlgo::GetOutputTensorNum(condition_switch_actor->kernel());
  for (size_t i = 0; i < output_num; ++i) {
    const auto &device_address = AnfAlgo::GetMutableOutputAddr(condition_switch_actor->kernel(), i);
    MS_EXCEPTION_IF_NULL(device_address);
    total_ref_count.emplace_back(device_address->original_ref_count());
    MS_LOG(DEBUG) << "For actor:" << condition_switch_actor->GetAID() << " output device address:" << device_address
                  << " output index:" << i << " ref_count:" << total_ref_count.back();
  }

  size_t input_num = common::AnfAlgo::GetInputTensorNum(condition_switch_actor->kernel());
  // Input num should same as the output num and the condition of switch node.
  if (input_num != output_num + 1) {
    MS_LOG(EXCEPTION) << "Invalid input num:" << input_num << " and output num:" << output_num
                      << " for actor:" << condition_switch_actor->GetAID();
  }

  // Add the ref count to the input of condition switch actor.
  for (size_t i = 1; i < input_num; ++i) {
    const auto &device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(condition_switch_actor->kernel(), i);
    MS_EXCEPTION_IF_NULL(device_address);
    MS_LOG(DEBUG) << "For actor::" << condition_switch_actor->GetAID() << " input device address:" << device_address
                  << " input index:" << i << " ref_count:" << device_address->original_ref_count();
    if (device_address->original_ref_count() == SIZE_MAX) {
      continue;
    }
    device_address->set_original_ref_count(device_address->original_ref_count() + total_ref_count[i - 1] - 1);
    device_address->ResetRefCount();
    MS_LOG(DEBUG) << "For actor::" << condition_switch_actor->GetAID() << " input device address:" << device_address
                  << " input index:" << i << " ref_count:" << device_address->original_ref_count();
  }
  FixRefCountByKernelGraphRefMap(condition_switch_actor, kernel_graph);
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
    if (to_actor->type() != KernelTransformType::kConditionGatherActor &&
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
        MS_LOG(EXCEPTION) << "Failed to get inline sub graph name by user node:"
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
    size_t branch_index = iter - condition_switch_actor->branch_names_.begin();
    if (IntToSize(data_arrow->from_output_index_) >= output_num ||
        branch_index >= condition_switch_actor->branch_names_.size()) {
      MS_LOG(EXCEPTION) << "Invalid output index:" << data_arrow->from_output_index_ << " total:" << output_num
                        << " and branch index:" << branch_index
                        << " total:" << condition_switch_actor->branch_names_.size()
                        << " for actor:" << condition_switch_actor->GetAID();
    }
    condition_switch_actor->branch_origin_ref_count_[branch_index][data_arrow->from_output_index_]++;
    condition_switch_actor->output_data_branch_indexes_[i] = branch_index;
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
      MS_LOG(EXCEPTION) << "Failed to get inline sub graph name by user node:"
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
    size_t branch_index = iter - condition_switch_actor->branch_names_.begin();
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

  FixRefCountByConditionSwitchActor(condition_switch_actor, kernel_graph);
  InitOutputBranchInfoForConditionSwitchActor(condition_switch_actor, kernel_graph);
}

void InlineControlFlowScheduler::FixRefCountByKernelGraphRefMap(ConditionGatherActor *const condition_gather_actor,
                                                                const KernelGraphPtr &kernel_graph) {
  // If the input node of ref node pair is a condition gather node , the ref count of corresponding gather node input
  // should add 1.
  for (const auto &ref_pair : kernel_graph->GetRefMap()) {
    const auto &output_pair = ref_pair.first;
    const auto &origin_pair = ref_pair.second;
    MS_LOG(DEBUG) << "output node:" << output_pair.first->fullname_with_scope()
                  << " origin node:" << origin_pair.first->fullname_with_scope();
    const auto &recursive_origin_pair = kernel_graph->GetRefNodeRecursive(output_pair);
    if (recursive_origin_pair.first == condition_gather_actor->kernel() && output_pair.first != nullptr) {
      MS_LOG(DEBUG) << "Condtion gather node output index:" << recursive_origin_pair.second
                    << " is an input of ref node:" << output_pair.first->fullname_with_scope()
                    << " to index:" << output_pair.second
                    << " need update ref count for actor:" << condition_gather_actor->GetAID();
      for (size_t i = recursive_origin_pair.second; i < common::AnfAlgo::GetInputNum(condition_gather_actor->kernel());
           i += condition_gather_actor->branch_output_num_) {
        const auto &device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(condition_gather_actor->kernel(), i);
        MS_EXCEPTION_IF_NULL(device_address);
        MS_LOG(DEBUG) << "For actor::" << condition_gather_actor->GetAID() << " input device address:" << device_address
                      << " input index:" << i << " ref_count:" << device_address->original_ref_count();
        if (device_address->original_ref_count() == SIZE_MAX) {
          continue;
        }
        size_t pre_origin_ref_count = device_address->original_ref_count();
        device_address->set_original_ref_count(device_address->original_ref_count() + 1);
        device_address->ResetRefCount();
        MS_LOG(DEBUG) << "For actor::" << condition_gather_actor->GetAID() << " input device address:" << device_address
                      << " input index:" << i << " fix ref count from:" << pre_origin_ref_count
                      << " to:" << device_address->original_ref_count();
      }
    }
  }
}

void InlineControlFlowScheduler::FixRefCountByConditionGatherActor(ConditionGatherActor *const condition_gather_actor,
                                                                   const KernelGraphPtr &kernel_graph) {
  std::vector<size_t> total_ref_count;
  size_t output_num = AnfAlgo::GetOutputTensorNum(condition_gather_actor->kernel());
  for (size_t i = 0; i < output_num; ++i) {
    const auto &device_address = AnfAlgo::GetMutableOutputAddr(condition_gather_actor->kernel(), i);
    MS_EXCEPTION_IF_NULL(device_address);
    total_ref_count.emplace_back(device_address->original_ref_count());
    MS_LOG(DEBUG) << "For actor:" << condition_gather_actor->GetAID() << " output device address:" << device_address
                  << " output index:" << i << " ref_count:" << total_ref_count.back();
  }
  size_t input_num = common::AnfAlgo::GetInputNum(condition_gather_actor->kernel());
  if (input_num == 0 || input_num % condition_gather_actor->branch_output_num_ != 0) {
    MS_LOG(EXCEPTION) << "Invalid input num:" << input_num
                      << " branch output num:" << condition_gather_actor->branch_output_num_
                      << " for actor:" << condition_gather_actor->GetAID();
  }
  for (size_t i = 0; i < input_num; ++i) {
    const auto &device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(condition_gather_actor->kernel(), i);
    MS_EXCEPTION_IF_NULL(device_address);
    MS_LOG(DEBUG) << "For actor::" << condition_gather_actor->GetAID() << " input device address:" << device_address
                  << " input index:" << i << " ref_count:" << device_address->original_ref_count();
    if (device_address->original_ref_count() == SIZE_MAX) {
      continue;
    }
    size_t pre_origin_ref_count = device_address->original_ref_count();
    // The real ref count is the relative position of this branch output.
    device_address->set_original_ref_count(device_address->original_ref_count() +
                                           total_ref_count[i % condition_gather_actor->branch_output_num_] - 1);
    device_address->ResetRefCount();
    MS_LOG(DEBUG) << "For actor::" << condition_gather_actor->GetAID() << " input device address:" << device_address
                  << " input index:" << i << " fix ref count from:" << pre_origin_ref_count
                  << " to:" << device_address->original_ref_count();
  }
  FixRefCountByKernelGraphRefMap(condition_gather_actor, kernel_graph);
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
        from_actor->type() != KernelTransformType::kConditionSwitchActor) {
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
        MS_LOG(EXCEPTION) << "Failed to get inline sub graph name by user node:"
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
      MS_LOG(EXCEPTION) << "Failed to get inline sub graph name by user node:"
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
  MS_EXCEPTION_IF_NULL(condition_gather_actor->kernel());
  const auto &graph = condition_gather_actor->kernel()->func_graph();
  if (graph == nullptr || !graph->isa<KernelGraph>()) {
    MS_LOG(EXCEPTION) << "Failed to get kernel graph by actor:" << condition_gather_actor->GetAID();
  }
  const auto &kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const auto &gather_to_switch_iter = kernel_graph->condition_gather_to_switch().find(condition_gather_actor->kernel());
  if (gather_to_switch_iter == kernel_graph->condition_gather_to_switch().end()) {
    MS_LOG(EXCEPTION) << "Failed to get switch node by gather node:"
                      << condition_gather_actor->kernel()->fullname_with_scope();
  }
  MS_EXCEPTION_IF_NULL(gather_to_switch_iter->second);
  const auto &actor = FetchActor(gather_to_switch_iter->second->fullname_with_scope());
  MS_EXCEPTION_IF_NULL(actor);
  const auto &condition_switch_actor = dynamic_cast<ConditionSwitchActor *>(actor);
  MS_EXCEPTION_IF_NULL(condition_switch_actor);
  condition_switch_actor->gather_aid_ = const_cast<AID *>(&condition_gather_actor->GetAID());

  if (!condition_gather_actor->kernel()->HasAttr(kAttrBranchOutputNum)) {
    MS_LOG(EXCEPTION) << "Failed to get branch output num by actor:" << condition_gather_actor->GetAID();
  }
  const auto &output_value = condition_gather_actor->kernel()->GetAttr(kAttrBranchOutputNum);
  MS_EXCEPTION_IF_NULL(output_value);
  condition_gather_actor->branch_output_num_ = GetValue<size_t>(output_value);

  if (!condition_gather_actor->kernel()->HasAttr(kAttrBranchGraphName)) {
    MS_LOG(EXCEPTION) << "Failed to get inline graph name by actor:" << condition_gather_actor->GetAID();
  }
  const auto &branch_graph_names = condition_gather_actor->kernel()->GetAttr(kAttrBranchGraphName);
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

void InlineControlFlowScheduler::Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL) != kOptimizeO0) {
    for (const auto &graph : graph_compiler_info.graphs_) {
      MS_EXCEPTION_IF_NULL(graph);
      LinkControlArrowByExecutionOrder(graph, graph_compiler_info);
    }
  }
  for (const auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    if (kernel_actor->type() == KernelTransformType::kConditionSwitchActor) {
      HandleConditionSwitchActor(kernel_actor);
    } else if (kernel_actor->type() == KernelTransformType::kConditionGatherActor) {
      HandleConditionGatherActor(kernel_actor);
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
