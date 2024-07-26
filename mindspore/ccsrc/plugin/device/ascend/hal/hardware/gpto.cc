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
#include "plugin/device/ascend/hal/hardware/gpto.h"

#include <cmath>
#include <algorithm>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <string>

#include "ops/math_ops.h"
#include "ops/conv_pool_op_name.h"
#include "ops/ascend_op_name.h"
#include "utils/anf_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/misc.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "plugin/device/ascend/hal/device/ascend_memory_adapter.h"

namespace mindspore {
namespace gpto {
bool Overlap(const Time &start1, const Time &end1, const Time &start2, const Time &end2) {
  return (start1 >= start2 && start1 < end2) ||
         (start2 >= start1 && start2 < end1);  // if equal start and end for two intervals, then no overlap
}

std::pair<bool, std::string> GetDebugConfig() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto enable_save_graphs =
    (context_ptr->CanDump(kIntroductory)) || (common::GetEnv("MS_ENABLE_GPTO_VERIFICATION") != "");
  auto save_graphs_path = context_ptr->GetSaveGraphsPath();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  return std::make_pair(enable_save_graphs, save_graphs_path);
}

std::vector<std::pair<CNodePtr, CNodePtr>> ScheduleToEvents(const SchedulingOutput &schedule) {
  std::vector<std::pair<CNodePtr, CNodePtr>> events;  // to return
  // Distinguish types and sort
  std::set<Interval, SortByStart> tasks_start;
  std::set<Interval, SortByEnd> tasks_end;
  for (const auto &task_time : schedule.task_times) {
    tasks_start.insert(task_time);
    tasks_end.insert(task_time);
  }
  // Main loop
  for (auto it = tasks_start.begin(); it != tasks_start.end(); ++it) {
    tasks_end.erase(*it);
    // Dismiss overlapping tasks: save min end value of non-overlapping task to the right
    std::unordered_map<GptoTaskPtr, bool> dismissed;
    auto it1 = std::next(it);
    for (; Overlap(it->start, it->end, it1->start, it1->end) && it1 != tasks_start.end(); ++it1) {
      dismissed[it1->task] = true;
    }
    Time min_end_value = 0;
    for (auto it2 = tasks_end.begin(); it2 != tasks_end.end(); ++it2) {
      if (!dismissed[it2->task]) {
        min_end_value = it2->end;
        break;
      }
    }
    // Add events to immediate right neighborhood
    for (; it1->start < min_end_value && it1 != tasks_start.end(); ++it1) {
      if (it->task->gpto_type() != it1->task->gpto_type()) {
        events.emplace_back(it->task->cnode(), it1->task->cnode());
      }
    }
  }
  MS_LOG(INFO) << "Generated " << events.size() << " events";
  return events;
}

// Sorting for tasks
bool SortByCostMax(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->cost() > task2->cost() || (task1->cost() == task2->cost() && task1->id() < task2->id())));
}

bool SortByCostMin(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->cost() < task2->cost() || (task1->cost() == task2->cost() && task1->id() < task2->id())));
}

bool SortBySuccDiff(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->succ_diff_type() > task2->succ_diff_type() ||
           (task1->succ_diff_type() == task2->succ_diff_type() && task1->cost() > task2->cost()) ||
           (task1->succ_diff_type() == task2->succ_diff_type() && task1->cost() == task2->cost() &&
            task1->id() < task2->id())));
}

bool SortByBottomLevelMax(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->bottom_level() > task2->bottom_level() ||
           (task1->bottom_level() == task2->bottom_level() && task1->cost() > task2->cost()) ||
           (task1->bottom_level() == task2->bottom_level() && task1->cost() == task2->cost() &&
            task1->id() < task2->id())));
}

bool SortByBottomLevelMin(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->bottom_level() < task2->bottom_level() ||
           (task1->bottom_level() == task2->bottom_level() && task1->cost() > task2->cost()) ||
           (task1->bottom_level() == task2->bottom_level() && task1->cost() == task2->cost() &&
            task1->id() < task2->id())));
}

bool SortByTopLevelMax(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->top_level() > task2->top_level() ||
           (task1->top_level() == task2->top_level() && task1->cost() > task2->cost()) ||
           (task1->top_level() == task2->top_level() && task1->cost() == task2->cost() && task1->id() < task2->id())));
}

bool SortByTopLevelMin(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->top_level() < task2->top_level() ||
           (task1->top_level() == task2->top_level() && task1->cost() > task2->cost()) ||
           (task1->top_level() == task2->top_level() && task1->cost() == task2->cost() && task1->id() < task2->id())));
}

bool SortByBottomTopLevelMaxSum(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->top_level() + task1->bottom_level() > task2->top_level() + task2->bottom_level() ||
           (task1->top_level() + task1->bottom_level() == task2->top_level() + task2->bottom_level() &&
            task1->cost() > task2->cost()) ||
           (task1->top_level() + task1->bottom_level() == task2->top_level() + task2->bottom_level() &&
            task1->cost() == task2->cost() && task1->id() < task2->id())));
}

bool SortByBottomTopLevelMinSum(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->top_level() + task1->bottom_level() < task2->top_level() + task2->bottom_level() ||
           (task1->top_level() + task1->bottom_level() == task2->top_level() + task2->bottom_level() &&
            task1->cost() > task2->cost()) ||
           (task1->top_level() + task1->bottom_level() == task2->top_level() + task2->bottom_level() &&
            task1->cost() == task2->cost() && task1->id() < task2->id())));
}

bool SortByBottomTopLevelComposite(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->bottom_level() - task1->top_level() > task2->bottom_level() - task2->top_level() ||
           (task1->bottom_level() - task1->top_level() == task2->bottom_level() - task2->top_level() &&
            task1->cost() > task2->cost()) ||
           (task1->bottom_level() - task1->top_level() == task2->bottom_level() - task2->top_level() &&
            task1->cost() == task2->cost() && task1->id() < task2->id())));
}

bool SortByWeightedLength(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->weighted_length() > task2->weighted_length() ||
           (task1->weighted_length() == task2->weighted_length() && task1->id() < task2->id())));
}

bool SortByDepthMax(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->depth() > task2->depth() || (task1->depth() == task2->depth() && task1->cost() > task2->cost()) ||
           (task1->depth() == task2->depth() && task1->cost() == task2->cost() && task1->id() < task2->id())));
}

// BFS with costs for tie breaking
bool SortByDepthMin(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->depth() < task2->depth() || (task1->depth() == task2->depth() && task1->cost() > task2->cost()) ||
           (task1->depth() == task2->depth() && task1->cost() == task2->cost() && task1->id() < task2->id())));
}

// Sort by predecessor to comm
bool SortByPredComm(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->pred_comm() < task2->pred_comm() ||
           (task1->pred_comm() == task2->pred_comm() && task1->bottom_level() > task2->bottom_level()) ||
           (task1->pred_comm() == task2->pred_comm() && task1->bottom_level() == task2->bottom_level() &&
            task1->id() < task2->id())));
}

// Sort by predecessor to comm + DFS
bool SortByPredCommDepth(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->pred_comm() < task2->pred_comm() ||
           (task1->pred_comm() == task2->pred_comm() && task1->depth() > task2->depth()) ||
           (task1->pred_comm() == task2->pred_comm() && task1->depth() == task2->depth() &&
            task1->id() < task2->id())));
}

// Sort by predecessor to cube + bottom level
bool SortByPredCube(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->pred_cube() < task2->pred_cube() ||
           (task1->pred_cube() == task2->pred_cube() && task1->bottom_level() > task2->bottom_level()) ||
           (task1->pred_cube() == task2->pred_cube() && task1->bottom_level() == task2->bottom_level() &&
            task1->id() < task2->id())));
}

// Sort by greedy height of memory (maintained dynamically)
bool SortByGreedyHeight(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->current_mem_impact() < task2->current_mem_impact() ||
           (task1->current_mem_impact() == task2->current_mem_impact() && SortByBottomLevelMax(task1, task2))));
}

// Get PEs description
std::unordered_map<GptoTaskType, int32_t> GetPEs() {
  std::unordered_map<GptoTaskType, int32_t> new_pem;
  if (gpto_mode == kSingle) {
    new_pem[kComp] = 1;
  } else if (gpto_mode == kCompComm) {
    new_pem[kComp] = 1;
    new_pem[kComm] = 1;
  } else if (gpto_mode == kMulti) {
    new_pem[kComp] = 1;
    new_pem[kComm] = 1;
    new_pem[kCube] = 1;
  }
  return new_pem;
}

// Auxiliary subroutines and lower bounds
void ComputeDepthAndTopLevel(const std::vector<GptoTaskPtr> &tasks) {
  std::unordered_map<GptoTaskId, size_t> unprocessed_parents;
  std::queue<GptoTaskPtr> tasks_to_visit;
  // Initialization loop
  for (size_t j = 0; j < tasks.size(); ++j) {
    const auto &id = tasks[j]->id();
    unprocessed_parents[id] = tasks[j]->parents().size();
    if (unprocessed_parents[id] == 0) {
      tasks[j]->set_top_level(tasks[j]->cost());
      tasks_to_visit.push(tasks[j]);
    }
  }
  // Traversal loop
  while (!tasks_to_visit.empty()) {
    const auto &selected_task = tasks_to_visit.front();
    // Update candidate tasks
    for (auto &successor : selected_task->children()) {
      const auto &succ_id = successor->id();
      successor->set_depth(std::max(successor->depth(), selected_task->depth() + 1));
      successor->set_top_level(std::max(successor->top_level(), selected_task->top_level() + successor->cost()));
      unprocessed_parents[succ_id] -= 1;
      if (unprocessed_parents[succ_id] == 0) {
        tasks_to_visit.push(successor);
      }
    }
    tasks_to_visit.pop();
  }
}

void ComputeBottomLevelAndWeightedLength(const std::vector<GptoTaskPtr> &tasks) {
  std::unordered_map<GptoTaskId, size_t> unprocessed_children;
  std::unordered_map<GptoTaskId, double> children_sum;
  std::unordered_map<GptoTaskId, double> children_max;
  std::queue<GptoTaskPtr> tasks_to_visit;
  // Initialization loop: bottom and weighted_length already initialized to cost when AssignCost() is called
  for (auto &task : tasks) {
    const auto &id = task->id();
    unprocessed_children[id] = task->children().size();
    if (unprocessed_children[id] == 0) {
      tasks_to_visit.push(task);
    }
  }
  // Traversal loop
  while (!tasks_to_visit.empty()) {
    const auto &selected_task = tasks_to_visit.front();
    // Update candidate tasks
    for (auto &predecessor : selected_task->parents()) {
      const auto &pred_id = predecessor.lock()->id();
      predecessor.lock()->set_bottom_level(
        std::max(predecessor.lock()->bottom_level(), selected_task->bottom_level() + predecessor.lock()->cost()));
      children_sum[pred_id] += selected_task->weighted_length();
      children_max[pred_id] = std::max(children_max[pred_id], selected_task->weighted_length());
      unprocessed_children[pred_id] -= 1;
      if (unprocessed_children[pred_id] == 0) {
        if (children_max[pred_id] == 0) {
          MS_LOG(EXCEPTION) << "divisor children_max[pred_id] cannot be 0!";
        }
        predecessor.lock()->set_weighted_length(predecessor.lock()->cost() + children_max[pred_id] +
                                                children_sum[pred_id] / children_max[pred_id]);
        tasks_to_visit.push(predecessor.lock());
      }
    }
    tasks_to_visit.pop();
  }
}

void ComputePredComm(const std::vector<GptoTaskPtr> &tasks) {
  for (auto &task : tasks) {
    task->set_pred_comm(0);
    for (auto &predecessor : task->parents()) {
      if (predecessor.lock()->gpto_type() == kComm) {
        task->set_pred_comm(task->pred_comm() + 1);
      }
    }
  }
}

void ComputePredCube(const std::vector<GptoTaskPtr> &tasks) {
  for (auto &task : tasks) {
    task->set_pred_cube(0);
    for (auto &predecessor : task->parents()) {
      if (predecessor.lock()->gpto_type() == kCube) {
        task->set_pred_cube(task->pred_cube() + 1);
      }
    }
  }
}

void ComputeInitialMemoryImpact(const std::vector<GptoTaskPtr> &tasks) {
  for (auto &task : tasks) {
    Memory out_weight = 0, workspace_weight = 0;
    for (auto &tensor : task->out_tensors()) {
      if (tensor->type() == kWorkspace) {
        workspace_weight += tensor->weight();
      } else {
        out_weight += tensor->weight();
      }
    }
    for (auto &tensor : task->workspace_tensors()) {
      workspace_weight += tensor->weight();
    }
    task->set_workspace_memory(workspace_weight);
    task->set_initial_mem_impact(out_weight + workspace_weight);
    MS_LOG(DEBUG) << "Initial memory impact for task " << task->id() << " is " << task->initial_mem_impact()
                  << ", workspace is " << workspace_weight;
  }
}

Time LowerBoundBottomLevel(const std::vector<GptoTaskPtr> &tasks) {
  Time max_bottom_level = 0;
  for (const auto &task : tasks) {
    max_bottom_level = std::max(max_bottom_level, task->bottom_level());
  }
  return max_bottom_level;
}

Time LowerBoundPEs(const std::vector<GptoTaskPtr> &tasks,
                   const std::unordered_map<GptoTaskType, int32_t> &type_to_num_cores_map) {
  double lower_bound = 0;

  std::unordered_map<GptoTaskType, Time> type_task_sum;
  for (const auto &task : tasks) {
    type_task_sum[task->gpto_type()] += task->cost();
  }
  for (const auto &type_to_num : type_to_num_cores_map) {
    const auto &type = type_to_num.first;
    const auto &num_cores = type_to_num.second;
    if (num_cores == 0) {
      MS_LOG(EXCEPTION) << "divisor num_cores cannot be 0!";
    }
    lower_bound = std::max(lower_bound, type_task_sum[type] / (1.0 * num_cores));
  }
  return std::ceil(lower_bound);
}

// Main algorithms/subroutines
std::pair<PeId, Time> SelectPEandTime(const GptoTask &task, Time can_start,
                                      std::set<ProcessingElement, SortByLoad> *PEs_ptr) {
  MS_EXCEPTION_IF_NULL(PEs_ptr);
  auto &PEs = *PEs_ptr;
  std::pair<PeId, Time> return_pair = std::make_pair(0, 0);
  for (auto it = PEs.begin(); it != PEs.end(); ++it) {
    auto &mut_pe = const_cast<ProcessingElement &>(*it);
    // Put in first idle that fits it
    for (auto idle_it = mut_pe.idle.begin(); idle_it != mut_pe.idle.end(); ++idle_it) {
      Time start_time;
      bool case_flag = false;
      // Distinguish cases based on can_start constraint
      if (can_start <= idle_it->first) {
        start_time = idle_it->first;
      } else if (can_start <= idle_it->second) {
        start_time = can_start;
        case_flag = true;
      } else {
        continue;
      }
      // If the task fits, then place it here
      if (idle_it->second - start_time >= task.cost()) {
        // Save info to return: start task at time idle_it->first
        return_pair.first = (*it).id;
        return_pair.second = start_time;
        // Update idle list
        if (!case_flag) {
          if (idle_it->second - idle_it->first == task.cost()) {
            mut_pe.idle.erase(idle_it);
          } else {
            idle_it->first += task.cost();
          }
        } else {
          Time upper = idle_it->second;
          idle_it->second = can_start;
          if (upper - can_start - task.cost() > 0) {
            std::pair<Time, Time> new_idle = std::make_pair(can_start + task.cost(), upper);
            mut_pe.idle.emplace(std::next(idle_it), new_idle);
          }
        }
        // Update load and PEs set
        auto updated_PE = PEs.extract(it);
        updated_PE.value().load += task.cost();
        PEs.insert(std::move(updated_PE));
        return return_pair;
      }
    }
  }
  return return_pair;
}

std::pair<PeId, Time> SelectPEandTimeAvailableStart(const GptoTask &task, Time can_start,
                                                    std::vector<ProcessingElement> *PEs_ptr) {
  MS_EXCEPTION_IF_NULL(PEs_ptr);
  auto &PEs = *PEs_ptr;
  // Precompute min first available start for task
  Time min_start = SIZE_MAX;
  bool min_case = false;
  std::vector<ProcessingElement>::iterator min_it;
  std::list<std::pair<Time, Time>>::iterator min_idle_it;
  for (auto it = PEs.begin(); it != PEs.end(); ++it) {
    for (auto idle_it = it->idle.begin(); idle_it != it->idle.end(); ++idle_it) {
      Time start_time;
      bool case_flag = false;
      // Distinguish cases based on can_start constraint
      if (can_start <= idle_it->first) {
        start_time = idle_it->first;
      } else if (can_start <= idle_it->second) {
        start_time = can_start;
        case_flag = true;
      } else {
        continue;
      }
      if (idle_it->second - start_time >= task.cost()) {
        if (min_start > start_time) {
          min_start = start_time;
          min_case = case_flag;
          min_it = it;
          min_idle_it = idle_it;
          break;
        }
      }
    }
  }
  // Assign task to min PE
  std::pair<PeId, Time> return_pair = std::make_pair(0, 0);
  // Save info to return: start task at time idle_it->first
  return_pair.first = (*min_it).id;
  return_pair.second = min_start;
  // Update idle list
  if (!min_case) {
    if (min_idle_it->second - min_idle_it->first == task.cost()) {
      min_it->idle.erase(min_idle_it);
    } else {
      min_idle_it->first += task.cost();
    }
  } else {
    Time upper = min_idle_it->second;
    min_idle_it->second = can_start;
    if (upper - can_start - task.cost() > 0) {
      std::pair<Time, Time> new_idle = std::make_pair(can_start + task.cost(), upper);
      min_it->idle.emplace(std::next(min_idle_it), new_idle);
    }
  }
  // Update load
  min_it->load += task.cost();
  return return_pair;
}

constexpr TaskSortFunction TASK_SORT[] = {SortByCostMax,
                                          SortByCostMin,
                                          SortBySuccDiff,
                                          SortByBottomLevelMax,
                                          SortByBottomLevelMin,
                                          SortByTopLevelMax,
                                          SortByTopLevelMin,
                                          SortByBottomTopLevelMaxSum,
                                          SortByBottomTopLevelMinSum,
                                          SortByBottomTopLevelComposite,
                                          SortByWeightedLength,
                                          SortByDepthMax,
                                          SortByDepthMin,
                                          SortByPredComm,
                                          SortByPredCommDepth,
                                          SortByPredCube,
                                          SortByGreedyHeight};

constexpr std::string_view TASK_SORT_NAMES[] = {"SortByCostMax",
                                                "SortByCostMin",
                                                "SortBySuccDiff",
                                                "SortByBottomLevelMax",
                                                "SortByBottomLevelMin",
                                                "SortByTopLevelMax",
                                                "SortByTopLevelMin",
                                                "SortByBottomTopLevelMaxSum",
                                                "SortByBottomTopLevelMinSum",
                                                "SortByBottomTopLevelComposite",
                                                "SortByWeightedLength",
                                                "SortByDepthMax",
                                                "SortByDepthMin",
                                                "SortByPredComm",
                                                "SortByPredCommDepth",
                                                "SortByPredCube",
                                                "SortByGreedyHeight"};

constexpr std::string_view PE_NAME_SORT[] = {"SortByLoad", "SortByValidStart"};

SchedulingOutput Process(const SchedulingInput &input) {
  const std::vector<GptoTaskPtr> *tasks = &(input.tasks);
  auto type_to_num_cores_map = GetPEs();
  SchedulingOutput output{{}, SIZE_MAX, HARD_MEMORY_LIMIT};
  output.task_times.reserve(input.tasks.size());

  // Optional: verify input task graph is a DAG
  auto can_debug = GetDebugConfig();
  if (can_debug.first) {
    if (VerifyDAG(*tasks)) {
      MS_LOG(INFO) << "Verification of DAG: SUCCESS";
    } else {
      MS_LOG(ERROR) << "Verification of DAG: FAILURE";
      return output;
    }
  }

  // Preprocessing: values computation for necessary sorting
  ComputeBottomLevelAndWeightedLength(*tasks);
  // ComputeDepthAndTopLevel(*tasks); // already called earlier, necessary for nested conditional blocks
  ComputePredComm(*tasks);
  if (gpto_mode == kMulti) {
    ComputePredCube(*tasks);
  }
  ComputeInitialMemoryImpact(*tasks);

  // Loop over all sorting combinations
  std::unordered_map<GptoTaskPtr, Time> best_start;  // to use in verify dependencies only
  std::unordered_map<GptoTaskPtr, Time> best_end;
  std::string best_solution = "";
  MS_LOG(INFO) << "Start looping multiple scheduling functions";
  for (size_t task_sort = 0; task_sort < static_cast<size_t>(kNumTaskSort); ++task_sort) {
    for (size_t pes_sort = 0; pes_sort < static_cast<size_t>(PEsSort::kNumPEsSort); ++pes_sort) {
      if (common::GetEnv("MS_ENABLE_GPTO_ALGO") != "") {  // force specific algorithm
        if (common::GetEnv("MS_ENABLE_GPTO_ALGO") != TASK_SORT_NAMES[task_sort]) {
          continue;
        }
      }
      if (gpto_mode != kMulti && TASK_SORT_NAMES[task_sort] == "SortByPredCube") {
        continue;
      }
      if (pes_sort == static_cast<size_t>(PEsSort::kSortByValidStart)) {  // same solution for current default modes
        continue;
      }

      MS_LOG(INFO) << TASK_SORT_NAMES[task_sort] << " and " << PE_NAME_SORT[pes_sort];
      SchedulingOutput solution = ProcessCore(*tasks, type_to_num_cores_map, TASK_SORT[task_sort],
                                              (pes_sort == static_cast<size_t>(PEsSort::kSortByLoad)));
      if ((solution.makespan < output.makespan ||
           (solution.makespan == output.makespan && solution.memory_peak < output.memory_peak)) &&
          solution.memory_peak + PARAMETER_SIZE <= HARD_MEMORY_LIMIT) {
        output = solution;
        best_solution = TASK_SORT_NAMES[task_sort];
        for (const auto &task : *tasks) {
          best_start[task] = task->start();
          best_end[task] = task->end();
        }
      }
      for (const auto &task : *tasks) {
        task->ResetStartEnd();
      }
    }
  }
  MS_LOG(INFO) << "End looping multiple scheduling functions";

  if (best_solution == "") {
    output.makespan = SIZE_MAX;
    return output;
  }

  // Print stats about best solution
  MS_LOG(INFO) << "Memory-aware heuristics with soft memory limit " << SOFT_MEMORY_LIMIT << " and hard memory limit "
               << HARD_MEMORY_LIMIT;
  MS_LOG(INFO) << "Best solution is: " << best_solution;
  MS_LOG(INFO) << "Makespan of best solution is " << output.makespan;
  MS_LOG(INFO) << "Bottom level lower bound is " << LowerBoundBottomLevel(*tasks);
  MS_LOG(INFO) << "Max type lower bound is " << LowerBoundPEs(*tasks, type_to_num_cores_map);
  const size_t kPrecision = 5;
  const size_t kHundred = 100;
  MS_LOG(INFO) << "Solution relative error is " << std::setprecision(kPrecision)
               << ((output.makespan /
                      (1.0 * std::max(LowerBoundBottomLevel(*tasks), LowerBoundPEs(*tasks, type_to_num_cores_map))) -
                    1) *
                   kHundred)
               << "%";
  MS_LOG(INFO) << "GptoTensor peak memory estimate of best solution " << output.memory_peak << " ("
               << output.memory_peak / kMBToByte << " MB)";
  MS_LOG(INFO) << "Parameter memory estimate " << PARAMETER_SIZE << " (" << PARAMETER_SIZE / kMBToByte << " MB)";
  MS_LOG(INFO) << "Total memory estimate " << output.memory_peak + PARAMETER_SIZE << " ("
               << (output.memory_peak + PARAMETER_SIZE) / kMBToByte << " MB)";
  // Save best solution start/end values
  for (const auto &task : *tasks) {
    task->set_start(best_start[task]);
    task->set_end(best_end[task]);
  }

  return output;
}

void InitializeTasks(const std::vector<GptoTaskPtr> &tasks, std::unordered_map<GptoTaskId, Time> *can_start,
                     std::unordered_map<GptoTaskId, size_t> *unprocessed_parents,
                     std::set<GptoTaskPtr, TaskSortFunction> *candidate_tasks,
                     std::unordered_set<GptoTaskPtr> *switch_candidates) {
  MS_EXCEPTION_IF_NULL(can_start);
  MS_EXCEPTION_IF_NULL(unprocessed_parents);
  MS_EXCEPTION_IF_NULL(candidate_tasks);
  MS_EXCEPTION_IF_NULL(switch_candidates);
  for (auto &task : tasks) {
    const auto &id = task->id();
    (*can_start)[id] = static_cast<Time>(0);
    (*unprocessed_parents)[id] = static_cast<size_t>(task->parents().size());
    if ((*unprocessed_parents)[id] == static_cast<size_t>(0)) {
      candidate_tasks->insert(task);
      if (task->condition_switch()) {
        (*switch_candidates).insert(task);
      }
    }
  }
}

void InitializeProcessingElement(const std::unordered_map<GptoTaskType, int32_t> &type_to_num_cores_map, size_t *count,
                                 std::unordered_map<GptoTaskType, std::set<ProcessingElement, SortByLoad>> *PEs_load,
                                 std::unordered_map<GptoTaskType, std::vector<ProcessingElement>> *PEs_start,
                                 bool pe_load_sort) {
  MS_EXCEPTION_IF_NULL(count);
  MS_EXCEPTION_IF_NULL(PEs_load);
  MS_EXCEPTION_IF_NULL(PEs_start);

  for (const auto &type_to_num : type_to_num_cores_map) {
    const auto &type = type_to_num.first;
    const auto &num_cores = type_to_num.second;
    for (int i = 0; i < num_cores; ++i) {
      ProcessingElement new_pe;
      new_pe.id = *count + i;
      new_pe.gpto_type = type;
      new_pe.load = 0;
      new_pe.idle.emplace_back(0, SIZE_MAX);
      if (pe_load_sort) {
        (*PEs_load)[type].insert(new_pe);
      } else {
        (*PEs_start)[type].push_back(new_pe);
      }
    }
    *count += num_cores;
  }
}

GptoTaskPtr SelectTaskToSchedule(Memory cur_mem_peak, std::set<GptoTaskPtr, TaskSortFunction> *candidate_tasks,
                                 std::unordered_map<GptoTaskType, Memory> *last_workspace_memory) {
  MS_EXCEPTION_IF_NULL(candidate_tasks);
  MS_EXCEPTION_IF_NULL(last_workspace_memory);

  bool flag = false;
  GptoTaskPtr selected_task;
  for (auto it = (*candidate_tasks).begin(); it != (*candidate_tasks).end(); ++it) {
    selected_task = *it;
    if ((PARAMETER_SIZE + cur_mem_peak + selected_task->current_mem_impact() -
           (*last_workspace_memory)[selected_task->gpto_type()] <=
         SOFT_MEMORY_LIMIT) ||
        (selected_task->subgraph_id() < SIZE_MAX)) {
      flag = true;
      break;
    }
  }
  if (!flag) {
    selected_task = *((*candidate_tasks).begin());
  }
  return selected_task;
}

void UpdateMemoryImpactAndCandidates(
  std::set<GptoTaskPtr, TaskSortFunction> *candidate_tasks, const GptoTaskPtr &selected_task,
  std::unordered_map<size_t, std::set<std::weak_ptr<GptoTask>, GptoTask::SortByIdWeak>> *left_consumers,
  std::unordered_map<GptoTaskId, size_t> *unprocessed_parents, std::unordered_map<GptoTaskId, Time> *can_start,
  Time *last_end, Time *last_gather_end, std::unordered_set<GptoTaskPtr> *switch_candidates) {
  MS_EXCEPTION_IF_NULL(candidate_tasks);

  MS_EXCEPTION_IF_NULL(left_consumers);
  MS_EXCEPTION_IF_NULL(unprocessed_parents);
  MS_EXCEPTION_IF_NULL(can_start);
  MS_EXCEPTION_IF_NULL(last_end);
  MS_EXCEPTION_IF_NULL(last_gather_end);
  MS_EXCEPTION_IF_NULL(switch_candidates);

  for (auto &in_tensor : selected_task->in_tensors()) {
    const size_t &tid = in_tensor->id();
    (*left_consumers)[tid].erase(selected_task);
    if ((*left_consumers)[tid].size() == 1) {
      if (in_tensor->type() == kGraphOutput) {
        continue;
      }
      auto last_consumer = *((*left_consumers)[tid].begin());
      auto it = candidate_tasks->find(last_consumer.lock());
      if (it != candidate_tasks->end()) {
        auto updated_candidate = candidate_tasks->extract(it);
        updated_candidate.value()->set_current_mem_impact(updated_candidate.value()->current_mem_impact() -
                                                          in_tensor->weight());
        candidate_tasks->insert(std::move(updated_candidate));
      } else {
        last_consumer.lock()->set_current_mem_impact(last_consumer.lock()->current_mem_impact() - in_tensor->weight());
      }
    }
  }
  // Update out-tensors of selected node
  for (auto &out_tensor : selected_task->out_tensors()) {
    if (out_tensor->consumers().size() == 1 && out_tensor->type() != kGraphOutput) {
      auto last_consumer = *(out_tensor->consumers().begin());
      last_consumer.lock()->set_current_mem_impact(last_consumer.lock()->current_mem_impact() - out_tensor->weight());
    }
  }

  // Update candidate tasks
  candidate_tasks->erase(selected_task);
  if (selected_task->condition_switch()) {
    (*switch_candidates).erase(selected_task);
  }

  // Update can_start with special processing for ConditionalSwitch/Gather cases
  if (selected_task->condition_gather()) {
    for (const auto &candidate : *candidate_tasks) {
      (*can_start)[candidate->id()] =
        std::max((*can_start)[candidate->id()], static_cast<uint64_t>(selected_task->end()));
    }
    *last_gather_end = std::max(*last_gather_end, selected_task->end());
  }

  *last_end = std::max(*last_end, selected_task->end());
  for (const auto &candidate : *switch_candidates) {
    (*can_start)[candidate->id()] = std::max((*can_start)[candidate->id()], static_cast<uint64_t>(*last_end));
  }

  for (const auto &successor : selected_task->children()) {
    const auto &succ_id = successor->id();
    (*can_start)[succ_id] = std::max((*can_start)[succ_id], static_cast<uint64_t>(selected_task->end()));
    (*can_start)[succ_id] = std::max((*can_start)[succ_id], static_cast<uint64_t>(*last_gather_end));
    if (successor->condition_switch()) {
      (*can_start)[succ_id] = std::max((*can_start)[succ_id], static_cast<uint64_t>(*last_end));
    }
    (*unprocessed_parents)[succ_id] -= 1;
    if ((*unprocessed_parents)[succ_id] == 0) {
      candidate_tasks->insert(successor);
      if (successor->condition_switch()) {
        (*switch_candidates).insert(successor);
      }
    }
  }
}

SchedulingOutput ProcessCore(const std::vector<GptoTaskPtr> &tasks,
                             const std::unordered_map<GptoTaskType, int32_t> &type_to_num_cores_map,
                             const TaskSortFunction &sortPtr, bool pe_load_sort) {
  SchedulingOutput output{{}, 0, 0};
  output.task_times.reserve(tasks.size());
  // Initializations for tasks
  std::set<GptoTaskPtr, TaskSortFunction> candidate_tasks(sortPtr);
  std::unordered_map<GptoTaskId, Time> can_start;
  std::unordered_map<GptoTaskId, size_t> unprocessed_parents;
  std::unordered_set<GptoTaskPtr> switch_candidates;
  InitializeTasks(tasks, &can_start, &unprocessed_parents, &candidate_tasks, &switch_candidates);

  std::unordered_map<size_t, std::set<std::weak_ptr<GptoTask>, GptoTask::SortByIdWeak>> left_consumers;
  for (const auto &task : tasks) {
    for (const auto &in_tensor : task->in_tensors()) {
      left_consumers[in_tensor->id()].insert(in_tensor->consumers().begin(), in_tensor->consumers().end());
    }
  }

  // Initialization for memory impact handling
  for (auto &task : tasks) {
    task->set_current_mem_impact(task->initial_mem_impact());
  }

  // Initializations for processing elements
  // Pick a sorting for processing elements
  // Implemented: SortByLoad, SortByAvailableStart
  // Only one structure to be used depending on argument; we define both here
  std::unordered_map<GptoTaskType, std::set<ProcessingElement, SortByLoad>> PEs_load;
  std::unordered_map<GptoTaskType, std::vector<ProcessingElement>> PEs_start;
  size_t count = 0;
  InitializeProcessingElement(type_to_num_cores_map, &count, &PEs_load, &PEs_start, pe_load_sort);

  // Task graph scheduling loop
  output.memory_peak = 0;
  Memory cur_mem_peak = 0;
  std::unordered_map<GptoTaskType, Memory> last_workspace_memory;  // comp/comm for now
  last_workspace_memory[kComp] = 0;
  if (gpto_mode != kSingle) {
    last_workspace_memory[kComm] = 0;
  }
  Time last_end = 0;
  Time last_gather_end = 0;
  while (!candidate_tasks.empty()) {
    // Select task and schedule it (memory-aware), save info for output
    GptoTaskPtr selected_task = SelectTaskToSchedule(cur_mem_peak, &candidate_tasks, &last_workspace_memory);
    const auto &selected_id = selected_task->id();

    // Maintain memory peak information
    cur_mem_peak += selected_task->current_mem_impact() - last_workspace_memory[selected_task->gpto_type()];
    MS_LOG(DEBUG) << "Current memory peak " << cur_mem_peak << " after task " << selected_task->id();
    last_workspace_memory[selected_task->gpto_type()] = selected_task->workspace_memory();
    output.memory_peak = std::max(output.memory_peak, cur_mem_peak);

    // Selected PE and start time
    std::pair<PeId, Time> PE_and_time;
    if (pe_load_sort) {
      PE_and_time = SelectPEandTime(*selected_task, can_start[selected_id], &PEs_load[selected_task->gpto_type()]);
    } else {
      PE_and_time =
        SelectPEandTimeAvailableStart(*selected_task, can_start[selected_id], &PEs_start[selected_task->gpto_type()]);
    }
    const auto &sigma = PE_and_time.second;

    // Maintenance of task interval
    selected_task->set_start(sigma);
    selected_task->set_end(sigma + selected_task->cost());
    // New interval for task in output
    Interval new_interval{selected_task, selected_task->start(), selected_task->end()};
    output.task_times.push_back(new_interval);
    // Update makespan
    output.makespan = std::max(output.makespan, selected_task->end());
    // Update memory impact values (no need for workspace memory removal here)
    UpdateMemoryImpactAndCandidates(&candidate_tasks, selected_task, &left_consumers, &unprocessed_parents, &can_start,
                                    &last_end, &last_gather_end, &switch_candidates);
  }
  MS_LOG(INFO) << "Makespan is " << output.makespan;
  MS_LOG(INFO) << "Peak mem is " << output.memory_peak;

  // Verification of scheduling solution (optional)
  auto can_debug = GetDebugConfig();
  if (can_debug.first) {
    if (VerifyScheduling(tasks)) {
      MS_LOG(INFO) << "Verification of Scheduling: SUCCESS";
    } else {
      MS_LOG(ERROR) << "Verification of Scheduling: FAILURE";
      output.makespan = SIZE_MAX;
    }
  }
  return output;
}

bool VerifyScheduling(const std::vector<GptoTaskPtr> &tasks) {
  bool flag = true;
  MS_LOG(INFO) << "Start Verification of Scheduling";
  for (const auto &task : tasks) {
    // Check if task is scheduled before its children
    for (auto child = task->children().begin(); child != task->children().end(); ++child) {
      if (!(task->start() < task->end() && task->end() <= (*child)->start() &&
            (*child)->start() < (*child)->end())) {  // assume open-rightpoint intervals and non-zero size
        MS_LOG(INFO) << "Verification violation: task " << task->id() << " [" << task->start() << "," << task->end()
                     << "] and task " << (*child)->id() << " [" << (*child)->start() << "," << (*child)->end() << "]";
        flag = false;
      }
    }
  }
  MS_LOG(INFO) << "End Verification of Scheduling";
  return flag;
}

bool VerifyDAG(const std::vector<GptoTaskPtr> &tasks) {
  // simple verifier: no directed cycle exists
  std::unordered_map<GptoTaskId, bool> visited;
  std::unordered_map<GptoTaskId, size_t> unprocessed_parents;
  std::deque<GptoTaskPtr> to_visit;
  MS_LOG(INFO) << "Start Verification of DAG";
  for (const auto &task : tasks) {
    const auto &id = task->id();
    visited[id] = false;
    unprocessed_parents[id] = task->parents().size();
    if (unprocessed_parents[id] == 0) {
      to_visit.push_back(task);
    }
  }
  while (!to_visit.empty()) {
    const auto selected_task = *(to_visit.begin());
    const auto &selected_id = selected_task->id();
    if (visited[selected_id]) {
      MS_LOG(INFO) << "Cycle including task " << selected_id;
      return false;
    } else {
      visited[selected_id] = true;
    }
    to_visit.pop_front();
    for (const auto &successor : selected_task->children()) {
      const auto &succ_id = successor->id();
      unprocessed_parents[succ_id] -= 1;
      if (unprocessed_parents[succ_id] == 0) {
        to_visit.push_back(successor);
      }
    }
  }
  MS_LOG(INFO) << "End Verification of DAG";
  return true;
}

void LogSchedulingOutput(const SchedulingOutput &output, const std::unordered_map<CNodePtr, GptoTaskPtr> &cnode_to_task,
                         const std::vector<std::pair<CNodePtr, CNodePtr>> &events, const KernelGraphPtr &kernel_graph,
                         const std::set<GptoTensorPtr, GptoTensorIdSort> &tensors, const Time lower,
                         const Memory memory_lower_bound, const std::string &path) {
  const size_t graph_id = kernel_graph->graph_id();
  std::stringstream ss;
  std::ostringstream oss;
  std::string filename;
  ss << kernel_graph;
  filename = path + std::string("/") + std::string("gpto_out_") + std::to_string(graph_id) + std::string("_") +
             ss.str() + std::string(".log");
  // Print info for tasks
  const auto &intervals = output.task_times;
  for (const auto &interval : intervals) {
    oss << "TASK id=" << std::to_string(interval.task->id()) << ", name=" << interval.task->name()
        << ", type=" << std::to_string(interval.task->gpto_type()) << ", start=" << std::to_string(interval.start)
        << ", end=" << std::to_string(interval.end) << "\n";
  }
  // Print events (scheduling dependencies)
  for (const auto &event : events) {
    const auto &source = event.first;
    const auto &dst = event.second;
    oss << "EVENT " << std::to_string(cnode_to_task.at(source)->id()) << " "
        << std::to_string(cnode_to_task.at(dst)->id()) << "\n";
  }

  // Print makespan and memory bounds
  oss << "UPPER " << output.makespan << "\n";
  oss << "LOWER " << lower << "\n";
  oss << "SOFT_MEMORY_LIMIT " << SOFT_MEMORY_LIMIT << "\n";
  oss << "HARD_MEMORY_LIMIT " << HARD_MEMORY_LIMIT << "\n";
  oss << "PARAMETER_SIZE " << PARAMETER_SIZE << "\n";
  oss << "MEMORY LOWER BOUND " << memory_lower_bound << "\n";

  // Print edges
  for (const auto &interval : intervals) {
    for (const auto &child : interval.task->children()) {
      oss << "EDGE " << std::to_string(interval.task->id()) << " " << std::to_string(child->id()) << "\n";
    }
  }

  // Print tensor info
  for (const auto &tensor : tensors) {
    std::string consumers = std::accumulate(tensor->consumers().begin(), tensor->consumers().end(), std::string{},
                                            [](std::string consumers_str, const auto &consumer) {
                                              return consumers_str += std::to_string(consumer.lock()->id()) + ";";
                                            });
    oss << "TENSOR id=" << std::to_string(tensor->id()) << ", weight=" << std::to_string(tensor->weight())
        << ", source=" << std::to_string(tensor->source().lock()->id()) << ", type=" << std::to_string(tensor->type())
        << ", consumers=" << consumers << "\n";
  }

  (void)Common::SaveStringToFile(filename, oss.str());
}

void ComputeAncestorsDescendants(const std::vector<GptoTaskPtr> &tasks,
                                 std::vector<mindspore::somas::DynamicBitSet> *nodes_dependency) {
  // Assume tasks are sorted by id (ie in BFS order); if not, sort them
  // Do we need each node to be ancestor/descendant of itself? No (for now)

  MS_EXCEPTION_IF_NULL(nodes_dependency);
  MS_EXCEPTION_IF_NULL(tasks.back());
  size_t count = tasks.back()->id() + 1;
  for (size_t i = 0; i < count; i++) {
    (void)nodes_dependency->emplace_back(count);
  }
  for (const auto &task : tasks) {
    for (const auto &parent : task->parents()) {
      auto &elem = (*nodes_dependency)[task->id()];
      elem.SetBitTrue(parent.lock()->id());
      Union(&((*nodes_dependency)[task->id()]), &((*nodes_dependency)[parent.lock()->id()]));
    }
    // Log message just for debugging
    MS_LOG(DEBUG) << "Task " << task->id() << " has " << (*nodes_dependency)[task->id()].CountOnesNum() << "ancestors";
  }
}

void InsertEdges(const KernelGraphPtr &kernel_graph, std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);

  const std::list<CNodePtr> &cnode_list = kernel_graph->GetOrderedCnodes();
  std::vector<CNodePtr> cnode_vec(cnode_list.cbegin(), cnode_list.cend());
  auto &cnode_to_task = *cnode_to_task_map_ptr;

  std::queue<CNodePtr> visit_queue;
  std::unordered_map<CNodePtr, size_t> unprocessed_children;
  std::unordered_map<CNodePtr, std::unordered_set<GptoTaskPtr>> real_children;

  // Initialization loops
  for (size_t i = 0; i < cnode_vec.size(); ++i) {
    unprocessed_children[cnode_vec[i]] = 0;
  }
  for (size_t i = 0; i < cnode_vec.size(); ++i) {
    for (size_t j = 0; j < cnode_vec[i]->size(); ++j) {
      if (!(cnode_vec[i]->input(j)->isa<CNode>())) {
        continue;
      }
      const auto &input_node = cnode_vec[i]->input(j)->cast<CNodePtr>();
      unprocessed_children[input_node] = unprocessed_children[input_node] + 1;
    }
  }
  for (size_t i = 0; i < cnode_vec.size(); ++i) {
    if (unprocessed_children[cnode_vec[i]] == 0) {
      visit_queue.push(cnode_vec[i]);
    }
  }

  // CNode graph traversal loop
  while (!visit_queue.empty()) {
    const auto &visit_cnode = visit_queue.front();
    MS_LOG(DEBUG) << "Visit cnode " << visit_cnode->UniqueName();
    if (cnode_to_task.count(visit_cnode) > 0) {  // if real kernel, then add edges
      const auto &visit_task = cnode_to_task[visit_cnode];
      for (const auto &real_child : real_children[visit_cnode]) {
        visit_task->AddChild(real_child);
        real_child->AddParent(visit_task);
        MS_LOG(DEBUG) << "Edge " << visit_task->id() << " " << real_child->id();
        MS_LOG(DEBUG) << "Edge (UniqueName) " << visit_cnode->UniqueName() << " " << real_child->cnode()->UniqueName();
      }
      real_children[visit_cnode].clear();
      real_children[visit_cnode].insert(visit_task);
    }
    // Maintain real_children and visit_queue
    for (size_t j = 1; j < visit_cnode->size(); ++j) {
      if (!visit_cnode->input(j)->isa<CNode>()) {
        continue;
      }
      const auto &parent_cnode = visit_cnode->input(j)->cast<CNodePtr>();
      for (const auto &real_child : real_children[visit_cnode]) {
        real_children[parent_cnode].insert(real_child);
      }
      unprocessed_children[parent_cnode]--;
      if (unprocessed_children[parent_cnode] == 0) {
        visit_queue.push(parent_cnode);
      }
    }
    visit_queue.pop();
  }
}

bool IsCubeKernel(const CNodePtr &node) {
  static const std::unordered_set<std::string> kCubeKernelSet = {
    // matmul
    kMatMulOpName, kMatMulV2OpName, kBatchMatMulOpName, kBatchMatMulV2OpName,
    // conv
    kConv2DOpName, kConv3DOpName,
    // conv dx
    kConv2DBackpropInputOpName, kConv2DBackpropInputDOpName, kConv2DTransposeOpName, kConv2DTransposeDOpName,
    kDepthwiseConv2DBackpropInputOpName, kDepthwiseConv2DBackpropInputDOpName, kConv3DBackpropInputOpName,
    kConv3DBackpropInputDOpName, kConv3DTransposeOpName, kConv3DTransposeDOpName,
    // conv dw
    kConv2DBackpropFilterOpName, kConv2DBackpropFilterDOpName, kDepthwiseConv2DBackpropFilterOpName,
    kDepthwiseConv2DBackpropFilterDOpName, kConv3DBackpropFilterOpName, kConv3DBackpropFilterDOpName};

  auto op_name = common::AnfAlgo::GetCNodeName(node);
  return kCubeKernelSet.find(op_name) != kCubeKernelSet.end();
}

GptoTaskType GetType(const CNodePtr cnode) {
  if (gpto_mode == kSingle) {
    return kComp;
  } else if (common::AnfAlgo::IsCommunicationOp(cnode)) {
    return kComm;
  } else if (gpto_mode == kCompComm) {
    return kComp;
  } else {  // gpto_mode == kMulti && cnode is not KComm
    return IsCubeKernel(cnode) ? kCube : kComp;
  }
}

GptoTaskType GetRealType(const CNodePtr cnode) {
  if (common::AnfAlgo::IsCommunicationOp(cnode)) {
    return kComm;
  } else if (IsCubeKernel(cnode)) {
    return kCube;
  } else {
    return kComp;
  }
}

size_t GetAlignedSize(size_t original_size) {
  constexpr size_t alignment = 512;
  constexpr size_t alignment_complement = 31;
  size_t aligned_size =
    (original_size > 0) ? ((original_size + alignment + alignment_complement) / alignment) * alignment : 0;
  return aligned_size;
}

KernelWithIndex GetVisitKernelWithReturnType(const AnfNodePtr &ori_node, size_t ori_index,
                                             std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);

  auto prenode = common::AnfAlgo::VisitKernelWithReturnType(ori_node, ori_index, false);
  while (prenode.first->isa<CNode>() &&
         cnode_to_task_map_ptr->find(prenode.first->cast<CNodePtr>()) == cnode_to_task_map_ptr->end()) {
    auto cnode = prenode.first->cast<CNodePtr>();
    if (!common::AnfAlgo::IsNopNode(cnode) &&
        !(IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) && common::AnfAlgo::GetInputNum(cnode) == 1)) {
      MS_LOG(INTERNAL_EXCEPTION) << "Node[" << ori_node->fullname_with_scope() << "] find input node["
                                 << cnode->fullname_with_scope()
                                 << "] doesn't exist in cnode_to_task map and is not a nop node!";
    }
    prenode = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(1), 0, false);
  }
  return prenode;
}

void ExtractOutputWorkspaceTensors(const SchedulingInput &scheduling_input, const std::vector<GptoTaskPtr> &tasks) {
  size_t tensor_count = 0;
  // Looping over tasks to obtain output and workspace tensors (somas style)
  for (auto &task : tasks) {
    const auto &kernel_mod = AnfAlgo::GetKernelMod(task->cnode());
    MS_EXCEPTION_IF_NULL(kernel_mod);

    // Extract each node's output tensors
    task->out_tensors().reserve(kernel_mod->GetOutputSizeList().size());
    for (const auto &size : kernel_mod->GetOutputSizeList()) {
      Memory weight = GetAlignedSize(size);
      if (weight == 0) {
        weight = GetAlignedSize(1);
      }
      GptoTensorPtr new_tensor = std::make_shared<GptoTensor>(tensor_count, size, weight, task,
                                                              kWorkspace);  // initially kWorkspace, since no consumers
      task->out_tensors().push_back(new_tensor);
      MS_LOG(DEBUG) << "New output tensor " << tensor_count << " source id " << task->id() << " weight " << weight;
      tensor_count++;
    }

    // Extract each node's workspace tensor
    task->workspace_tensors().reserve(kernel_mod->GetWorkspaceSizeList().size());
    for (const auto &size : kernel_mod->GetWorkspaceSizeList()) {
      Memory weight = GetAlignedSize(size);
      if (weight == 0) {
        weight = GetAlignedSize(1);
      }
      GptoTensorPtr new_tensor = std::make_shared<GptoTensor>(tensor_count, size, weight, task, kWorkspace);
      task->workspace_tensors().push_back(new_tensor);
      MS_LOG(DEBUG) << "New workspace tensor " << tensor_count << " source id " << task->id() << " weight " << weight;
      tensor_count++;
    }
  }
}

void CleanWorkspace(CNodePtr pre_node, const GptoTaskPtr &pre_task, const GptoTaskPtr &task) {
  auto clean_workspace_indexs = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicWorkspaceIndexs);
  for (const auto &index : clean_workspace_indexs) {
    if (index > pre_task->out_tensors().size()) {
      MS_LOG(INFO) << "Workspace index " << index << " exceed input node [" << pre_node->fullname_with_scope()
                   << "]'s Workspace size " << pre_task->workspace_tensors().size();
      continue;
    }
    auto input_tensor = pre_task->workspace_tensors()[index];
    MS_EXCEPTION_IF_NULL(input_tensor);
    task->in_tensors().push_back(input_tensor);
    if (input_tensor->type() == GptoTensorType::kWorkspace) {
      input_tensor->set_type(GptoTensorType::kSimple);
    }
    input_tensor->consumers().insert(task);
  }
}

void CleanOutput(size_t index, CNodePtr pre_node, GptoTaskPtr pre_task, const GptoTaskPtr &task) {
  if (index > pre_task->out_tensors().size()) {
    MS_LOG(INFO) << "Output index " << index << " exceed input node [" << pre_node->fullname_with_scope()
                 << "]'s outputs size " << pre_task->out_tensors().size();
    return;
  }
  auto input_tensor = pre_task->out_tensors()[index];
  MS_EXCEPTION_IF_NULL(input_tensor);
  task->in_tensors().push_back(input_tensor);
  if (input_tensor->type() == GptoTensorType::kWorkspace) {
    input_tensor->set_type(GptoTensorType::kSimple);
  }
  input_tensor->consumers().insert(task);
}

void StandardInputCase(const GptoTaskPtr &task, std::unordered_set<void *> *parameter_set, const CNodePtr &kernel,
                       std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_gpto_map_ptr) {
  MS_EXCEPTION_IF_NULL(parameter_set);

  MS_EXCEPTION_IF_NULL(cnode_to_task_gpto_map_ptr);

  const auto &input_size_list = AnfAlgo::GetNodeInputSizeList(kernel);
  auto input_tensor_num = common::AnfAlgo::GetInputTensorNum(kernel);
  MS_LOG(DEBUG) << "StandardInputCase Task " << task->id() << " " << task->cnode()->fullname_with_scope();
  for (size_t i = 0; i < input_tensor_num; i++) {
    auto input_node = kernel->input(i + 1);
    MS_EXCEPTION_IF_NULL(input_node);
    KernelWithIndex prenode_index = GetVisitKernelWithReturnType(input_node, 0, cnode_to_task_gpto_map_ptr);
    MS_EXCEPTION_IF_NULL(prenode_index.first);
    if (common::AnfAlgo::CheckPrimitiveType(prenode_index.first, prim::kPrimMakeTuple)) {
      MS_LOG(INTERNAL_EXCEPTION) << "Node " << kernel->fullname_with_scope() << "'s input node ["
                                 << input_node->DebugString() << "]'s input " << i << " is MakeTuple";
    }

    if (!AnfUtils::IsRealCNodeKernel(prenode_index.first)) {  // somas input parameter case
      MS_LOG(DEBUG) << "Input  [" << prenode_index.first->fullname_with_scope() << "] is not a real cnode kernel.";
      MS_LOG(DEBUG) << "Checking input parameter";
      auto op_name = common::AnfAlgo::GetCNodeName(kernel);
      TypeId input_origin_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel, i);
      if ((op_name == kDynamicRNNOpName || op_name == kDynamicGRUV2OpName) && input_origin_type == kMetaTypeNone) {
        continue;
      }
      size_t input_size = 0;
      if (i >= input_size_list.size()) {
        MS_LOG(DEBUG) << "Node: " << kernel->fullname_with_scope() << " input idx: " << i
                      << " greater than the size of input_size_list: " << input_size_list.size()
                      << ", so use 0 as parameter size.";
      } else {
        input_size = input_size_list.at(i);
      }
      if (parameter_set->find(prenode_index.first.get()) == parameter_set->end()) {
        parameter_set->insert(prenode_index.first.get());
        PARAMETER_SIZE += input_size;
      }
      continue;
    }
    auto iter = cnode_to_task_gpto_map_ptr->find(prenode_index.first->cast<CNodePtr>());
    if (iter == cnode_to_task_gpto_map_ptr->end()) {
      MS_LOG(DEBUG) << "Kernel[" << kernel->fullname_with_scope() << "]'s input " << i << " ["
                    << prenode_index.first->fullname_with_scope() << "] not found in tasks";
      continue;
    }
    auto pre_task = iter->second;
    if (pre_task->out_tensors().size() == 0) {
      MS_LOG(DEBUG) << "Precedent task " << pre_task->name() << " does not have output tensors";
      continue;
    }
    if (prenode_index.second > pre_task->out_tensors().size()) {
      MS_LOG(DEBUG) << "Output index " << prenode_index.second << " exceeds input node ["
                    << prenode_index.first->fullname_with_scope() << "]'s outputs size "
                    << pre_task->out_tensors().size();
      continue;
    }
    auto input_tensor = pre_task->out_tensors()[prenode_index.second];
    MS_EXCEPTION_IF_NULL(input_tensor);
    input_tensor->consumers().insert(task);
    task->in_tensors().push_back(input_tensor);
    MS_LOG(DEBUG) << "GptoTensor " << input_tensor->id() << " has new consumer " << task->id();
    if (input_tensor->type() == GptoTensorType::kWorkspace) {
      input_tensor->set_type(GptoTensorType::kSimple);
    }
  }
}
void ExtractRealTensors(const SchedulingInput &scheduling_input,
                        std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_gpto_map_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_gpto_map_ptr);
  const auto &tasks = scheduling_input.tasks;

  ExtractOutputWorkspaceTensors(scheduling_input, tasks);

  // Looping over tasks to obtain input tensors after all outputs have been saved
  PARAMETER_SIZE = 0;
  static std::unordered_set<void *> parameter_set;
  for (auto &task : tasks) {
    const auto &kernel = task->cnode();
    const auto &kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);

    if (common::AnfAlgo::GetCNodeName(kernel) != kMemSetOpName) {  // standard input case
      StandardInputCase(task, &parameter_set, kernel, cnode_to_task_gpto_map_ptr);
    } else {  // atomic clean input case
      auto input_tensor_num = common::AnfAlgo::GetInputTensorNum(kernel);
      for (size_t i = 0; i < input_tensor_num; i++) {
        auto pre_node = kernel->input(i + 1)->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(pre_node);

        auto iter = cnode_to_task_gpto_map_ptr->find(pre_node);
        if (iter == cnode_to_task_gpto_map_ptr->end()) {
          MS_LOG(DEBUG) << "Kernel[" << kernel->fullname_with_scope() << "]'s input " << i << " ["
                        << pre_node->fullname_with_scope() << "] not found in tasks";
          continue;
        }
        auto pre_task = iter->second;

        if (common::AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, pre_node)) {  // clean output
          auto clean_output_indexs =
            common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicOutputIndexs);
          for (auto index : clean_output_indexs) {
            CleanOutput(index, pre_node, pre_task, task);
          }
        }

        if (common::AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, pre_node)) {  // clean workspace
          CleanWorkspace(pre_node, pre_task, task);
        }
      }
    }
  }
  parameter_set.clear();
}

size_t CalculateVectorCost(const CNodePtr &cnode) {
  Time cost = 0;
  if (common::AnfAlgo::GetInputTensorNum(cnode) == 0) {
    return cost;
  }
  KernelWithIndex kernel_with_index_1 = common::AnfAlgo::GetPrevNodeOutput(cnode, 0);
  ShapeVector shape_1 = common::AnfAlgo::GetOutputInferShape(kernel_with_index_1.first, kernel_with_index_1.second);
  const TypeId type_1 = common::AnfAlgo::GetOutputInferDataType(kernel_with_index_1.first, 0);
  size_t type_size_1 = GetDataTypeSize(type_1);
  size_t compute_count = std::accumulate(shape_1.cbegin(), shape_1.cend(), 1, std::multiplies<size_t>{});
  const double kLatency = 0.5;
  const size_t kParallel = 128;
  cost = kLatency + (compute_count * type_size_1 / kParallel);
  return cost;
}

size_t CalculateCubeCost(const CNodePtr &cnode) {
  Time cost = 0;
  // Get info of input 1
  size_t batch = 1;
  KernelWithIndex kernel_with_index_1 = common::AnfAlgo::GetPrevNodeOutput(cnode, 0);
  ShapeVector shape_1 = common::AnfAlgo::GetOutputInferShape(kernel_with_index_1.first, kernel_with_index_1.second);

  // Get info of input 2
  KernelWithIndex kernel_with_index_2 = common::AnfAlgo::GetPrevNodeOutput(cnode, 1);
  ShapeVector shape_2 = common::AnfAlgo::GetOutputInferShape(kernel_with_index_2.first, kernel_with_index_2.second);

  // Get info of output
  ShapeVector shape_out = common::AnfAlgo::GetOutputInferShape(cnode, 0);

  // Remove batch if operator is batchmatmul
  const size_t kShapeSizeFour = 4;
  if (IsPrimitiveCNode(cnode, prim::kPrimBatchMatMul) || IsPrimitiveCNode(cnode, prim::kPrimBatchMatMulV2)) {
    batch = shape_1.front();
    if (shape_1.size() == kShapeSizeFour) {
      shape_1.erase(shape_1.begin());
      shape_1.erase(shape_1.begin());
      shape_out.erase(shape_out.begin());
      shape_out.erase(shape_out.begin());
    } else {
      shape_1.erase(shape_1.begin());
      shape_2.erase(shape_2.begin());
      shape_out.erase(shape_out.begin());
    }
  }

  // Find MKN
  size_t k = 0;
  size_t m = 0;
  size_t n = 0;
  std::vector<size_t> tmp;
  std::copy(shape_1.begin(), shape_1.end(), back_inserter(tmp));
  for (auto dim : shape_2) {
    bool found_in_input = std::find(tmp.begin(), tmp.end(), dim) != tmp.end();
    bool found_in_output = std::find(shape_out.begin(), shape_out.end(), dim) != shape_out.end();
    if (found_in_input && k == 0 && !found_in_output) {
      k = dim;
      tmp.erase(std::remove(tmp.begin(), tmp.end(), dim), tmp.end());
    } else if (found_in_input && k == 0 && found_in_output && n != 0) {
      k = dim;
    } else {
      n = dim;
    }
  }
  m = tmp[0];

  // Get info of dtype
  const TypeId type_1 = common::AnfAlgo::GetOutputInferDataType(kernel_with_index_1.first, 0);
  size_t type_size_1 = GetDataTypeSize(type_1);
  const Time kLatency = 21;
  const Time kParallel = 8192;
  cost = kLatency + batch * m * k * n * type_size_1 / kParallel;
  return cost;
}

size_t CalculateCommCost(const CNodePtr &cnode) {
  Time cost = 0;
  size_t output_num = AnfUtils::GetOutputTensorNum(cnode);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);

  // For each operator we get the inputs and outputs
  // For each inputs, we multiply the shape to have the total size and we multiply the size by the data type
  // We then sum all inputs
  // If there is more than 1 output, we do the same for the outputs
  // If output == 1 then cost is 0. We then sum all outputs
  // We sum inputs cost + outputs cost
  for (size_t j = 0; j < input_num; j++) {
    KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, j);
    if (dyn_cast<abstract::BaseShape>(kernel_with_index.first->Shape()) == nullptr ||
        dyn_cast<Type>(kernel_with_index.first->Type()) == nullptr) {
      MS_LOG(DEBUG) << "shape or type is nullptr, ignore";
      continue;
    }
    ShapeVector shape = common::AnfAlgo::GetOutputInferShape(kernel_with_index.first, kernel_with_index.second);
    if (shape.size() <= 0) {
      continue;
    }

    const TypeId type = common::AnfAlgo::GetOutputInferDataType(kernel_with_index.first, 0);
    if (type == kObjectTypeUMonad || type == kObjectTypeMonad || type == kObjectTypeFunction) {
      continue;
    }

    size_t type_size = GetDataTypeSize(type);
    cost += std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * type_size;
  }

  if (output_num > 1) {
    for (size_t j = 0; j < output_num; j++) {
      ShapeVector shape = common::AnfAlgo::GetOutputInferShape(cnode, j);
      if (shape.size() <= 0) {
        continue;
      }

      const TypeId type = common::AnfAlgo::GetOutputInferDataType(cnode, j);
      if (type == kObjectTypeUMonad || type == kObjectTypeMonad || type == kObjectTypeFunction) {
        continue;
      }

      size_t type_size = GetDataTypeSize(type);
      cost += std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * type_size;
    }
  }

  return cost;
}

void LogBaseline(const SchedulingInput &input, std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_gpto_map_ptr,
                 const KernelGraphPtr &kernel_graph, const std::string &path) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_gpto_map_ptr);

  const size_t graph_id = kernel_graph->graph_id();
  std::stringstream ss;
  std::ostringstream oss;
  std::string filename;
  ss << kernel_graph;
  filename = path + std::string("/") + std::string("gpto_baseline_") + std::to_string(graph_id) + std::string("_") +
             ss.str() + std::string(".log");

  std::unordered_map<GptoTaskId, Time> taskid_to_end_value;
  std::unordered_map<GptoTaskId, Time> taskid_to_start_value;
  size_t makespan = 0;
  const std::vector<CNodePtr> execution_order = kernel_graph->execution_order();
  for (size_t i = 0; i < execution_order.size(); i++) {
    const auto &cnode = execution_order[i];

    GptoTaskPtr current_task = (*cnode_to_task_gpto_map_ptr)[cnode];

    // Find the latest executed task which has the same type as the current task
    GptoTaskPtr last_task = nullptr;
    MS_LOG(DEBUG) << "Current value loop: " << i << " with node: " << execution_order[i]->UniqueName();
    for (int j = i - 1; j >= 0; j--) {
      MS_LOG(DEBUG) << "Current value loop j: " << j;
      GptoTaskPtr tmp_task = (*cnode_to_task_gpto_map_ptr)[execution_order[j]];
      MS_LOG(DEBUG) << "With node: " << tmp_task->name();
      if (tmp_task->gpto_type() == current_task->gpto_type()) {
        MS_LOG(DEBUG) << "Found node same type";
        last_task = tmp_task;
        break;
      }
    }

    // Find the latest parent of the current task
    for (const auto &parent : (*cnode_to_task_gpto_map_ptr)[cnode]->parents()) {
      if (last_task == nullptr || taskid_to_end_value[parent.lock()->id()] >= taskid_to_end_value[last_task->id()]) {
        last_task = parent.lock();
        MS_LOG(DEBUG) << "Found parent " << last_task->name();
      }
    }

    if (last_task == nullptr) {
      last_task = current_task;
      taskid_to_start_value[current_task->id()] = 0;
      taskid_to_end_value[current_task->id()] = 0 + current_task->cost();
    } else {
      taskid_to_start_value[current_task->id()] = taskid_to_end_value[last_task->id()];
      taskid_to_end_value[current_task->id()] = taskid_to_start_value[current_task->id()] + current_task->cost();
    }

    size_t current_task_end = taskid_to_end_value[current_task->id()];
    if (current_task_end > makespan) {
      makespan = taskid_to_end_value[current_task->id()];
    }
    oss << "TASK id=" << std::to_string(current_task->id()) << ", name=" << current_task->name()
        << ", type=" << std::to_string(current_task->gpto_type())
        << ", start=" << std::to_string(taskid_to_start_value[current_task->id()])
        << ", end=" << std::to_string(current_task_end) << "\n";
  }
  MS_LOG(INFO) << "Makespan estimate of baseline is " + std::to_string(makespan);
  (void)Common::SaveStringToFile(filename, oss.str());
}

void InitializeTaskInlineCondition(const CNodePtr &cnode, GptoTaskPtr *task,
                                   std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> *switch_attribute_ids,
                                   std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> *gather_attribute_ids) {
  MS_EXCEPTION_IF_NULL(task);
  MS_EXCEPTION_IF_NULL(switch_attribute_ids);
  MS_EXCEPTION_IF_NULL(gather_attribute_ids);

  const size_t kPrefixLength = 12;
  if (cnode->HasAttr(kInlineSubGraphName)) {  // ConditionSwitch
    (*task)->set_condition_switch(true);
    std::string s = cnode->GetAttr(kInlineSubGraphName)->ToString();
    std::string s1 = s.substr(s.find('(') + 1, s.find(',') - 1);
    std::string s2 = s.substr(s.find(',') + 1, s.find(')') - 1);
    (*switch_attribute_ids)[(*task)] = std::make_pair(std::stoll(s1.substr(s1.find("kernel_graph") + kPrefixLength)),
                                                      std::stoll(s2.substr(s2.find("kernel_graph") + kPrefixLength)));
    MS_LOG(DEBUG) << "Task ConditionSwitch " << (*task)->id() << " with attribute kInlineSubGraphName" << s;
  } else if (cnode->HasAttr(kAttrBranchGraphName)) {  // ConditionGather
    (*task)->set_condition_gather(true);
    std::string s = cnode->GetAttr(kAttrBranchGraphName)->ToString();
    std::string s1 = s.substr(s.find('(') + 1, s.find(',') - 1);
    std::string s2 = s.substr(s.find(',') + 1, s.find(')') - 1);
    (*gather_attribute_ids)[(*task)] = std::make_pair(std::stoll(s2.substr(s2.find("kernel_graph") + kPrefixLength)),
                                                      std::stoll(s1.substr(s1.find("kernel_graph") + kPrefixLength)));
    MS_LOG(DEBUG) << "Task ConditionGather " << (*task)->id() << " with attribute kAttrBranchGraphName" << s;
  }
}

void UpdateTasksInlineCondition(std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr,
                                std::map<GptoTaskPtr, GptoTaskPtr, TaskDepthSort> *switch_gather) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);
  MS_EXCEPTION_IF_NULL(switch_gather);

  size_t count_condition = SIZE_MAX - 1;
  std::unordered_map<size_t, size_t> unprocessed_parents;
  std::queue<GptoTaskPtr> tasks_to_visit;

  for (const auto &key_val : *cnode_to_task_map_ptr) {
    auto &task = key_val.second;
    unprocessed_parents[task->id()] = task->parents().size();
  }

  for (auto &it : (*switch_gather)) {
    const auto &switch_task = it.first;
    const auto &gather_task = it.second;
    MS_LOG(DEBUG) << "Assign subgraph id " << count_condition << " to tasks under ConditionSwitch task "
                  << switch_task->id() << " name " << switch_task->name()
                  << " up to (and including) ConditionGather task " << gather_task->id() << " name "
                  << gather_task->name();

    for (auto child : switch_task->children()) {
      if (child == gather_task) {
        child->set_subgraph_id(count_condition);
        MS_LOG(DEBUG) << "Assign subgraph id " << count_condition << " to task " << gather_task->id() << " name "
                      << gather_task->name();
      } else {
        tasks_to_visit.push(child);
      }
    }

    while (!tasks_to_visit.empty()) {
      const auto &selected_task = tasks_to_visit.front();
      selected_task->set_subgraph_id(count_condition);
      MS_LOG(DEBUG) << "Assign subgraph id " << count_condition << " to task " << selected_task->id() << " name "
                    << selected_task->name();
      if (selected_task->name().find("ConditionSwitch") != std::string::npos) {
        for (auto gather_child : (*switch_gather)[selected_task]->children()) {
          unprocessed_parents[gather_child->id()] -= 1;
          if (unprocessed_parents[gather_child->id()] == 0) {
            if (gather_child != gather_task) {
              tasks_to_visit.push(gather_child);
            } else {
              if (gather_task->subgraph_id() != count_condition) {
                gather_task->set_subgraph_id(count_condition);
                MS_LOG(DEBUG) << "Assign subgraph id " << count_condition << " to task " << gather_task->id()
                              << " name " << gather_task->name();
              }
            }
          }
        }
      } else {
        for (auto &child : selected_task->children()) {
          unprocessed_parents[child->id()] -= 1;
          if (unprocessed_parents[child->id()] == 0) {
            if (child != gather_task) {
              tasks_to_visit.push(child);
            } else {
              if (gather_task->subgraph_id() != count_condition) {
                gather_task->set_subgraph_id(count_condition);
                MS_LOG(DEBUG) << "Assign subgraph id " << count_condition << " to task " << gather_task->id()
                              << " name " << gather_task->name();
              }
            }
          }
        }
      }
      tasks_to_visit.pop();
    }
    count_condition--;
  }
}

SchedulingInput ExtractSchedulingInput(const KernelGraphPtr &kernel_graph,
                                       std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);
  SchedulingInput scheduling_input;  // to fill in and return
  std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> switch_attribute_ids;
  std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> gather_attribute_ids;

  // Create a task per node
  MS_LOG(INFO) << "Start Extract GPTO Tasks";
  const auto &real_kernels = kernel_graph->execution_order();
  scheduling_input.tasks.reserve(real_kernels.size());
  for (size_t i = 0; i < real_kernels.size(); ++i) {
    const auto &cnode = real_kernels[i];

    if (common::AnfAlgo::IsDynamicShape(cnode) || common::AnfAlgo::IsDynamicSequence(cnode) ||
        common::AnfAlgo::IsDynamicValue(cnode)) {
      MS_LOG(INFO) << "GPTO can't parse graph with dynamic shape or dynamic value now.";
      scheduling_input.tasks.clear();
      return scheduling_input;
    }

    GptoTaskPtr task = std::make_shared<GptoTask>(i, GetRealType(cnode), GetType(cnode), cnode->fullname_with_scope());
    task->set_cnode(cnode);
    Time cost = 0;

    if (task->real_type() == kComp) {  // comp node of type Vector
      cost = CalculateVectorCost(cnode);
    } else if (task->real_type() == kCube) {  // comp node of type Cube
      cost = CalculateCubeCost(cnode);
    } else {  // comm node
      cost = CalculateCommCost(cnode);
    }

    task->AssignCost(cost);

    // Start Step 1 ConditionSwitch/Gather for inline: save attributes
    InitializeTaskInlineCondition(cnode, &task, &switch_attribute_ids, &gather_attribute_ids);
    // End Step 1 ConditionSwitch/Gather for inline

    MS_LOG(DEBUG) << "Task " << task->id() << " with name " << cnode->UniqueName() << " and CNodePtr " << cnode
                  << " with cost " << task->cost() << " and type " << GetType(cnode);
    scheduling_input.tasks.push_back(task);
    (*cnode_to_task_map_ptr)[cnode] = task;
  }
  MS_LOG(INFO) << "End Extract GPTO Tasks";

  MS_LOG(INFO) << "Start Extract GPTO Edges";
  InsertEdges(kernel_graph, cnode_to_task_map_ptr);
  MS_LOG(INFO) << "End Extract GPTO Edges";

  MS_LOG(INFO) << "Start Extract GPTO Tensors";
  ExtractRealTensors(scheduling_input, cnode_to_task_map_ptr);
  MS_LOG(INFO) << "End Extract GPTO Tensors";

  // Start Step 2 ConditionSwitch/Gather for inline: identify matching switch/gather pairs
  MS_LOG(INFO) << "Start Extract GPTO Switch/Gather";
  ComputeDepthAndTopLevel(scheduling_input.tasks);  // if kept here, do not call again later in Process()

  std::map<GptoTaskPtr, GptoTaskPtr, TaskDepthSort> switch_gather;
  for (auto &switch_it : switch_attribute_ids) {
    const auto &switch_task = switch_it.first;
    auto switch_pair = switch_it.second;

    std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>>::iterator gather_it;
    for (gather_it = gather_attribute_ids.begin(); gather_it != gather_attribute_ids.end(); ++gather_it) {
      if (gather_it->second == switch_pair) {
        break;
      }
    }
    if (gather_it == gather_attribute_ids.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Could not find matching ConditionGather for a given ConditionSwitch "
                                 << switch_pair;
    }
    const auto &gather_task = gather_it->first;
    switch_gather[switch_task] = gather_task;
    MS_LOG(DEBUG) << "Mapped ConditionSwitch task " << switch_task->id() << " to ConditionGather task "
                  << gather_task->id();
  }
  MS_LOG(INFO) << "End Extract GPTO Switch/Gather";
  // End Step 2 ConditionSwitch/Gather for inline

  // Start Step 3 ConditionSwitch/Gather for inline: traverse each Condition/Switch gather block to assign proper ids
  // Assumption 1: switch and nodes before gather have no predecessors/descendants outside the block
  // Assumption 2: conditional switch does not have conditional gather as a child
  MS_LOG(INFO) << "Start Update Inline";
  UpdateTasksInlineCondition(cnode_to_task_map_ptr, &switch_gather);
  MS_LOG(INFO) << "End Update Inline";
  // End Step 3 ConditionSwitch/Gather for inline

  return scheduling_input;
}

Memory MemoryLowerBound(const std::vector<GptoTaskPtr> &tasks,
                        const std::vector<mindspore::somas::DynamicBitSet> &nodes_dependency,
                        const std::set<GptoTensorPtr, GptoTensorIdSort> &tensors) {
  Memory max_lb = 0;

  for (const auto &task : tasks) {
    Memory task_lb = 0;
    for (const auto &tensor : tensors) {
      if (tensor->weight() == 0) {
        continue;
      }
      const auto &source = tensor->source().lock();
      const auto &consumers = tensor->consumers();

      if (task == source || consumers.count(task) > 0) {  // tensor is source or consumer of task
        task_lb += tensor->weight();
      } else {
        if (nodes_dependency[task->id()].IsBitTrue(source->id())) {
          if (tensor->type() == kGraphOutput) {  // semilifelong end logic
            task_lb += tensor->weight();
          } else {
            if (std::any_of(consumers.cbegin(), consumers.cend(), [&](auto &consumer) {
                  return nodes_dependency[consumer.lock()->id()].IsBitTrue(task->id());
                })) {
              task_lb += tensor->weight();
            }
          }
        }
      }
    }
    task->set_lower_bound(task_lb);
    max_lb = std::max(max_lb, task_lb);
  }
  MS_LOG(INFO) << "Memory Lower bound for tensors " << max_lb << " (" << max_lb / kMBToByte << " MB)";
  return max_lb;
}

void GraphOutputProcess(const KernelGraphPtr &graph, std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);

  size_t count = 0;
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  auto &cnode_to_task_map = *cnode_to_task_map_ptr;
  for (auto &output : outputs) {
    auto output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto output_kernel = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output_kernel);
    while (AnfUtils::IsRealCNodeKernel(output_kernel) &&
           cnode_to_task_map.find(output_kernel->cast<CNodePtr>()) == cnode_to_task_map.end()) {
      auto cnode = output_kernel->cast<CNodePtr>();
      if (!common::AnfAlgo::IsNopNode(cnode)) {
        MS_LOG(INTERNAL_EXCEPTION) << "Node[" << cnode->fullname_with_scope()
                                   << "] doesn't exist in cnode_to_task_map and is not a nop node!!!";
      }
      output_with_index = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(1), 0, false);
      output_kernel = output_with_index.first;
    }

    if (!AnfUtils::IsRealCNodeKernel(output_kernel)) {
      continue;
    }

    auto output_index = output_with_index.second;
    auto iter = cnode_to_task_map.find(output_kernel->cast<CNodePtr>());
    if (iter != cnode_to_task_map.end()) {
      auto &node = iter->second;
      MS_EXCEPTION_IF_NULL(node);
      if (node->out_tensors().size() == 0) {
        MS_LOG(DEBUG) << "Node " << node->name() << " does not have output tensors";
        continue;
      } else if (output_index < node->out_tensors().size()) {
        auto &tensor = node->out_tensors()[output_index];
        tensor->set_type(kGraphOutput);  // if need_reuse_graph_output (default is true), then treat as semilifelong
                                         // end, otherwise set to 0
        MS_LOG(DEBUG) << "GPTO Tensor " << tensor->id() << " with size " << tensor->weight() << " is kGraphOutput";
        count++;
      } else {
        MS_LOG(INTERNAL_EXCEPTION) << "Graph's output node " << output_kernel->fullname_with_scope()
                                   << "'s output index " << output_index << " is larger than its output tensor number "
                                   << node->out_tensors().size();
      }
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "Can't find task for graph output node " << output_kernel->fullname_with_scope();
    }
  }
  MS_LOG(INFO) << "Found " << count << " graph output tensors for GPTO";
}

void RefNodeProcess(const KernelGraphPtr &graph, std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);
  auto &cnode_to_task_map = *cnode_to_task_map_ptr;
  const auto &kernel_cnodes = graph->execution_order();
  size_t total_output_size = 0;
  size_t total_input_size = 0;
  std::vector<std::pair<GptoTensorPtr, GptoTensorPtr>> in_out_vector;

  // Loop to obtain ref node pairs
  for (const auto &kernel : kernel_cnodes) {
    auto mod = AnfAlgo::GetKernelMod(kernel);
    if (mod == nullptr) {
      MS_LOG(WARNING) << "Null mod for kernel " << kernel->fullname_with_scope();
      continue;
    }
    size_t index = 0;
    for (const auto &size : mod->GetOutputSizeList()) {
      auto out_index = index++;
      session::AnfWithOutIndex out_pair(kernel, out_index);
      if (graph->IsInRefOutputMap(out_pair)) {
        auto origin_pair = graph->GetRefCorrespondOutput(out_pair);
        MS_EXCEPTION_IF_NULL(origin_pair.first);
        auto &node = cnode_to_task_map[kernel];
        MS_EXCEPTION_IF_NULL(node);
        auto output_tensor = node->out_tensors()[out_index];
        MS_EXCEPTION_IF_NULL(output_tensor);
        total_output_size += size;

        if (!AnfUtils::IsRealCNodeKernel(origin_pair.first)) {
          output_tensor->set_type(kGraphInput);
          output_tensor->set_weight(0);
          continue;
        }

        if (cnode_to_task_map.find(origin_pair.first->cast<CNodePtr>()) == cnode_to_task_map.end()) {
          auto cnode = origin_pair.first->cast<CNodePtr>();
          if (!common::AnfAlgo::IsNopNode(cnode)) {
            MS_LOG(INTERNAL_EXCEPTION) << "Node[" << origin_pair.first->fullname_with_scope() << "] find input node["
                                       << cnode->fullname_with_scope()
                                       << "] doesn't exist in nodes_map and is not a nop node!!!!";
          }
          origin_pair = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(1), 0, false);
        }
        if (!origin_pair.first->isa<CNode>()) {
          MS_LOG(INTERNAL_EXCEPTION) << "The origin_pair.first is not a cnode. Info origin_pair.first: "
                                     << origin_pair.first->DebugString();
        }
        auto ori_node = origin_pair.first->cast<CNodePtr>();
        auto ori_index = origin_pair.second;
        if (cnode_to_task_map.find(ori_node.get()->cast<CNodePtr>()) == cnode_to_task_map.end()) {
          MS_LOG(EXCEPTION)
            << "The ori_node is not included in cnode_to_task_map constructed from exec_order of graph. Info ori_node: "
            << ori_node->DebugString();
        }
        auto &repeat_node = cnode_to_task_map[ori_node];
        MS_EXCEPTION_IF_NULL(repeat_node);
        auto input_tensor = repeat_node->out_tensors()[ori_index];
        MS_EXCEPTION_IF_NULL(input_tensor);
        in_out_vector.push_back(std::make_pair(input_tensor, output_tensor));
        total_input_size += input_tensor->weight();
        MS_LOG(DEBUG) << "RefNode: input " << input_tensor->id() << " output " << output_tensor->id();
      }
    }
  }

  // Loop to update ref node tensor sizes and update graph logic
  for (auto &in_out : in_out_vector) {
    auto input_tensor = in_out.first;
    auto output_tensor = in_out.second;

    if (input_tensor->original_weight() == 0 || output_tensor->original_weight() == 0) {
      input_tensor->set_weight(0);
      output_tensor->set_weight(0);
    } else if (input_tensor->weight() < output_tensor->weight()) {
      if (output_tensor->source().lock()->gpto_type() != kComm) {
        input_tensor->set_weight(output_tensor->weight());
      }
      output_tensor->set_weight(0);
    }

    for (auto &out_consumer : output_tensor->consumers()) {
      input_tensor->consumers().insert(out_consumer);
      auto it =
        std::find(out_consumer.lock()->in_tensors().begin(), out_consumer.lock()->in_tensors().end(), output_tensor);
      if (it != out_consumer.lock()->in_tensors().end()) {
        out_consumer.lock()->in_tensors().erase(it);
      }
      out_consumer.lock()->in_tensors().push_back(input_tensor);
    }
    auto it = std::find(output_tensor->source().lock()->out_tensors().begin(),
                        output_tensor->source().lock()->out_tensors().end(), output_tensor);
    output_tensor->source().lock()->out_tensors().erase(it);
  }
  MS_LOG(INFO) << "RefNode tensor total size: input " << total_input_size << " output " << total_output_size;
}

void ExtractTensors(const std::vector<GptoTaskPtr> &tasks, std::set<GptoTensorPtr, GptoTensorIdSort> *tensors) {
  MS_EXCEPTION_IF_NULL(tensors);
  for (const auto &task : tasks) {
    const auto &out_tensors = task->out_tensors();
    const auto &ws_tensors = task->workspace_tensors();
    tensors->insert(out_tensors.begin(), out_tensors.end());
    tensors->insert(ws_tensors.begin(), ws_tensors.end());
  }
}

void GPTO(const KernelGraphPtr &kernel_graph, std::vector<std::pair<CNodePtr, CNodePtr>> *events) {
  MS_EXCEPTION_IF_NULL(events);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);

  if (kernel_graph->is_from_single_op()) {
    MS_LOG(INFO) << "GPTO is not used when pynative forward.";
    return;
  }

  if (kernel_graph->is_dynamic_shape()) {
    MS_LOG(INFO) << "GPTO can't parse graph with dynamic shape now.";
    return;
  }

  if (common::GetEnv("MS_ENABLE_GPTO_MODE") != "") {
    gpto_mode = static_cast<GPTO_MODE>(stoll(common::GetEnv("MS_ENABLE_GPTO_MODE")));
  }

  if (common::GetEnv("MS_ENABLE_GPTO_MEMORY_LIMIT") != "") {
    SOFT_MEMORY_LIMIT = static_cast<Memory>(stoll(common::GetEnv("MS_ENABLE_GPTO_MEMORY_LIMIT")) * kGBToByte);
  } else {
    SOFT_MEMORY_LIMIT = static_cast<Memory>(context->get_param<float>(MS_CTX_MAX_DEVICE_MEMORY) * kGBToByte);
  }
  HARD_MEMORY_LIMIT = static_cast<Memory>(context->get_param<float>(MS_CTX_MAX_DEVICE_MEMORY) * kGBToByte);

  MS_LOG(INFO) << "Soft Memory value: " << SOFT_MEMORY_LIMIT;
  MS_LOG(INFO) << "Hard Memory value: " << HARD_MEMORY_LIMIT;

  MS_LOG(INFO) << "Start Scheduling Subgraph " << kernel_graph << " with id " << kernel_graph->graph_id()
               << " and Execution order size " << kernel_graph->execution_order().size();

  std::unordered_map<CNodePtr, GptoTaskPtr> cnode_to_task;

  MS_LOG(INFO) << "Start ExtractSchedulingInput";
  SchedulingInput scheduling_input = ExtractSchedulingInput(kernel_graph, &cnode_to_task);
  MS_LOG(INFO) << "End ExtractSchedulingInput";
  if (scheduling_input.tasks.size() == 0) {
    MS_LOG(WARNING) << "Scheduling input doesn't have any tasks: skipping";
    return;
  }

  MS_LOG(INFO) << "Start Graph Output Process";
  GraphOutputProcess(kernel_graph,
                     &cnode_to_task);  //  kGraphOutput: "semilifelong end" functionality by default in memory
                                       //  estimated in source's memory impact and never "deallocated"
  MS_LOG(INFO) << "End Graph Output Process";

  MS_LOG(INFO) << "Start Ref Node Process";
  RefNodeProcess(kernel_graph, &cnode_to_task);
  MS_LOG(INFO) << "End Ref Node Process";

  MS_LOG(INFO) << "Start GPTO Process";
  auto scheduling_output = Process(scheduling_input);
  MS_LOG(INFO) << "End GPTO Process";

  if (scheduling_output.makespan == SIZE_MAX) {
    MS_LOG(INFO) << "Hard memory limit is not satisfied by any solution's memory estimate, exiting GPTO...";
    return;
  }

  // Update execution order based on computed schedule
  std::vector<Interval> task_times = scheduling_output.task_times;
  std::sort(task_times.begin(), task_times.end(),
            [](Interval x, Interval y) { return x.start < y.start || (x.start == y.start && x.end < y.end); });
  std::vector<CNodePtr> new_order;
  std::transform(task_times.cbegin(), task_times.cend(), std::back_inserter(new_order),
                 [](const auto &interval) { return interval.task->cnode(); });
  kernel_graph->set_execution_order(new_order);

  // Get dependencies (events) corresponding to computed schedule
  std::vector<std::pair<CNodePtr, CNodePtr>> dependencies = ScheduleToEvents(scheduling_output);

  auto can_debug = GetDebugConfig();
  if (can_debug.first) {
    // Memory lower bound (optional: for analysis only)
    std::vector<mindspore::somas::DynamicBitSet> nodes_dependency;
    std::set<GptoTensorPtr, GptoTensorIdSort> tensors;

    MS_LOG(INFO) << "Start Compute Ancestors Descendants";
    ComputeAncestorsDescendants(scheduling_input.tasks, &nodes_dependency);
    MS_LOG(INFO) << "End Compute Ancestors Descendants";

    MS_LOG(INFO) << "Start Memory Lower Bound";
    ExtractTensors(scheduling_input.tasks, &tensors);
    Memory memory_lower_bound = MemoryLowerBound(scheduling_input.tasks, nodes_dependency, tensors);
    MS_LOG(INFO) << "End Memory Lower Bound";

    // Graph execution order
    MS_LOG(INFO) << "Start GPTO PrintGraphExecuteOrder";
    kernel_graph->PrintGraphExecuteOrder();
    MS_LOG(INFO) << "End GPTO PrintGraphExecuteOrder";

    // Log files
    MS_LOG(INFO) << "Start Baseline Greedy Scheduling";
    LogBaseline(scheduling_input, &cnode_to_task, kernel_graph, can_debug.second);
    MS_LOG(INFO) << "End Baseline Greedy Scheduling";

    MS_LOG(INFO) << "Start printing output log file";
    auto lower_makespan =
      std::max(LowerBoundBottomLevel(scheduling_input.tasks), LowerBoundPEs(scheduling_input.tasks, GetPEs()));
    LogSchedulingOutput(scheduling_output, cnode_to_task, dependencies, kernel_graph, tensors, lower_makespan,
                        memory_lower_bound, can_debug.second);
    MS_LOG(INFO) << "End printing output log file";
  }
}
}  // namespace gpto
}  // namespace mindspore
