/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <unordered_map>
#include <set>
#include <map>
#include <deque>
#include <queue>
#include <iostream>
#include <string>
#include <cmath>
#include <iomanip>
#include <functional>
#include <sstream>

#include "mindspore/core/ops/math_op_name.h"
#include "mindspore/core/ops/conv_pool_op_name.h"
#include "mindspore/core/ops/ascend_op_name.h"
#include "mindspore/core/utils/anf_utils.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "mindspore/ccsrc/frontend/parallel/step_parallel.h"
#include "mindspore/core/utils/misc.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/device/ascend/hal/hardware/gpto.h"
#include "plugin/device/ascend/hal/device/ascend_memory_adapter.h"

static mindspore::opt::Memory SOFT_MEMORY_LIMIT;
static mindspore::opt::Memory HARD_MEMORY_LIMIT; // preset some value to capture max size of 910B
constexpr size_t kGBToByte = 1073741824; // 1GB

namespace mindspore {
namespace opt {
// Subroutines Implementing "Scheduling to Dependencies"
struct SortByStart {
  bool operator()(const Interval &interval1, const Interval &interval2) const {
    const auto &id1 = interval1.id;
    const auto &start1 = interval1.start;
    const auto &end1 = interval1.end;
    const auto &id2 = interval2.id;
    const auto &start2 = interval2.start;
    const auto &end2 = interval2.end;
    return start1 < start2 || (start1 == start2 && end1 < end2) || (start1 == start2 && end1 == end2 && id1 < id2);
  }
};

struct SortByEnd {
  bool operator()(const Interval &interval1, const Interval &interval2) const {
    const auto &id1 = interval1.id;
    const auto &start1 = interval1.start;
    const auto &end1 = interval1.end;
    const auto &id2 = interval2.id;
    const auto &start2 = interval2.start;
    const auto &end2 = interval2.end;
    return end1 < end2 || (end1 == end2 && start1 < start2) || (end1 == end2 && start1 == start2 && id1 < id2);
  }
};

bool Overlap(const Time &start1, const Time &end1, const Time &start2, const Time &end2) {
  return (start1 >= start2 && start1 < end2) ||
         (start2 >= start1 && start2 < end1);  // if equal start and end for two intervals, then no overlap
}

std::vector<std::pair<TaskId, TaskId>> gpto::ScheduleToDependencies(const SchedulingOutput &schedule) {
  std::vector<std::pair<TaskId, TaskId>> dependencies;  // to return
  MS_LOG(INFO) << "Started Preprocessing of Intervals";
  // Distinguish types and sort
  std::unordered_map<TaskType, std::set<Interval, SortByStart>> tasks_start;
  std::unordered_map<TaskType, std::set<Interval, SortByEnd>> tasks_end;
  for (const auto &task_time : schedule.task_times) {
    tasks_start[task_time.gpto_type].insert(task_time);
    tasks_end[task_time.gpto_type].insert(task_time);
  }
  MS_LOG(INFO) << "Finished Preprocessing of Intervals";
  MS_LOG(INFO) << "Started Main Loop";
  // Main loop: check each task for potential dependencies in its right neighborhood
  for (const auto &type_to_set : tasks_start) {
    const auto &type = type_to_set.first;
    for (auto it = tasks_start[type].begin(); it != tasks_start[type].end(); ++it) {
      tasks_end[type].erase(*it);
      // Dismiss overlapping tasks: save min end value of non-overlapping task to the right
      std::unordered_map<TaskId, bool> dismissed;
      auto it1 = std::next(it);
      for (; Overlap(it->start, it->end, it1->start, it1->end) && it1 != tasks_start[type].end(); ++it1) {
        dismissed[it1->id] = true;
      }
      Time min_end_value = 0;
      for (auto it2 = tasks_end[type].begin(); it2 != tasks_end[type].end(); ++it2) {
        if (!dismissed[it2->id]) {
          min_end_value = it2->end;
          break;
        }
      }
      // Add dependencies to immediate right neighborhood
      for (; it1->start < min_end_value && it1 != tasks_start[type].end(); ++it1) {
	dependencies.emplace_back(it->id, it1->id);
      }
    }
  }
  MS_LOG(INFO) << "Finished Main Loop";
  MS_LOG(INFO) << "Generated " << dependencies.size() << " dependencies";
  return dependencies;
}

std::vector<std::pair<TaskId, TaskId>> gpto::ScheduleToDependenciesDifferentTypes(const SchedulingOutput &schedule) {
  std::vector<std::pair<TaskId, TaskId>> dependencies;  // to return
  MS_LOG(INFO) << "Started Preprocessing of Intervals";
  // Distinguish types and sort
  std::set<Interval, SortByStart> tasks_start;
  std::set<Interval, SortByEnd> tasks_end;
  for (const auto &task_time : schedule.task_times) {
    tasks_start.insert(task_time);
    tasks_end.insert(task_time);
  }
  MS_LOG(INFO) << "Finished Preprocessing of Intervals";
  MS_LOG(INFO) << "Started Main Loop";
  // Main loop: check each task for potential dependencies in its right neighborhood
  //for (const auto &type_to_set : tasks_start) {
    //const auto &type = type_to_set.first;
    for (auto it = tasks_start.begin(); it != tasks_start.end(); ++it) {
      tasks_end.erase(*it);
      // Dismiss overlapping tasks: save min end value of non-overlapping task to the right
      std::unordered_map<TaskId, bool> dismissed;
      auto it1 = std::next(it);
      for (; Overlap(it->start, it->end, it1->start, it1->end) && it1 != tasks_start.end(); ++it1) {
        dismissed[it1->id] = true;
      }
      Time min_end_value = 0;
      for (auto it2 = tasks_end.begin(); it2 != tasks_end.end(); ++it2) {
        if (!dismissed[it2->id]) {
          min_end_value = it2->end;
          break;
        }
      }
      // Add dependencies to immediate right neighborhood
      for (; it1->start < min_end_value && it1 != tasks_start.end(); ++it1) {
        if (it->gpto_type != it1->gpto_type){
          dependencies.emplace_back(it->id, it1->id);
        }
      }
    }
  //}
  MS_LOG(INFO) << "Finished Main Loop";
  MS_LOG(INFO) << "Generated " << dependencies.size() << " dependencies";
  return dependencies;
}

// Sorting for tasks
bool SortByWeightMax(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->weight() > task2->weight() || (task1->weight() == task2->weight() && task1->id() < task2->id());
}

bool SortByWeightMin(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->weight() < task2->weight() || (task1->weight() == task2->weight() && task1->id() < task2->id());
}

bool SortBySuccDiff(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->succ_diff_type() > task2->succ_diff_type() ||
         (task1->succ_diff_type() == task2->succ_diff_type() && task1->weight() > task2->weight()) ||
         (task1->succ_diff_type() == task2->succ_diff_type() && task1->weight() == task2->weight() &&
          task1->id() < task2->id());
}

bool SortByBottomLevelMax(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->bottom_level() > task2->bottom_level() ||
         (task1->bottom_level() == task2->bottom_level() && task1->weight() > task2->weight()) ||
         (task1->bottom_level() == task2->bottom_level() && task1->weight() == task2->weight() &&
          task1->id() < task2->id());
}

bool SortByBottomLevelMin(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->bottom_level() < task2->bottom_level() ||
         (task1->bottom_level() == task2->bottom_level() && task1->weight() > task2->weight()) ||
         (task1->bottom_level() == task2->bottom_level() && task1->weight() == task2->weight() &&
          task1->id() < task2->id());
}

bool SortByTopLevelMax(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->top_level() > task2->top_level() ||
         (task1->top_level() == task2->top_level() && task1->weight() > task2->weight()) ||
         (task1->top_level() == task2->top_level() && task1->weight() == task2->weight() && task1->id() < task2->id());
}

bool SortByTopLevelMin(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->top_level() < task2->top_level() ||
         (task1->top_level() == task2->top_level() && task1->weight() > task2->weight()) ||
         (task1->top_level() == task2->top_level() && task1->weight() == task2->weight() && task1->id() < task2->id());
}

bool SortByBottomTopLevelMaxSum(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->top_level() + task1->bottom_level() > task2->top_level() + task2->bottom_level() ||
         (task1->top_level() + task1->bottom_level() == task2->top_level() + task2->bottom_level() &&
          task1->weight() > task2->weight()) ||
         (task1->top_level() + task1->bottom_level() == task2->top_level() + task2->bottom_level() &&
          task1->weight() == task2->weight() && task1->id() < task2->id());
}

bool SortByBottomTopLevelMinSum(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->top_level() + task1->bottom_level() < task2->top_level() + task2->bottom_level() ||
         (task1->top_level() + task1->bottom_level() == task2->top_level() + task2->bottom_level() &&
          task1->weight() > task2->weight()) ||
         (task1->top_level() + task1->bottom_level() == task2->top_level() + task2->bottom_level() &&
          task1->weight() == task2->weight() && task1->id() < task2->id());
}

// Ishfaq Ahmad, Yu-Kwong Kwok, and Min-You Wu.
// Analysis, evaluation, and comparison of algorithms for scheduling task graphs on parallel processors.
// Second  International  Symposium  on Parallel Architectures, Algorithms, and Networks (I-SPAN'96),
// pages 207-213. IEEE, 1996.
bool SortByBottomTopLevelComposite(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->bottom_level() - task1->top_level() > task2->bottom_level() - task2->top_level() ||
         (task1->bottom_level() - task1->top_level() == task2->bottom_level() - task2->top_level() &&
          task1->weight() > task2->weight()) ||
         (task1->bottom_level() - task1->top_level() == task2->bottom_level() - task2->top_level() &&
          task1->weight() == task2->weight() && task1->id() < task2->id());
}

// Behrooz Shirazi, Mingfang Wang, and Girish Pathak.
// Analysis and evaluation of heuristic methods for static task scheduling.
// Journal of Parallel and Distributed Computing, 10(3):222-232, 1990.
bool SortByWeightedLength(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->weighted_length() > task2->weighted_length() ||
         (task1->weighted_length() == task2->weighted_length() && task1->id() < task2->id());
}

// DFS with weights for tie breaking
bool SortByDepthMax(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->depth() > task2->depth() || (task1->depth() == task2->depth() && task1->weight() > task2->weight()) ||
         (task1->depth() == task2->depth() && task1->weight() == task2->weight() && task1->id() < task2->id());
}

// BFS with weights for tie breaking
bool SortByDepthMin(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->depth() < task2->depth() || (task1->depth() == task2->depth() && task1->weight() > task2->weight()) ||
         (task1->depth() == task2->depth() && task1->weight() == task2->weight() && task1->id() < task2->id());
}

// Sort by predecessor to comm
bool SortByPredComm(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->pred_comm() < task2->pred_comm() ||
         (task1->pred_comm() == task2->pred_comm() && task1->bottom_level() > task2->bottom_level()) ||
         (task1->pred_comm() == task2->pred_comm() && task1->bottom_level() == task2->bottom_level() &&
          task1->id() < task2->id());
}

// Sort by predecessor to comm + DFS
bool SortByPredCommDepth(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->pred_comm() < task2->pred_comm() ||
         (task1->pred_comm() == task2->pred_comm() && task1->depth() > task2->depth()) ||
         (task1->pred_comm() == task2->pred_comm() && task1->depth() == task2->depth() && task1->id() < task2->id());
}

// Sort by predecessor to cube + bottom level
bool SortByPredCube(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->pred_cube() < task2->pred_cube() ||
         (task1->pred_cube() == task2->pred_cube() && task1->bottom_level() > task2->bottom_level()) ||
         (task1->pred_cube() == task2->pred_cube() && task1->bottom_level() == task2->bottom_level() &&
          task1->id() < task2->id());
}

// Sort by greedy height of memory (maintained dynamically)
bool SortByGreedyHeight(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->mem_impact() < task2->mem_impact() || (
         task1->mem_impact() == task2->mem_impact() && SortByBottomLevelMax(task1, task2));
}

// Sorting by load for processing elements
struct SortByLoad {
  bool operator()(const ProcessingElement &pe1, const ProcessingElement &pe2) const {
    return pe1.load < pe2.load || (pe1.load == pe2.load && pe1.id < pe2.id);
  }
};

// Get PEs description
std::unordered_map<TaskType, int32_t> GetTestPEs() {
  std::unordered_map<TaskType, int32_t> new_pem;
  new_pem[kComm] = 1;
  if (common::GetEnv("MS_ENABLE_GPTO_SINGLESTREAM") == "1") {
    return new_pem;
  }
  new_pem[kComp] = 1;
  if (common::GetEnv("MS_ENABLE_GPTO_MULTISTREAM") == "0") {  
    return new_pem;
  }
  new_pem[kCube] = 1;
  return new_pem;
}

// Auxiliary subroutines and lower bounds
void gpto::ComputeDepthAndTopLevel(std::vector<std::shared_ptr<Task>> &tasks) {
  MS_LOG(INFO) << "Top Level: Start Initialization";
  std::unordered_map<TaskId, size_t> unprocessed_parents;
  std::queue<std::shared_ptr<Task>> tasks_to_visit;
  // Initialization loop
  for (size_t j = 0; j < tasks.size(); ++j) {
    const auto &id = tasks[j]->id();
    unprocessed_parents[id] = tasks[j]->parents().size();
    if (unprocessed_parents[id] == 0) {
      tasks[j]->set_top_level(tasks[j]->weight());
      tasks_to_visit.push(tasks[j]);
    }
  }
  MS_LOG(INFO) << "Top Level: End Initialization";
  MS_LOG(INFO) << "Top Level: Start Traversal Loop";
  while (!tasks_to_visit.empty()) {
    const auto &selected_task = tasks_to_visit.front();
    // Update candidate tasks
    for (auto &successor : selected_task->children()) {
      const auto &succ_id = successor->id();
      successor->set_depth(std::max(successor->depth(), selected_task->depth() + 1));
      successor->set_top_level(
        std::max(successor->top_level(), selected_task->top_level() + successor->weight()));
      unprocessed_parents[succ_id] -= 1;
      if (unprocessed_parents[succ_id] == 0) {
        tasks_to_visit.push(successor);
      }
    }
    tasks_to_visit.pop();
  }
  MS_LOG(INFO) << "Top Level: End Traversal Loop";
}

void gpto::ComputeBottomLevelAndWeightedLength(std::vector<std::shared_ptr<Task>> &tasks) {
  MS_LOG(INFO) << "Bottom Level: Start Initialization";
  std::unordered_map<TaskId, size_t> unprocessed_children;
  std::unordered_map<TaskId, double> children_sum;
  std::unordered_map<TaskId, double> children_max;
  std::queue<std::shared_ptr<Task>> tasks_to_visit;
  // Initialization loop
  for (auto &task : tasks) {
    const auto &id = task->id();
    task->set_bottom_level(task->weight());
    task->set_weighted_length(task->weight());
    unprocessed_children[id] = task->children().size();
    if (unprocessed_children[id] == 0) {
      tasks_to_visit.push(task);
    }
  }
  MS_LOG(INFO) << "Bottom Level: End Initialization";
  MS_LOG(INFO) << "Bottom Level: Start Traversal Loop";
  while (!tasks_to_visit.empty()) {
    const auto &selected_task = tasks_to_visit.front();
    // Update candidate tasks
    for (auto &predecessor : selected_task->parents()) {
      const auto &pred_id = predecessor.lock()->id();
      predecessor.lock()->set_bottom_level(std::max(
        predecessor.lock()->bottom_level(), selected_task->bottom_level() + predecessor.lock()->weight()));
      children_sum[pred_id] += selected_task->weighted_length();
      children_max[pred_id] = std::max(children_max[pred_id], selected_task->weighted_length());
      unprocessed_children[pred_id] -= 1;
      if (unprocessed_children[pred_id] == 0) {
        if (children_max[pred_id] == 0) {
          MS_LOG(EXCEPTION) << "divisor children_max[pred_id] cannot be 0!";
        }
        predecessor.lock()->set_weighted_length(predecessor.lock()->weight() + children_max[pred_id] +
                                                children_sum[pred_id] / children_max[pred_id]);
        tasks_to_visit.push(predecessor.lock());
      }
    }
    tasks_to_visit.pop();
  }
  MS_LOG(INFO) << "Bottom Level: End Traversal Loop";
}

void gpto::ComputePredComm(std::vector<std::shared_ptr<Task>> &tasks) {
  for (auto &task : tasks) {
    task->set_pred_comm(0);
    for (auto &predecessor : task->parents()) {
      if (predecessor.lock()->gpto_type() == kComm) {
        task->set_pred_comm(task->pred_comm() + 1);
      }
    }
  }
}

void gpto::ComputePredCube(std::vector<std::shared_ptr<Task>> &tasks) {
  for (auto &task : tasks) {
    task->set_pred_cube(0);
    for (auto &predecessor : task->parents()) {
      if (predecessor.lock()->gpto_type() == kCube) {
        task->set_pred_cube(task->pred_cube() + 1);
      }
    }
  }
}

void gpto::InitializeMemoryImpact(std::vector<std::shared_ptr<Task>> &tasks){
  for (auto &task : tasks) {
    Memory out_weight = 0, workspace_weight = 0;
    for (auto &tensor : task->out_tensors()){
      if (tensor->type() == kWorkspace){       //TODO: Ioannis(later make them into lifelong end (new logic)???
        workspace_weight += tensor->weight();
      } else {
        out_weight += tensor->weight();
      }
    }
		for (auto &tensor : task->workspace_tensors()){
      workspace_weight += tensor->weight();
    }
		task->set_workspace_memory(workspace_weight);
		//task->set_mem_impact(out_weight);
		task->set_mem_impact(out_weight + workspace_weight);
  }
}

Time gpto::LowerBoundBottomLevel(std::vector<std::shared_ptr<Task>> &tasks) {
  Time max_bottom_level = 0;
  for (const auto &task : tasks) {
    max_bottom_level = std::max(max_bottom_level, task->bottom_level());
  }
  return max_bottom_level;
}

Time gpto::LowerBoundPEs(std::vector<std::shared_ptr<Task>> &tasks,
                                        std::unordered_map<TaskType, int32_t> &type_to_num_cores_map) {
  double lower_bound = 0;

  std::unordered_map<TaskType, Time> type_task_sum;
  for (const auto &task : tasks) {
    type_task_sum[task->gpto_type()] += task->weight();
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
std::pair<PeId, Time> SelectPEandTime(const Task &task, Time can_start,
                                      std::set<ProcessingElement, SortByLoad> *PEs_ptr) {
  auto &PEs = *PEs_ptr;
  std::pair<PeId, Time> return_pair = std::make_pair(0, 0);
  for (auto it = PEs.begin(); it != PEs.end(); ++it) {
    // unsafe use of const_cast, but we modify only idle list and not key sorting parameters like load, id, etc.
    // cf: https://stackoverflow.com/questions/43340050/modification-of-elements-of-stdset-defined-behavior
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
      } else {  // can_start > idle_it->second means we are not allowed to schedule the task here
        continue;
      }
      // If the task fits, then place it here
      if (idle_it->second - start_time >= task.weight()) {
        // Save info to return: start task at time idle_it->first
        return_pair.first = (*it).id;
        return_pair.second = start_time;
        // Update idle list
        if (!case_flag) {
          if (idle_it->second - idle_it->first == task.weight()) {  // whole idle interval is filled in, erase it
            mut_pe.idle.erase(idle_it);
          } else {  // idle_it->second - idle_it->first > task.weight()
            idle_it->first += task.weight();
          }
        } else {  // case_flag = true, idle interval is broken into two sub-blocks [idle_it->first, can_start] and
                  // (maybe empty) [can_start + weight, idle_it->second]
          Time upper = idle_it->second;
          idle_it->second = can_start;
          if (upper - can_start - task.weight() > 0) {
            std::pair<Time, Time> new_idle = std::make_pair(can_start + task.weight(), upper);
            mut_pe.idle.emplace(std::next(idle_it), new_idle);
          }
        }
        // Update load and PEs set
        auto updated_PE = PEs.extract(it);
        updated_PE.value().load += task.weight();
        PEs.insert(std::move(updated_PE));
        return return_pair;
      }
    }
  }
  return return_pair;
}

std::pair<PeId, Time> SelectPEandTimeAvailableStart(const Task &task, Time can_start,
                                                    std::vector<ProcessingElement> *PEs_ptr) {
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
      } else {  // can_start > idle_it->second means we are not allowed to schedule the task here
        continue;
      }
      if (idle_it->second - start_time >= task.weight()) {
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
    if (min_idle_it->second - min_idle_it->first == task.weight()) {  // whole idle interval is filled in, erase it
      min_it->idle.erase(min_idle_it);
    } else {  // idle_it->second - idle_it->first > task.weight()
      min_idle_it->first += task.weight();
    }
  } else {  // min_case = true, idle interval is broken into two sub-blocks [idle_it->first, can_start] and
            // (maybe empty)[can_start + task.weight(), idle_it->second]
    Time upper = min_idle_it->second;
    min_idle_it->second = can_start;
    if (upper - can_start - task.weight() > 0) {
      std::pair<Time, Time> new_idle = std::make_pair(can_start + task.weight(), upper);
      min_it->idle.emplace(std::next(min_idle_it), new_idle);
    }
  }
  // Update load
  min_it->load += task.weight();
  return return_pair;
}

constexpr TaskSortFunction TASK_SORT[] = {SortByWeightMax,
                                            SortByWeightMin,
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

constexpr std::string_view TASK_SORT_NAMES[] = {"SortByWeightMax",
                                                  "SortByWeightMin",
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

enum class PEsSort { kSortByLoad = 0, kSortByValidStart, kNumPEsSort };

constexpr std::string_view PE_NAME_SORT[] = {"SortByLoad", "SortByValidStart"};

SchedulingOutput gpto::Process(SchedulingInput &input, const size_t graph_id, const FuncGraphPtr &graph, const std::set<std::shared_ptr<Tensor>> &tensors) {
  std::vector<std::shared_ptr<Task>> *tasks = &(input.tasks);
  auto type_to_num_cores_map = GetTestPEs();
  SchedulingOutput output{{}, SIZE_MAX, HARD_MEMORY_LIMIT};

  // Optional: verify input task graph is a DAG
  if (VerifyDAG(*tasks)) {
    MS_LOG(INFO) << "Verification of DAG: SUCCESS";
  } else {
    MS_LOG(INFO) << "Verification of DAG: FAILURE";
  }

  // Preprocessing: values computation for necessary sorting
  ComputeBottomLevelAndWeightedLength(*tasks);
  ComputeDepthAndTopLevel(*tasks);
  ComputePredComm(*tasks);
  if (common::GetEnv("MS_ENABLE_GPTO_MULTISTREAM") == "1") {
      ComputePredCube(*tasks);
  }

  // Loop over all sorting combinations
  std::unordered_map<std::shared_ptr<Task>, Time> best_start, best_end;  // to use in verify dependencies only
  std::string best_solution = "";
  MS_LOG(INFO) << "Start loop multiple scheduling functions";
  for (size_t task_sort = 0; task_sort < static_cast<size_t>(kNumTaskSort); ++task_sort) {
    for (size_t pes_sort = 0; pes_sort < static_cast<size_t>(PEsSort::kNumPEsSort); ++pes_sort) {
      // Etienne: Add variable here to force algo
      if (common::GetEnv("MS_ENABLE_GPTO_ALGO") != ""){
          if (common::GetEnv("MS_ENABLE_GPTO_ALGO") != TASK_SORT_NAMES[task_sort]) {
              continue;
          }
      }
      MS_LOG(INFO) << TASK_SORT_NAMES[task_sort] << " and " << PE_NAME_SORT[pes_sort];
      SchedulingOutput solution = ProcessCore(*tasks, type_to_num_cores_map, TASK_SORT[task_sort],
                                              (pes_sort == static_cast<size_t>(PEsSort::kSortByLoad)));      
      if ((solution.makespan < output.makespan || (solution.makespan == output.makespan && solution.memory_peak < output.memory_peak)) 
            && solution.memory_peak <= HARD_MEMORY_LIMIT) {
        output = solution;
        best_solution = TASK_SORT_NAMES[task_sort];
        for (const auto &task : *tasks) {  // to use in verify dependencies only
          best_start[task] = task->start();
          best_end[task] = task->end();
        }
      }
      for (const auto &task : *tasks) {
        task->ResetStartEnd();
      }
    }
  }
  MS_LOG(INFO) << "End loop multiple scheduling functions";

  if (best_solution == "") {
    MS_LOG(EXCEPTION) << "Hard memory limit is not satisfied by any scheduling memory estimate, exiting...";
  }

  // Print stats about best solution
	MS_LOG(INFO) << "Memory-aware heuristics with soft memory limit " << SOFT_MEMORY_LIMIT << " and hard memory limit " << HARD_MEMORY_LIMIT;
  MS_LOG(INFO) << "Best solution is: " << best_solution;
  MS_LOG(INFO) << "Makespan of best solution is " << output.makespan;
  MS_LOG(INFO) << "Bottom level lower bound is " << LowerBoundBottomLevel(*tasks);
  MS_LOG(INFO) << "Max type lower bound is " << LowerBoundPEs(*tasks, type_to_num_cores_map);
  MS_LOG(INFO) << "Solution relative error is " << std::setprecision(5)
               << ((output.makespan /
                      (1.0 * std::max(LowerBoundBottomLevel(*tasks), LowerBoundPEs(*tasks, type_to_num_cores_map))) -
                    1) *
                   100)
               << "%";
	MS_LOG(INFO) << "Peak memory estimate of best solution is " << output.memory_peak;

  // Create and (optionally) verify dependencies (here only for testing)
  //MS_LOG(INFO) << "Start Schedule to Dependencies";
  //auto dependencies = ScheduleToDependencies(output);
  //MS_LOG(INFO) << "End Schedule to Dependencies";
	
	// Save best solution (intervals)
  for (const auto &task : *tasks) {
    task->set_start(best_start[task]);
    task->set_end(best_end[task]);
  }

  // Output log files
  //MS_LOG(INFO) << "Start printing output log file";
  //PrintLog(output, dependencies, graph_id, tensors);
  //MS_LOG(INFO) << "End printing output log file";
  auto lower = std::max(LowerBoundBottomLevel(*tasks), LowerBoundPEs(*tasks, type_to_num_cores_map));
  PrintLogForILP(input, output, graph_id, graph, lower, tensors);
  return output;
}

SchedulingOutput gpto::ProcessCore(std::vector<std::shared_ptr<Task>> &tasks,
                                                  std::unordered_map<TaskType, int32_t> &type_to_num_cores_map,
                                                  const TaskSortFunction &sortPtr, bool pe_load_sort) {
  SchedulingOutput output{{}, 0, 0};
  // Initializations for tasks
  MS_LOG(INFO) << "Started Task Initialization";
  std::set<std::shared_ptr<Task>, TaskSortFunction> candidate_tasks(sortPtr);
  std::unordered_map<TaskId, Time> can_start;
  std::unordered_map<TaskId, size_t> unprocessed_parents;
  for (auto &task : tasks) {
    const auto &id = task->id();
    can_start[id] = 0;
    unprocessed_parents[id] = task->parents().size();
    if (unprocessed_parents[id] == 0) {
      candidate_tasks.insert(task);
    }
  }

  // Initialization for memory impact handling
	InitializeMemoryImpact(tasks);
	std::unordered_map<size_t, std::set<std::shared_ptr<Task>>> left_consumers;
  for (auto &task : tasks) {
    for (auto &in_tensor : task->in_tensors()) {
      left_consumers[in_tensor->id()].insert(in_tensor->consumers().begin(), in_tensor->consumers().end());
    }
  }

  MS_LOG(INFO) << "Finished Task Initialization";

  // Initializations for processing elements
  // Pick a sorting for processing elements
  // Implemented: SortByLoad, SortByAvailableStart
  // Only one structure to be used depending on argument; we define both here
  std::unordered_map<TaskType, std::set<ProcessingElement, SortByLoad>> PEs_load;
  std::unordered_map<TaskType, std::vector<ProcessingElement>> PEs_start;
  MS_LOG(INFO) << "Started Processing Element Initialization";
  size_t count = 0;
  for (const auto &type_to_num : type_to_num_cores_map) {
    const auto &type = type_to_num.first;
    const auto &num_cores = type_to_num.second;
    for (int i = 0; i < num_cores; ++i) {
      ProcessingElement new_pe;
      new_pe.id = count + i;
      new_pe.gpto_type = type;
      new_pe.load = 0;
      new_pe.idle.emplace_back(0, SIZE_MAX);
      if (pe_load_sort) {
        PEs_load[type].insert(new_pe);
      } else {
        PEs_start[type].push_back(new_pe);
      }
    }
    count += num_cores;
  }
  MS_LOG(INFO) << "Finished Processing Element Initialization";

  // Task graph scheduling loop
  MS_LOG(INFO) << "Started Scheduling Main Loop";
	output.memory_peak = 0;
	Memory cur_mem_peak = 0;
	std::unordered_map<TaskType, Memory> last_workspace_memory; // comp/comm for now -> originally 0 by definition here
  while (!candidate_tasks.empty()) {
    // Select task and schedule it (memory-aware), save info for output
    bool flag = false;
		TaskPtr selected_task;
		for (auto it = candidate_tasks.begin(); it != candidate_tasks.end(); ++it){
			selected_task = *it;
			if (cur_mem_peak + selected_task->mem_impact() <= SOFT_MEMORY_LIMIT){
				flag = true;
				break;
			}				
		}
		if (flag == false){
			selected_task = *(candidate_tasks.begin());
		}
    const auto &selected_id = selected_task->id();
		
		// Maintain memory peak information
		cur_mem_peak += selected_task->mem_impact() - last_workspace_memory[selected_task->gpto_type()];
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
    selected_task->set_end(sigma + selected_task->weight());
    // New interval for task in output
    Interval new_interval{selected_id, selected_task->name(), selected_task->gpto_type(), selected_task->start(), selected_task->end()};
    output.task_times.push_back(new_interval);
    // Update makespan
    output.makespan = std::max(output.makespan, selected_task->end());

		// Update memory impact values (no need for workspace memory removal here; only using as first estimate)
    for (auto &in_tensor : selected_task->in_tensors()) {
      const auto &tid = in_tensor->id();
      left_consumers[tid].erase(selected_task);
      if (left_consumers[tid].size() == 1) {
        auto last_consumer = *(left_consumers[tid].begin());
				auto it = candidate_tasks.find(last_consumer);
				if (it != candidate_tasks.end()) {
					auto updated_candidate = candidate_tasks.extract(it);
					updated_candidate.value()->set_mem_impact(updated_candidate.value()->mem_impact() - in_tensor->weight());
					candidate_tasks.insert(std::move(updated_candidate));
				} else {
					last_consumer->set_mem_impact(last_consumer->mem_impact() - in_tensor->weight());
        }
      }
    }
    // Update out-tensors of selected node
    for (auto &out_tensor : selected_task->out_tensors()) {
      if (out_tensor->consumers().size() == 1) {
        auto last_consumer = *(out_tensor->consumers().begin());
        last_consumer->set_mem_impact(last_consumer->mem_impact() - out_tensor->weight());
      }
    }

    // Update candidate tasks
    candidate_tasks.erase(selected_task);
    for (const auto &successor : selected_task->children()) {
      const auto &succ_id = successor->id();
      can_start[succ_id] = std::max(can_start[succ_id], selected_task->end());
      unprocessed_parents[succ_id] -= 1;
      if (unprocessed_parents[succ_id] == 0) {
        candidate_tasks.insert(successor);
      }
    }
  }
  MS_LOG(INFO) << "Finished Scheduling Main Loop";
  MS_LOG(INFO) << "Makespan is " << output.makespan;
	MS_LOG(INFO) << "Peak mem is " << output.memory_peak;
  // Verification of scheduling solution (optional)
  if (VerifyScheduling(tasks)) {
    MS_LOG(INFO) << "Verification of Scheduling: SUCCESS";
  } else {
    MS_LOG(INFO) << "Verification of Scheduling: FAILURE";
  }

  return output;
}

bool gpto::VerifyScheduling(std::vector<std::shared_ptr<Task>> &tasks) {
  bool flag = true;
  MS_LOG(INFO) << "Start Verification of Scheduling";
  for (auto &task : tasks) {
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

bool BFSsort(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->depth() < task2->depth() || (task1->depth() == task2->depth() && task1->id() < task2->id());
}

bool gpto::VerifyDependencies(std::vector<std::shared_ptr<Task>> &tasks,
                                             std::vector<std::pair<TaskId, TaskId>> &dependencies) {
  bool flag = true;

  MS_LOG(INFO) << "Start Verification of Dependencies";
  // Traverse graph by depth to maintain ancestor info
  auto tasks_sorted = tasks;
  std::sort(tasks_sorted.begin(), tasks_sorted.end(), BFSsort);
  std::map<TaskId, std::map<TaskId, bool>> exists_path;
  std::map<TaskId, std::shared_ptr<Task>> id_to_ptr;
  for (auto current = tasks_sorted.begin(); current != tasks_sorted.end(); ++current) {
    id_to_ptr[(*current)->id()] = *current;
    for (auto parent = (*current)->parents().begin(); parent != (*current)->parents().end(); ++parent) {
      exists_path[(*parent).lock()->id()][(*current)->id()] = true;
      for (auto &it : tasks_sorted) {
        if (exists_path[it->id()][(*parent).lock()->id()]) {
          exists_path[it->id()][(*current)->id()] = true;
        }
      }
    }
  }
  // For each dependency, check if redundant it forms a directed cycle and if corresponding tasks are scheduled
  // correctly
  size_t redundant_count = 0;
  for (auto it = dependencies.begin(); it != dependencies.end(); ++it) {
    const auto &source = id_to_ptr[it->first];
    const auto &dst = id_to_ptr[it->second];
    if (exists_path[it->first][it->second]) {
      redundant_count++;
    }
    if (exists_path[it->second][it->first]) {
      MS_LOG(INFO) << "Dependency cycle formation: task " << source->id() << " [" << source->start() << ","
                   << source->end() << "] and task " << dst->id() << " [" << dst->start() << "," << dst->end() << "]";
    }
    if (!(source->start() < source->end() && source->end() <= dst->start() && dst->start() < dst->end())) {
      // allow weights of size 0
      MS_LOG(INFO) << "Dependency scheduling violation: task " << source->id() << " [" << source->start() << ","
                   << source->end() << "] and task " << dst->id() << " [" << dst->start() << "," << dst->end() << "]";
    }
  }
  MS_LOG(INFO) << "End Verification of Dependencies";
  MS_LOG(INFO) << redundant_count << " dependencies are redundant, " << dependencies.size() - redundant_count
               << " are real";

  return flag;
}

bool gpto::VerifyDAG(std::vector<std::shared_ptr<Task>> &tasks) {
  // simple verifier that no directed cycle exists
  std::unordered_map<TaskId, bool> visited;
  std::unordered_map<TaskId, size_t> unprocessed_parents;
  std::deque<std::shared_ptr<Task>> to_visit;
  MS_LOG(INFO) << "Start Verification of DAG";
  for (auto &task : tasks) {
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

void gpto::PrintLog(const SchedulingOutput &output,
                                   const std::vector<std::pair<TaskId, TaskId>> &dependencies, const FuncGraphPtr &graph,
                                   const size_t graph_id, std::set<std::shared_ptr<Tensor>> &tensors) {
  std::stringstream ss;
  ss << graph;
  std::ofstream out_file("gpto_out_" + std::to_string(graph_id) + "_" + ss.str() + ".log", std::ios::out | std::ios::trunc);
  if (!out_file.is_open()) {
    MS_LOG(ERROR) << "Could not open comp_comm_scheduling_out.log";
    return;
  }

  // Print info for tasks
  const auto &tasks = output.task_times;
  for (const auto &task : tasks) {
    out_file << "TASK id=" << std::to_string(task.id) << ", name=" << task.name << ", type=" << std::to_string(task.gpto_type)
             << ", start=" << std::to_string(task.start) << ", end=" << std::to_string(task.end) << "\n";
  }
  // Print dependencies (or events depending on function used)
  for (const auto &dependency : dependencies) {
    const auto &source = dependency.first;
    const auto &dst = dependency.second;
    out_file << "DEPENDENCY " << std::to_string(source) << " " << std::to_string(dst) << "\n";
  }
	// Print tensor info
  // Change set of TensorPtr to vector of TensorPtr to be able to sort the list
  std::vector<TensorPtr> tensors_vec;
  std::copy(tensors.begin(), tensors.end(), back_inserter(tensors_vec));
  std::sort(tensors_vec.begin(), tensors_vec.end(), [](const TensorPtr lhs, const TensorPtr rhs) { return lhs->id() < rhs->id(); });
  for (const auto &tensor : tensors_vec) {
    std::string consumers = "";
    for (const auto &consumer: tensor->consumers()){
      consumers += std::to_string(consumer->id()) + ";";
    }
    out_file << "TENSOR id=" << std::to_string(tensor->id()) << ", weight=" << std::to_string(tensor->weight()) << ", source=" << std::to_string(tensor->source()->id()) << ", consumers=" << consumers << "\n";
  }

  out_file.close();
}

void gpto::PrintLogForILP(const SchedulingInput &input, const SchedulingOutput &output,
                                         const size_t graph_id, const FuncGraphPtr &graph, const Time lower,
                                         const std::set<TensorPtr> &tensors) {

  std::stringstream ss;
  ss << graph;
  std::ofstream out_file("gpto_out_ilp_" + std::to_string(graph_id) + "_" + ss.str() + ".log", std::ios::out | std::ios::trunc);
  if (!out_file.is_open()) {
    MS_LOG(ERROR) << "Could not open gpto_out file";
    return;
  }
  // Print info for tasks
  const auto &tasks = input.tasks;
  for (const auto &task : tasks) {
    out_file << "TASK id=" << std::to_string(task->id()) << ", name=" << task->name() << ", type=" << std::to_string(task->gpto_type())
             << ", cost=" << std::to_string(task->weight()) << ", top=" << std::to_string(task->top_level()-task->weight())
             << ", bottom=" << std::to_string(task->bottom_level()-task->weight())
             << "\n";
  }

  // Print makespan and memory bounds
  out_file << "UPPER " << output.makespan << "\n";
  out_file << "LOWER " << lower           << "\n";
	out_file << "SOFT_MEMORY_LIMIT " << SOFT_MEMORY_LIMIT << "\n";
  out_file << "HARD_MEMORY_LIMIT " << HARD_MEMORY_LIMIT << "\n";

  // Print edges
  for (const auto &task : tasks) {
    for (const auto &child : task->children()){
      out_file << "EDGE " << std::to_string(task->id()) << " " << std::to_string(child->id()) << "\n";
    }
  }

  // Print same-type task pairs which can be executed in parallel
  std::vector<DynamicBitSet> nodes_dependency;
  gpto::ComputeAncestorsDescendants(tasks, nodes_dependency);
  for (size_t i = 0; i < tasks.size(); i++){
    for (size_t j = i+1; j < tasks.size(); j++){
      const auto &task1 = tasks[i];
      const auto &task2 = tasks[j];
      if (task1->gpto_type() != task2->gpto_type()) continue;
      if (nodes_dependency[task2->id()].IsBitTrue(task1->id()) || nodes_dependency[task1->id()].IsBitTrue(task2->id())) continue;
      out_file << "NO_OVERLAP " << std::to_string(task1->id()) << " " << std::to_string(task2->id()) << "\n";
    }
  }
  // Print tensor info
  for (const auto &tensor : tensors) {
    std::string consumers = "";
    for (const auto &consumer: tensor->consumers()){
      consumers += std::to_string(consumer->id()) + ";";
    }
    out_file << "TENSOR id=" << std::to_string(tensor->id()) << ", weight=" << std::to_string(tensor->weight())
             << ", source=" << std::to_string(tensor->source()->id()) << ", consumers=" << consumers << "\n";
  }

  out_file.close();
}

void gpto::ComputeAncestorsDescendants(const std::vector<std::shared_ptr<Task>> &tasks, std::vector<DynamicBitSet> &nodes_dependency) {
  // Ioannis: Assume tasks are sorted by id (ie in BFS order); if not, sort them
  // Ioannis: Do we need each node to be ancestor/descendant of itself? No (for now)
  size_t count = tasks.back()->id() + 1;
  for (size_t i = 0; i < count; i++) {
    (void)nodes_dependency.emplace_back(count);
  }
  for (const auto &task : tasks) {
    for (const auto &parent : task->parents()){
      nodes_dependency[task->id()].SetBitTrue(parent.lock()->id());
      Union(&nodes_dependency[task->id()], &nodes_dependency[parent.lock()->id()]);
    }
    // Ioannis: log message just for debugging
    MS_LOG(DEBUG) << "Task " << task->id() << " has " << nodes_dependency[task->id()].CountOnesNum() << "ancestors";
  }
}

void InsertEdges(const std::vector<CNodePtr> &cnode_vec,
                     std::unordered_map<CNodePtr, TaskPtr> *cnode_to_task_map_ptr) {
  for (size_t i = 0; i < cnode_vec.size(); ++i) {
    for (size_t j = 0; j < cnode_vec[i]->size(); ++j) {
      const auto &input_node = cnode_vec[i]->input(j)->cast<CNodePtr>();
      if ((*cnode_to_task_map_ptr).count(input_node) == 0) continue;

      ((*cnode_to_task_map_ptr)[cnode_vec[i]])->AddParent((*cnode_to_task_map_ptr)[input_node]);
      ((*cnode_to_task_map_ptr)[input_node])->AddChild((*cnode_to_task_map_ptr)[cnode_vec[i]]);
      MS_LOG(INFO) << "Edge " << (*cnode_to_task_map_ptr)[input_node]->id() << " "
                   << (*cnode_to_task_map_ptr)[cnode_vec[i]]->id();
      MS_LOG(INFO) << "Edge (UniqueName) " << input_node->UniqueName() << " " << cnode_vec[i]->UniqueName();
    }
  }
}

bool IsCubeKernel(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
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

// To-do: rename the function
TaskType GetGPTOTaskTypeFromCNode(const CNodePtr cnode){
  if(common::AnfAlgo::IsCommunicationOp(cnode)){
    return kComm;
  }
  if (common::GetEnv("MS_ENABLE_GPTO_SINGLESTREAM") == "1") {
    return kComm;
  }  
  if (common::GetEnv("MS_ENABLE_GPTO_MULTISTREAM") == "0") {
    return kComp;
  }
  if(IsCubeKernel(cnode)){
    return kCube;
  } else {
    return kComp;
  }
}

TaskType GetRealTaskTypeFromCNode(const CNodePtr cnode){
  if(common::AnfAlgo::IsCommunicationOp(cnode)){
    return kComm;
  } else if(IsCubeKernel(cnode)){
    return kCube;
  } else {
    return kComp;
  }
}

size_t GetAlignSize(size_t original_size) {
  constexpr size_t alignment = 512;
  constexpr size_t alignment_complement = 31;
  size_t aligned_size =
    (original_size > 0) ? ((original_size + alignment + alignment_complement) / alignment) * alignment : 0;
  return aligned_size;
}

void ContractUnrealTasks(const std::vector<CNodePtr> &cnode_vec,
                 std::unordered_map<CNodePtr, TaskPtr> *cnode_to_task_map_ptr) {

  std::unordered_set<CNodePtr> nodes_to_be_removed;

  for (const auto &cnode : cnode_vec) {
    if (AnfUtils::IsRealKernel(cnode)) continue;

    auto &task_to_remove = (*cnode_to_task_map_ptr)[cnode];
    auto &task_parents = task_to_remove->parents();
    auto &task_children = task_to_remove->children();

    // Case:  void --> Load --> Add     ==>     void --> Add
    if (task_parents.empty()){
      for (auto &task_child : task_children) {
        task_child->RemoveParent(task_to_remove);
      }
      task_to_remove->ClearChildren();
      //TEST
      nodes_to_be_removed.insert(cnode);
      // (*cnode_to_task_map_ptr).erase(cnode);
      //TEST
      continue;
    }

    // Case: Add --> MakeTuple --> Return --> void     ==>     Add --> void
    if (task_children.empty()){
        MS_LOG(INFO) << "Current task doesn't have any children, process parents instead";
        for (auto &task_parent: task_parents) {
            task_parent.lock()->RemoveChild(task_to_remove);
        }
        task_to_remove->ClearParents();
        //TEST
        nodes_to_be_removed.insert(cnode);
        // (*cnode_to_task_map_ptr).erase(cnode);
        //TEST
        continue;
    }

    for (auto &task_parent : task_parents) {
      task_parent.lock()->RemoveChild(task_to_remove);
      for (auto &task_child : task_children) {
        task_parent.lock()->AddChild(task_child);
        task_child->AddParent(task_parent);
        task_child->RemoveParent(static_cast<std::weak_ptr<Task>>(task_to_remove));
        MS_LOG(INFO) << "Contraction (id): " << task_parent.lock()->id() << " - " << task_child->id();
      }
    }
    task_to_remove->ClearParents();
    task_to_remove->ClearChildren();

    //TEST
    nodes_to_be_removed.insert(cnode);
    //(*cnode_to_task_map_ptr).erase(cnode);
    //TEST


  }

  for (const auto &cnode : cnode_vec) {

    //TEST
    if (nodes_to_be_removed.find(cnode) != nodes_to_be_removed.end()) {
       cnode_to_task_map_ptr->erase(cnode);
       continue;
    }
    //TEST

    auto &to_remove_task  = (*cnode_to_task_map_ptr)[cnode];
    auto &task_parents = to_remove_task->parents();
    auto &task_children = to_remove_task->children();

    MS_LOG(INFO) << "END Processing task " << to_remove_task->id() << ", cnode name " << cnode->UniqueName() << " with " << task_parents.size() << " parents and " << task_children.size() << " children";
    for (auto &task_parent: task_parents) { MS_LOG(INFO) << "Parent: " << task_parent.lock()->id();}
    for (auto &task_child: task_children) { MS_LOG(INFO) << "Child: " << task_child->id();}
  }
}

// To-Do: Review the function later
void ContractionEtienne(const std::vector<CNodePtr> &cnode_vec,
                 std::unordered_map<CNodePtr, TaskPtr> *cnode_to_task_map_ptr) {

  std::unordered_map<TaskPtr, CNodePtr> task_to_cnode_map_ptr;
  for (auto& it: *cnode_to_task_map_ptr) {
    task_to_cnode_map_ptr[it.second] = it.first;
  }

  for (const auto &cnode : cnode_vec) {
    auto &to_remove_task  = (*cnode_to_task_map_ptr)[cnode];
    auto &task_parents = to_remove_task->parents();
    auto &task_children = to_remove_task->children();

    MS_LOG(INFO) << "Processing task " << to_remove_task->id() << ", cnode name " << cnode->UniqueName() << " with " << task_parents.size() << " parents and " << task_children.size() << " children";

    if (AnfUtils::IsRealKernel(cnode)){
      MS_LOG(INFO) << "Task skipped as IsRealKernel";
      continue;
    }
    MS_LOG(INFO) << "Task is processing as not IsRealKernel";

    // Case:  void --> Load --> Add     ==>     void --> Add
    if (task_parents.empty()){
      MS_LOG(INFO) << "Current task doesn't have any parents, process children instead";
      for (auto &task_child : task_children) {
        MS_LOG(INFO) << "Process deletion of current kernel from child: " << task_to_cnode_map_ptr[task_child]->UniqueName();
        task_child->RemoveParent(to_remove_task);
      }
      to_remove_task->ClearChildren();
      continue;
    }

    // Case: Add --> MakeTuple --> Return --> void     ==>     Add --> Return --> void
    if (task_children.empty()){
        MS_LOG(INFO) << "Current task doesn't have any children, process parents instead";
        for (auto &task_parent: task_parents) {
            if (AnfUtils::IsRealKernel(task_to_cnode_map_ptr[task_parent.lock()])){continue;}
            MS_LOG(INFO) << "Process deletion of current kernel from parent: " << task_to_cnode_map_ptr[task_parent.lock()]->UniqueName();
            task_parent.lock()->RemoveChild(to_remove_task);
            to_remove_task->RemoveParent(task_parent.lock());
        }
        continue;
    }

    // Case: Complex case with parents and children for Non Real Kernel
    std::vector<TaskPtr> real_kernel_children;
    std::vector<TaskPtr> kernel_to_visit;
    std::vector<TaskPtr> kernel_visited;
    kernel_to_visit.insert(kernel_to_visit.begin(),task_children.begin(),task_children.end());
    auto cur_child = kernel_to_visit.back();
    while(cur_child != NULL){
			if (std::find(kernel_visited.begin(), kernel_visited.end(), cur_child) == kernel_visited.end()){
					if(AnfUtils::IsRealKernel(task_to_cnode_map_ptr[cur_child]) || IsPrimitiveCNode(task_to_cnode_map_ptr[cur_child], prim::kPrimReturn)){ // If real kernel, need to flag it and continue to visit the remaining kernel
							real_kernel_children.push_back(cur_child);
					} else { // Add all children of current child to visit
							kernel_to_visit.insert(kernel_to_visit.begin(),cur_child->children().begin(),cur_child->children().end());
					}
					kernel_visited.push_back(cur_child);
			}

			// Get the next kernel, C++ error if back() an empty vector
			if (!kernel_to_visit.empty()) {
					cur_child = kernel_to_visit.back();
					kernel_to_visit.pop_back();
			} else {
					cur_child = NULL;
			}
    }

    for (auto &task_parent: task_parents) {
      MS_LOG(INFO) << "Remove parents -> to_remove_task relationship: " << task_parent.lock()->id();
      task_parent.lock()->RemoveChild(to_remove_task);
      //to_remove_task->ClearParents();
      if(!AnfUtils::IsRealKernel(task_to_cnode_map_ptr[task_parent.lock()])){
        continue;
      }
      for (auto &task_child : real_kernel_children) {
        MS_LOG(INFO) << "Contraction - New link - Task " << task_parent.lock()->id() << " - " << task_child->id();
        task_parent.lock()->AddChild(task_child);
        task_child->AddParent(task_parent);
        task_child->RemoveParent(static_cast<std::weak_ptr<Task>>(to_remove_task));
      }
    }
    to_remove_task->ClearParents();
    to_remove_task->ClearChildren();
  }

  for (const auto &cnode : cnode_vec) {
    auto &to_remove_task  = (*cnode_to_task_map_ptr)[cnode];
    auto &task_parents = to_remove_task->parents();
    auto &task_children = to_remove_task->children();

    MS_LOG(INFO) << "END Processing task " << to_remove_task->id() << ", cnode name " << cnode->UniqueName() << " with " << task_parents.size() << " parents and " << task_children.size() << " children";
    for (auto &task_parent: task_parents) { MS_LOG(INFO) << "Parent: " << task_parent.lock()->id();}
    for (auto &task_child: task_children) { MS_LOG(INFO) << "Child: " << task_child->id();}
  }
}

KernelWithIndex GetVisitKernelWithReturnType(const AnfNodePtr &ori_node, size_t ori_index, std::unordered_map<CNodePtr, TaskPtr> *cnode_to_task_map_ptr) {
  auto prenode = common::AnfAlgo::VisitKernelWithReturnType(ori_node, ori_index, false);
  //auto xx = prenode.first->cast<CNodePtr>();
  while (prenode.first->isa<CNode>() && cnode_to_task_map_ptr->find(prenode.first->cast<CNodePtr>()) == cnode_to_task_map_ptr->end()) {
    auto cnode = prenode.first->cast<CNodePtr>();
//    if (!common::AnfAlgo::IsNopNode(cnode)) {
//      MS_LOG(INTERNAL_EXCEPTION) << "Node[" << ori_node->fullname_with_scope() << "] find input node["
//                                 << cnode->fullname_with_scope()
//                                 << "] doesn't exist in nodes_map and is not a nop node!!!!";
//    }
    prenode = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(1), 0, false);
  }
  return prenode;
}

void ExtractRealTensors(const SchedulingInput &scheduling_input, std::unordered_map<CNodePtr, TaskPtr> *cnode_to_task_map_ptr, std::set<std::shared_ptr<Tensor>> &tensors) {
	size_t tensor_count = 0;
	const auto &tasks = scheduling_input.tasks;
	
	// Looping over tasks to obtain output and workspace tensors (somas style)
	for (auto &task : tasks) {
		const auto &kernel_mod = AnfAlgo::GetKernelMod(task->cnode());
		MS_EXCEPTION_IF_NULL(kernel_mod);
		
		// Extract each node's output tensors 
		for (const auto &size : kernel_mod->GetOutputSizeList()) {
			Time weight = GetAlignSize(size);
			if (weight == 0){
				weight = GetAlignSize(1);
			}
			std::shared_ptr<Tensor> new_tensor = std::make_shared<Tensor>(tensor_count, weight, task, kWorkspace);	// initially kWorkspace, since no consumers
			task->out_tensors().push_back(new_tensor);
			MS_LOG(INFO) << "New output tensor " << tensor_count << " source id " << task->id() << " weight " << weight;
			tensor_count++;
      tensors.insert(new_tensor);
		}
		
		// Extract each node's workspace tensor
		for (const auto &size : kernel_mod->GetWorkspaceSizeList()) {
			Time weight = GetAlignSize(size);
			if (weight == 0){
				weight = GetAlignSize(1);
			}
			std::shared_ptr<Tensor> new_tensor = std::make_shared<Tensor>(tensor_count, weight, task, kWorkspace);
			task->workspace_tensors().push_back(new_tensor);
			MS_LOG(INFO) << "New workspace tensor " << tensor_count << " source id " << task->id() << " weight " << weight;
			tensor_count++;
      tensors.insert(new_tensor);
		}
	}

	// Looping over tasks to obtain input tensors after all outputs have been saved (somas style)
	for (auto &task : tasks) {
		const auto &kernel = task->cnode();
		const auto &kernel_mod = AnfAlgo::GetKernelMod(kernel);
		MS_EXCEPTION_IF_NULL(kernel_mod);
		
		if (common::AnfAlgo::GetCNodeName(kernel) != kMemSetOpName) {	// standard input case
			auto input_tensor_num = common::AnfAlgo::GetInputTensorNum(kernel);
			for (size_t i = 0; i < input_tensor_num; i++) {
				auto input_node = kernel->input(i+1);
				MS_EXCEPTION_IF_NULL(input_node);
				KernelWithIndex prenode_index = GetVisitKernelWithReturnType(input_node, 0, cnode_to_task_map_ptr);
				MS_EXCEPTION_IF_NULL(prenode_index.first);
				if (common::AnfAlgo::CheckPrimitiveType(prenode_index.first, prim::kPrimMakeTuple)) {
					MS_LOG(INTERNAL_EXCEPTION) << "Node " << kernel->fullname_with_scope() << "'s input node [" << input_node->DebugString()
                                     << "]'s input " << i << " is MakeTuple";
				}

				if (!AnfUtils::IsRealCNodeKernel(prenode_index.first)) { // somas input parameter case, ignore for now
					MS_LOG(INFO) << "Input  [" << prenode_index.first->fullname_with_scope() << "] is not a real cnode kernel.";
					continue;
				}
				auto iter = cnode_to_task_map_ptr->find(prenode_index.first->cast<CNodePtr>());
				if (iter == cnode_to_task_map_ptr->end()){
					MS_LOG(INFO) << "Kernel[" << kernel->fullname_with_scope() << "]'s input " << i << " ["
                       << prenode_index.first->fullname_with_scope() << "] not found in tasks";
					continue;
				}
				auto pre_task = iter->second;
				if (prenode_index.second > pre_task->out_tensors().size()){
					MS_LOG(INFO) << "Output index " << prenode_index.second << " exceeds input node ["
                       << prenode_index.first->fullname_with_scope() << "]'s outputs size "
                       << pre_task->out_tensors().size();
					continue;
				}
				auto input_tensor = pre_task->out_tensors()[prenode_index.second];
				MS_EXCEPTION_IF_NULL(input_tensor);
				input_tensor->consumers().insert(task);
				task->in_tensors().push_back(input_tensor);
				MS_LOG(INFO) << "Tensor " << input_tensor->id() << " has new consumer " << task->id();
				if(input_tensor->type() == TensorType::kWorkspace){
					input_tensor->set_type(TensorType::kOutput);
				}
				tensors.insert(input_tensor);	// TODO: remove eventually
			}	
		} else { // atomic clean input case 
			auto input_tensor_num = common::AnfAlgo::GetInputTensorNum(kernel);
			for (size_t i = 0; i < input_tensor_num; i++) {
				auto pre_node = kernel->input(i+1)->cast<CNodePtr>();
				MS_EXCEPTION_IF_NULL(pre_node);
				
				auto iter = cnode_to_task_map_ptr->find(pre_node);
				if (iter == cnode_to_task_map_ptr->end()){
					MS_LOG(INFO) << "Kernel[" << kernel->fullname_with_scope() << "]'s input " << i << " ["
                                        << pre_node->fullname_with_scope() << "] not found in tasks";
					continue;
				}
				auto pre_task = iter->second;
				
				if (common::AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, pre_node)) {	// clean output
					auto clean_output_indexs = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicOutputIndexs);
					for (auto index : clean_output_indexs) {
						if (index > pre_task->out_tensors().size()) {
							MS_LOG(INFO) << "Output index " << index << " exceed input node ["
													<< pre_node->fullname_with_scope() << "]'s outputs size "
													<< pre_task->out_tensors().size();
							continue; // TODO: replace above INFO by INTERNAL_EXCEPTION and remove continue (everywhere)
						}
						auto input_tensor = pre_task->out_tensors()[index];
						MS_EXCEPTION_IF_NULL(input_tensor);
						task->in_tensors().push_back(input_tensor);
						//
            if(input_tensor->type() == TensorType::kWorkspace){
					    input_tensor->set_type(TensorType::kOutput);
				    }
						input_tensor->consumers().insert(task);
						tensors.insert(input_tensor);	// TODO: remove eventually
						//
					}
				}
				
				if (common::AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, pre_node)) {	// clean workspace
					auto clean_workspace_indexs = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicWorkspaceIndexs);
					for (const auto &index : clean_workspace_indexs) {
						if (index > pre_task->out_tensors().size()) {
						  MS_LOG(INFO) << "Workspace index " << index << " exceed input node ["
                                                  << pre_node->fullname_with_scope() << "]'s Workspace size "
                                                  << pre_task->workspace_tensors().size();
					          continue;
						}
						auto input_tensor = pre_task->workspace_tensors()[index];
						MS_EXCEPTION_IF_NULL(input_tensor);
						task->in_tensors().push_back(input_tensor);
						//
            if(input_tensor->type() == TensorType::kWorkspace){
					    input_tensor->set_type(TensorType::kOutput);
				    }
						input_tensor->consumers().insert(task);
						tensors.insert(input_tensor);	// TODO: remove eventually (not use extra storage)
						//
					}
				}
			}	
		}
	}
}

size_t CalculateVectorCost(CNodePtr cnode){
    Time weight = 0;
    if (common::AnfAlgo::GetInputTensorNum(cnode) == 0) {
        return weight;
    }
    KernelWithIndex kernel_with_index_1 = common::AnfAlgo::GetPrevNodeOutput(cnode, 0);
    ShapeVector shape_1 = common::AnfAlgo::GetOutputInferShape(kernel_with_index_1.first, kernel_with_index_1.second);
    const TypeId type_1 = common::AnfAlgo::GetOutputInferDataType(kernel_with_index_1.first, 0);
    size_t type_size_1 = GetDataTypeSize(type_1);
    size_t compute_count = std::accumulate(shape_1.cbegin(), shape_1.cend(), 1, std::multiplies<size_t>{});
    weight = 0.5 + (compute_count*type_size_1/128);
    return weight;
}


size_t CalculateCubeCost(CNodePtr cnode){
    Time weight = 0;
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
    if (IsPrimitiveCNode(cnode, prim::kPrimBatchMatMul) || IsPrimitiveCNode(cnode, prim::kPrimBatchMatMulV2)){
        batch = shape_1.front();
        if (shape_1.size() == 4) {
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
    for(auto dim: shape_1){ tmp.push_back(dim); }
    for(auto dim: shape_2){
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

    weight = 21 + batch*m*k*n*type_size_1/8192;
    return weight;
}

size_t CalculateCommCost(CNodePtr cnode){
    Time weight = 0;
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
        MS_LOG(INFO) << "shape or type is nullptr, ignore";
        continue;
      }
      ShapeVector shape = common::AnfAlgo::GetOutputInferShape(kernel_with_index.first, kernel_with_index.second);
      if (shape.size() <= 0) continue;

      const TypeId type = common::AnfAlgo::GetOutputInferDataType(kernel_with_index.first, 0);
      if (type == kObjectTypeUMonad || type == kObjectTypeMonad || type == kObjectTypeFunction) continue;

      size_t type_size = GetDataTypeSize(type);
      weight += std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * type_size;
    }

    if (output_num > 1) {
        for (size_t j = 0; j < output_num; j++) {
            ShapeVector shape = common::AnfAlgo::GetOutputInferShape(cnode, j);
            if (shape.size() <= 0) continue;

            const TypeId type = common::AnfAlgo::GetOutputInferDataType(cnode, j);
            if (type == kObjectTypeUMonad || type == kObjectTypeMonad || type == kObjectTypeFunction) continue;

            size_t type_size = GetDataTypeSize(type);
            weight += std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * type_size;
        }
    }

    return weight;
}

size_t CalculateProfilingCost(const CNodePtr &cnode){
    Time weight = 0;
    std::ifstream file;
    file.open(common::GetEnv("MS_ENABLE_GPTO_PROF_FILE"));

    std::string line;
    while(std::getline(file, line)) {
        std::istringstream s(line);
        std::string field;
        std::vector<std::string> fields;
        while(getline(s, field,',')) {
            fields.push_back(field);
        }
        if (cnode->fullname_with_scope() == fields[0]){
            weight = stoi(fields[3]);
            break;
        }
    }
    return weight;
}

void gpto::PrintLogBaseline(const SchedulingInput &input,
                                   const std::vector<CNodePtr> &execution_order,
                                   std::unordered_map<CNodePtr, TaskPtr> *cnode_to_task_gpto_map_ptr, const FuncGraphPtr &graph,
                                   const size_t graph_id) {
  std::stringstream ss;
  ss << graph;
  std::ofstream out_file("comp_comm_scheduling_baseline_" + std::to_string(graph_id) + "_" + ss.str() + ".log", std::ios::out | std::ios::trunc);
  if (!out_file.is_open()) {
    MS_LOG(ERROR) << "Could not open comp_comm_scheduling_baseline_" << graph_id << ".log";
    return;
  }

  std::unordered_map<TaskId, Time> taskid_to_end_value;
  std::unordered_map<TaskId, Time> taskid_to_start_value;
  size_t makespan = 0;

  for(size_t i=0; i<execution_order.size(); i++){
    const auto &cnode = execution_order[i];

    TaskPtr current_task = (*cnode_to_task_gpto_map_ptr)[cnode];

    // Find the latest executed task which has the same type as the current task
    TaskPtr last_task = nullptr;
    MS_LOG(INFO) << "Current value loop: " << i << " with node: " << execution_order[i]->UniqueName();
    for(int j=i-1; j>=0; j--){
      MS_LOG(INFO) << "Current value loop j: " << j;
      TaskPtr tmp_task = (*cnode_to_task_gpto_map_ptr)[execution_order[j]];
      MS_LOG(INFO) << "With node: " << tmp_task->name();
      if(tmp_task->gpto_type() == current_task->gpto_type()){
        MS_LOG(INFO) << "Found node same type";
        last_task = tmp_task;
        break;
      }
    }

    // Find the latest parent of the current task
    for(const auto &parent: (*cnode_to_task_gpto_map_ptr)[cnode]->parents()){
      if(last_task == nullptr || taskid_to_end_value[parent.lock()->id()] >= taskid_to_end_value[last_task->id()]){
        last_task = parent.lock();
        MS_LOG(INFO) << "Find parent " << last_task->name();
      }
    }
    
    if (last_task == nullptr) {
      last_task = current_task;
      taskid_to_start_value[current_task->id()] = 0;
      taskid_to_end_value[current_task->id()] = 0 + current_task->weight();
    } else {    
      taskid_to_start_value[current_task->id()] = taskid_to_end_value[last_task->id()];
      taskid_to_end_value[current_task->id()] = taskid_to_start_value[current_task->id()] + current_task->weight();
    }

    size_t current_task_end = taskid_to_end_value[current_task->id()];

    if(current_task_end > makespan){
        makespan = taskid_to_end_value[current_task->id()];
    }

    out_file << "TASK id=" << std::to_string(current_task->id()) << ", name=" << current_task->name() << ", type=" << std::to_string(current_task->gpto_type())
             << ", start=" << std::to_string(taskid_to_start_value[current_task->id()]) << ", end=" << std::to_string(current_task_end) << "\n";
   
  }

  MS_LOG(INFO) << "Makespan of baseline is " + std::to_string(makespan);

  out_file.close();
}

SchedulingInput gpto::ExtractSchedulingInput(const std::vector<CNodePtr> &cnode_vec, std::unordered_map<CNodePtr, TaskPtr> *cnode_to_task_map_ptr, 
                                       std::set<std::shared_ptr<Tensor>> &tensors) {
  SchedulingInput scheduling_input;  // to fill in and return

  // Create a task per node
  for (size_t i = 0; i < cnode_vec.size(); ++i) {
    const auto &cnode = cnode_vec[i];

    std::shared_ptr<Task> task = std::make_shared<Task>(i, GetRealTaskTypeFromCNode(cnode), GetGPTOTaskTypeFromCNode(cnode), cnode->fullname_with_scope());
    Time weight = 0;
    if (common::GetEnv("MS_ENABLE_GPTO_PROF_FILE") != ""){    
      weight = CalculateProfilingCost(cnode);
    } else {
      if (!AnfUtils::IsRealKernel(cnode)){	// CASE 1: not real kernel node
        weight = 1;
      } else if(task->real_type() == kComp){    // CASE 2: comp node of type Vector
        weight = CalculateVectorCost(cnode);
      } else if (task->real_type() == kCube) {  // CASE 3: comp node of type Cube
        weight = CalculateCubeCost(cnode);
      } else {                                  // CASE 4: comm node
        weight = CalculateCommCost(cnode);
      }
    }

    //std::shared_ptr<Task> task = std::make_shared<Task>(i, GetRealTaskTypeFromCNode(cnode), GetGPTOTaskTypeFromCNode(cnode), cnode->fullname_with_scope());
    task->AssignWeight(weight);
    task->set_cnode(cnode);
    (*cnode_to_task_map_ptr)[cnode] = task;

    MS_LOG(INFO) << "Task " << task->id() << " with name " << cnode->UniqueName() << " and CNodePtr " << cnode
                 << " with weight " << task->weight() << " and type " << GetGPTOTaskTypeFromCNode(cnode);

    if (AnfUtils::IsRealKernel(cnode)){	// only maintain real kernels in vector of tasks, the rest will be contracted later
      scheduling_input.tasks.push_back(task);
    }
  }

  InsertEdges(cnode_vec, cnode_to_task_map_ptr);
  //ContractionEtienne(cnode_vec, cnode_to_task_map_ptr);
  ContractUnrealTasks(cnode_vec, cnode_to_task_map_ptr);
  ExtractRealTensors(scheduling_input, cnode_to_task_map_ptr, tensors);
  return scheduling_input;
}

Memory MemoryLowerBound(std::vector<std::shared_ptr<Task>> &tasks, std::vector<DynamicBitSet> &nodes_dependency, std::set<std::shared_ptr<Tensor>> &tensors){
 Memory max_lb = 0;
 
 for (const auto &task: tasks){
	 Memory task_lb = 0;
	 for (const auto &tensor : tensors){
		  //if (tensor->type() == 1) continue; // ignore workspace for now
			const auto &source = tensor->source();
			const auto &consumers = tensor->consumers();
			
			if (task == source || consumers.count(task) > 0) {
				task_lb += tensor->weight();
			} else {
				if (nodes_dependency[task->id()].IsBitTrue(source->id())){
					for (const auto &consumer : consumers){
						if (nodes_dependency[consumer->id()].IsBitTrue(task->id())){
							task_lb += tensor->weight();
							break;
						}
					}
				}
			}	
		}
		task->set_lower_bound(task_lb);
		max_lb = std::max(max_lb, task_lb);
	}
	return max_lb;
}

void gpto::AddRealDependencies(const FuncGraphManagerPtr &manager, const std::vector<CNodePtr> &cnode_vec, const std::vector<std::pair<TaskId, TaskId>> &dependencies,
                         std::unordered_map<CNodePtr, TaskPtr> *cnode_to_task) {
  size_t count = 0, redundant_count = 0;
  for (const auto &dependency : dependencies) {

    if(count > (size_t)(stoi(common::GetEnv("MS_ENABLE_GPTO_COUNT")))){break;}
    MS_LOG(INFO) << "Checking dependency " << dependency.first << " " << dependency.second;
    const auto &source = cnode_vec[dependency.first];
    const auto &dest = cnode_vec[dependency.second];

    // Ignore dependencies already there
    if ((*cnode_to_task)[source]->HasChild((*cnode_to_task)[dest])) {
      MS_LOG(INFO) << "Dependency " << dependency.first << " " << dependency.second << " is redundant (already parent and child)";
      redundant_count++;
      continue;
    }

    // At least two inputs in destination node (input 0 is the node primitive)
    if (dest->size() < 2) {
      MS_LOG(INFO) << "Destination inputs size < 2: ignore";
      continue;
    }

    // If destination node (comp) has comm inputs, make dependency involving one of them
    for (size_t j = 1; j < dest->size(); ++j) {  // input 0 is node primitive: ignore
      if (!utils::isa<CNodePtr>(dest->input(j))) {
        MS_LOG(INFO) << "Not a cnodeptr at input " << j;
        continue;
      }

      //bool is_same_input = false;
      for (size_t k = 1; k < source->size(); ++k) {
	if (!utils::isa<CNodePtr>(source->input(k))) {
          MS_LOG(INFO) << "Not a cnodeptr at input " << j;
          continue;
        }
      }

      // Add real dependency logic here
      const auto &input_node = dest->input(j)->cast<CNodePtr>();
      std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), input_node, source};
      auto depend_node = dest->func_graph()->NewCNode(depend_inputs);
      depend_node->set_abstract(input_node->abstract()->Clone());
      depend_node->AddAttr("multistream_scheduling_depend", MakeValue(true));
      MS_EXCEPTION_IF_NULL(depend_node);
      auto &nodes = manager->node_users()[input_node];
      auto it = std::find_if(nodes.begin(), nodes.end(), [dest](const auto &user) { return user.first == dest; });
      if (it != nodes.end()) {
        int idx = (*it).second;
        manager->SetEdge(dest, idx, depend_node);
        MS_LOG(INFO) << "Added dependency from " << dependency.first << ", unique name " << source->UniqueName()
                     << ", to " << dependency.second << ", unique name " << dest->UniqueName();
        count++;
        break;  // add dependency involving only one destination node input
      } else {
        MS_LOG(INFO) << "User index not found: Ignore dependency and continue";
        continue;
      }
    }
  }
  MS_LOG(INFO) << "Num of real dependencies added is " << count;
  MS_LOG(INFO) << "Num of redundant dependencies (HasChild) is " << redundant_count;
}

std::vector<std::pair<CNodePtr,CNodePtr>> GPTO(const FuncGraphPtr &graph) {
  std::vector<std::pair<CNodePtr, CNodePtr>> events;
  if (common::GetEnv("MS_ENABLE_GPTO") != "1") {
    return events;
  }

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);

  if (common::GetEnv("MS_ENABLE_GPTO_MEMORY_LIMIT") != "") {
    SOFT_MEMORY_LIMIT = static_cast<Memory>(stoll(common::GetEnv("MS_ENABLE_GPTO_MEMORY_LIMIT")));
  } else {
    SOFT_MEMORY_LIMIT = static_cast<Memory>(device::ascend::AscendMemAdapter::GetInstance().FreeDevMemSize());
  }
  HARD_MEMORY_LIMIT = static_cast<Memory>(context->get_param<float>(MS_CTX_MAX_DEVICE_MEMORY)*kGBToByte);

  MS_LOG(INFO) << "Soft Memory value: " << SOFT_MEMORY_LIMIT;
  MS_LOG(INFO) << "Hard Memory value: " << HARD_MEMORY_LIMIT;

  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_LOG(INFO) << "Graph pointer: " << graph;
  MS_EXCEPTION_IF_NULL(manager);

  KernelGraphPtr kernel_graph = graph->cast<KernelGraphPtr>();
  const size_t graph_id = kernel_graph->graph_id();
  MS_LOG(INFO) << "Start Scheduling Subgraph " << graph << " with id " << graph_id << " and Execution order size= " << kernel_graph->execution_order().size();

  std::list<CNodePtr> cnode_list = graph->GetOrderedCnodes();
  std::vector<CNodePtr> cnode_vec(cnode_list.cbegin(), cnode_list.cend());

  MS_LOG(INFO) << "Start ExtractSchedulingInput";
  std::unordered_map<CNodePtr, TaskPtr> cnode_to_task;
  std::set<std::shared_ptr<Tensor>> tensors;	// TODO: remove this data structure eventually, and only use out_tensors/in_tensors within tasks
  SchedulingInput scheduling_input = gpto::ExtractSchedulingInput(cnode_vec, &cnode_to_task, tensors);
  MS_LOG(INFO) << "End ExtractSchedulingInput";
  if (scheduling_input.tasks.size() == 0){
    MS_LOG(WARNING) << "Scheduling input doesn't have tasks to continue... skip";
    MS_LOG(WARNING) << "Etienne PrintGraphExecuteOrder start";
    kernel_graph->PrintGraphExecuteOrder();
    MS_LOG(WARNING) << "Etienne PrintGraphExecuteOrder end";
    return events;  
  }

  MS_LOG(INFO) << "Start Baseline Greedy Scheduling";
  gpto::PrintLogBaseline(scheduling_input, kernel_graph->execution_order(), &cnode_to_task, graph, graph_id);
  MS_LOG(INFO) << "End Baseline Greedy Scheduling";

  auto scheduling_output = gpto::Process(scheduling_input, graph_id, graph, tensors);

	// Memory lower bound (for comparison only)
	std::vector<DynamicBitSet> nodes_dependency;

	MS_LOG(INFO) << "Start Compute Ancestors Descendants";
  gpto::ComputeAncestorsDescendants(scheduling_input.tasks, nodes_dependency);
	MS_LOG(INFO) << "End Compute Ancestors Descendants";

	MS_LOG(INFO) << "Start Memory Lower Bound";
	Time memory_lb = MemoryLowerBound(scheduling_input.tasks, nodes_dependency, tensors);
  MS_LOG(INFO) << "Memory Lower Bound value: " << memory_lb;
	MS_LOG(INFO) << "End Memory Lower Bound";

  std::vector<std::pair<TaskId, TaskId>> dependencies;
  if (common::GetEnv("MS_ENABLE_GPTO_EVENTS") != "1") {
    dependencies = gpto::ScheduleToDependencies(scheduling_output);
    gpto::AddRealDependencies(manager, cnode_vec, dependencies, &cnode_to_task);
    graph->cast<KernelGraphPtr>()->SetExecOrderByDefault();
  } else {
    dependencies = gpto::ScheduleToDependenciesDifferentTypes(scheduling_output);
    std::vector<CNodePtr> new_order;
    std::vector<Interval> task_times = scheduling_output.task_times;
    std::sort(task_times.begin(), task_times.end(), [](Interval x, Interval y) {
      return x.start < y.start || (x.start == y.start && x.end < y.end);
    });
    for (auto interval : task_times){
      new_order.push_back(cnode_vec[interval.id]);
    }
    MS_LOG(WARNING) << "Etienne PrintGraphExecuteOrder start";
    graph->cast<KernelGraphPtr>()->set_execution_order(new_order);
    kernel_graph->PrintGraphExecuteOrder();
    MS_LOG(WARNING) << "Etienne PrintGraphExecuteOrder end";
    for (auto dep : dependencies){
      events.push_back(std::make_pair(cnode_vec[dep.first], cnode_vec[dep.second]));
    }
  }
  // Output log file with all info (scheduling and dependencies)
  MS_LOG(INFO) << "Start printing output log file";
  gpto::PrintLog(scheduling_output, dependencies, graph, graph_id, tensors);
  MS_LOG(INFO) << "End printing output log file";
  return events;
}
}  // namespace opt
}  // namespace mindspore

