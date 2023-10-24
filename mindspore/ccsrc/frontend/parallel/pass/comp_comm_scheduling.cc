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

#include "mindspore/ccsrc/frontend/parallel/pass/comp_comm_scheduling.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "mindspore/ccsrc/frontend/parallel/step_parallel.h"
#include "mindspore/core/utils/misc.h"

#include "include/backend/optimizer/helper.h"

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
         (start2 >= start1 && start2 < end1);  // if equal start and end for two interval, then no overlap
}

std::vector<std::pair<TaskId, TaskId>> FastGreedyScheduler::ScheduleToDependencies(const SchedulingOutput &schedule) {
  std::vector<std::pair<TaskId, TaskId>> dependencies;  // to return
  MS_LOG(INFO) << "Started Preprocessing of Intervals";
  // Distinguish types and sort
  std::unordered_map<TaskType, std::set<Interval, SortByStart>> tasks_start;
  std::unordered_map<TaskType, std::set<Interval, SortByEnd>> tasks_end;
  for (const auto &task_time : schedule.task_times) {
    tasks_start[task_time.type].insert(task_time);
    tasks_end[task_time.type].insert(task_time);
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
// Second  International  Symposium  on Parallel Architectures, Algorithms, and Networks (I-SPAN?96),
// pages 207?213. IEEE, 1996.
bool SortByBottomTopLevelComposite(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->bottom_level() - task1->top_level() > task2->bottom_level() - task2->top_level() ||
         (task1->bottom_level() - task1->top_level() == task2->bottom_level() - task2->top_level() &&
          task1->weight() > task2->weight()) ||
         (task1->bottom_level() - task1->top_level() == task2->bottom_level() - task2->top_level() &&
          task1->weight() == task2->weight() && task1->id() < task2->id());
}

// Behrooz Shirazi, Mingfang Wang, and Girish Pathak.
// Analysis and evaluation of heuristic methods for static task scheduling.
// Journal of Parallel and Distributed Computing, 10(3):222?232, 1990.
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

// Sort by predecessor to comm DFS
bool SortByPredCommDepth(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) {
  return task1->pred_comm() < task2->pred_comm() ||
         (task1->pred_comm() == task2->pred_comm() && task1->depth() > task2->depth()) ||
         (task1->pred_comm() == task2->pred_comm() && task1->depth() == task2->depth() && task1->id() < task2->id());
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
  new_pem[kComp] = 1;
  return new_pem;
}

// Auxiliary subroutines and lower bounds
void FastGreedyScheduler::ComputeDepthAndTopLevel(std::vector<std::shared_ptr<Task>> &tasks) {
  MS_LOG(INFO) << "Top Level: Start Initialization";
  std::unordered_map<TaskId, size_t> unprocessed_parents;
  std::queue<std::shared_ptr<Task>> tasks_to_visit;
  // Initialization loop
  for (size_t j = 0; j < tasks.size(); ++j) {
    const auto &id = tasks[j]->id();
    unprocessed_parents[id] = tasks[j]->parents().size();
    if (unprocessed_parents[id] == 0) {
      tasks[j]->set_top_level(tasks[j]->parallel_weight());
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
        std::max(successor->top_level(), selected_task->top_level() + successor->parallel_weight()));
      unprocessed_parents[succ_id] -= 1;
      if (unprocessed_parents[succ_id] == 0) {
        tasks_to_visit.push(successor);
      }
    }
    tasks_to_visit.pop();
  }
  MS_LOG(INFO) << "Top Level: End Traversal Loop";
}

void FastGreedyScheduler::ComputeBottomLevelAndWeightedLength(std::vector<std::shared_ptr<Task>> &tasks) {
  MS_LOG(INFO) << "Bottom Level: Start Initialization";
  std::unordered_map<TaskId, size_t> unprocessed_children;
  std::unordered_map<TaskId, double> children_sum;
  std::unordered_map<TaskId, double> children_max;
  std::queue<std::shared_ptr<Task>> tasks_to_visit;
  // Initialization loop
  for (auto &task : tasks) {
    const auto &id = task->id();
    task->set_bottom_level(task->parallel_weight());
    task->set_weighted_length(task->parallel_weight());
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
        predecessor.lock()->bottom_level(), selected_task->bottom_level() + predecessor.lock()->parallel_weight()));
      children_sum[pred_id] += selected_task->weighted_length();
      children_max[pred_id] = std::max(children_max[pred_id], selected_task->weighted_length());
      unprocessed_children[pred_id] -= 1;
      if (unprocessed_children[pred_id] == 0) {
        if (children_max[pred_id] == 0) {
          MS_LOG(EXCEPTION) << "divisor children_max[pred_id] cannot be 0!";
        }
        predecessor.lock()->set_weighted_length(predecessor.lock()->parallel_weight() + children_max[pred_id] +
                                                children_sum[pred_id] / children_max[pred_id]);
        tasks_to_visit.push(predecessor.lock());
      }
    }
    tasks_to_visit.pop();
  }
  MS_LOG(INFO) << "Bottom Level: End Traversal Loop";
}

void FastGreedyScheduler::ComputePredComm(std::vector<std::shared_ptr<Task>> &tasks) {
  for (auto &task : tasks) {
    task->set_pred_comm(0);
    for (auto &predecessor : task->parents()) {
      if (predecessor.lock()->type() == kComm) {
        task->set_pred_comm(task->pred_comm() + 1);
      }
    }
  }
}

Time FastGreedyScheduler::LowerBoundBottomLevel(std::vector<std::shared_ptr<Task>> &tasks) {
  Time max_bottom_level = 0;
  for (const auto &task : tasks) {
    max_bottom_level = std::max(max_bottom_level, task->bottom_level());
  }
  return max_bottom_level;
}

Time FastGreedyScheduler::LowerBoundPEs(std::vector<std::shared_ptr<Task>> &tasks,
                                        std::unordered_map<TaskType, int32_t> &type_to_num_cores_map) {
  double lower_bound = 0;

  std::unordered_map<TaskType, Time> type_task_sum;
  for (const auto &task : tasks) {
    type_task_sum[task->type()] += task->weight();
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

constexpr TaskSortFunction THREAD_SORT[] = {SortByWeightMax,
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
                                            SortByPredCommDepth};

constexpr std::string_view THREAD_SORT_NAMES[] = {"SortByWeightMax",
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
                                                  "SortByPredCommDepth"};

enum class PEsSort { kSortByLoad = 0, kSortByValidStart, kNumPEsSort };

constexpr std::string_view PE_NAME_SORT[] = {"SortByLoad", "SortByValidStart"};

SchedulingOutput FastGreedyScheduler::Process(SchedulingInput &input, const std::string &graph_name) {
  std::vector<std::shared_ptr<Task>> *tasks = &(input.tasks);
  auto type_to_num_cores_map = GetTestPEs();
  SchedulingOutput output{{}, SIZE_MAX};
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

  // Loop over all sorting combinations
  std::unordered_map<std::shared_ptr<Task>, Time> best_start, best_end;  // to use in verify dependencies only
  std::string best_solution;
  MS_LOG(INFO) << "Start loop multiple scheduling functions";
  for (size_t task_sort = 0; task_sort < static_cast<size_t>(kNumTaskSort); ++task_sort) {
    for (size_t pes_sort = 0; pes_sort < static_cast<size_t>(PEsSort::kNumPEsSort); ++pes_sort) {
      MS_LOG(INFO) << THREAD_SORT_NAMES[task_sort] << " and " << PE_NAME_SORT[pes_sort];
      SchedulingOutput solution = ProcessCore(*tasks, type_to_num_cores_map, THREAD_SORT[task_sort],
                                              (pes_sort == static_cast<size_t>(PEsSort::kSortByLoad)));
      if (solution.makespan < output.makespan) {
        output = solution;
        best_solution = THREAD_SORT_NAMES[task_sort];
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

  // Print stats about best solution
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

  // Create and (optionally) verify dependencies (here only for testing)
  MS_LOG(INFO) << "Start Schedule to Dependencies";
  auto dependencies = ScheduleToDependencies(output);
  MS_LOG(INFO) << "End Schedule to Dependencies";
  for (const auto &task : *tasks) {
    task->set_start(best_start[task]);
    task->set_end(best_end[task]);
  }

  // Output log file with all info (scheduling and dependencies)
  MS_LOG(INFO) << "Start printing output log file";
  PrintLog(output, dependencies, graph_name);
  MS_LOG(INFO) << "End printing output log file";

  return output;
}

SchedulingOutput FastGreedyScheduler::ProcessCore(std::vector<std::shared_ptr<Task>> &tasks,
                                                  std::unordered_map<TaskType, int32_t> &type_to_num_cores_map,
                                                  const TaskSortFunction &sortPtr, bool pe_load_sort) {
  SchedulingOutput output{{}, 0};

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
      new_pe.type = type;
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
  while (!candidate_tasks.empty()) {
    // Select task and schedule it, save info for output
    const auto selected_task = *(candidate_tasks.begin());
    const auto &selected_id = selected_task->id();
    // Selected PE and start time
    std::pair<PeId, Time> PE_and_time;
    if (pe_load_sort) {
      PE_and_time = SelectPEandTime(*selected_task, can_start[selected_id], &PEs_load[selected_task->type()]);
    } else {
      PE_and_time =
        SelectPEandTimeAvailableStart(*selected_task, can_start[selected_id], &PEs_start[selected_task->type()]);
    }

    const auto &sigma = PE_and_time.second;

    // Maintenance of task interval
    selected_task->set_start(sigma);
    selected_task->set_end(sigma + selected_task->weight());
    // New interval for task in output
    Interval new_interval{selected_id, selected_task->type(), selected_task->start(), selected_task->end()};
    output.task_times.push_back(new_interval);
    // Update makespan
    output.makespan = std::max(output.makespan, selected_task->end());
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
  // Verification of scheduling solution (optional)
  if (VerifyScheduling(tasks)) {
    MS_LOG(INFO) << "Verification of Scheduling: SUCCESS";
  } else {
    MS_LOG(INFO) << "Verification of Scheduling: FAILURE";
  }

  return output;
}

SchedulingOutput FastGreedyScheduler::ProcessSingle(const SchedulingInput &input, const TaskSortFunction &sortPtr,
                                                    bool pe_load_sort, const std::string &graph_name) {
  auto tasks = input.tasks;
  auto type_to_num_cores_map = GetTestPEs();
  SchedulingOutput output{{}, 0};
  // Optional: verify input task graph is a DAG
  if (VerifyDAG(tasks)) {
    MS_LOG(INFO) << "Verification of DAG: SUCCESS";
  } else {
    MS_LOG(INFO) << "Verification of DAG: FAILURE";
  }
  // Preprocessing: values computation for sorting necessary
  ComputeBottomLevelAndWeightedLength(tasks);
  ComputeDepthAndTopLevel(tasks);
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
      new_pe.type = type;
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
  while (!candidate_tasks.empty()) {
    // Select task and schedule its blocks, save info for output
    const auto selected_task = *(candidate_tasks.begin());
    const auto &selected_id = selected_task->id();
    // Selected PE and start time
    std::pair<PeId, Time> PE_and_time;
    if (pe_load_sort) {
      PE_and_time = SelectPEandTime(*selected_task, can_start[selected_id], &PEs_load[selected_task->type()]);
    } else {
      PE_and_time =
        SelectPEandTimeAvailableStart(*selected_task, can_start[selected_id], &PEs_start[selected_task->type()]);
    }
    const auto &sigma = PE_and_time.second;

    // Maintenance of task interval
    selected_task->set_start(sigma);
    selected_task->set_end(sigma + selected_task->weight());
    // New interval for task in output
    Interval new_interval{selected_id, selected_task->type(), selected_task->start(), selected_task->end()};
    output.task_times.push_back(new_interval);
    // Update makespan
    output.makespan = std::max(output.makespan, selected_task->end());
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
  MS_LOG(INFO) << "Bottom level lower bound is " << LowerBoundBottomLevel(tasks);
  MS_LOG(INFO) << "Max type lower bound is " << LowerBoundPEs(tasks, type_to_num_cores_map);
  MS_LOG(INFO) << "Solution relative error is " << std::setprecision(5)
               << ((output.makespan /
                      (1.0 * std::max(LowerBoundBottomLevel(tasks), LowerBoundPEs(tasks, type_to_num_cores_map))) -
                    1) *
                   100)
               << "%";
  // Verification of scheduling solution (optional)
  if (VerifyScheduling(tasks)) {
    MS_LOG(INFO) << "Verification of Scheduling: SUCCESS";
  } else {
    MS_LOG(INFO) << "Verification of Scheduling: FAILURE";
  }
  // Scheduling to Dependencies (here only for testing)
  MS_LOG(INFO) << "Start Schedule to Dependencies";
  auto dependencies = ScheduleToDependencies(output);
  MS_LOG(INFO) << "End Schedule to Dependencies";
  if (VerifyDependencies(tasks, dependencies)) {
    MS_LOG(INFO) << "Verification of Dependencies: SUCCESS";
  } else {
    MS_LOG(INFO) << "Verification of Dependencies: FAILURE";
  }
  PrintLog(output, dependencies, graph_name);

  return output;
}

bool FastGreedyScheduler::VerifyScheduling(std::vector<std::shared_ptr<Task>> &tasks) {
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

bool FastGreedyScheduler::VerifyDependencies(std::vector<std::shared_ptr<Task>> &tasks,
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

bool FastGreedyScheduler::VerifyDAG(std::vector<std::shared_ptr<Task>> &tasks) {
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

void FastGreedyScheduler::PrintLog(const SchedulingOutput &output,
                                   const std::vector<std::pair<TaskId, TaskId>> &dependencies,
                                   const std::string &graph_name) {
  std::ofstream out_file("comp_comm_scheduling_out_" + graph_name + ".log", std::ios::out | std::ios::trunc);
  if (!out_file.is_open()) {
    MS_LOG(ERROR) << "Could not open comp_comm_scheduling_out.log";
    return;
  }

  // Print info for tasks
  const auto &tasks = output.task_times;
  for (const auto &task : tasks) {
    out_file << "THREAD id=" << std::to_string(task.id) << ", type=" << std::to_string(task.type)
             << ", start=" << std::to_string(task.start) << ", end=" << std::to_string(task.end) << "\n";
  }
  // Print dependencies
  for (const auto &dependency : dependencies) {
    const auto &source = dependency.first;
    const auto &dst = dependency.second;
    out_file << "DEPENDENCY " << std::to_string(source) << " " << std::to_string(dst) << "\n";
  }
  out_file.close();
}

void InsertTaskGraph(const std::vector<CNodePtr> &cnode_vec,
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

SchedulingInput ExtractSchedulingInput(const FuncGraphManagerPtr &manager, const std::vector<CNodePtr> &cnode_vec,
                                       std::unordered_map<CNodePtr, TaskPtr> *cnode_to_task_map_ptr) {
  SchedulingInput scheduling_input;  // to fill in and return

  // Create a task per node
  for (size_t i = 0; i < cnode_vec.size(); ++i) {
    std::shared_ptr<Task> task1 =
      std::make_shared<Task>(i, common::AnfAlgo::IsCommunicationOp(cnode_vec[i]) ? kComm : kComp);
    MS_LOG(INFO) << "Start Assign Weight";
    const auto &cnode = cnode_vec[i];
    size_t output_num = AnfUtils::GetOutputTensorNum(cnode);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
    Time weight = 0;

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
      // size_t type_size = convert_type_to_num(type);
      weight += std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * type_size;
    }

    if (output_num > 1) {
      for (size_t j = 0; j < output_num; j++) {
        ShapeVector shape = common::AnfAlgo::GetOutputInferShape(cnode, j);
        if (shape.size() <= 0) continue;

        const TypeId type = common::AnfAlgo::GetOutputInferDataType(cnode, j);
        if (type == kObjectTypeUMonad || type == kObjectTypeMonad || type == kObjectTypeFunction) continue;

        size_t type_size = GetDataTypeSize(type);
        // size_t type_size = convert_type_to_num(type);
        weight += std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * type_size;
      }
    }

    if (weight < 0) MS_LOG(EXCEPTION) << "Weight < 0, replace by SIZE_MAX";

    task1->AssignWeight(weight);
    MS_LOG(INFO) << "End Assign Weight";

    (*cnode_to_task_map_ptr)[cnode_vec[i]] = task1;
    scheduling_input.tasks.push_back(task1);
    MS_LOG(INFO) << "Task " << task1->id() << " with name " << cnode->UniqueName() << " and CNodePtr " << cnode_vec[i]
                 << " with weight " << weight;
  }

  // Insert task graph edges
  InsertTaskGraph(cnode_vec, cnode_to_task_map_ptr);

  return scheduling_input;
}

void AddRealDependencies(const FuncGraphManagerPtr &manager, const std::vector<CNodePtr> &cnode_vec,
                         const std::vector<std::pair<TaskId, TaskId>> &dependencies,
                         std::unordered_map<CNodePtr, TaskPtr> *cnode_to_task) {
  size_t count = 0, redundant_count = 0;
  for (const auto &dependency : dependencies) {
    MS_LOG(INFO) << "Checking dependency " << dependency.first << " " << dependency.second;
    const auto &source = cnode_vec[dependency.first];
    const auto &dest = cnode_vec[dependency.second];

    // Ignore dependencies already there
    if ((*cnode_to_task)[source]->HasChild((*cnode_to_task)[dest])) {
      MS_LOG(INFO) << "Dependency " << dependency.first << " " << dependency.second
                   << " is redundant (already parent and child)";
      redundant_count++;
      continue;
    }

    // Add dependency only between Comp (check also between Comm later)
    bool comp_comp = true;
    if (common::AnfAlgo::IsCommunicationOp(source) && common::AnfAlgo::IsCommunicationOp(dest)) {
      comp_comp = false;
      MS_LOG(INFO) << "Ignore Comm to Comm dependency " << dependency.first << " " << dependency.second;
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
      if (comp_comp && !common::AnfAlgo::IsCommunicationOp(dest->input(j))) {
        MS_LOG(INFO) << "Dest " << dest << " Task " << dependency.second << " Input " << j
                     << " is not CommunicationOp: Ignore";
        continue;
      }
      if (!comp_comp && common::AnfAlgo::IsCommunicationOp(dest->input(j))) {
        MS_LOG(INFO) << "Dest " << dest << " Task " << dependency.second << " Input " << j
                     << " is CommunicationOp: Ignore";
        continue;
      }

      // Add real dependency logic here
      const auto &input_node = dest->input(j)->cast<CNodePtr>();
      std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), input_node, source};
      auto depend_node = dest->func_graph()->NewCNode(depend_inputs);
      depend_node->set_abstract(input_node->abstract()->Clone());
      depend_node->AddAttr("comp_comm_scheduling_depend", MakeValue(true));
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
  return;
}

void CompCommScheduling(const FuncGraphPtr &graph) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    return;
  }
  if (common::GetEnv("MS_ENABLE_FRONTEND_SCHEDULING_OPTIMIZATION") != "1") {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_LOG(INFO) << "Main graph pointer: " << graph;
  MS_EXCEPTION_IF_NULL(manager);

  FuncGraphSet graphs = manager->func_graphs();
  for (const auto &subgraph : graphs) {
    MS_LOG(INFO) << "Start Scheduling Subgraph " << subgraph;
    std::stringstream graph_ss;
    graph_ss << subgraph;
    std::string graph_name = graph_ss.str();
    std::list<CNodePtr> cnode_list = subgraph->GetOrderedCnodes();
    std::vector<CNodePtr> cnode_vec(cnode_list.cbegin(), cnode_list.cend());

    MS_LOG(INFO) << "Start ExtractSchedulingInput";
    std::unordered_map<CNodePtr, TaskPtr> cnode_to_task;
    SchedulingInput scheduling_input = ExtractSchedulingInput(manager, cnode_vec, &cnode_to_task);
    MS_LOG(INFO) << "End ExtractSchedulingInput";

    auto scheduling_output = FastGreedyScheduler::Process(scheduling_input, graph_name);
    auto dependencies = FastGreedyScheduler::ScheduleToDependencies(scheduling_output);

    MS_LOG(INFO) << "Start AddRealDependencies";
    AddRealDependencies(manager, cnode_vec, dependencies, &cnode_to_task);
    MS_LOG(INFO) << "End AddRealDependencies";
  }
}
}  // namespace opt
}  // namespace mindspore
