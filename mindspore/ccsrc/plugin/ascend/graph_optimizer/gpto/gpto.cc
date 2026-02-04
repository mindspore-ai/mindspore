/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/graph_optimizer/gpto/gpto.h"

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
#include <stack>
#include <tuple>

#include "primitive/math_ops.h"
#include "primitive/conv_pool_op_name.h"
#include "primitive/ascend_op_name.h"
#include "primitive/other_op_name.h"
#include "utils/anf_utils.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "include/utils/common.h"
#include "include/runtime/utils/runtime_conf/runtime_env.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_memory_adapter.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "include/utils/thread_pool.h"
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
#include "utils/numa_interface.h"
#endif
#include "include/utils/parallel_context.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace gpto {
bool Overlap(const Time &start1, const Time &end1, const Time &start2, const Time &end2) {
  return (start1 >= start2 && start1 < end2) ||
         (start2 >= start1 && start2 < end1);  // if equal start and end for two intervals, then no overlap
}

size_t ScheduleToEvents(const SchedulingOutput &schedule) {
  size_t count = 0;

  std::set<Interval, SortByStartMin> tasks_start;
  std::set<Interval, SortByEndMin> tasks_end;
  for (const auto &task_time : schedule.task_times) {
    tasks_start.insert(task_time);
    tasks_end.insert(task_time);
  }

  for (auto it = tasks_start.begin(); it != tasks_start.end(); ++it) {
    tasks_end.erase(*it);
    // Dismiss overlapping tasks: save min end value of non-overlapping task to the right
    std::unordered_map<GptoTaskPtr, bool> dismissed;
    auto it1 = std::next(it);
    for (; Overlap(it->start, it->end, it1->start, it1->end) && it1 != tasks_start.end(); ++it1) {
      dismissed[it1->task] = true;
    }
    Time min_end_value = 0;
    auto it2 = std::find_if(tasks_end.begin(), tasks_end.end(), [&](const auto &i) { return !dismissed[i.task]; });
    if (it2 != tasks_end.end()) {
      min_end_value = it2->end;
    }
    // Add events to immediate right neighborhood
    for (; it1 != tasks_start.end() && it1->start < min_end_value; ++it1) {
      if (it->task->gpto_type() != it1->task->gpto_type()) {
        it1->task->recv_events().insert(it->task);
        MS_LOG(DEBUG) << "GPTO scheduling event from " << it->task->id() << " to " << it1->task->id();
        count++;
      }
    }
  }
  MS_LOG(INFO) << "Generated " << count << " events to impose GPTO schedule";
  tasks_start.clear();
  tasks_end.clear();
  return count;
}

size_t LiftEvents(const SchedulingOutput &schedule, const std::unordered_set<uint32_t> &stream_ids) {
  size_t count = 0;
  std::set<Interval, SortByStartMin> tasks_start;
  std::unordered_map<GptoTaskPtr, GptoTaskPtr> next_in_stream;
  std::unordered_map<GptoTaskPtr, GptoTaskPtr> prev_in_stream;
  std::unordered_map<GptoTaskPtr, Interval> interval;

  for (const auto &task_time : schedule.task_times) {
    tasks_start.insert(task_time);
    next_in_stream[task_time.task] = nullptr;
    prev_in_stream[task_time.task] = nullptr;
    interval[task_time.task] = task_time;
  }

  size_t max_original_stream = *(std::max_element(stream_ids.begin(), stream_ids.end()));
  MS_LOG(INFO) << "Max original stream id: " << max_original_stream;
  std::unordered_map<GptoTaskType, GptoTaskPtr> last_in_stream;
  std::unordered_map<GptoTaskType, size_t> cantor_to_real;
  size_t num_new_streams = 0;

  for (uint32_t s1 : stream_ids) {
    last_in_stream[s1] = nullptr;
    for (uint32_t s2 : stream_ids) {
      if (s1 != s2) {
        const size_t cantor_pairing = (s1 + s2) * (s1 + s2 + 1) / 2 + s2;
        size_t cantor_stream = max_original_stream + cantor_pairing;
        size_t real_stream;
        device::ascend::AscendStreamMng::GetInstance().CreateStream(&real_stream);
        cantor_to_real[cantor_stream] = real_stream;
        MS_LOG(INFO) << "Created new stream id: " << real_stream << " for cantor pairing: " << cantor_pairing;
        last_in_stream[real_stream] = nullptr;
        num_new_streams++;
      }
    }
  }
  MS_LOG(INFO) << "Total new streams created for lifting: " << num_new_streams;

  for (auto it = tasks_start.begin(); it != tasks_start.end(); ++it) {
    const auto &stream = it->task->gpto_type();
    if (last_in_stream[stream] != nullptr) {
      next_in_stream[last_in_stream[stream]] = it->task;
      prev_in_stream[it->task] = last_in_stream[stream];
    }
    last_in_stream[stream] = it->task;
  }

  for (auto it = tasks_start.begin(); it != tasks_start.end(); ++it) {
    for (auto send_event_task : it->task->recv_events()) {
      for (auto task = next_in_stream[send_event_task.lock()]; task != nullptr; task = next_in_stream[task]) {
        if (!(it->start >= interval[task].start && it->start < interval[task].end)) {
          break;
        }
        // Assign new stream id (gpto type) based on cantor pairing function
        const auto &send_stream_id = send_event_task.lock()->gpto_original_type();
        const auto &recv_stream_id = it->task->gpto_original_type();
        const size_t cantor_pairing =
          (send_stream_id + recv_stream_id) * (send_stream_id + recv_stream_id + 1) / 2 + recv_stream_id;
        const size_t real_stream = cantor_to_real[max_original_stream + cantor_pairing];
        MS_LOG(DEBUG) << "GPTO task id " << task->id() << " lifted to new stream id: " << real_stream
                      << " from original stream id: " << task->gpto_original_type();
        task->set_gpto_type(real_stream);
        AnfAlgo::SetStreamId(real_stream, task->cnode().get());
        common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue(real_stream), task->cnode());

        // Add new events to and from task with respect to its previous stream
        if (prev_in_stream[task] != nullptr) {
          task->recv_events().insert(prev_in_stream[task]);
          count++;
        }
        if (next_in_stream[task] != nullptr) {
          next_in_stream[task]->recv_events().insert(task);
          count++;
        }

        // Update stream ordering for original stream
        if (prev_in_stream[task] != nullptr) {
          next_in_stream[prev_in_stream[task]] = next_in_stream[task];
        }
        if (next_in_stream[task] != nullptr) {
          prev_in_stream[next_in_stream[task]] = prev_in_stream[task];
        }

        // Update stream ordering for new stream
        if (last_in_stream[real_stream] != nullptr) {
          next_in_stream[last_in_stream[real_stream]] = task;
        }
        prev_in_stream[task] = last_in_stream[real_stream];
        next_in_stream[task] = nullptr;
        last_in_stream[real_stream] = task;
      }
    }
  }

  // Clear data structure and return
  tasks_start.clear();
  next_in_stream.clear();
  prev_in_stream.clear();
  interval.clear();

  MS_LOG(INFO) << "Lifted " << count << " events by stream lifting";
  return count;
}

size_t MemorySafetyEvents(const SchedulingOutput &schedule) {
  size_t count = 0;

  std::set<Interval, SortByEndMax> tasks_end;
  for (const auto &task_time : schedule.task_times) {
    tasks_end.insert(task_time);
  }

  for (auto it = tasks_end.begin(); it != tasks_end.end(); ++it) {
    if (it->task->gpto_type() == 0) {
      continue;
    }
    for (auto it1 = std::next(it); it1 != tasks_end.end(); ++it1) {
      if (Overlap(it->start, it->end, it1->start, it1->end)) {
        continue;
      }
      if (it1->task->gpto_type() == 0) {
        if (it->task->recv_events().find(it1->task) == it->task->recv_events().end()) {
          it->task->recv_events().insert(it1->task);
          count++;
          MS_LOG(DEBUG) << "Memory safety event from " << it1->task->id() << " to " << it->task->id();
        }
        break;
      }
    }
  }
  MS_LOG(INFO) << "Generated " << count << " events to ensure dynamic memory safety";
  tasks_end.clear();
  return count;
}

std::map<GptoTaskType, size_t> GetPEs(const std::unordered_set<uint32_t> &stream_ids) {
  std::map<GptoTaskType, size_t> new_pem;
  for (const auto &stream_id : stream_ids) {
    new_pem[stream_id] = 1;
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
  while (!tasks_to_visit.empty()) {
    const auto &selected_task = tasks_to_visit.front();
    // Update candidate tasks
    for (auto &successor : selected_task->children()) {
      const auto &succ_id = successor->id();
      successor->set_depth(std::max(successor->depth(), selected_task->depth() + 1));
      Time extra_top = successor->cost();
      if (!selected_task->child_is_data_dep(successor)) {
        if (successor->cost() > selected_task->cost() - static_cast<Time>(1)) {
          extra_top = successor->cost() - (selected_task->cost() - static_cast<Time>(1));
        } else {
          extra_top = static_cast<Time>(0);
        }
      }
      successor->set_top_level(std::max(successor->top_level(), selected_task->top_level() + extra_top));

      unprocessed_parents[succ_id] -= 1;
      if (unprocessed_parents[succ_id] == 0) {
        tasks_to_visit.push(successor);
      }
    }
    tasks_to_visit.pop();
  }
  unprocessed_parents.clear();
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
  while (!tasks_to_visit.empty()) {
    const auto &selected_task = tasks_to_visit.front();
    // Update candidate tasks
    for (auto &predecessor : selected_task->parents()) {
      const auto &pred_id = predecessor.lock()->id();
      Time start_bottom =
        (predecessor.lock()->child_is_data_dep(selected_task)) ? predecessor.lock()->cost() : static_cast<Time>(1);
      predecessor.lock()->set_bottom_level(
        std::max(predecessor.lock()->bottom_level(), selected_task->bottom_level() + start_bottom));
      children_sum[pred_id] += selected_task->weighted_length();
      children_max[pred_id] = std::max(children_max[pred_id], selected_task->weighted_length());
      unprocessed_children[pred_id] -= 1;
      if (unprocessed_children[pred_id] == 0) {
        if (children_max[pred_id] == 0) {
          MS_LOG(EXCEPTION) << "Divisor children_max[pred_id] cannot be 0!";
        }
        predecessor.lock()->set_weighted_length(start_bottom + children_max[pred_id] +
                                                children_sum[pred_id] / children_max[pred_id]);
        tasks_to_visit.push(predecessor.lock());
      }
    }
    tasks_to_visit.pop();
  }
  unprocessed_children.clear();
  children_sum.clear();
  children_max.clear();
}

void ComputePostOrder(const std::vector<GptoTaskPtr> &tasks) {
  std::stack<GptoTaskPtr> tasks_to_visit;
  std::unordered_map<GptoTaskId, bool> visited;

  for (auto &task : tasks) {
    visited[task->id()] = false;
    if (task->parents().empty()) {
      tasks_to_visit.push(task);
    }
  }

  size_t step = 0;
  while (!tasks_to_visit.empty()) {
    auto &selected_task = tasks_to_visit.top();
    bool zero_unvisited_children = true;
    for (auto &child : selected_task->children()) {
      if (!visited[child->id()]) {
        tasks_to_visit.push(child);
        zero_unvisited_children = false;
      }
    }
    if (zero_unvisited_children) {
      tasks_to_visit.pop();
      if (!visited[selected_task->id()]) {
        selected_task->set_post_order_time(step++);
        visited[selected_task->id()] = true;
      }
    }
  }
  visited.clear();
}

void ComputePredComm(const std::vector<GptoTaskPtr> &tasks) {
  for (auto &task : tasks) {
    task->set_pred_comm(0);
    for (auto &predecessor : task->parents()) {
      if (predecessor.lock()->gpto_type() >= kCommTrue) {
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

void ComputeSuccDiffType(const std::vector<GptoTaskPtr> &tasks) {
  for (auto &task : tasks) {
    task->set_succ_diff_type(0);
    for (auto &successor : task->children()) {
      if (successor->gpto_type() != task->gpto_type()) {
        task->set_succ_diff_type(task->succ_diff_type() + 1);
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
                  << ", workspace is " << task->workspace_memory();
  }
}

Time LowerBoundBottomLevel(const std::vector<GptoTaskPtr> &tasks) {
  return std::accumulate(tasks.begin(), tasks.end(), static_cast<Time>(0), [](Time max_bottom_level, const auto &task) {
    return std::max(max_bottom_level, task->bottom_level());
  });
}

Time TotalTime(const std::vector<GptoTaskPtr> &tasks) {
  return std::accumulate(tasks.begin(), tasks.end(), static_cast<Time>(0),
                         [](Time acc, const auto &task) { return acc + task->cost(); });
}

Time LowerBoundPEs(const std::vector<GptoTaskPtr> &tasks, const std::map<GptoTaskType, size_t> &type_to_num_cores_map) {
  double lower_bound = 0;

  std::unordered_map<GptoTaskType, Time> type_task_sum;
  for (const auto &task : tasks) {
    type_task_sum[task->gpto_type()] += task->cost();
  }
  for (const auto &type_to_num : type_to_num_cores_map) {
    const auto &type = type_to_num.first;
    const auto &num_cores = type_to_num.second;
    if (num_cores == 0) {
      MS_LOG(EXCEPTION) << "Divisor num_cores cannot be 0!";
    }
    lower_bound = std::max(lower_bound, type_task_sum[type] / (1.0 * num_cores));
  }
  type_task_sum.clear();
  return std::ceil(lower_bound);
}

// Task sorting functions
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
          (static_cast<int64_t>(task1->bottom_level()) - static_cast<int64_t>(task1->top_level()) >
             static_cast<int64_t>(task2->bottom_level()) - static_cast<int64_t>(task2->top_level()) ||
           (static_cast<int64_t>(task1->bottom_level()) - static_cast<int64_t>(task1->top_level()) ==
              static_cast<int64_t>(task2->bottom_level()) - static_cast<int64_t>(task2->top_level()) &&
            task1->cost() > task2->cost()) ||
           (static_cast<int64_t>(task1->bottom_level()) - static_cast<int64_t>(task1->top_level()) ==
              static_cast<int64_t>(task2->bottom_level()) - static_cast<int64_t>(task2->top_level()) &&
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

bool SortByDepthMin(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->depth() < task2->depth() || (task1->depth() == task2->depth() && task1->cost() > task2->cost()) ||
           (task1->depth() == task2->depth() && task1->cost() == task2->cost() && task1->id() < task2->id())));
}

// Sort by type (prioritize comm)
bool SortByTypePriority(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->type_priority() > task2->type_priority() ||
           (task1->type_priority() == task2->type_priority() && task1->bottom_level() > task2->bottom_level()) ||
           (task1->type_priority() == task2->type_priority() && task1->bottom_level() == task2->bottom_level() &&
            task1->id() < task2->id())));
}

bool SortByPredComm(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->pred_comm() < task2->pred_comm() ||
           (task1->pred_comm() == task2->pred_comm() && task1->bottom_level() > task2->bottom_level()) ||
           (task1->pred_comm() == task2->pred_comm() && task1->bottom_level() == task2->bottom_level() &&
            task1->id() < task2->id())));
}

bool SortByPredCommDepth(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->pred_comm() < task2->pred_comm() ||
           (task1->pred_comm() == task2->pred_comm() && task1->depth() > task2->depth()) ||
           (task1->pred_comm() == task2->pred_comm() && task1->depth() == task2->depth() &&
            task1->id() < task2->id())));
}

bool SortByPredCube(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->pred_cube() < task2->pred_cube() ||
           (task1->pred_cube() == task2->pred_cube() && task1->bottom_level() > task2->bottom_level()) ||
           (task1->pred_cube() == task2->pred_cube() && task1->bottom_level() == task2->bottom_level() &&
            task1->id() < task2->id())));
}

bool SortByGreedyHeight(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->initial_mem_impact() - task1->minus_mem_impact_greedy_height() <
             task2->initial_mem_impact() - task2->minus_mem_impact_greedy_height() ||
           (task1->initial_mem_impact() - task1->minus_mem_impact_greedy_height() ==
              task2->initial_mem_impact() - task2->minus_mem_impact_greedy_height() &&
            task1->id() < task2->id())));
}

bool SortByReversePostOrder(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->post_order_time() > task2->post_order_time() ||
           (task1->post_order_time() == task2->post_order_time() && task1->id() < task2->id())));
}

bool SortBySValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->s_value() > task2->s_value() || (task1->s_value() == task2->s_value() && task1->id() < task2->id())));
}

bool SortByAValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->a_value() > task2->a_value() || (task1->a_value() == task2->a_value() && task1->id() < task2->id())));
}

bool SortByMValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->m_value() > task2->m_value() || (task1->m_value() == task2->m_value() && task1->id() < task2->id())));
}

bool SortByWeightedSValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->sw_value() > task2->sw_value() ||
           (task1->sw_value() == task2->sw_value() && task1->id() < task2->id())));
}

bool SortByWeightedAValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->aw_value() > task2->aw_value() ||
           (task1->aw_value() == task2->aw_value() && task1->id() < task2->id())));
}

bool SortByWeightedMValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->mw_value() > task2->mw_value() ||
           (task1->mw_value() == task2->mw_value() && task1->id() < task2->id())));
}

bool SortByCostSValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->sc_value() > task2->sc_value() ||
           (task1->sc_value() == task2->sc_value() && task1->id() < task2->id())));
}

bool SortByCostAValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->ac_value() > task2->ac_value() ||
           (task1->ac_value() == task2->ac_value() && task1->id() < task2->id())));
}

bool SortByCostMValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->mc_value() > task2->mc_value() ||
           (task1->mc_value() == task2->mc_value() && task1->id() < task2->id())));
}

GPTO_OPTIONS_MODE GetGptoOptionsMode() {
  const auto &backend_jit_config = backend::BackendJitConfig::ParseBackendJitConfig();
  std::string gpto_mode = backend_jit_config.gpto_options.at("mode");
  if (gpto_mode == "basic") {
    return kBasic;
  } else if (gpto_mode == "advance") {
    return kAdvance;
  } else {
    MS_LOG(EXCEPTION) << "GPTO mode is invalid. Expected in ['advance', 'basic', 'off'], given: " << gpto_mode;
  }
}

bool IsGptoDebug() {
  const auto &backend_jit_config = backend::BackendJitConfig::ParseBackendJitConfig();
  auto it = backend_jit_config.gpto_options.find("verification");
  if (it != backend_jit_config.gpto_options.end()) {
    if (it->second == "off") {
      MS_LOG(INFO) << "GPTO verification is disabled";
      return false;
    } else if (it->second == "on") {
      MS_LOG(INFO) << "GPTO verification is enabled";
      return true;
    }
  }
  MS_LOG(WARNING) << "Invalid or unset value for 'verification' for GPTO, Disabling GPTO verification by default";
  return false;
}

bool SkipAlgorithm(const size_t &task_sort, const size_t &pes_sort, const bool &multi_pe_for_one_type) {
  std::string gpto_algo;
  const auto &backend_jit_config = backend::BackendJitConfig::ParseBackendJitConfig();
  auto it = backend_jit_config.gpto_options.find("algo");
  if (it != backend_jit_config.gpto_options.end()) {
    gpto_algo = it->second;
  }
  if (!gpto_algo.empty()) {
    auto exists = std::find(std::begin(TASK_SORT_NAMES), std::end(TASK_SORT_NAMES), gpto_algo);
    if (exists == std::end(TASK_SORT_NAMES)) {
      MS_LOG(WARNING) << "Unknown GPTO algo: '" << gpto_algo << "'. Ignoring";
    } else if (gpto_algo != TASK_SORT_NAMES[task_sort]) {
      return true;
    }
  }

  if (gpto_mode == kComp &&
      (TASK_SORT_NAMES[task_sort] == "SortByPredComm" || TASK_SORT_NAMES[task_sort] == "SortByPredCommDepth")) {
    return true;
  }
  if (gpto_mode != kCompCubeComm && TASK_SORT_NAMES[task_sort] == "SortByPredCube") {
    return true;
  }
  if (!multi_pe_for_one_type &&
      pes_sort == static_cast<size_t>(PEsSort::kSortByValidStart)) {  // same solution if 1 PE per type
    return true;
  }
  if ((task_sort == static_cast<size_t>(kSortByGreedyHeight)) ||
      (task_sort >= static_cast<size_t>(kSortBySValue) && task_sort <= static_cast<size_t>(kSortByCostMValue))) {
    if (pes_sort == static_cast<size_t>(
                      PEsSort::kSortByValidStart)) {  // greedy height and SAM not thread-safe for multiple PE sort
      return true;
    }
  }
  return false;
}

namespace {  // for multi-threading
std::vector<int> InnerGetCoreList() {
  std::vector<int> core_list;
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__) && !defined(ENABLE_ANDROID)
  uint32_t rank_id = CommManager::GetInstance().GetRank();
  MS_LOG(INFO) << "Gpto bind numa node for rank " << rank_id;
  auto numa_handle = GetNumaAdapterHandle();
  if (numa_handle) {
    (void)LoadNumaCpuInfo(numa_handle.get(), rank_id + 1, &core_list);
  } else {
    MS_LOG(INFO) << "Gpto load numa library failed";
  }
  std::ostringstream oss;
  for (auto core_id : core_list) {
    oss << ", " << core_id;
  }
  MS_LOG(INFO) << "Gpto bind core list" << oss.str();
#endif
  return core_list;
}
std::vector<int> GetCoreList() {
  static std::vector<int> core_list = InnerGetCoreList();
  return core_list;
}
}  // namespace

void MemAwareSchedulerLockFreeThreadSafe(
  std::unordered_map<size_t, std::unordered_map<size_t, SchedulingOutput>> *output,
  const std::vector<GptoTaskPtr> &tasks, const bool &multi_pe_for_one_type) {
  for (size_t task_sort = 0; task_sort < static_cast<size_t>(kNumTaskSort); ++task_sort) {
    for (size_t pes_sort = 0; pes_sort < static_cast<size_t>(PEsSort::kNumPEsSort); ++pes_sort) {
      if ((task_sort > 0 || pes_sort > 0) && SkipAlgorithm(task_sort, pes_sort, multi_pe_for_one_type)) {
        continue;
      }
      (*output)[task_sort][pes_sort].task_times.reserve(tasks.size());
      (*output)[task_sort][pes_sort].makespan = static_cast<Time>(SIZE_MAX);
      (*output)[task_sort][pes_sort].memory_peak = static_cast<Memory>(SIZE_MAX);
    }
  }
  for (const auto &task : tasks) {
    task->init_start(static_cast<size_t>(kNumTaskSort), static_cast<size_t>(PEsSort::kNumPEsSort));
    task->init_end(static_cast<size_t>(kNumTaskSort), static_cast<size_t>(PEsSort::kNumPEsSort));
    task->init_minus_mem_impact(static_cast<size_t>(kNumTaskSort), static_cast<size_t>(PEsSort::kNumPEsSort));
    task->init_position(static_cast<size_t>(kNumTaskSort), static_cast<size_t>(PEsSort::kNumPEsSort));
    for (const auto &tensor : task->out_tensors()) {
      tensor->init_lifetime_end(static_cast<size_t>(kNumTaskSort), static_cast<size_t>(PEsSort::kNumPEsSort));
      tensor->init_last_consumer(static_cast<size_t>(kNumTaskSort), static_cast<size_t>(PEsSort::kNumPEsSort));
    }
    for (const auto &tensor : task->workspace_tensors()) {
      tensor->init_lifetime_end(static_cast<size_t>(kNumTaskSort), static_cast<size_t>(PEsSort::kNumPEsSort));
      tensor->init_last_consumer(static_cast<size_t>(kNumTaskSort), static_cast<size_t>(PEsSort::kNumPEsSort));
    }
  }
}

void MemAwareSchedulerReleaseMem(std::unordered_map<size_t, std::unordered_map<size_t, SchedulingOutput>> *output,
                                 const std::vector<GptoTaskPtr> &tasks, const bool &multi_pe_for_one_type,
                                 const std::string &best_sol, const size_t &best_task_sort,
                                 const size_t &best_pes_sort) {
  for (const auto &task : tasks) {
    task->start().clear();
    task->end().clear();
    task->minus_mem_impact().clear();
    task->position().clear();
    for (const auto &tensor : task->out_tensors()) {
      tensor->lifetime_end().clear();
      tensor->last_consumer().clear();
    }
    for (const auto &tensor : task->workspace_tensors()) {
      tensor->lifetime_end().clear();
      tensor->last_consumer().clear();
    }
  }

  for (size_t task_sort = 0; task_sort < static_cast<size_t>(kNumTaskSort); ++task_sort) {
    for (size_t pes_sort = 0; pes_sort < static_cast<size_t>(PEsSort::kNumPEsSort); ++pes_sort) {
      if ((best_sol == "" && task_sort == 0 && pes_sort == 0) ||
          (best_sol != "" && task_sort == best_task_sort && pes_sort == best_pes_sort) ||
          SkipAlgorithm(task_sort, pes_sort, multi_pe_for_one_type)) {
        continue;
      }
      (*output)[task_sort][pes_sort].task_times.clear();
      (*output)[task_sort].erase(pes_sort);
    }
  }
}

SchedulingOutput MemAwareScheduler(const SchedulingInput &input, const std::unordered_set<uint32_t> &stream_ids,
                                   size_t *lower_makespan) {
  const std::vector<GptoTaskPtr> &tasks = input.tasks;
  const auto &type_to_num_cores_map = GetPEs(stream_ids);
  std::unordered_map<size_t, std::unordered_map<size_t, SchedulingOutput>> output;

  // Optional: verify input task graph is a DAG
  if (IsGptoDebug()) {
    if (VerifyDAG(tasks)) {
      MS_LOG(INFO) << "Verification of DAG: SUCCESS";
    } else {
      MS_LOG(ERROR) << "Verification of DAG: FAILURE";
      output[0][0].makespan = static_cast<Time>(SIZE_MAX);
      return output[0][0];
    }
  }

  // Multi-PE check for algorithm skipping
  const bool multi_pe_for_one_type = std::any_of(type_to_num_cores_map.begin(), type_to_num_cores_map.end(),
                                                 [](const auto &type_num) { return type_num.second > 1; });

  // Preprocessing: auxiliary values computation necessary for task sorting algorithms
  auto start_time = std::chrono::system_clock::now();
  MemAwareSchedulerPre(tasks);
  MS_LOG(INFO)
    << "MemAwareScheduler preprocessing "
    << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count()
    << "ms";

  // Initialize indices for safe lock-free multi-threading
  start_time = std::chrono::system_clock::now();
  MemAwareSchedulerLockFreeThreadSafe(&output, tasks, multi_pe_for_one_type);
  MS_LOG(INFO)
    << "MemAwareScheduler initialize for thread safety "
    << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count()
    << "ms";

  // Multiple scheduling algorithms in parallel
  start_time = std::chrono::system_clock::now();
  try {
    std::vector<common::Task> common_tasks;
    for (size_t task_sort = 0; task_sort < static_cast<size_t>(kNumTaskSort); ++task_sort) {
      for (size_t pes_sort = 0; pes_sort < static_cast<size_t>(PEsSort::kNumPEsSort); ++pes_sort) {
        if (SkipAlgorithm(task_sort, pes_sort, multi_pe_for_one_type)) {
          continue;
        }
        auto common_task = [&output, &tasks, &type_to_num_cores_map, task_sort, pes_sort]() {
          output[task_sort][pes_sort] = MemAwareSchedulerCore(tasks, type_to_num_cores_map, task_sort, pes_sort);
          return common::SUCCESS;
        };
        (void)common_tasks.emplace_back(common_task);
      }
    }
    auto core_list = GetCoreList();
    (void)common::ThreadPool::GetInstance().SyncRun(common_tasks, core_list);
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "mindspore::gpto::MemAwareScheduler FAILED: " << e.what();
  }
  MS_LOG(INFO)
    << "MemAwareScheduler multiple algorithms "
    << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count()
    << "ms";

  // Save best solution
  std::string best_solution = "";
  std::pair<std::string, Memory> best_memory_solution =
    std::make_pair(best_solution, static_cast<Memory>(MEMORY_LIMIT));
  std::tuple<size_t, size_t, size_t, size_t> best_sols =
    GetBestSolutions(&output, &best_solution, &best_memory_solution);
  MemAwareSchedulerReleaseMem(&output, tasks, multi_pe_for_one_type, best_solution, std::get<0>(best_sols),
                              std::get<1>(best_sols));

  if (best_solution == "") {
    output[0][0].makespan = static_cast<Time>(SIZE_MAX);
    return output[0][0];
  }

  PrintBestSolutionStats(output[std::get<0>(best_sols)][std::get<1>(best_sols)], tasks, type_to_num_cores_map,
                         best_solution, best_memory_solution, lower_makespan);
  return output[std::get<0>(best_sols)][std::get<1>(best_sols)];
}

std::tuple<size_t, size_t, size_t, size_t> GetBestSolutions(
  std::unordered_map<size_t, std::unordered_map<size_t, SchedulingOutput>> *output, std::string *best_solution,
  std::pair<std::string, Memory> *best_memory_solution) {
  // Data for min makespan
  Time min_makespan = static_cast<Time>(SIZE_MAX);
  Memory memory_of_min_makespan = static_cast<Memory>(SIZE_MAX);
  size_t taskSort_min_makespan = SIZE_MAX;
  size_t pesSort_min_makespan = SIZE_MAX;

  // Data for min memory
  Memory min_memory = static_cast<Memory>(MEMORY_LIMIT);
  Time makespan_of_min_memory = static_cast<Time>(SIZE_MAX);
  size_t taskSort_min_memory = SIZE_MAX;
  size_t pesSort_min_memory = SIZE_MAX;

  for (auto &taskSort_val : *output) {
    for (auto &pe_solution : taskSort_val.second) {
      const auto &solution = pe_solution.second;
      if (((solution.makespan < min_makespan ||
            (solution.makespan < min_makespan ||
             (solution.makespan == min_makespan && solution.memory_peak < memory_of_min_makespan)))) &&
          solution.memory_peak + PARAMETER_SIZE <= MEMORY_LIMIT) {
        min_makespan = solution.makespan;
        memory_of_min_makespan = solution.memory_peak;
        taskSort_min_makespan = taskSort_val.first;
        pesSort_min_makespan = pe_solution.first;
        *best_solution =
          std::string(TASK_SORT_NAMES[taskSort_val.first]) + " and " + std::string(PE_NAME_SORT[pe_solution.first]);
      }
      if ((solution.memory_peak < min_memory ||
           (solution.memory_peak == min_memory && solution.makespan < makespan_of_min_memory)) &&
          solution.memory_peak + PARAMETER_SIZE <= MEMORY_LIMIT) {
        min_memory = solution.memory_peak;
        makespan_of_min_memory = solution.makespan;
        taskSort_min_memory = taskSort_val.first;
        pesSort_min_memory = pe_solution.first;
        best_memory_solution->first =
          std::string(TASK_SORT_NAMES[taskSort_val.first]) + " and " + std::string(PE_NAME_SORT[pe_solution.first]);
        best_memory_solution->second = solution.memory_peak;
      }
    }
  }
  return std::make_tuple(taskSort_min_makespan, pesSort_min_makespan, taskSort_min_memory, pesSort_min_memory);
}

void PrintBestSolutionStats(const SchedulingOutput &output, const std::vector<GptoTaskPtr> &tasks,
                            const std::map<GptoTaskType, size_t> &type_to_num_cores_map,
                            const std::string &best_solution,
                            const std::pair<std::string, Memory> &best_memory_solution, size_t *lower_makespan) {
  const size_t kPrecision = 5;
  MS_LOG(INFO) << "Memory-aware scheduler with memory limit " << MEMORY_LIMIT;
  MS_LOG(INFO) << "Best solution is: " << best_solution;
  MS_LOG(INFO) << "Makespan of best solution is " << output.makespan;
  const auto &bottom_level_lb = LowerBoundBottomLevel(tasks);
  const auto &pes_lb = LowerBoundPEs(tasks, type_to_num_cores_map);
  *lower_makespan = std::max(bottom_level_lb, pes_lb);
  MS_LOG(INFO) << "Bottom level lower bound is " << bottom_level_lb;
  MS_LOG(INFO) << "Max type lower bound is " << pes_lb;
  const Time &total_time = TotalTime(tasks);
  MS_LOG(INFO) << "Makespan of sequential solution is " << total_time;
  MS_LOG(INFO) << "Speedup upper bound is " << std::setprecision(kPrecision) << 1.0 * total_time / (*lower_makespan);
  MS_LOG(INFO) << "Speedup of best solution is " << std::setprecision(kPrecision) << 1.0 * total_time / output.makespan;
  const size_t kHundred = 100;
  MS_LOG(INFO) << "Solution relative error is " << std::setprecision(kPrecision)
               << ((output.makespan / (1.0 * (*lower_makespan)) - 1) * kHundred) << "%";
  MS_LOG(INFO) << "GptoTensor peak memory estimate of best solution " << output.memory_peak << " ("
               << output.memory_peak / kMBToByte << " MB)";
  MS_LOG(INFO) << "Parameter memory estimate " << PARAMETER_SIZE << " (" << PARAMETER_SIZE / kMBToByte << " MB)";
  MS_LOG(INFO) << "Total memory estimate " << output.memory_peak + PARAMETER_SIZE << " ("
               << (output.memory_peak + PARAMETER_SIZE) / kMBToByte << " MB)";
  MS_LOG(INFO) << "Best solution for memory is: " << best_memory_solution.first << " with peak memory estimate "
               << best_memory_solution.second;
}

void InitializeTasks(
  const std::vector<GptoTaskPtr> &tasks, std::vector<Time> *can_start, std::vector<size_t> *unprocessed_parents,
  std::set<GptoTaskPtr, TaskSortFunction> *candidate_tasks, std::unordered_set<GptoTaskPtr> *switch_candidates,
  std::unordered_map<size_t, std::set<std::weak_ptr<GptoTask>, GptoTask::SortByIdWeak>> *left_consumers,
  const size_t &task_sort, const size_t &pes_sort) {
  MS_EXCEPTION_IF_NULL(can_start);
  MS_EXCEPTION_IF_NULL(unprocessed_parents);
  MS_EXCEPTION_IF_NULL(candidate_tasks);
  MS_EXCEPTION_IF_NULL(switch_candidates);
  MS_EXCEPTION_IF_NULL(left_consumers);
  std::unordered_set<GptoTensorPtr> all_in_tensors;
  for (auto &task : tasks) {
    const auto &id = task->id();
    (*can_start)[id] = static_cast<Time>(0);
    task->set_minus_mem_impact(static_cast<Memory>(0), task_sort, pes_sort);
    if (task_sort == static_cast<size_t>(kSortByGreedyHeight)) {
      task->set_minus_mem_impact_greedy_height(static_cast<Memory>(0));
    }
    (*unprocessed_parents)[id] = static_cast<size_t>(task->parents().size());
    if ((*unprocessed_parents)[id] == static_cast<size_t>(0)) {
      candidate_tasks->insert(task);
      if (task->condition_switch()) {
        (*switch_candidates).insert(task);
      }
    }
    all_in_tensors.insert(task->in_tensors().begin(), task->in_tensors().end());
  }
  for (const auto &in_tensor : all_in_tensors) {
    (*left_consumers)[in_tensor->id()].insert(in_tensor->consumers().begin(), in_tensor->consumers().end());
  }
}

void InitializeProcessingElements(const std::map<GptoTaskType, size_t> &type_to_num_cores_map,
                                  std::unordered_map<GptoTaskType, std::set<ProcessingElement, SortByLoad>> *PEs_load,
                                  std::unordered_map<GptoTaskType, std::vector<ProcessingElement>> *PEs_start,
                                  const size_t &pes_sort) {
  MS_EXCEPTION_IF_NULL(PEs_load);
  MS_EXCEPTION_IF_NULL(PEs_start);

  size_t count = 0;
  for (const auto &type_to_num : type_to_num_cores_map) {
    const auto &type = type_to_num.first;
    const auto &num_cores = type_to_num.second;
    for (size_t i = 0; i < num_cores; ++i) {
      ProcessingElement new_pe;
      new_pe.id = count + i;
      new_pe.gpto_type = type;
      new_pe.load = 0;
      new_pe.idle.emplace_back(0, SIZE_MAX);
      if (pes_sort == static_cast<size_t>(PEsSort::kSortByLoad)) {
        (*PEs_load)[type].insert(new_pe);
      } else if (pes_sort == static_cast<size_t>(PEsSort::kSortByValidStart)) {
        (*PEs_start)[type].push_back(new_pe);
      }
      MS_LOG(DEBUG) << "Initialized PE " << new_pe.id << " of type " << new_pe.gpto_type;
    }
    count += num_cores;
  }
}

void SubtractMemory(
  std::set<GptoTaskPtr, TaskSortFunction> *candidate_tasks, const GptoTaskPtr &selected_task,
  std::unordered_map<size_t, std::set<std::weak_ptr<GptoTask>, GptoTask::SortByIdWeak>> *left_consumers,
  std::map<Time, Memory> *cur_mem_peak_ptr, const size_t &task_sort, const size_t &pes_sort) {
  MS_EXCEPTION_IF_NULL(left_consumers);
  MS_EXCEPTION_IF_NULL(cur_mem_peak_ptr);

  for (auto &in_tensor : selected_task->in_tensors()) {
    if (selected_task->end(task_sort, pes_sort) > in_tensor->lifetime_end(task_sort, pes_sort)) {
      in_tensor->set_lifetime_end(selected_task->end(task_sort, pes_sort), task_sort, pes_sort);
      in_tensor->set_last_consumer(selected_task, task_sort, pes_sort);
    }
    (*left_consumers)[in_tensor->id()].erase(selected_task);
    if ((*left_consumers)[in_tensor->id()].size() == 0) {
      if (in_tensor->type() == kGraphOutput) {
        continue;
      }
      for (auto it = (*cur_mem_peak_ptr)
                       .lower_bound(in_tensor->last_consumer(task_sort, pes_sort).lock()->end(task_sort, pes_sort));
           it != (*cur_mem_peak_ptr).end(); it++) {
        it->second -= in_tensor->weight();
      }
    }
  }

  for (auto &out_tensor : selected_task->out_tensors()) {
    out_tensor->set_lifetime_end(static_cast<Time>(0), task_sort, pes_sort);
    if (out_tensor->consumers().size() == 1 && out_tensor->type() != kGraphOutput) {
      auto last_consumer = *(out_tensor->consumers().begin());
      last_consumer.lock()->set_minus_mem_impact(
        last_consumer.lock()->minus_mem_impact(task_sort, pes_sort) + out_tensor->weight(), task_sort, pes_sort);
      if (task_sort == static_cast<size_t>(kSortByGreedyHeight)) {
        auto it = candidate_tasks->find(last_consumer.lock());
        if (it != candidate_tasks->end()) {
          auto updated_candidate = candidate_tasks->extract(it);
          updated_candidate.value()->set_minus_mem_impact_greedy_height(
            last_consumer.lock()->minus_mem_impact(task_sort, pes_sort));
          candidate_tasks->insert(std::move(updated_candidate));
        } else {
          last_consumer.lock()->set_minus_mem_impact_greedy_height(
            last_consumer.lock()->minus_mem_impact(task_sort, pes_sort));
        }
      }
      out_tensor->set_last_consumer(last_consumer, task_sort, pes_sort);
    }
  }
}

void UpdateCandidates(std::set<GptoTaskPtr, TaskSortFunction> *candidate_tasks, const GptoTaskPtr &selected_task,
                      std::vector<size_t> *unprocessed_parents, std::vector<Time> *can_start, Time *last_end,
                      std::map<GptoTaskType, Time> *last_end_type, Time *last_gather_end,
                      std::unordered_set<GptoTaskPtr> *switch_candidates, const size_t &position,
                      const size_t &task_sort, const size_t &pes_sort) {
  MS_EXCEPTION_IF_NULL(candidate_tasks);
  MS_EXCEPTION_IF_NULL(unprocessed_parents);
  MS_EXCEPTION_IF_NULL(can_start);
  MS_EXCEPTION_IF_NULL(last_end);
  MS_EXCEPTION_IF_NULL(last_end_type);
  MS_EXCEPTION_IF_NULL(last_gather_end);
  MS_EXCEPTION_IF_NULL(switch_candidates);

  // Update candidate tasks
  candidate_tasks->erase(selected_task);
  if (selected_task->condition_switch()) {
    (*switch_candidates).erase(selected_task);
  }

  // Update can_start with special processing for ConditionalSwitch/Gather cases
  if (selected_task->condition_gather()) {
    for (const auto &candidate : *candidate_tasks) {
      (*can_start)[candidate->id()] =
        std::max((*can_start)[candidate->id()], static_cast<Time>(selected_task->end(task_sort, pes_sort)));
    }
    *last_gather_end = std::max(*last_gather_end, selected_task->end(task_sort, pes_sort));
  }

  *last_end = std::max(*last_end, selected_task->end(task_sort, pes_sort));
  (*last_end_type)[selected_task->gpto_type()] =
    std::max((*last_end_type)[selected_task->gpto_type()], selected_task->end(task_sort, pes_sort));
  for (const auto &candidate : *switch_candidates) {
    (*can_start)[candidate->id()] = std::max((*can_start)[candidate->id()], static_cast<Time>(*last_end));
  }

  // Special case for GPTO topological sorting
  if (GetGptoOptionsMode() == kBasic) {
    for (const auto &candidate : *candidate_tasks) {
      (*can_start)[candidate->id()] =
        std::max((*can_start)[candidate->id()], static_cast<Time>((*last_end_type)[candidate->gpto_type()]));
    }
  }

  // SAM update values
  const bool SAM_algo =
    (task_sort >= static_cast<size_t>(kSortBySValue) && task_sort <= static_cast<size_t>(kSortByCostMValue));
  if (SAM_algo) {
    auto candidates = *candidate_tasks;
    for (auto it = candidates.begin(); it != candidates.end(); it++) {
      auto updated_candidate = candidate_tasks->extract(*it);
      UPDATE_SAM[TASK_SORT_NAMES[task_sort]](updated_candidate.value(), task_sort, pes_sort, position);
      candidate_tasks->insert(std::move(updated_candidate));
    }
  }

  for (auto &successor : selected_task->children()) {
    const auto &succ_id = successor->id();
    if (GetGptoOptionsMode() == kBasic) {
      (*can_start)[succ_id] =
        std::max((*can_start)[succ_id], static_cast<Time>((*last_end_type)[successor->gpto_type()]));
    }
    if (selected_task->child_is_data_dep(successor)) {
      (*can_start)[succ_id] =
        std::max((*can_start)[succ_id], static_cast<Time>(selected_task->end(task_sort, pes_sort)));
    } else {
      (*can_start)[succ_id] =
        std::max((*can_start)[succ_id], static_cast<Time>(1 + selected_task->start(task_sort, pes_sort)));
    }
    (*can_start)[succ_id] = std::max((*can_start)[succ_id], static_cast<Time>(*last_gather_end));
    if (successor->condition_switch()) {
      (*can_start)[succ_id] = std::max((*can_start)[succ_id], static_cast<Time>(*last_end));
    }
    (*unprocessed_parents)[succ_id] -= 1;
    if ((*unprocessed_parents)[succ_id] == 0) {
      if (SAM_algo) {
        INIT_SAM[TASK_SORT_NAMES[task_sort]](successor, task_sort, pes_sort, position);
      }

      candidate_tasks->insert(successor);
      if (successor->condition_switch()) {
        (*switch_candidates).insert(successor);
      }
    }
  }
}

void InitializeInTensorsWeight(const GptoTaskPtr &task) {
  // can instead associate (save) list of tensors to edge when extracting scheduling input
  std::unordered_map<GptoTaskPtr, Memory> in_weight;
  Memory in_weights_sum = 0;
  for (const auto &parent : task->parents()) {
    in_weight[parent.lock()] = 0;
  }
  for (const auto &in_tensor : task->in_tensors()) {
    const auto &source = in_tensor->source();
    if (source.lock() != task) {
      in_weight[source.lock()] += in_tensor->weight();
      in_weights_sum += in_tensor->weight();
    }
  }
  task->set_in_weights(in_weight);
  task->set_in_weights_sum(in_weights_sum);
  in_weight.clear();
}

void InitializeS(const GptoTaskPtr &task, const size_t &task_sort, const size_t &pes_sort,
                 const size_t &current_position) {
  task->set_s_value(task->parents().size() * current_position -
                    std::accumulate(task->parents().begin(), task->parents().end(), static_cast<size_t>(0),
                                    [task_sort, pes_sort](size_t acc, const auto &parent) {
                                      return acc + parent.lock()->position(task_sort, pes_sort);
                                    }));
}

void InitializeSW(const GptoTaskPtr &task, const size_t &task_sort, const size_t &pes_sort,
                  const size_t &current_position) {
  task->set_sw_value(std::accumulate(task->parents().begin(), task->parents().end(), static_cast<Memory>(0),
                                     [current_position, task, task_sort, pes_sort](Memory acc, const auto &parent) {
                                       return acc + (current_position - parent.lock()->position(task_sort, pes_sort)) *
                                                      task->in_weights()[parent.lock()];
                                     }));
}

void InitializeSC(const GptoTaskPtr &task, const size_t &task_sort, const size_t &pes_sort,
                  const size_t &current_position) {
  task->set_sc_value(task->cost() *
                     (task->parents().size() * current_position -
                      std::accumulate(task->parents().begin(), task->parents().end(), static_cast<size_t>(0),
                                      [task_sort, pes_sort](size_t acc, const auto &parent) {
                                        return acc + parent.lock()->position(task_sort, pes_sort);
                                      })));
}

void InitializeA(const GptoTaskPtr &task, const size_t &task_sort, const size_t &pes_sort,
                 const size_t &current_position) {
  if (task->parents().size() > 0) {
    auto s_value = task->parents().size() * current_position -
                   std::accumulate(task->parents().begin(), task->parents().end(), static_cast<size_t>(0),
                                   [task_sort, pes_sort](size_t acc, const auto &parent) {
                                     return acc + parent.lock()->position(task_sort, pes_sort);
                                   });
    task->set_a_value(1.0f * s_value / task->parents().size());
  } else {
    task->set_a_value(0.0f);
  }
}

void InitializeAW(const GptoTaskPtr &task, const size_t &task_sort, const size_t &pes_sort,
                  const size_t &current_position) {
  if (task->in_weights_sum() > 0) {
    auto sw_value = std::accumulate(task->parents().begin(), task->parents().end(), static_cast<Memory>(0),
                                    [current_position, task, task_sort, pes_sort](Memory acc, const auto &parent) {
                                      return acc + (current_position - parent.lock()->position(task_sort, pes_sort)) *
                                                     task->in_weights()[parent.lock()];
                                    });
    task->set_aw_value(1.0f * sw_value / task->in_weights_sum());
  } else {
    task->set_aw_value(0.0f);
  }
}

void InitializeAC(const GptoTaskPtr &task, const size_t &task_sort, const size_t &pes_sort,
                  const size_t &current_position) {
  if (task->parents().size() > 0) {
    auto s_value = task->parents().size() * current_position -
                   std::accumulate(task->parents().begin(), task->parents().end(), static_cast<size_t>(0),
                                   [task_sort, pes_sort](size_t acc, const auto &parent) {
                                     return acc + parent.lock()->position(task_sort, pes_sort);
                                   });
    task->set_ac_value(1.0f * task->cost() * s_value / task->parents().size());
  } else {
    task->set_ac_value(0.0f);
  }
}

void InitializeM(const GptoTaskPtr &task, const size_t &task_sort, const size_t &pes_sort,
                 const size_t &current_position) {
  size_t m = 0;
  m = std::accumulate(task->parents().begin(), task->parents().end(), m,
                      [current_position, task_sort, pes_sort](size_t m, const auto &parent) {
                        return std::max(m, current_position - parent.lock()->position(task_sort, pes_sort));
                      });
  task->set_m_value(m);
}

void InitializeMW(const GptoTaskPtr &task, const size_t &task_sort, const size_t &pes_sort,
                  const size_t &current_position) {
  Memory mw = 0;
  Memory weight_in_mw = 0;
  for (const auto &parent : task->parents()) {
    Memory cand_mw = static_cast<Memory>((current_position - parent.lock()->position(task_sort, pes_sort)) *
                                         task->in_weights()[parent.lock()]);
    if (cand_mw > mw) {
      mw = cand_mw;
      weight_in_mw = task->in_weights()[parent.lock()];
    }
  }
  task->set_mw_value(mw);

  // For update, only parents with larger weight than weight_in_mw, may cause mw to change
  if (weight_in_mw > 0) {
    for (const auto &parent : task->parents()) {
      if (task->in_weights()[parent.lock()] >= weight_in_mw) {
        task->AddMWParent(parent);
      }
    }
  }
}

void InitializeMC(const GptoTaskPtr &task, const size_t &task_sort, const size_t &pes_sort,
                  const size_t &current_position) {
  size_t m = 0;
  m = std::accumulate(task->parents().begin(), task->parents().end(), m,
                      [current_position, task_sort, pes_sort](size_t m, const auto &parent) {
                        return std::max(m, current_position - parent.lock()->position(task_sort, pes_sort));
                      });

  task->set_mc_value(m * task->cost());
}

void UpdateS(const GptoTaskPtr &task, [[maybe_unused]] const size_t &task_sort, [[maybe_unused]] const size_t &pes_sort,
             [[maybe_unused]] const size_t &current_position = 0) {
  if (task->parents().size() > 0) {
    task->set_s_value(task->s_value() + task->parents().size());
  }
}

void UpdateSW(const GptoTaskPtr &task, [[maybe_unused]] const size_t &task_sort,
              [[maybe_unused]] const size_t &pes_sort, [[maybe_unused]] const size_t &current_position = 0) {
  if (task->in_weights_sum() > 0) {
    task->set_sw_value(task->sw_value() + task->in_weights_sum());
  }
}

void UpdateSC(const GptoTaskPtr &task, [[maybe_unused]] const size_t &task_sort,
              [[maybe_unused]] const size_t &pes_sort, [[maybe_unused]] const size_t &current_position = 0) {
  if (task->parents().size() > 0) {
    task->set_sc_value(task->sc_value() + task->cost() * task->parents().size());
  }
}

void UpdateA(const GptoTaskPtr &task, [[maybe_unused]] const size_t &task_sort, [[maybe_unused]] const size_t &pes_sort,
             [[maybe_unused]] const size_t &current_position = 0) {
  if (task->parents().size() > 0) {
    task->set_a_value(task->a_value() + 1.0);
  }
}

void UpdateAW(const GptoTaskPtr &task, [[maybe_unused]] const size_t &task_sort,
              [[maybe_unused]] const size_t &pes_sort, [[maybe_unused]] const size_t &current_position = 0) {
  if (task->in_weights_sum() > 0) {
    task->set_aw_value(task->aw_value() + 1.0);
  }
}

void UpdateAC(const GptoTaskPtr &task, [[maybe_unused]] const size_t &task_sort,
              [[maybe_unused]] const size_t &pes_sort, [[maybe_unused]] const size_t &current_position = 0) {
  if (task->parents().size() > 0) {
    task->set_ac_value(task->ac_value() + 1.0 * task->cost());
  }
}

void UpdateM(const GptoTaskPtr &task, [[maybe_unused]] const size_t &task_sort, [[maybe_unused]] const size_t &pes_sort,
             [[maybe_unused]] const size_t &current_position = 0) {
  if (task->parents().size() > 0) {
    task->set_m_value(task->m_value() + 1);
  }
}

void UpdateMW(const GptoTaskPtr &task, [[maybe_unused]] const size_t &task_sort,
              [[maybe_unused]] const size_t &pes_sort, [[maybe_unused]] const size_t &current_position = 0) {
  Memory mw = 0;

  mw = std::accumulate(task->mw_parents().begin(), task->mw_parents().end(), mw,
                       [current_position, task_sort, pes_sort, task](size_t mw, const auto &parent) {
                         return std::max(
                           mw, static_cast<Memory>((current_position - parent.lock()->position(task_sort, pes_sort)) *
                                                   task->in_weights()[parent.lock()]));
                       });

  task->set_mw_value(mw);
}

void UpdateMC(const GptoTaskPtr &task, [[maybe_unused]] const size_t &task_sort,
              [[maybe_unused]] const size_t &pes_sort, [[maybe_unused]] const size_t &current_position = 0) {
  if (task->parents().size() > 0) {
    task->set_mc_value(task->mc_value() + task->cost());
  }
}

bool VerifyS(const std::vector<GptoTaskPtr> &tasks, const size_t &task_sort, const size_t &pes_sort) {
  bool success = true;
  for (const auto &task : tasks) {
    size_t s = 0;
    for (auto &parent : task->parents()) {
      size_t d = task->position(task_sort, pes_sort) - parent.lock()->position(task_sort, pes_sort);
      s += d;
    }
    if (s != task->s_value()) {
      std::ostringstream error_msg;
      error_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort] << "] Task " << task->id()
                << " s value: " << task->s_value() << " verify: " << s;
      MS_LOG(ERROR) << error_msg.str();
      success = false;
      break;
    }
  }
  return success;
}

bool VerifySW(const std::vector<GptoTaskPtr> &tasks, const size_t &task_sort, const size_t &pes_sort) {
  bool success = true;
  for (const auto &task : tasks) {
    Memory sw = 0;
    for (auto &parent : task->parents()) {
      size_t d = task->position(task_sort, pes_sort) - parent.lock()->position(task_sort, pes_sort);
      Memory dw = d * task->in_weights()[parent.lock()];
      sw += dw;
    }
    if (sw != task->sw_value()) {
      std::ostringstream error_msg;
      error_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort] << "] Task " << task->id()
                << " sw value: " << task->sw_value() << " verify: " << sw;
      MS_LOG(ERROR) << error_msg.str();
      success = false;
      break;
    }
  }
  return success;
}

bool VerifySC(const std::vector<GptoTaskPtr> &tasks, const size_t &task_sort, const size_t &pes_sort) {
  bool success = true;
  for (const auto &task : tasks) {
    size_t s = 0;
    Time sc = 0;
    for (auto &parent : task->parents()) {
      size_t d = task->position(task_sort, pes_sort) - parent.lock()->position(task_sort, pes_sort);
      s += d;
    }
    sc = s * task->cost();
    if (sc != task->sc_value()) {
      std::ostringstream error_msg;
      error_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort] << "] Task " << task->id()
                << " sc value: " << task->sc_value() << " verify: " << sc;
      MS_LOG(ERROR) << error_msg.str();
      success = false;
      break;
    }
  }
  return success;
}

bool VerifyA(const std::vector<GptoTaskPtr> &tasks, const size_t &task_sort, const size_t &pes_sort) {
  bool success = true;
  for (const auto &task : tasks) {
    size_t s = 0;
    double a = 0.0;
    const double EPSILON = 1e-3;
    for (auto &parent : task->parents()) {
      size_t d = task->position(task_sort, pes_sort) - parent.lock()->position(task_sort, pes_sort);
      s += d;
    }
    if (task->parents().size() > 0) {
      a = 1.0 * s / (1.0 * task->parents().size());
    }
    if ((std::fabs(a - task->a_value()) / std::max(a, task->a_value()) > EPSILON)) {
      std::ostringstream error_msg;
      error_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort] << "] Task " << task->id()
                << " a value: " << task->a_value() << " verify: " << a;
      MS_LOG(ERROR) << error_msg.str();
      success = false;
      break;
    }
  }
  return success;
}

bool VerifyAW(const std::vector<GptoTaskPtr> &tasks, const size_t &task_sort, const size_t &pes_sort) {
  bool success = true;
  for (const auto &task : tasks) {
    double aw = 0.0;
    const double EPSILON = 1e-3;
    Memory sw = 0;
    for (auto &parent : task->parents()) {
      size_t d = task->position(task_sort, pes_sort) - parent.lock()->position(task_sort, pes_sort);
      Memory dw = d * task->in_weights()[parent.lock()];
      sw += dw;
    }
    if (task->in_weights_sum() > 0) {
      aw = 1.0 * sw / (1.0 * task->in_weights_sum());
    }
    if ((std::fabs(aw - task->aw_value()) / std::max(aw, task->aw_value()) > EPSILON)) {
      std::ostringstream error_msg;
      error_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort] << "] Task " << task->id()
                << " aw value: " << task->aw_value() << " verify: " << aw;
      MS_LOG(ERROR) << error_msg.str();
      success = false;
      break;
    }
  }
  return success;
}

bool VerifyAC(const std::vector<GptoTaskPtr> &tasks, const size_t &task_sort, const size_t &pes_sort) {
  bool success = true;
  for (const auto &task : tasks) {
    size_t s = 0;
    double ac = 0.0;
    const double EPSILON = 1e-3;
    for (const auto &parent : task->parents()) {
      size_t d = task->position(task_sort, pes_sort) - parent.lock()->position(task_sort, pes_sort);
      s += d;
    }
    if (task->parents().size() > 0) {
      const double a = 1.0 * s / (1.0 * task->parents().size());
      ac = a * task->cost();
    }
    if ((std::fabs(ac - task->ac_value()) > EPSILON) / std::max(ac, task->ac_value()) > EPSILON) {
      std::ostringstream error_msg;
      error_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort] << "] Task " << task->id()
                << " ac value: " << task->ac_value() << " verify: " << ac;
      MS_LOG(ERROR) << error_msg.str();
      success = false;
      break;
    }
  }
  return success;
}

bool VerifyM(const std::vector<GptoTaskPtr> &tasks, const size_t &task_sort, const size_t &pes_sort) {
  bool success = true;
  for (const auto &task : tasks) {
    size_t m = 0;
    for (auto &parent : task->parents()) {
      size_t d = task->position(task_sort, pes_sort) - parent.lock()->position(task_sort, pes_sort);
      if (d > m) {
        m = d;
      }
    }
    if (m != task->m_value()) {
      std::ostringstream error_msg;
      error_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort] << "] Task " << task->id()
                << " m value: " << task->m_value() << " verify: " << m;
      MS_LOG(ERROR) << error_msg.str();
      success = false;
      break;
    }
  }
  return success;
}

bool VerifyMW(const std::vector<GptoTaskPtr> &tasks, const size_t &task_sort, const size_t &pes_sort) {
  bool success = true;
  for (const auto &task : tasks) {
    Memory mw = 0;
    for (auto &parent : task->parents()) {
      size_t d = task->position(task_sort, pes_sort) - parent.lock()->position(task_sort, pes_sort);
      Memory dw = d * task->in_weights()[parent.lock()];
      if (dw > mw) {
        mw = dw;
      }
    }
    if (mw != task->mw_value()) {
      std::ostringstream error_msg;
      error_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort] << "] Task " << task->id()
                << " mw value: " << task->mw_value() << " verify: " << mw;
      MS_LOG(ERROR) << error_msg.str();
      success = false;
      break;
    }
  }
  return success;
}

bool VerifyMC(const std::vector<GptoTaskPtr> &tasks, const size_t &task_sort, const size_t &pes_sort) {
  bool success = true;
  for (const auto &task : tasks) {
    size_t m = 0;
    Time mc = 0;
    for (auto &parent : task->parents()) {
      size_t d = task->position(task_sort, pes_sort) - parent.lock()->position(task_sort, pes_sort);
      if (d > m) {
        m = d;
      }
    }
    mc = m * task->cost();
    if (mc != task->mc_value()) {
      std::ostringstream error_msg;
      error_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort] << "] Task " << task->id()
                << " mc value: " << task->mc_value() << " verify: " << mc;
      MS_LOG(ERROR) << error_msg.str();
      success = false;
      break;
    }
  }
  return success;
}

void MemAwareSchedulerPre(const std::vector<GptoTaskPtr> &tasks) {
  ComputeBottomLevelAndWeightedLength(tasks);
  if (gpto_mode == kCompComm || gpto_mode == kCompCubeComm || gpto_mode == kCompCommGroup) {
    ComputePredComm(tasks);
  }
  if (gpto_mode == kCompCubeComm) {
    ComputePredCube(tasks);
  }
  ComputeSuccDiffType(tasks);
  ComputePostOrder(tasks);
  for (const auto &task : tasks) {
    InitializeInTensorsWeight(task);
  }
  ComputeInitialMemoryImpact(tasks);
}

SchedulingOutput MemAwareSchedulerCore(const std::vector<GptoTaskPtr> &tasks,
                                       const std::map<GptoTaskType, size_t> &type_to_num_cores_map,
                                       const size_t &task_sort, const size_t &pes_sort) {
  SchedulingOutput output;  // to return
  output.task_times.reserve(tasks.size());
  output.makespan = static_cast<Time>(0);
  output.memory_peak = static_cast<Memory>(0);

  // Initializations for tasks
  std::set<GptoTaskPtr, TaskSortFunction> candidate_tasks(TASK_SORT[task_sort]);
  // Assume GptoTaskId goes from 0 to tasks.size()-1
  std::vector<Time> can_start(tasks.size());
  std::vector<size_t> unprocessed_parents(tasks.size());
  std::unordered_set<GptoTaskPtr> switch_candidates;
  std::unordered_map<size_t, std::set<std::weak_ptr<GptoTask>, GptoTask::SortByIdWeak>> left_consumers;
  InitializeTasks(tasks, &can_start, &unprocessed_parents, &candidate_tasks, &switch_candidates, &left_consumers,
                  task_sort, pes_sort);

  // Initializations for processing elements
  // Pick a sorting for processing elements
  // Implemented: SortByLoad, SortByAvailableStart
  // Only one structure to be used depending on argument; we define both here
  std::unordered_map<GptoTaskType, std::set<ProcessingElement, SortByLoad>> PEs_load;
  std::unordered_map<GptoTaskType, std::vector<ProcessingElement>> PEs_start;
  InitializeProcessingElements(type_to_num_cores_map, &PEs_load, &PEs_start, pes_sort);

  // Task graph scheduling loop
  std::map<Time, Memory> cur_mem_peak;
  Time last_end = 0;
  std::map<GptoTaskType, Time> last_end_type;
  for (const auto &type_num : type_to_num_cores_map) {
    last_end_type[type_num.first] = static_cast<Time>(0);
  }
  Time last_gather_end = 0;
  size_t position = 0;
  size_t last_switch_subgraph = SIZE_MAX;
  while (!candidate_tasks.empty()) {
    // Schedule a task (if possible)
    std::tuple<GptoTaskPtr, Time, PeId> scheduled_info;
    if (pes_sort == static_cast<size_t>(PEsSort::kSortByLoad)) {
      scheduled_info = ScheduleTaskLoad(&candidate_tasks, &PEs_load, &can_start, &cur_mem_peak, &last_switch_subgraph,
                                        task_sort, pes_sort);
    } else if (pes_sort == static_cast<size_t>(PEsSort::kSortByValidStart)) {
      scheduled_info = ScheduleTaskStart(&candidate_tasks, &PEs_start, &can_start, &cur_mem_peak, &last_switch_subgraph,
                                         task_sort, pes_sort);
    }

    GptoTaskPtr &selected_task = std::get<0>(scheduled_info);
    Time &selected_time = std::get<1>(scheduled_info);
    PeId &selected_pe = std::get<2>(scheduled_info);
    if (selected_task == nullptr) {  // Out-of-memory for all candidates
      output.makespan = static_cast<Time>(SIZE_MAX);
      output.memory_peak = static_cast<Memory>(SIZE_MAX);
      std::ostringstream oom_msg;
      oom_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort] << "] Out of memory estimated!";
      MS_LOG(INFO) << oom_msg.str();
      return output;
    }

    // Switch/gather logic
    if (selected_task->condition_switch()) {
      last_switch_subgraph = selected_task->subgraph_id();
    } else if (selected_task->condition_gather()) {
      last_switch_subgraph = selected_task->subgraph_id_parent();
    }

    // Update output
    selected_task->set_position(position++, task_sort, pes_sort);
    selected_task->set_start(selected_time, task_sort, pes_sort);
    selected_task->set_end(selected_time + selected_task->cost(), task_sort, pes_sort);
    Interval new_interval{selected_task, selected_task->start(task_sort, pes_sort),
                          selected_task->end(task_sort, pes_sort), selected_pe};
    output.task_times.push_back(new_interval);
    output.makespan = std::max(output.makespan, selected_task->end(task_sort, pes_sort));

    // Bookkeeping for memory consumption and candidates
    AddMemory(selected_task, selected_time, &cur_mem_peak);
    SubtractMemory(&candidate_tasks, selected_task, &left_consumers, &cur_mem_peak, task_sort, pes_sort);
    UpdateCandidates(&candidate_tasks, selected_task, &unprocessed_parents, &can_start, &last_end, &last_end_type,
                     &last_gather_end, &switch_candidates, position, task_sort, pes_sort);
  }  // end-while candidates not empty

  // Clear auxiliary data structures
  can_start.clear();
  unprocessed_parents.clear();
  switch_candidates.clear();
  left_consumers.clear();
  PEs_load.clear();
  PEs_start.clear();

  // End of time: reset memory to zero (equivalent to subtracting kGraphOutput tensors)
  cur_mem_peak.rbegin()->second = 0;

  // Output memory peak is the max of cur_mem_peak
  output.memory_peak = 0;
  for (const auto &time_mem : cur_mem_peak) {
    if (time_mem.second > output.memory_peak) {
      output.memory_peak = time_mem.second;
    }
  }
  // Print result
  std::ostringstream makespan_msg;
  makespan_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort] << "] Makespan is "
               << output.makespan;
  MS_LOG(INFO) << makespan_msg.str();
  std::ostringstream memory_msg;
  memory_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort] << "] Peak mem is "
             << output.memory_peak;
  MS_LOG(INFO) << memory_msg.str();

  // Verification of scheduling solution (optional)
  if (IsGptoDebug()) {
    if (!VerifyAll(tasks, &cur_mem_peak, task_sort, pes_sort)) {
      output.makespan = SIZE_MAX;
    }
  }
  cur_mem_peak.clear();

  return output;
}

std::tuple<GptoTaskPtr, Time, PeId> ScheduleTaskLoad(
  std::set<GptoTaskPtr, TaskSortFunction> *candidate_tasks_ptr,
  std::unordered_map<GptoTaskType, std::set<ProcessingElement, SortByLoad>> *PEs_load_ptr,
  std::vector<Time> *can_start_ptr, std::map<Time, Memory> *cur_mem_peak_ptr, size_t *last_switch_subgraph_ptr,
  const size_t &task_sort, const size_t &pes_sort) {
  auto &candidate_tasks = *candidate_tasks_ptr;
  auto &PEs_load = *PEs_load_ptr;
  const auto &can_start = *can_start_ptr;
  const auto &cur_mem_peak = *cur_mem_peak_ptr;
  const auto &last_switch_subgraph = *last_switch_subgraph_ptr;

  bool exists_subgraph = false;
  for (auto task_it = candidate_tasks.begin(); task_it != candidate_tasks.end(); ++task_it) {
    const auto &selected_task = *task_it;
    const auto task_can_start = can_start[selected_task->id()];
    auto &PEs = PEs_load[selected_task->gpto_type()];
    // Pick a PE amongst PEs of same type
    for (auto pe_it = PEs.begin(); pe_it != PEs.end(); pe_it++) {
      auto &mut_pe = const_cast<ProcessingElement &>(*pe_it);
      // Put in first idle window that fits the task (if memory limit is not violated)
      for (auto idle_it = mut_pe.idle.begin(); idle_it != mut_pe.idle.end(); ++idle_it) {
        Time start_time;
        bool case_flag = false;
        // Distinguish cases based on can_start constraint
        if (task_can_start <= idle_it->first) {
          start_time = idle_it->first;
        } else if (task_can_start <= idle_it->second) {
          start_time = task_can_start;
          case_flag = true;
        } else {  // task_can_start > idle_it->second (not allowed to place here)
          continue;
        }
        if (idle_it->second - start_time < selected_task->cost() ||
            MemoryViolated(selected_task, start_time, cur_mem_peak, last_switch_subgraph, &exists_subgraph, task_sort,
                           pes_sort)) {
          continue;
        }
        // Place task in idle window here
        // Save info to return: start task at time idle_it->first
        PeId selected_pe = (*pe_it).id;
        Time selected_time = start_time;
        // Update idle list
        if (!case_flag) {
          if (idle_it->second - idle_it->first ==
              selected_task->cost()) {  // whole idle interval is filled in, erase it
            mut_pe.idle.erase(idle_it);
          } else {  // idle_it->second - idle_it->first > selected_task->cost()
            idle_it->first += selected_task->cost();
          }
        } else {  // case_flag = true, idle interval is broken into two
                  // sub-blocks [idle_it->first, can_start] and
                  // (maybe empty) [can_start + cost, idle_it->second]
          Time upper = idle_it->second;
          idle_it->second = task_can_start;
          if (upper - task_can_start - selected_task->cost() > 0) {
            std::pair<Time, Time> new_idle = std::make_pair(task_can_start + selected_task->cost(), upper);
            mut_pe.idle.emplace(std::next(idle_it), new_idle);
          }
        }
        // Update load, PEs, and memory peaks
        auto updated_PE = PEs.extract(pe_it);
        updated_PE.value().load += selected_task->cost();
        PEs.insert(std::move(updated_PE));
        return std::make_tuple(selected_task, selected_time, selected_pe);
      }  // end-for idle windows
    }
  }
  return std::make_tuple(nullptr, 0, 0);
}

std::tuple<GptoTaskPtr, Time, PeId> ScheduleTaskStart(
  std::set<GptoTaskPtr, TaskSortFunction> *candidate_tasks_ptr,
  std::unordered_map<GptoTaskType, std::vector<ProcessingElement>> *PEs_start_ptr, std::vector<Time> *can_start,
  std::map<Time, Memory> *cur_mem_peak_ptr, size_t *last_switch_subgraph_ptr, const size_t &task_sort,
  const size_t &pes_sort) {
  auto &candidate_tasks = *candidate_tasks_ptr;
  auto &PEs_start = *PEs_start_ptr;
  const auto &cur_mem_peak = *cur_mem_peak_ptr;
  const auto &last_switch_subgraph = *last_switch_subgraph_ptr;

  bool exists_subgraph = false;
  for (auto task_it = candidate_tasks.begin(); task_it != candidate_tasks.end(); ++task_it) {
    const auto &selected_task = *task_it;
    const auto &task_can_start = (*can_start)[selected_task->id()];
    auto &PEs = PEs_start[selected_task->gpto_type()];
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
        if (task_can_start <= idle_it->first) {
          start_time = idle_it->first;
        } else if (task_can_start <= idle_it->second) {
          start_time = task_can_start;
          case_flag = true;
        } else {  // task_can_start > idle_it->second (not allowed to place here)
          continue;
        }
        if (idle_it->second - start_time >= selected_task->cost()) {
          if (MemoryViolated(selected_task, start_time, cur_mem_peak, last_switch_subgraph, &exists_subgraph, task_sort,
                             pes_sort)) {
            continue;
          }
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
    if (min_start == SIZE_MAX) {  // cannot place selected_task anywhere
      if (std::next(task_it) == candidate_tasks.end()) {
        return std::make_tuple(nullptr, 0, 0);
      } else {
        continue;
      }
    }
    // Assign task to min-start-time PE
    auto selected_pe = (*min_it).id;
    auto selected_time = min_start;
    // Update idle list
    if (!min_case) {
      if (min_idle_it->second - min_idle_it->first ==
          selected_task->cost()) {  // whole idle interval is filled in, erase it
        min_it->idle.erase(min_idle_it);
      } else {  // idle_it->second - idle_it->first > selected_task->cost()
        min_idle_it->first += selected_task->cost();
      }
    } else {  // min_case = true, idle interval is broken into two sub-blocks
              // [idle_it->first, can_start] and
              // (maybe empty) [can_start + task.cost(), idle_it->second]
      Time upper = min_idle_it->second;
      min_idle_it->second = task_can_start;
      if (upper - task_can_start - selected_task->cost() > 0) {
        std::pair<Time, Time> new_idle = std::make_pair(task_can_start + selected_task->cost(), upper);
        min_it->idle.emplace(std::next(min_idle_it), new_idle);
      }
    }
    // Update and break
    min_it->load += selected_task->cost();
    return std::make_tuple(selected_task, selected_time, selected_pe);
  }
  return std::make_tuple(nullptr, 0, 0);
}

void AddMemory(const GptoTaskPtr &selected_task, const Time &selected_time, std::map<Time, Memory> *cur_mem_peak_ptr) {
  auto &cur_mem_peak = *cur_mem_peak_ptr;
  bool found_start = false;
  bool found_end = false;
  Time prev_end = 0;
  auto selected_time_it = cur_mem_peak.lower_bound(selected_time);
  for (auto it = selected_time_it; it != cur_mem_peak.end(); it++) {
    auto &time = it->first;
    auto &mem = it->second;
    if (time == selected_time) {
      found_start = true;
      prev_end = std::max(prev_end, time);
      mem += selected_task->initial_mem_impact();
    } else if (time < selected_time + selected_task->cost()) {
      prev_end = std::max(prev_end, time);
      mem += selected_task->initial_mem_impact();
    } else if (time == selected_time + selected_task->cost()) {
      found_end = true;
      mem += selected_task->initial_mem_impact() - selected_task->workspace_memory();
    } else if (time > selected_time + selected_task->cost()) {
      mem += selected_task->initial_mem_impact() - selected_task->workspace_memory();
    }
  }
  if (!found_start) {
    if (selected_time_it != cur_mem_peak.begin() && selected_time_it != cur_mem_peak.end()) {
      cur_mem_peak[selected_time] = std::prev(selected_time_it)->second + selected_task->initial_mem_impact();
    } else {
      cur_mem_peak[selected_time] = selected_task->initial_mem_impact();
    }
  }
  if (!found_end) {
    cur_mem_peak[selected_time + selected_task->cost()] =
      cur_mem_peak[std::max(prev_end, selected_time)] - selected_task->workspace_memory();
  }
}

bool MemoryViolated(const GptoTaskPtr &selected_task, const Time &start_time,
                    const std::map<Time, Memory> &cur_mem_peak, const size_t &last_switch_subgraph,
                    bool *exists_subgraph, const size_t &task_sort, const size_t &pes_sort) {
  if (selected_task->subgraph_id() < last_switch_subgraph) {
    *exists_subgraph = true;
    return MemoryViolatedCore(selected_task, start_time, cur_mem_peak, task_sort, pes_sort);
  } else {  // selected_task->subgraph_id() == *last_switch_subgraph
    if (*exists_subgraph) {
      return false;
    } else {
      return MemoryViolatedCore(selected_task, start_time, cur_mem_peak, task_sort, pes_sort);
    }
  }
}

bool MemoryViolatedCore(const GptoTaskPtr &selected_task, const Time &start_time,
                        const std::map<Time, Memory> &cur_mem_peak, const size_t &task_sort, const size_t &pes_sort) {
  auto start_time_it = cur_mem_peak.lower_bound(start_time);
  for (auto it = start_time_it; it != cur_mem_peak.end(); it++) {
    const auto &time = it->first;
    const auto &mem = it->second;
    if (time < start_time + selected_task->cost()) {
      if (PARAMETER_SIZE + mem + selected_task->initial_mem_impact() > MEMORY_LIMIT) {
        return true;
      }
    } else {  // time >= start_time + selected_task->cost()
      if (PARAMETER_SIZE + mem + selected_task->initial_mem_impact() -
            selected_task->minus_mem_impact(task_sort, pes_sort) - selected_task->workspace_memory() >
          MEMORY_LIMIT) {
        return true;
      }
    }
  }
  if (cur_mem_peak.find(start_time) == cur_mem_peak.end()) {
    if (start_time_it != cur_mem_peak.begin() && start_time_it != cur_mem_peak.end()) {
      if (PARAMETER_SIZE + std::prev(start_time_it)->second + selected_task->initial_mem_impact() > MEMORY_LIMIT) {
        return true;
      }
    }
  }
  if (cur_mem_peak.find(start_time + selected_task->cost()) == cur_mem_peak.end()) {
    auto end_time_it = cur_mem_peak.lower_bound(start_time + selected_task->cost());
    if (end_time_it != cur_mem_peak.begin() && end_time_it != cur_mem_peak.end()) {
      if (PARAMETER_SIZE + std::prev(end_time_it)->second + selected_task->initial_mem_impact() -
            selected_task->minus_mem_impact(task_sort, pes_sort) - selected_task->workspace_memory() >
          MEMORY_LIMIT) {
        return true;
      }
    }
  }
  return false;
}

bool VerifyAll(const std::vector<GptoTaskPtr> &tasks, std::map<Time, Memory> *mem_peak, const size_t &task_sort,
               const size_t &pes_sort) {
  if (VerifyScheduling(tasks, task_sort, pes_sort)) {
    std::ostringstream success_msg;
    success_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort]
                << "] Verification of Scheduling: SUCCESS";
    MS_LOG(INFO) << success_msg.str();
  } else {
    std::ostringstream failure_msg;
    failure_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort]
                << "] Verification of Scheduling: FAILURE";
    MS_LOG(ERROR) << failure_msg.str();
    return false;
  }

  if (VerifyMemory(tasks, mem_peak, task_sort, pes_sort)) {
    std::ostringstream success_msg;
    success_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort]
                << "] Verification of Memory: SUCCESS";
    MS_LOG(INFO) << success_msg.str();
  } else {
    std::ostringstream failure_msg;
    failure_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort]
                << "] Verification of Memory: FAILURE";
    MS_LOG(ERROR) << failure_msg.str();
    return false;
  }

  if ((task_sort >= static_cast<size_t>(kSortBySValue) && task_sort <= static_cast<size_t>(kSortByCostMValue))) {
    if (VERIFY_SAM[TASK_SORT_NAMES[task_sort]](tasks, task_sort, pes_sort)) {
      std::ostringstream success_msg;
      success_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort]
                  << "] Verification of SAM: SUCCESS";
      MS_LOG(INFO) << success_msg.str();
    } else {
      std::ostringstream failure_msg;
      failure_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort]
                  << "] Verification of SAM: FAILURE";
      MS_LOG(ERROR) << failure_msg.str();
      return false;
    }
  }
  return true;
}

bool VerifyScheduling(const std::vector<GptoTaskPtr> &tasks, const size_t &task_sort, const size_t &pes_sort) {
  bool flag = true;
  for (const auto &task : tasks) {
    if (!flag) {
      break;
    }
    // Check if task is scheduled before its children
    for (auto child = task->children().begin(); child != task->children().end(); ++child) {
      Time constraint =
        task->child_is_data_dep(*child) ? task->end(task_sort, pes_sort) : task->start(task_sort, pes_sort) + 1;
      if (!(task->start(task_sort, pes_sort) < constraint && constraint <= (*child)->start(task_sort, pes_sort) &&
            (*child)->start(task_sort, pes_sort) <
              (*child)->end(task_sort, pes_sort))) {  // assume open-rightpoint intervals and non-zero size
        std::ostringstream violation_msg;
        violation_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort]
                      << "] Verification violation: task " << task->id() << " [" << task->start(task_sort, pes_sort)
                      << "," << constraint << "] (child_is_data_dep: " << task->child_is_data_dep(*child)
                      << ") and task " << (*child)->id() << " [" << (*child)->start(task_sort, pes_sort) << ","
                      << (*child)->end(task_sort, pes_sort) << "]";
        MS_LOG(ERROR) << violation_msg.str();
        flag = false;
        break;
      }
    }
  }
  return flag;
}

bool VerifyDAG(const std::vector<GptoTaskPtr> &tasks) {
  // simple verifier: no directed cycle exists
  std::unordered_map<GptoTaskId, bool> visited;
  std::unordered_map<GptoTaskId, size_t> unprocessed_parents;
  std::deque<GptoTaskPtr> to_visit;
  size_t count_visited = 0;
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
    visited[selected_id] = true;
    count_visited++;
    to_visit.pop_front();
    for (const auto &successor : selected_task->children()) {
      const auto &succ_id = successor->id();
      unprocessed_parents[succ_id] -= 1;
      if (unprocessed_parents[succ_id] == 0) {
        to_visit.push_back(successor);
      }
    }
  }
  if (count_visited < tasks.size()) {
    MS_LOG(INFO) << "Cycle(s) present: " << tasks.size() - count_visited << " tasks remain unvisited";
    for (const auto &[id, v] : visited) {
      if (!v) {
        MS_LOG(INFO) << "Example unvisited task with id " << id;
        break;
      }
    }
    MS_LOG(INFO) << "End Verification of DAG";
    visited.clear();
    unprocessed_parents.clear();
    return false;
  }
  MS_LOG(INFO) << "End Verification of DAG";
  visited.clear();
  unprocessed_parents.clear();
  return true;
}

bool VerifyMemory(const std::vector<GptoTaskPtr> &tasks, std::map<Time, Memory> *mem_peak, const size_t &task_sort,
                  const size_t &pes_sort) {
  bool verified = true;
  std::map<Time, Memory> verify_peak;
  std::vector<GptoTensorPtr> graph_output_tensors;
  Memory graph_output_mem = 0;

  for (const auto &time_mem : *mem_peak) {
    verify_peak[time_mem.first] = 0;
  }

  // Recompute peaks based on tensor lifetimes
  for (auto &task : tasks) {
    std::vector<GptoTensorPtr> tensors;
    tensors.reserve(task->out_tensors().size() + task->workspace_tensors().size());
    std::copy(task->out_tensors().begin(), task->out_tensors().end(), std::back_inserter(tensors));
    std::copy(task->workspace_tensors().begin(), task->workspace_tensors().end(), std::back_inserter(tensors));
    for (auto &tensor : tensors) {
      const auto &start = tensor->source().lock()->start(task_sort, pes_sort);
      Time end;
      if (tensor->consumers().size() > 0) {
        end = tensor->lifetime_end(task_sort, pes_sort);
      } else {
        end = tensor->source().lock()->end(task_sort, pes_sort);
      }
      if (tensor->type() == kGraphOutput) {
        graph_output_mem += tensor->weight();
        graph_output_tensors.push_back(tensor);
        continue;
      }
      for (auto it = mem_peak->lower_bound(start); it != mem_peak->end() && it->first < end; it++) {
        verify_peak[it->first] += tensor->weight();
      }
    }
  }

  // Special care for semi-lifelong end graph output tensors
  for (auto &tensor : graph_output_tensors) {
    const auto &start = tensor->source().lock()->start(task_sort, pes_sort);
    for (auto it = mem_peak->lower_bound(start); it != mem_peak->end(); it++) {
      verify_peak[it->first] += tensor->weight();
    }
  }
  if (verify_peak.rbegin()->second != graph_output_mem) {
    std::ostringstream violation_msg;
    violation_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort] << "] Makespan time "
                  << verify_peak.rbegin()->first << " verify peak memory " << verify_peak.rbegin()->second
                  << " kGraphOutput memory " << graph_output_mem;
    MS_LOG(ERROR) << violation_msg.str();
    verified = false;
  }
  verify_peak.rbegin()->second = 0;

  // Compare to originally saved peaks
  auto it = std::find_if(mem_peak->begin(), mem_peak->end(), [&verify_peak](const auto &time_mem) {
    return time_mem.second != verify_peak[time_mem.first];
  });
  if (it != mem_peak->end()) {
    std::ostringstream violation_msg;
    violation_msg << "[" << TASK_SORT_NAMES[task_sort] << "," << PE_NAME_SORT[pes_sort] << "] Time " << it->first
                  << " peak " << it->second << " verify peak " << verify_peak[it->first];
    MS_LOG(ERROR) << violation_msg.str();
    verified = false;
  }

  verify_peak.clear();
  graph_output_tensors.clear();
  return verified;
}

void PrintParameterInfo(const std::vector<Interval> &intervals, std::ostringstream *oss_ptr) {
  std::unordered_map<AnfNodePtr, std::pair<std::vector<GptoTaskPtr>, std::vector<size_t>>> consumers_inputs;
  for (const auto &interval : intervals) {
    for (const auto &param_input : interval.task->in_params()) {
      const auto &param = param_input.first;
      const auto &in = param_input.second;
      consumers_inputs[param].first.push_back(interval.task);
      consumers_inputs[param].second.push_back(in);
    }
  }

  size_t param_count = 0;
  for (const auto &param_info : consumers_inputs) {
    const auto &param = param_info.first;
    const auto &consumers = param_info.second.first;
    const auto &inputs = param_info.second.second;

    std::string consumers_string = std::accumulate(
      consumers.begin(), consumers.end(), std::string{},
      [](std::string str, const auto &consumer) { return str += std::to_string(consumer->id()) + ";"; });

    std::string consumers_inputs_string =
      std::accumulate(inputs.begin(), inputs.end(), std::string{},
                      [](std::string str, const auto &in) { return str += std::to_string(in) + ";"; });

    const auto &shape = common::AnfAlgo::GetPrevNodeOutputInferShape(consumers[0]->cnode(), inputs[0] - 1);
    std::string shape_string;
    if (!shape.empty()) {
      shape_string = std::accumulate(
        shape.cbegin(), shape.cend(), std::string{},
        [](std::string shape_str, const auto &shape_i) { return shape_str += std::to_string(shape_i) + ";"; });
    } else {
      shape_string = "1;";
    }
    const auto &dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(consumers[0]->cnode(), inputs[0] - 1);
    Memory param_weight = static_cast<Memory>(0);
    if (inputs[0] - 1 >= AnfAlgo::GetNodeInputSizeList(consumers[0]->cnode()).size()) {
      MS_LOG(DEBUG) << "Node: " << consumers[0]->cnode()->fullname_with_scope() << " input idx: " << inputs[0] - 1
                    << " greater than the size of input_size_list: "
                    << AnfAlgo::GetNodeInputSizeList(consumers[0]->cnode()).size() << ", so use 0 as parameter size.";
    } else {
      param_weight =
        static_cast<Memory>(GetAlignedSize(AnfAlgo::GetNodeInputSizeList(consumers[0]->cnode()).at(inputs[0] - 1)));
    }

    std::string val = "";
    if (param->isa<ValueNode>() && dtype == kNumberTypeBool) {
      val = std::to_string(GetValue<bool>(param->cast<ValueNodePtr>()->value()));
    }

    *oss_ptr << "PARAM id=" << std::to_string(param_count++) << ", name=" << param->UniqueName()
             << ", weight=" << std::to_string(param_weight) << ", dtype=" << TypeIdToString(dtype)
             << ", shape=" << shape_string << ", consumers=" << consumers_string << ", in=" << consumers_inputs_string
             << ", val=" << val << "\n";
  }
  consumers_inputs.clear();
}

void LogSchedulingOutput(const SchedulingOutput &output, const KernelGraphPtr &kernel_graph,
                         const std::set<GptoTensorPtr, GptoTensorIdSort> &tensors, const Memory memory_lower_bound,
                         std::unordered_set<uint32_t> *stream_ids, const size_t &lower_makespan,
                         const std::string &suffix = "") {
  const size_t &graph_id = kernel_graph->graph_id();
  std::stringstream ss;
  std::ostringstream oss;
  std::string filename;
  ss << kernel_graph;
  filename = GetSaveGraphsPathName(std::string("/") + std::string("gpto_out_") + std::to_string(graph_id) +
                                   std::string("_") + ss.str() + suffix + std::string(".log"));
  // Print info for tasks
  const auto &intervals = output.task_times;
  for (const auto &interval : intervals) {
    oss << "TASK id=" << std::to_string(interval.task->id()) << ", name=" << interval.task->name()
        << ", type=" << std::to_string(interval.task->gpto_type()) << ", cost=" << std::to_string(interval.task->cost())
        << ", start=" << std::to_string(interval.start) << ", end=" << std::to_string(interval.end)
        << ", pe=" << std::to_string(interval.pid) << ", subgraph=" << std::to_string(interval.task->subgraph_id())
        << ", subgraph_parent=" << std::to_string(interval.task->subgraph_id_parent()) << "\n";
  }
  // Print events
  for (const auto &interval : intervals) {
    const auto &dst_id = interval.task->id();
    for (const auto &source : interval.task->recv_events()) {
      oss << "EVENT " << std::to_string(source.lock()->id()) << " " << std::to_string(dst_id) << "\n";
    }
  }

  // Print makespan and memory bounds
  oss << "UPPER " << output.makespan << "\n";
  oss << "LOWER " << lower_makespan << "\n";
  oss << "MEMORY_LIMIT " << MEMORY_LIMIT << "\n";
  oss << "PARAMETER_SIZE " << PARAMETER_SIZE << "\n";
  oss << "MEMORY LOWER BOUND " << memory_lower_bound << "\n";

  // Print edges
  for (const auto &interval : intervals) {
    for (const auto &child : interval.task->children()) {
      oss << "EDGE " << std::to_string(interval.task->id()) << " " << std::to_string(child->id()) << " "
          << interval.task->child_is_data_dep(child) << "\n";
    }
  }

  // Print tensor info
  for (const auto &tensor : tensors) {
    std::string shape{};
    std::string dtype{};
    if (tensor->type() != kWorkspace) {
      if (tensor->weight() > 0 && tensor->shape().size() > 0) {
        shape = std::accumulate(
          tensor->shape().cbegin(), tensor->shape().cend(), std::string{},
          [](std::string shape_str, const auto &shape_i) { return shape_str += std::to_string(shape_i) + ";"; });
      } else {
        shape = "1;";
      }
      dtype = TypeIdToString(tensor->dtype());
    }
    std::string consumers = std::accumulate(tensor->consumers().begin(), tensor->consumers().end(), std::string{},
                                            [](std::string consumers_str, const auto &consumer) {
                                              return consumers_str += std::to_string(consumer.lock()->id()) + ";";
                                            });
    std::string consumers_inputs =
      std::accumulate(tensor->consumers().begin(), tensor->consumers().end(), std::string{},
                      [&tensor](std::string consumers_str, const auto &consumer) {
                        return consumers_str += std::to_string(tensor->consumers_inputs()[consumer]) + ";";
                      });

    oss << "TENSOR id=" << std::to_string(tensor->id()) << ", weight=" << std::to_string(tensor->weight())
        << ", source=" << std::to_string(tensor->source().lock()->id()) << ", type=" << std::to_string(tensor->type())
        << ", dtype=" << dtype << ", shape=" << shape << ", consumers=" << consumers << ", in=" << consumers_inputs
        << "\n";
  }

  PrintParameterInfo(intervals, &oss);

  (void)Common::SaveStringToFile(filename, oss.str());
}

void ComputeAncestorsDescendants(const std::vector<GptoTaskPtr> &tasks,
                                 std::vector<mindspore::somas::DynamicBitSet> *nodes_dependency) {
  // Assume tasks are sorted in vector such that a task appears after all its parents
  // A task is NOT ancestor/descendant of itself
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
      Union(&elem, &((*nodes_dependency)[parent.lock()->id()]));
    }
    // Log message just for debugging
    MS_LOG(DEBUG) << "Task " << task->id() << " has " << (*nodes_dependency)[task->id()].CountOnesNum() << "ancestors";
  }
}

void InsertEdges(const KernelGraphPtr &kernel_graph, std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);

  const std::list<CNodePtr> &cnode_list = kernel_graph->GetOrderedCnodes();
  auto &cnode_to_task = *cnode_to_task_map_ptr;
  size_t num_GptoTask = cnode_to_task.size();
  std::unordered_map<CNodePtr, mindspore::somas::DynamicBitSet> real_children;
  std::unordered_map<size_t, GptoTaskPtr> GptoTask_id_to_GptoTaskPtr;

  std::queue<CNodePtr> visit_queue;
  std::unordered_map<CNodePtr, size_t> unprocessed_children;

  // Initialization loops
  for (const auto &cnode : cnode_list) {
    unprocessed_children[cnode] = 0;
    real_children.emplace(cnode, mindspore::somas::DynamicBitSet(num_GptoTask));
  }
  for (const auto &cnode : cnode_list) {
    for (size_t j = 0; j < cnode->size(); ++j) {
      if (!(cnode->input(j)->isa<CNode>())) {
        continue;
      }
      const auto &input_node = cnode->input(j)->cast<CNodePtr>();
      unprocessed_children[input_node] = unprocessed_children[input_node] + 1;
    }
  }
  for (const auto &cnode : cnode_list) {
    if (unprocessed_children[cnode] == 0) {
      visit_queue.push(cnode);
    }
  }
  for (const auto &it : cnode_to_task) {
    const auto &task = it.second;
    GptoTask_id_to_GptoTaskPtr[task->id()] = task;
  }

  // CNode graph traversal loop
  while (!visit_queue.empty()) {
    const auto &visit_cnode = visit_queue.front();
    MS_LOG(DEBUG) << "Visit cnode " << visit_cnode->UniqueName();
    auto it = cnode_to_task.find(visit_cnode);
    auto &children = real_children.at(visit_cnode);
    if (it != cnode_to_task.end()) {  // if real kernel, then add edges
      auto &visit_task = it->second;
      mindspore::somas::DynamicBitSet all_children(num_GptoTask);
      auto set_bits = children.find_all();
      for (const auto idx : set_bits) {
        const auto &real_child = GptoTask_id_to_GptoTaskPtr[idx];
        if (real_child->condition_switch() || real_child->condition_gather() || !all_children.IsBitTrue(idx)) {
          visit_task->AddChild(real_child);
          real_child->AddParent(visit_task);
          visit_task->set_child_is_data_dep(real_child, false);  // initially all assumed as control edge
          MS_LOG(DEBUG) << "Edge " << visit_task->id() << " " << real_child->id();
          MS_LOG(DEBUG) << "Edge (UniqueName)" << visit_cnode->UniqueName() << " " << real_child->cnode()->UniqueName();
        }
        for (const auto &child : real_child->children()) {
          all_children.SetBitTrue(child->id());
        }
      }
      children.Reset(0x0);
      children.SetBitTrue(visit_task->id());
    }
    // Maintain real_children and visit_queue
    for (size_t j = 1; j < visit_cnode->size(); ++j) {
      if (!visit_cnode->input(j)->isa<CNode>()) {
        continue;
      }
      const auto &parent_cnode = visit_cnode->input(j)->cast<CNodePtr>();
      auto &parent_bitset = real_children.at(parent_cnode);
      Union(&parent_bitset, &children);
      unprocessed_children[parent_cnode]--;
      if (unprocessed_children[parent_cnode] == 0) {
        visit_queue.push(parent_cnode);
      }
    }
    visit_queue.pop();
  }
  unprocessed_children.clear();
  real_children.clear();
}

GptoTaskType GetType(const CNodePtr cnode, std::unordered_set<uint32_t> *stream_ids) {
  if (common::AnfAlgo::HasNodeAttr(kAttrStreamId, cnode)) {
    const auto prim = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    const auto stream_id_attr = prim->GetAttr(kAttrStreamId);
    if (stream_id_attr == nullptr) {
      MS_LOG(EXCEPTION) << "stream_id is nullptr, node: " << cnode->fullname_with_scope();
    }
    uint32_t stream_id = common::AnfAlgo::GetNodeAttr<uint32_t>(cnode, kAttrStreamId);
    stream_ids->insert(stream_id);
    return stream_id;
  } else {
    stream_ids->insert(0);
    MS_LOG(DEBUG) << "Stream id value will be 0 for node: " << cnode->fullname_with_scope();
    return 0;
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

std::tuple<std::unordered_map<std::string, std::vector<Memory>>, std::unordered_map<std::string, std::vector<Memory>>>
GetTensorsWeight() {
  std::unordered_map<std::string, std::vector<Memory>> offline_out_tensors_map;
  std::unordered_map<std::string, std::vector<Memory>> offline_work_tensors_map;
  std::string gpto_tensors_file;

  const auto &backend_jit_config = backend::BackendJitConfig::ParseBackendJitConfig();
  auto it = backend_jit_config.gpto_options.find("dynamic_tensor_info");
  if (it != backend_jit_config.gpto_options.end()) {
    gpto_tensors_file = it->second;
  }
  if (gpto_tensors_file.empty()) {
    return {offline_out_tensors_map, offline_work_tensors_map};
  }

  std::ifstream file(gpto_tensors_file);
  if (!file) {
    MS_LOG(WARNING) << "Unable to open dynamic tensor information file: " << gpto_tensors_file;
    return {offline_out_tensors_map, offline_work_tensors_map};
  }
  MS_LOG(INFO) << "Opening dynamic tensor information file: " << gpto_tensors_file;

  auto parse_numbers = [](const std::string &s) -> std::vector<size_t> {
    std::vector<size_t> nums;
    std::istringstream num_streams(s);
    size_t num;
    while (num_streams >> num) {
      nums.push_back(num);
    }
    return nums;
  };

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    std::string op_name, out_tensors_str, work_tensors_str;
    if (!std::getline(iss, op_name, ',') || !std::getline(iss, out_tensors_str, ',') ||
        !std::getline(iss, work_tensors_str, ',')) {
      MS_LOG(WARNING) << "Malformed line (expected 3 fields): " << line;
      continue;
    }
    // Parse output tensors
    auto out_tensors = parse_numbers(out_tensors_str);
    if (!out_tensors.empty()) {
      offline_out_tensors_map.emplace(op_name, std::move(out_tensors));
    }
    // Parse workspace tensors
    auto work_tensors = parse_numbers(work_tensors_str);
    if (!work_tensors.empty()) {
      offline_work_tensors_map.emplace(op_name, std::move(work_tensors));
    }
  }
  return {offline_out_tensors_map, offline_work_tensors_map};
}

void ExtractOutputWorkspaceTensors(const std::vector<GptoTaskPtr> &tasks) {
  size_t tensor_count = 0;

  // Fill output and workspace tensor maps to handle dynamic shape operators
  auto [offline_out_tensors_map, offline_work_tensors_map] = GetTensorsWeight();

  // Looping over tasks to obtain output and workspace tensors (somas style)
  for (auto &task : tasks) {
    const auto &kernel_mod = AnfAlgo::GetKernelMod(task->cnode());
    MS_EXCEPTION_IF_NULL(kernel_mod);
    MS_LOG(DEBUG) << "Current task: " << task->name();

    bool is_dynamic = common::AnfAlgo::IsDynamicShape(task->cnode());
    std::vector<size_t> out_sizes;
    std::vector<size_t> work_sizes;

    if (is_dynamic) {
      if (offline_out_tensors_map.find(task->name()) != offline_out_tensors_map.end()) {
        out_sizes = offline_out_tensors_map[task->name()];
      } else {
        MS_LOG(DEBUG) << "Did not find output tensor for node: " << task->name();
      }
      if (offline_work_tensors_map.find(task->name()) != offline_work_tensors_map.end()) {
        work_sizes = offline_work_tensors_map[task->name()];
      } else {
        MS_LOG(DEBUG) << "Did not find workspace tensor for node: " << task->name();
      }
    } else {
      out_sizes = kernel_mod->GetOutputSizeList();
      work_sizes = kernel_mod->GetWorkspaceSizeList();
    }

    task->out_tensors().reserve(out_sizes.size());
    size_t count_out = 0;
    for (const auto &size : out_sizes) {
      Memory weight;
      if (is_dynamic) {
        weight = size;
      } else {
        weight = GetAlignedSize(size);
        if (weight == 0) {
          weight = GetAlignedSize(1);
        }
      }
      const ShapeVector &shape = common::AnfAlgo::GetOutputInferShape(task->cnode(), count_out);
      const TypeId &dtype = common::AnfAlgo::GetOutputInferDataType(task->cnode(), count_out);
      GptoTensorPtr new_tensor = std::make_shared<GptoTensor>(tensor_count, size, weight, task, kWorkspace, shape,
                                                              dtype);  // initially kWorkspace, since no consumers
      task->out_tensors().push_back(new_tensor);
      MS_LOG(DEBUG) << "New output tensor " << tensor_count << " source id " << task->id() << " weight " << weight
                    << " - " << task->name() << " - is dynamic: " << is_dynamic;
      tensor_count++;
      count_out++;
    }
    out_sizes.clear();

    task->workspace_tensors().reserve(work_sizes.size());
    for (const auto &size : work_sizes) {
      Memory weight;
      if (is_dynamic) {
        weight = size;
      } else {
        weight = GetAlignedSize(size);
        if (weight == 0) {
          weight = GetAlignedSize(1);
        }
      }
      GptoTensorPtr new_tensor = std::make_shared<GptoTensor>(tensor_count, size, weight, task, kWorkspace);
      task->workspace_tensors().push_back(new_tensor);
      MS_LOG(DEBUG) << "New workspace tensor " << tensor_count << " source id " << task->id() << " weight " << weight
                    << " - " << task->name() << " - is dynamic: " << is_dynamic;
      tensor_count++;
    }
    work_sizes.clear();
  }
  offline_out_tensors_map.clear();
  offline_work_tensors_map.clear();
  MAX_TENSOR_ID = tensor_count - 1;
}

void CleanWorkspace(CNodePtr pre_node, const GptoTaskPtr &pre_task, const GptoTaskPtr &task) {
  auto clean_workspace_indexs = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicWorkspaceIndexs);
  for (const auto &index : clean_workspace_indexs) {
    if (index > pre_task->out_tensors().size()) {
      MS_LOG(INFO) << "Workspace index " << index << " exceed input node [" << pre_node->fullname_with_scope()
                   << "]'s Workspace size " << pre_task->workspace_tensors().size();
      continue;
    }
    auto &input_tensor = pre_task->workspace_tensors()[index];
    MS_EXCEPTION_IF_NULL(input_tensor);
    task->in_tensors().insert(input_tensor);
    if (input_tensor->type() == GptoTensorType::kWorkspace) {
      input_tensor->set_type(GptoTensorType::kSimple);
    }
    input_tensor->consumers().insert(task);
    pre_task->AddChild(task);
    task->AddParent(pre_task);
    pre_task->set_child_is_data_dep(task, true);
  }
}

void CleanOutput(size_t index, CNodePtr pre_node, GptoTaskPtr pre_task, const GptoTaskPtr &task) {
  if (index > pre_task->out_tensors().size()) {
    MS_LOG(INFO) << "Output index " << index << " exceed input node [" << pre_node->fullname_with_scope()
                 << "]'s outputs size " << pre_task->out_tensors().size();
    return;
  }
  auto &input_tensor = pre_task->out_tensors()[index];
  MS_EXCEPTION_IF_NULL(input_tensor);
  task->in_tensors().insert(input_tensor);
  if (input_tensor->type() == GptoTensorType::kWorkspace) {
    input_tensor->set_type(GptoTensorType::kSimple);
  }
  input_tensor->consumers().insert(task);
  pre_task->AddChild(task);
  task->AddParent(pre_task);
  pre_task->set_child_is_data_dep(task, true);
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

    if (!AnfUtils::IsRealCNodeKernel(prenode_index.first)) {  // input parameter case
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
        input_size = GetAlignedSize(input_size_list.at(i));
      }
      if (parameter_set->find(prenode_index.first.get()) == parameter_set->end()) {
        parameter_set->insert(prenode_index.first.get());
        PARAMETER_SIZE += input_size;
      }
      task->in_params()[prenode_index.first] = i + 1;  // save parameters for log printing
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
    auto &input_tensor = pre_task->out_tensors()[prenode_index.second];
    MS_EXCEPTION_IF_NULL(input_tensor);
    input_tensor->consumers().insert(task);
    input_tensor->consumers_inputs()[task] = i + 1;
    task->in_tensors().insert(input_tensor);
    pre_task->AddChild(task);
    task->AddParent(pre_task);
    pre_task->set_child_is_data_dep(task, true);
    MS_LOG(DEBUG) << "GptoTensor " << input_tensor->id() << " has new consumer " << task->id() << " "
                  << task->cnode()->fullname_with_scope();
    if (input_tensor->type() == GptoTensorType::kWorkspace) {
      input_tensor->set_type(GptoTensorType::kSimple);
    }
  }
}

void ExtractRealTensors(const SchedulingInput &scheduling_input,
                        std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_gpto_map_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_gpto_map_ptr);
  const auto &tasks = scheduling_input.tasks;
  ExtractOutputWorkspaceTensors(tasks);

  // Looping over tasks to obtain input tensors after all outputs have been saved
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

        const auto &iter = cnode_to_task_gpto_map_ptr->find(pre_node);
        if (iter == cnode_to_task_gpto_map_ptr->end()) {
          MS_LOG(DEBUG) << "Kernel[" << kernel->fullname_with_scope() << "]'s input " << i << " ["
                        << pre_node->fullname_with_scope() << "] not found in tasks";
          continue;
        }
        const auto &pre_task = iter->second;

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

std::pair<Time, Memory> LogBaseline(const std::unordered_map<CNodePtr, GptoTaskPtr> &cnode_to_task_gpto_map,
                                    const KernelGraphPtr &kernel_graph, bool print_log_file) {
  const size_t &graph_id = kernel_graph->graph_id();
  std::ostringstream oss;
  std::string filename;

  if (print_log_file) {
    std::stringstream ss;
    ss << kernel_graph;
    filename = GetSaveGraphsPathName(std::string("/") + std::string("gpto_baseline_") + std::to_string(graph_id) +
                                     std::string("_") + ss.str() + std::string(".log"));
  }

  std::unordered_map<GptoTaskId, Time> taskid_to_end_value;
  std::unordered_map<GptoTaskId, Time> taskid_to_start_value;
  size_t makespan = 0;
  const std::vector<CNodePtr> &execution_order = kernel_graph->execution_order();
  for (size_t i = 0; i < execution_order.size(); i++) {
    const auto &cnode = execution_order[i];
    GptoTaskPtr current_task = cnode_to_task_gpto_map.at(cnode);

    // Find the latest executed task which has the same type as the current task
    GptoTaskPtr last_task = nullptr;
    MS_LOG(DEBUG) << "Current value loop: " << i << " with node: " << execution_order[i]->UniqueName();
    for (int j = i - 1; j >= 0; j--) {
      MS_LOG(DEBUG) << "Current value loop j: " << j;
      GptoTaskPtr tmp_task = cnode_to_task_gpto_map.at(execution_order[j]);
      MS_LOG(DEBUG) << "With node: " << tmp_task->name();
      if (tmp_task->gpto_type() == current_task->gpto_type()) {
        MS_LOG(DEBUG) << "Found node same type";
        last_task = tmp_task;
        break;
      }
    }

    // Find the latest parent of the current task
    Time can_start = 0;
    if (last_task != nullptr) {
      can_start = taskid_to_end_value[last_task->id()];
    }

    for (const auto &parent : cnode_to_task_gpto_map.at(cnode)->parents()) {
      if (parent.lock()->child_is_data_dep(current_task) && taskid_to_end_value[parent.lock()->id()] > can_start) {
        can_start = taskid_to_end_value[parent.lock()->id()];
        MS_LOG(DEBUG) << "Found parent (data dep) to update can_start " << parent.lock()->name();
      } else {
        if (!parent.lock()->child_is_data_dep(current_task) &&
            taskid_to_start_value[parent.lock()->id()] + 1 > can_start) {
          can_start = taskid_to_start_value[parent.lock()->id()] + 1;
          MS_LOG(DEBUG) << "Found parent (control dep) to update can_start " << parent.lock()->name();
        }
      }
    }

    taskid_to_start_value[current_task->id()] = can_start;
    taskid_to_end_value[current_task->id()] = can_start + current_task->cost();

    const size_t &current_task_end = taskid_to_end_value[current_task->id()];
    if (current_task_end > makespan) {
      makespan = taskid_to_end_value[current_task->id()];
    }

    if (print_log_file) {
      oss << "TASK id=" << std::to_string(current_task->id()) << ", name=" << current_task->name()
          << ", type=" << std::to_string(current_task->gpto_type()) << ", cost=" << std::to_string(current_task->cost())
          << ", start=" << std::to_string(taskid_to_start_value[current_task->id()])
          << ", end=" << std::to_string(current_task_end) << "\n";
    }
  }
  MS_LOG(INFO) << "Makespan estimate of baseline is " + std::to_string(makespan);

  if (print_log_file) {
    (void)Common::SaveStringToFile(filename, oss.str());
  }

  Memory memory =
    MemoryEstimateBaseline(execution_order, cnode_to_task_gpto_map, &taskid_to_start_value, &taskid_to_end_value);
  taskid_to_start_value.clear();
  taskid_to_end_value.clear();
  return std::make_pair(makespan, memory);
}

Memory MemoryEstimateBaseline(const std::vector<CNodePtr> &execution_order,
                              const std::unordered_map<CNodePtr, GptoTaskPtr> &cnode_to_task_gpto_map,
                              std::unordered_map<GptoTaskId, Time> *taskid_to_start_value_ptr,
                              std::unordered_map<GptoTaskId, Time> *taskid_to_end_value_ptr) {
  std::map<Time, Memory> mem_peak;
  Memory graph_output_mem = 0;
  for (size_t i = 0; i < execution_order.size(); i++) {
    const auto &cnode = execution_order[i];
    GptoTaskPtr task = cnode_to_task_gpto_map.at(cnode);
    mem_peak[(*taskid_to_start_value_ptr)[task->id()]] = 0;
    mem_peak[(*taskid_to_end_value_ptr)[task->id()]] = 0;
  }

  for (size_t i = 0; i < execution_order.size(); i++) {
    const auto &cnode = execution_order[i];
    const GptoTaskPtr task = cnode_to_task_gpto_map.at(cnode);
    std::vector<GptoTensorPtr> tensors;
    tensors.reserve(task->out_tensors().size() + task->workspace_tensors().size());
    std::copy(task->out_tensors().begin(), task->out_tensors().end(), std::back_inserter(tensors));
    std::copy(task->workspace_tensors().begin(), task->workspace_tensors().end(), std::back_inserter(tensors));
    for (auto &tensor : tensors) {
      const auto &start = (*taskid_to_start_value_ptr)[task->id()];
      Time end = 0;
      if (tensor->consumers().size() > 0) {
        for (auto &consumer : tensor->consumers()) {
          end = std::max(end, (*taskid_to_end_value_ptr)[consumer.lock()->id()]);
        }
      } else {
        end = (*taskid_to_end_value_ptr)[task->id()];
      }
      if (tensor->type() == kGraphOutput) {
        graph_output_mem += tensor->weight();
      }
      for (const auto &time_mem : mem_peak) {
        if ((start <= time_mem.first && time_mem.first < end && tensor->type() != kGraphOutput) ||
            (start <= time_mem.first && tensor->type() == kGraphOutput)) {
          mem_peak[time_mem.first] += tensor->weight();
        }
      }
    }
  }
  if (!mem_peak.empty()) {
    mem_peak.rbegin()->second -= graph_output_mem;

    if (mem_peak.rbegin()->second > 0) {
      MS_LOG(WARNING) << "Memory peak " << mem_peak.rbegin()->second << " at makespan time "
                      << mem_peak.rbegin()->first;
    }
  }

  Memory memory = std::max_element(mem_peak.begin(), mem_peak.end(),
                                   [](const std::pair<Time, Memory> &a, const std::pair<Time, Memory> &b) -> bool {
                                     return a.second < b.second;
                                   })
                    ->second;
  MS_LOG(INFO) << "Memory estimate of baseline is " << memory;
  mem_peak.clear();
  return memory;
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

void PushTasksToVisit(std::queue<std::weak_ptr<GptoTask>> *tasks_to_visit,
                      std::unordered_map<size_t, size_t> *unprocessed_children, const std::weak_ptr<GptoTask> &parent,
                      GptoTaskPtr switch_task, const size_t &count_condition) {
  (*unprocessed_children)[parent.lock()->id()] -= 1;
  if ((*unprocessed_children)[parent.lock()->id()] == 0) {
    if (parent.lock() != switch_task) {
      tasks_to_visit->push(parent);
    } else {
      if (switch_task->subgraph_id() == count_condition) {
        switch_task->set_subgraph_id(count_condition + 1);
        MS_LOG(DEBUG) << "Assign subgraph id " << count_condition << " to task " << switch_task->id() << " name "
                      << switch_task->name();
      }
    }
  }
}

void ExtractSwitchGather(std::map<GptoTaskPtr, GptoTaskPtr, TaskDepthSort> *switch_gather,
                         std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> *switch_attribute_ids,
                         std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> *gather_attribute_ids) {
  MS_EXCEPTION_IF_NULL(switch_gather);
  MS_EXCEPTION_IF_NULL(switch_attribute_ids);
  MS_EXCEPTION_IF_NULL(gather_attribute_ids);
  for (const auto &switch_it : *switch_attribute_ids) {
    const auto &switch_task = switch_it.first;
    const auto &switch_pair = switch_it.second;
    std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>>::iterator gather_it = std::find_if(
      gather_attribute_ids->begin(), gather_attribute_ids->end(),
      [&switch_pair](const std::pair<GptoTaskPtr, std::pair<size_t, size_t>> &it) { return it.second == switch_pair; });
    if (gather_it == gather_attribute_ids->end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Could not find matching ConditionGather for a given ConditionSwitch "
                                 << switch_pair;
    }
    const auto &gather_task = gather_it->first;
    (*switch_gather)[switch_task] = gather_task;
    MS_LOG(DEBUG) << "Mapped ConditionSwitch task " << switch_task->id() << " to ConditionGather task "
                  << gather_task->id();
  }
}

void UpdateTasksInlineCondition(std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr,
                                std::map<GptoTaskPtr, GptoTaskPtr, TaskDepthSort> *switch_gather) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);
  MS_EXCEPTION_IF_NULL(switch_gather);

  size_t count_condition = SIZE_MAX - 1;
  std::unordered_map<size_t, size_t> unprocessed_children;
  std::queue<std::weak_ptr<GptoTask>> tasks_to_visit;

  for (const auto &key_val : *cnode_to_task_map_ptr) {
    auto &task = key_val.second;
    unprocessed_children[task->id()] = task->children().size();
  }

  for (const auto &it : (*switch_gather)) {
    const auto &switch_task = it.first;
    const auto &gather_task = it.second;
    MS_LOG(INFO) << "Assign subgraph id " << count_condition << " to tasks under ConditionSwitch task "
                 << switch_task->id() << " name " << switch_task->name()
                 << " up to (and including) ConditionGather task " << gather_task->id() << " name "
                 << gather_task->name();
    gather_task->set_subgraph_id(count_condition);

    for (auto parent : gather_task->parents()) {
      if (parent.lock() == switch_task) {
        parent.lock()->set_subgraph_id(count_condition + 1);
        MS_LOG(INFO) << "Assign subgraph id " << count_condition + 1 << " to task " << switch_task->id() << " name "
                     << switch_task->name();
      } else {
        tasks_to_visit.push(parent);
      }
    }

    while (!tasks_to_visit.empty()) {
      const auto &selected_task = tasks_to_visit.front().lock();
      selected_task->set_subgraph_id(count_condition);

      if (selected_task->name().find("ConditionGather") != std::string::npos) {
        // Get the switch node of the nested gather node
        GptoTaskPtr selected_task_switch =
          std::find_if(switch_gather->begin(), switch_gather->end(), [selected_task](const auto &i) {
            return i.second->name() == selected_task->name();
          })->first;
        // Get all nested switch parents to visit
        for (const auto switch_parent : selected_task_switch->parents()) {
          PushTasksToVisit(&tasks_to_visit, &unprocessed_children, switch_parent, switch_task, count_condition);
        }
      } else if (selected_task->parents().size() == 0) {
        selected_task->AddParent(switch_task);
        switch_task->AddChild(selected_task);
        switch_task->set_child_is_data_dep(selected_task, false);
      } else {
        for (const auto &parent : selected_task->parents()) {
          PushTasksToVisit(&tasks_to_visit, &unprocessed_children, parent, switch_task, count_condition);
        }
      }
      tasks_to_visit.pop();
    }
    gather_task->set_subgraph_id_parent(switch_task->subgraph_id());

    count_condition--;
  }
  unprocessed_children.clear();
}

std::unordered_map<std::string, Time> GetProfCost(const std::string &profile_as_cost_file) {
  std::unordered_map<std::string, Time> profiling_map;

  std::ifstream file(profile_as_cost_file);
  if (file) {
    MS_LOG(INFO) << "Opening profiling cost file: " << profile_as_cost_file;
  } else {
    MS_LOG(WARNING) << "Unable to open profiling cost file: " << profile_as_cost_file;
    return profiling_map;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream s(line);
    std::string key, value;
    if (std::getline(s, key, ',') && std::getline(s, value)) {
      try {
        profiling_map[key] = stoi(value);
      } catch (const std::invalid_argument &e) {
        MS_LOG(WARNING) << "Invalid integer: " << value << " in line: " << line;
        continue;
      } catch (const std::out_of_range &e) {
        MS_LOG(WARNING) << "Integer out of range: " << value << " in line: " << line;
        continue;
      }
    } else {
      MS_LOG(WARNING) << "Malformed line: " << line;
      continue;
    }
  }
  return profiling_map;
}

SchedulingInput ExtractSchedulingInput(const KernelGraphPtr &kernel_graph,
                                       std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr,
                                       std::unordered_set<uint32_t> *stream_ids) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);
  SchedulingInput scheduling_input;  // to fill in and return
  std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> switch_attribute_ids;
  std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> gather_attribute_ids;
  GptoTaskPtr root_task;
  GptoTaskPtr broadcast_dataset;

  // Create a task per node
  MS_LOG(INFO) << "Start Extract GPTO Tasks";
  const auto &real_kernels = kernel_graph->execution_order();
  scheduling_input.tasks.reserve(real_kernels.size());

  std::unordered_map<std::string, Time> profiling_map;
  const auto &backend_jit_config = kernel_graph->backend_jit_config();
  if (backend_jit_config.gpto_options.find("profile_cost_info") != backend_jit_config.gpto_options.end()) {
    std::string profile_as_cost_file = backend_jit_config.gpto_options.at("profile_cost_info");
    if (profile_as_cost_file != "") {
      profiling_map = GetProfCost(profile_as_cost_file);
    }
  }

  for (size_t i = 0; i < real_kernels.size(); ++i) {
    const auto &cnode = real_kernels[i];
    const std::string cnode_name = AnfUtils::GetCNodeName(cnode);
    GptoTaskPtr task = std::make_shared<GptoTask>(i, GetType(cnode, stream_ids), cnode->fullname_with_scope());
    task->set_cnode(cnode);

    // Find root node of graph
    if (root_task == nullptr && cnode_name == parallel::GET_NEXT) {
      root_task = task;
    }

    if (broadcast_dataset == nullptr && cnode_name == parallel::BROADCAST) {
      const auto &prim = GetCNodePrimitive(cnode);
      if (prim->HasAttr(parallel::DATASET_BROADCAST)) {
        broadcast_dataset = task;
      }
    }

    // Assign priority to task
    if (cnode_name == parallel::SEND) {
      task->set_type_priority(static_cast<size_t>(SIZE_MAX));
    } else if (cnode_name == parallel::RECEIVE) {
      task->set_type_priority(static_cast<size_t>(SIZE_MAX - 1));
    } else {
      task->set_type_priority(static_cast<size_t>(task->gpto_type()));
    }

    // Assign cost to task
    Time cost = 0;
    if (!profiling_map.empty()) {
      std::string node_name(cnode->fullname_with_scope());
      if (profiling_map.find(node_name) != profiling_map.end()) {
        cost = profiling_map[node_name];
      } else {
        MS_LOG(DEBUG) << "Node " << node_name << " not found in profiling file";
      }
    }
    task->AssignCost(cost);

    // Start Step 1 ConditionSwitch/Gather for inline: save attributes
    InitializeTaskInlineCondition(cnode, &task, &switch_attribute_ids, &gather_attribute_ids);
    // End Step 1 ConditionSwitch/Gather for inline

    MS_LOG(DEBUG) << "Task " << task->id() << " with name " << cnode->UniqueName() << " and CNodePtr " << cnode
                  << " with cost " << task->cost() << " and type " << task->gpto_type();
    scheduling_input.tasks.push_back(task);
    (*cnode_to_task_map_ptr)[cnode] = task;
  }
  profiling_map.clear();

  for (const auto &stream_id : *stream_ids) {
    MS_LOG(INFO) << "Get stream id " << stream_id;
  }

  MS_LOG(INFO) << "End Extract GPTO Tasks with " << stream_ids->size() << " streams";

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
  ExtractSwitchGather(&switch_gather, &switch_attribute_ids, &gather_attribute_ids);
  switch_attribute_ids.clear();
  gather_attribute_ids.clear();
  MS_LOG(INFO) << "End Extract GPTO Switch/Gather";
  // End Step 2 ConditionSwitch/Gather for inline

  // Start Step 3 ConditionSwitch/Gather for inline: traverse each Condition/Switch gather block to assign proper ids
  // Assumption 1: switch and nodes before gather have no predecessors/descendants outside the block
  // Assumption 2: conditional switch does not have conditional gather as a child
  MS_LOG(INFO) << "Start Update Inline";
  UpdateTasksInlineCondition(cnode_to_task_map_ptr, &switch_gather);
  switch_gather.clear();
  MS_LOG(INFO) << "End Update Inline";
  // End Step 3 ConditionSwitch/Gather for inline

  MakeGraphRoot(root_task, broadcast_dataset, cnode_to_task_map_ptr);

  return scheduling_input;
}

void MakeGraphRoot(const GptoTaskPtr &root_task, const GptoTaskPtr &broadcast_dataset,
                   std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  if (!root_task && !broadcast_dataset) {
    return;
  }

  // Link orphan node to one of the root task
  for (const auto &cnode_task : *cnode_to_task_map_ptr) {
    const auto &task = cnode_task.second;
    if (task == root_task || task == broadcast_dataset) {
      continue;
    }
    if (!task->parents().empty()) {
      continue;
    }

    if (std::any_of(task->children().begin(), task->children().end(),
                    [&](auto &child) { return child == root_task || child == broadcast_dataset; })) {
      continue;
    }

    if (broadcast_dataset) {  // add edge from broadcast to task
      task->AddParent(broadcast_dataset);
      broadcast_dataset->AddChild(task);
    } else {  // add edge from getnext to task
      task->AddParent(root_task);
      root_task->AddChild(task);
    }
    MS_LOG(DEBUG) << "Attached orphan CNode " << task->name();
  }

  // Re-route GetNext children to Broadcast
  if (broadcast_dataset && root_task) {
    std::vector<GptoTaskPtr> root_task_children(root_task->children().begin(), root_task->children().end());
    for (const auto &child : root_task_children) {
      if (child == broadcast_dataset) {
        continue;
      }

      broadcast_dataset->AddChild(child);
      child->AddParent(broadcast_dataset);
      child->parents().erase(root_task);
      root_task->children().erase(child);
    }
  }

  if (root_task) {
    for (const auto &child : root_task->children()) {
      root_task->set_child_is_data_dep(child, true);
    }
  }

  if (broadcast_dataset) {
    for (const auto &child : broadcast_dataset->children()) {
      broadcast_dataset->set_child_is_data_dep(child, true);
    }
  }
}

// Calculate the lower bound for a single task
Memory CalculateTaskLowerBound(const GptoTaskPtr &task, const std::set<GptoTensorPtr, GptoTensorIdSort> &tensors,
                               const std::vector<mindspore::somas::DynamicBitSet> &nodes_dependency) {
  Memory task_lb = 0;
  for (const auto &tensor : tensors) {
    if (tensor->weight() == 0) {
      continue;
    }
    const auto &source = tensor->source().lock();
    const auto &consumers = tensor->consumers();

    if (task == source || consumers.count(task) > 0) {
      task_lb += tensor->weight();
    } else {
      if (nodes_dependency[task->id()].IsBitTrue(source->id())) {
        if (tensor->type() == kGraphOutput) {
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
  return task_lb;
}

Memory MemoryLowerBound(const std::vector<GptoTaskPtr> &tasks,
                        const std::vector<mindspore::somas::DynamicBitSet> &nodes_dependency,
                        const std::set<GptoTensorPtr, GptoTensorIdSort> &tensors) {
  std::vector<std::atomic<Memory>> tasks_lb(tasks.size());
  size_t count = tasks.back()->id() + 1;
  std::vector<mindspore::somas::DynamicBitSet> suffix_tasks;
  suffix_tasks.reserve(count);
  for (size_t i = 0; i < count; i++) {
    suffix_tasks.emplace_back(count);
  }
  suffix_tasks[count - 1].SetBitTrue(tasks.back()->id());
  for (int j = tasks.size() - 2; j >= 0; j--) {
    suffix_tasks[j] = suffix_tasks[j + 1];
    suffix_tasks[j].SetBitTrue(tasks[j]->id());
  }
  std::vector<mindspore::somas::DynamicBitSet> nodes_prerequisite;
  nodes_prerequisite.reserve(count);
  for (size_t i = 0; i < count; i++) {
    nodes_prerequisite.emplace_back(count);
  }
  for (auto it = tasks.rbegin(); it != tasks.rend(); it++) {
    auto &task = *it;
    for (const auto &child : task->children()) {
      nodes_prerequisite[task->id()].SetBitTrue(child->id());
      Union(&(nodes_prerequisite[task->id()]), &(nodes_prerequisite[child->id()]));
    }
  }

  std::vector<GptoTensorPtr> tensors_vec(tensors.begin(), tensors.end());

  size_t index = 0;
  size_t batch_size = (tensors_vec.size() + 99) / 100;
  std::mutex mtx;
  auto worker = [&tasks, &nodes_dependency, &tasks_lb, &tensors_vec, &index, &mtx, &suffix_tasks, &nodes_prerequisite,
                 batch_size, count]() {
    while (true) {
      mtx.lock();
      if (index >= tensors_vec.size()) {
        mtx.unlock();
        break;
      }
      size_t start_index = index;
      index += batch_size;
      mtx.unlock();
      for (size_t i = start_index; i < std::min(start_index + batch_size, tensors_vec.size()); i++) {
        const auto &tensor = tensors_vec[i];
        if (tensor->weight() == 0) {
          continue;
        }
        const auto &source = tensor->source().lock();
        const auto &consumers = tensor->consumers();
        tasks_lb[source->id()].fetch_add(tensor->weight());
        mindspore::somas::DynamicBitSet tensors_dependency(count);
        for (const auto &consumer : consumers) {
          const auto &consumer_id = consumer.lock()->id();
          Union(&tensors_dependency, &(nodes_dependency[consumer_id]));
          if (consumer_id == source->id()) {
            continue;
          }
          tasks_lb[consumer_id].fetch_add(tensor->weight());
        }
        if (tensor->type() == kWorkspace) {
          continue;
        }
        size_t id = std::lower_bound(tasks.begin(), tasks.end(), source->depth(),
                                     [](const GptoTaskPtr p, size_t target_val) { return p->depth() < target_val; }) -
                    tasks.begin();
        mindspore::somas::DynamicBitSet bitset = suffix_tasks[id];
        And(&bitset, &(nodes_prerequisite[source->id()]));
        bitset.SetBitFalse(source->id());
        for (const auto &consumer : consumers) {
          bitset.SetBitFalse(consumer.lock()->id());
        }
        if (tensor->type() != kGraphOutput) {
          And(&bitset, &tensors_dependency);
        }
        std::vector<size_t> set_bits = bitset.find_all();
        for (const auto idx : set_bits) {
          tasks_lb[idx].fetch_add(tensor->weight());
        }
      }
    }
    return common::SUCCESS;
  };

  try {
    std::vector<common::Task> common_tasks;
    const auto &thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
    for (size_t i = 0; i < thread_num; i++) {
      (void)common_tasks.emplace_back(common::Task(worker));
    }
    (void)common::ThreadPool::GetInstance().SyncRun(common_tasks, GetCoreList());
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "mindspore::gpto::MemoryLowerBound FAILED: " << e.what();
  }
  auto max_lb = std::max_element(tasks_lb.begin(), tasks_lb.end());
  MS_LOG(INFO) << "Memory Lower bound for tensors " << (*max_lb).load() << " (" << (*max_lb).load() / kMBToByte
               << " MB) at task " << std::distance(tasks_lb.begin(), max_lb);
  return (*max_lb).load();
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

namespace {  // somas-like functionality for merging ref node (assuming unions of pairs of tensors originally)
size_t find_father(std::vector<size_t> *father, size_t x) {
  MS_EXCEPTION_IF_NULL(father);
  if (x >= father->size()) {
    MS_LOG(EXCEPTION) << "Index " << x << " out of range " << father->size();
  }
  if (x == (*father)[x]) {
    return x;
  }
  (*father)[x] = find_father(father, (*father)[x]);
  return (*father)[x];
}

std::vector<std::vector<GptoTensorPtr>> GetRegularUnionTensorsList(
  size_t max_tensor, const std::vector<std::pair<GptoTensorPtr, GptoTensorPtr>> &union_tensors) {
  std::vector<size_t> father;
  std::map<size_t, GptoTensorPtr> id_to_tensor;
  for (size_t i = 0; i < max_tensor; i++) {
    father.push_back(i);
  }
  for (auto union_pair : union_tensors) {
    const size_t &tid0 = union_pair.first->id();
    const size_t &tid1 = union_pair.second->id();
    father[find_father(&father, tid1)] = find_father(&father, tid0);
    id_to_tensor[tid0] = union_pair.first;
    id_to_tensor[tid1] = union_pair.second;
  }

  std::map<size_t, size_t> kv;
  std::vector<std::vector<GptoTensorPtr>> ret_union_list;
  std::vector<std::set<size_t>> union_tensor_sets;
  for (const auto &union_pair : union_tensors) {
    std::vector<size_t> tids = {union_pair.first->id(), union_pair.second->id()};
    for (const auto &tid : tids) {
      size_t fa = find_father(&father, tid);
      if (kv.find(fa) == kv.end()) {
        ret_union_list.emplace_back();
        union_tensor_sets.emplace_back();
        kv.emplace(fa, ret_union_list.size() - 1);
      }
      auto &union_tensor_set = union_tensor_sets[kv.at(fa)];
      if (union_tensor_set.find(tid) == union_tensor_set.end()) {
        ret_union_list[kv.at(fa)].push_back(id_to_tensor[tid]);
        union_tensor_set.insert(tid);
      }
    }
  }
  father.clear();
  id_to_tensor.clear();
  return ret_union_list;
}
}  // namespace

std::map<GptoTensorPtr, GptoTensorPtr> GetRefTensorsInContiguousList(
  const std::vector<std::vector<GptoTensorPtr>> &ref_node_vector) {
  std::map<GptoTensorPtr, GptoTensorPtr> ref_tensors_in_contiguous_map;  // key: refnode input, value: refnode output
  constexpr auto kRefNodeTensorNum = 2;
  for (auto ref_list : ref_node_vector) {
    auto contiguous_in_ref_list = std::count_if(ref_list.begin(), ref_list.end(), [](GptoTensorPtr t) {
      MS_EXCEPTION_IF_NULL(t);
      return t->contiguous();
    });
    if (ref_list.size() > kRefNodeTensorNum && contiguous_in_ref_list > 0) {
      MS_LOG(WARNING) << "Ref node of size greater than two with at least one contiguous tensor in";
    }
    if (ref_list.size() == kRefNodeTensorNum && contiguous_in_ref_list == 1) {
      MS_EXCEPTION_IF_NULL(ref_list[0]);
      MS_EXCEPTION_IF_NULL(ref_list[1]);
      MS_LOG(WARNING) << "Ref node of size two with only one contiguous tensor" << ref_list[0]->id() << ":"
                      << ref_list[0]->contiguous() << ", " << ref_list[1]->id() << ":" << ref_list[1]->contiguous();
    }
    if (ref_list.size() == kRefNodeTensorNum && static_cast<size_t>(contiguous_in_ref_list) == kRefNodeTensorNum) {
      ref_tensors_in_contiguous_map[ref_list[0]] = ref_list[1];
    }
  }
  return ref_tensors_in_contiguous_map;
}

std::map<size_t, size_t> GetContiguousRefIndexMap(
  const std::vector<std::vector<GptoTensorPtr>> &ref_node_vector,
  const std::vector<std::vector<GptoTensorPtr>> &contiguous_tensors_list) {
  std::map<size_t, size_t> contiguous_list_with_ref_index_map;
  std::map<GptoTensorPtr, GptoTensorPtr> ref_tensors_in_contiguous_map = GetRefTensorsInContiguousList(ref_node_vector);
  for (const auto &ref_pair : ref_tensors_in_contiguous_map) {
    const auto &ref_first = ref_pair.first;
    const auto &ref_second = ref_pair.second;
    bool found_first = false;
    bool found_second = false;
    size_t index_first = 0;
    size_t index_second = 0;
    size_t index_in_list_first = 0;
    size_t index_in_list_second = 0;
    for (size_t index = 0; index < contiguous_tensors_list.size() && (!found_first || !found_second); index++) {
      if (!found_first) {
        auto iterator_first =
          std::find(contiguous_tensors_list[index].begin(), contiguous_tensors_list[index].end(), ref_first);
        if (iterator_first != contiguous_tensors_list[index].end()) {
          index_first = index;
          index_in_list_first = static_cast<size_t>(iterator_first - contiguous_tensors_list[index].begin());
          found_first = true;
        }
      }
      if (!found_second) {
        auto iterator_second =
          std::find(contiguous_tensors_list[index].begin(), contiguous_tensors_list[index].end(), ref_second);
        if (iterator_second != contiguous_tensors_list[index].end()) {
          index_second = index;
          index_in_list_second = static_cast<size_t>(iterator_second - contiguous_tensors_list[index].begin());
          found_second = true;
        }
      }
    }
    if (!found_first && !found_second) {
      continue;
    }
    if (!found_first) {
      MS_LOG(WARNING) << "Contiguous ref tensor " << ref_first->id() << " not found in any contiguous list";
    }
    if (!found_second) {
      MS_LOG(WARNING) << "Contiguous ref tensor " << ref_second->id() << " not found in any contiguous list";
    }
    if (contiguous_list_with_ref_index_map.find(index_first) == contiguous_list_with_ref_index_map.end() ||
        contiguous_list_with_ref_index_map[index_first] == index_second) {
      contiguous_list_with_ref_index_map[index_first] = index_second;
      // Checking for error cases
      if (index_in_list_first != index_in_list_second) {
        MS_LOG(WARNING) << "Inconsistency in contiguous ref: tensor " << ref_first->id() << " in position "
                        << index_in_list_first << " of contiguous list " << index_first << " and tensor "
                        << ref_second->id() << " in position " << index_in_list_second << " of contiguous list "
                        << index_second;
      }
    } else {
      MS_LOG(WARNING) << "Contiguous list " << index_first << " associated (ref node) with two other contiguous lists: "
                      << contiguous_list_with_ref_index_map[index_first] << " and " << index_second;
    }
  }
  return contiguous_list_with_ref_index_map;
}

std::vector<std::pair<GptoTensorPtr, GptoTensorPtr>> RefNodeProcess(
  const KernelGraphPtr &graph, std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);
  auto &cnode_to_task_map = *cnode_to_task_map_ptr;
  const auto &kernel_cnodes = graph->execution_order();
  size_t total_output_size = 0;
  size_t total_input_size = 0;
  std::vector<std::pair<GptoTensorPtr, GptoTensorPtr>> in_out_vector;

  // Loop to obtain ref node pairs
  for (const auto &kernel : kernel_cnodes) {
    size_t index = 0;
    for (GptoTensorPtr &output_tensor : cnode_to_task_map[kernel]->out_tensors()) {
      const auto &size = output_tensor->weight();
      auto out_index = index++;
      session::AnfWithOutIndex out_pair(kernel, out_index);
      if (graph->IsInRefOutputMap(out_pair)) {
        auto origin_pair = graph->GetRefCorrespondOutput(out_pair);
        MS_EXCEPTION_IF_NULL(origin_pair.first);
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
          MS_LOG_WITH_NODE(EXCEPTION, ori_node)
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
  MS_LOG(INFO) << "RefNode tensor total size: input " << total_input_size << " output " << total_output_size;
  return in_out_vector;
}

std::vector<std::vector<GptoTensorPtr>> CommunicationNodeProcess(
  std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);
  const auto &cnode_to_task_map = *cnode_to_task_map_ptr;

  std::vector<std::vector<GptoTensorPtr>> contiguous_tensors_list;
  Memory comm_input_total_size = 0;
  Memory comm_output_total_size = 0;
  for (const auto &cnode_task : cnode_to_task_map) {
    const auto &task = cnode_task.second;
    MS_EXCEPTION_IF_NULL(task);
    if (!common::AnfAlgo::IsCommunicationOp(cnode_task.first)) {
      continue;
    }

    // Contiguous input
    if (task->in_tensors().size() > 1 && (*(task->in_tensors().begin()) != nullptr) &&
        (!(*(task->in_tensors().begin()))->contiguous())) {
      std::vector<GptoTensorPtr> inputs;
      for (const auto &in_tensor : task->in_tensors()) {
        MS_EXCEPTION_IF_NULL(in_tensor);
        if (in_tensor->weight() == 0) {
          MS_EXCEPTION(ValueError) << "The size of communication in_tensor is zero, tensor id: " +
                                        std::to_string(in_tensor->id());
        }
        comm_input_total_size += in_tensor->weight();
        in_tensor->set_contiguous(true);
        inputs.push_back(in_tensor);
      }
      contiguous_tensors_list.push_back(inputs);
    }

    // Contiguous output
    if (task->out_tensors().size() > 1 && (task->out_tensors()[0] != nullptr) &&
        (!task->out_tensors()[0]->contiguous())) {
      std::vector<GptoTensorPtr> outputs;
      for (const auto &out_tensor : task->out_tensors()) {
        MS_EXCEPTION_IF_NULL(out_tensor);
        if (out_tensor->weight() == 0) {
          MS_EXCEPTION(ValueError) << "The size of communication out_tensor is zero, tensor id: " +
                                        std::to_string(out_tensor->id());
        }
        comm_output_total_size += out_tensor->weight();
        out_tensor->set_contiguous(true);
        outputs.push_back(out_tensor);
      }
      if (outputs.size() != (std::set<GptoTensorPtr>(outputs.begin(), outputs.end())).size()) {
        MS_LOG(EXCEPTION) << task->name() << " has duplicate output tensors, please double check task output tensors";
      }
      contiguous_tensors_list.push_back(outputs);
    }

    // Check for duplication of tensors in lists
    std::set<GptoTensorPtr> all_contiguous_tensors_set;
    size_t all_contiguous_tensors_num = 0;
    for (auto &tensors : contiguous_tensors_list) {
      all_contiguous_tensors_num += tensors.size();
      all_contiguous_tensors_set.insert(tensors.begin(), tensors.end());
    }
    if (all_contiguous_tensors_num != all_contiguous_tensors_set.size()) {
      MS_LOG(EXCEPTION) << "Please check communication tasks, some tensors are in multiple contiguous lists";
    }
  }

  MS_LOG(INFO) << "Contiguous tensor total size: input " << comm_input_total_size << " output "
               << comm_output_total_size;
  return contiguous_tensors_list;
}

void UpdateRefNodeGpto(const KernelGraphPtr &graph, std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  // Retrieve necessary ref node and contiguous info
  auto in_out_vector = RefNodeProcess(graph, cnode_to_task_map_ptr);
  std::vector<std::vector<GptoTensorPtr>> ref_node_vector =
    GetRegularUnionTensorsList(MAX_TENSOR_ID + 1, in_out_vector);
  std::vector<std::vector<GptoTensorPtr>> contiguous_tensors_list;
  std::map<size_t, size_t> contiguous_ref_index_map;

  contiguous_tensors_list = CommunicationNodeProcess(cnode_to_task_map_ptr);
  contiguous_ref_index_map = GetContiguousRefIndexMap(ref_node_vector, contiguous_tensors_list);

  // Loop to update ref node tensor sizes and gpto graph logic
  for (auto &union_list : ref_node_vector) {
    if (std::any_of(union_list.begin(), union_list.end(), [](const GptoTensorPtr &t) { return t->weight() == 0; })) {
      for (auto &tensor : union_list) {
        tensor->set_weight(0);
      }
    } else {  // update head tensor weight
      const auto &tensor_head = union_list[0];
      for (size_t i = 1; i < union_list.size(); ++i) {
        const auto &tensor_ref = union_list[i];
        if (!tensor_ref->contiguous()) {
          if (tensor_head->weight() < tensor_ref->weight()) {
            tensor_head->set_weight(tensor_ref->weight());
          }
          tensor_ref->set_weight(0);
        }
        for (auto &consumer : tensor_ref->consumers()) {
          tensor_head->consumers().insert(consumer);
          consumer.lock()->in_tensors().insert(tensor_head);
          tensor_head->source().lock()->AddChild(consumer.lock());
          consumer.lock()->AddParent(tensor_head->source().lock());
          tensor_head->source().lock()->set_child_is_data_dep(consumer.lock(), true);
        }
      }
    }
  }

  // Loop to update ref node for contiguous
  for (auto ref_list_pair : contiguous_ref_index_map) {
    const auto &head_list = contiguous_tensors_list.at(ref_list_pair.first);
    const auto &ref_list = contiguous_tensors_list.at(ref_list_pair.second);
    if (head_list.size() != ref_list.size()) {
      MS_LOG(WARNING) << "Different head list " << ref_list_pair.first << " size " << head_list.size()
                      << " and ref list " << ref_list_pair.second << " size " << ref_list.size();
    }
    for (size_t i = 0; i < head_list.size(); i++) {
      const auto &tensor_head = head_list[i];
      const auto &tensor_ref = ref_list[i];
      MS_EXCEPTION_IF_NULL(tensor_head);
      MS_EXCEPTION_IF_NULL(tensor_ref);
      tensor_ref->set_weight(0);
      for (auto &consumer : tensor_ref->consumers()) {
        tensor_head->consumers().insert(consumer);
        consumer.lock()->in_tensors().insert(tensor_head);
        tensor_head->source().lock()->AddChild(consumer.lock());
        consumer.lock()->AddParent(tensor_head->source().lock());
        tensor_head->source().lock()->set_child_is_data_dep(consumer.lock(), true);
      }
    }
  }
  contiguous_tensors_list.clear();
  contiguous_ref_index_map.clear();
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

void MockExecutionOrder(const SchedulingOutput &scheduling_output,
                        std::vector<std::pair<CNodePtr, std::tuple<char, size_t, size_t, size_t>>> *mock_exec_order,
                        const size_t &num_events) {
  const size_t kSendRecv = 2;
  std::vector<Interval> task_times = scheduling_output.task_times;
  std::sort(task_times.begin(), task_times.end(), [](Interval x, Interval y) {
    return x.start < y.start || (x.start == y.start && x.task->gpto_type() > y.task->gpto_type());
  });
  mock_exec_order->resize(task_times.size() + kSendRecv * num_events);
  size_t mock_idx = mock_exec_order->capacity() - 1;
  size_t event_id = 0;
  for (std::vector<Interval>::reverse_iterator rit = task_times.rbegin(); rit != task_times.rend(); ++rit) {
    (*mock_exec_order)[mock_idx--] = std::make_pair(rit->task->cnode(), std::make_tuple('n', 0, 0, 0));
    size_t local_event_count = 0;
    for (auto task : rit->task->recv_events()) {
      (*mock_exec_order)[mock_idx - rit->task->recv_events().size() * kSendRecv + 1 + local_event_count] =
        std::make_pair(nullptr, std::make_tuple('s', task.lock()->gpto_type(), rit->task->gpto_type(), event_id));
      (*mock_exec_order)[mock_idx - local_event_count] =
        std::make_pair(nullptr, std::make_tuple('r', task.lock()->gpto_type(), rit->task->gpto_type(), event_id));
      event_id++;
      local_event_count++;
    }
    mock_idx -= rit->task->recv_events().size() * kSendRecv;
  }
  MS_LOG(INFO) << "Mock execution order event size: " << num_events;
}

void SetGPTOMode() {
  gpto_mode = kCompComm;  // default mode
  if (runtime::IsDisableRuntimeConfig(runtime::kRuntimeMultiStream)) {
    gpto_mode = kComp;
  } else if (runtime::IsEnableRuntimeConfig(runtime::kRuntimeMultiStream)) {
    gpto_mode = kCompComm;
  } else {
    gpto_mode = kCompCommGroup;
  }
  MS_LOG(INFO) << "Gpto mode value: " << gpto_mode;
}

void UpdateExecutionOrder(const KernelGraphPtr &kernel_graph, const SchedulingOutput &scheduling_output) {
  std::vector<Interval> task_times = scheduling_output.task_times;
  std::sort(task_times.begin(), task_times.end(), [](const Interval &x, const Interval &y) {
    return x.start < y.start || (x.start == y.start && x.task->gpto_type() > y.task->gpto_type());
  });
  std::vector<CNodePtr> new_exec_order;
  new_exec_order.reserve(task_times.size());
  for (size_t j = 0; j < task_times.size(); j++) {
    new_exec_order.push_back(task_times[j].task->cnode());
  }
  kernel_graph->set_execution_order(new_exec_order);
  new_exec_order.clear();
}

void CalculateMemoryLimit(const KernelGraphPtr &kernel_graph, const std::pair<Time, Memory> &baseline,
                          const std::unordered_map<CNodePtr, GptoTaskPtr> &cnode_to_task) {
  float memory_bias = 1.02;
  const auto &backend_jit_config = kernel_graph->backend_jit_config();
  auto it = backend_jit_config.gpto_options.find("bias");
  if (it != backend_jit_config.gpto_options.end() && !it->second.empty()) {
    try {
      memory_bias = stof(it->second);
    } catch (const std::invalid_argument &e) {
      MS_LOG(WARNING) << "Invalid 'bias' value: '" << it->second << "'. Using default: " << memory_bias;
    } catch (const std::out_of_range &e) {
      MS_LOG(WARNING) << "'bias' value out of range: '" << it->second << "'. Using default: " << memory_bias;
    }
  }

  if (IsGptoDebug()) {
    MEMORY_LIMIT = baseline.second;
  } else {
    MEMORY_LIMIT = LogBaseline(cnode_to_task, kernel_graph, false).second;
  }

  MEMORY_LIMIT = static_cast<Memory>(memory_bias * MEMORY_LIMIT + PARAMETER_SIZE);
  MS_LOG(INFO) << "PARAMETER_SIZE: " << PARAMETER_SIZE;
  MS_LOG(INFO) << "MEMORY_BIAS: " << memory_bias;
  MS_LOG(INFO) << "MEMORY_LIMIT: " << MEMORY_LIMIT;
}

void GPTO(const KernelGraphPtr &kernel_graph,
          std::vector<std::pair<CNodePtr, std::tuple<char, size_t, size_t, size_t>>> *mock_exec_order) {
  if (kernel_graph->is_from_single_op()) {
    MS_LOG(INFO) << "GPTO is not used when pynative forward now.";
    return;
  }

  std::pair<Time, Memory> baseline;

  SetGPTOMode();

  MS_LOG(INFO) << "Start Scheduling Subgraph " << kernel_graph << " with id " << kernel_graph->graph_id()
               << " and Execution order size " << kernel_graph->execution_order().size();

  std::unordered_map<CNodePtr, GptoTaskPtr> cnode_to_task;
  std::unordered_set<uint32_t> stream_ids;
  MS_LOG(INFO) << "Start ExtractSchedulingInput";
  PARAMETER_SIZE = 0;
  SchedulingInput scheduling_input = ExtractSchedulingInput(kernel_graph, &cnode_to_task, &stream_ids);
  MS_LOG(INFO) << "Parameter size: " << PARAMETER_SIZE;
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

  MS_LOG(INFO) << "Start Update Ref Node for Gpto";
  if (kernel_graph->is_dynamic_shape()) {
    MS_LOG(WARNING) << "UpdateRefNodeGpto skipped as kernel graph contains dynamic shape operators";
  } else {
    UpdateRefNodeGpto(kernel_graph, &cnode_to_task);
  }
  MS_LOG(INFO) << "End Update Ref Node for Gpto";

  Memory memory_lower_bound = 0;
  std::set<GptoTensorPtr, GptoTensorIdSort> tensors;
  if (IsGptoDebug()) {
    // Memory lower bound (optional: for analysis only)
    std::vector<mindspore::somas::DynamicBitSet> nodes_dependency;

    MS_LOG(INFO) << "Start Compute Ancestors Descendants";
    ComputeAncestorsDescendants(scheduling_input.tasks, &nodes_dependency);
    MS_LOG(INFO) << "End Compute Ancestors Descendants";

    MS_LOG(INFO) << "Start Memory Lower Bound";
    ExtractTensors(scheduling_input.tasks, &tensors);
    memory_lower_bound = MemoryLowerBound(scheduling_input.tasks, nodes_dependency, tensors);
    MS_LOG(INFO) << "End Memory Lower Bound";

    // Baseline log
    MS_LOG(INFO) << "Start Baseline Greedy Scheduling";
    baseline = LogBaseline(cnode_to_task, kernel_graph, true);
    MS_LOG(INFO) << "End Baseline Greedy Scheduling";

    const size_t kPrecision = 5;
    MS_LOG(INFO) << "Speedup of baseline solution is " << std::setprecision(kPrecision)
                 << 1.0 * TotalTime(scheduling_input.tasks) / baseline.first;
  }

  CalculateMemoryLimit(kernel_graph, baseline, cnode_to_task);
  cnode_to_task.clear();

  MS_LOG(INFO) << "Start GPTO Process";
  size_t lower_makespan;
  auto scheduling_output = MemAwareScheduler(scheduling_input, stream_ids, &lower_makespan);
  scheduling_input.tasks.clear();
  MS_LOG(INFO) << "End GPTO Process";

  if (scheduling_output.makespan == SIZE_MAX) {
    MS_LOG(INFO) << "Hard memory limit is not satisfied by any solution's memory estimate, exiting GPTO...";
    return;
  }

  for (auto &interval : scheduling_output.task_times) {
    interval.task->set_final_start(interval.start);
    interval.task->set_final_end(interval.end);
  }

  if (GetGptoOptionsMode() == kBasic) {
    // Update execution order and trigger default data dependency events
    UpdateExecutionOrder(kernel_graph, scheduling_output);
  } else {  // kAdvance
    // Explicitly generate events and generate mock execution order including them
    size_t num_schedule_events = ScheduleToEvents(scheduling_output);
    size_t num_memory_events = 0;
    if (runtime::RuntimeConf::GetInstance()->mem_optimize_level() == kOptimizeO0) {
      num_memory_events = MemorySafetyEvents(scheduling_output);
    }
    if (IsGptoDebug()) {
      MS_LOG(INFO) << "Start printing output log file before lifting events";
      LogSchedulingOutput(scheduling_output, kernel_graph, tensors, memory_lower_bound, &stream_ids, lower_makespan,
                          "_before_lift");
      MS_LOG(INFO) << "End printing output log file before lifting events";
    }
    size_t num_lift_events = LiftEvents(scheduling_output, stream_ids);
    MockExecutionOrder(scheduling_output, mock_exec_order, num_schedule_events + num_memory_events + num_lift_events);
  }

  if (IsGptoDebug()) {
    MS_LOG(INFO) << "Start printing output log file after lifting events";
    LogSchedulingOutput(scheduling_output, kernel_graph, tensors, memory_lower_bound, &stream_ids, lower_makespan,
                        "_after_lift");
    MS_LOG(INFO) << "End printing output log file after lifting events";
  }
  scheduling_output.task_times.clear();
  tensors.clear();
  stream_ids.clear();
}

void RunGPTO(const NotNull<KernelGraphPtr> &kernel_graph,
             std::vector<std::pair<CNodePtr, std::tuple<char, size_t, size_t, size_t>>> &mock_exec_order) {
  const auto &backend_jit_config = kernel_graph->backend_jit_config();
  auto it = backend_jit_config.gpto_options.find("mode");
  if (it == backend_jit_config.gpto_options.end()) {
    MS_LOG(INFO) << "'mode' not set in gpto_options. Disabling GPTO by default";
  } else {
    const auto &mode = it->second;
    if (mode == "basic" || mode == "advance") {
      MS_LOG(INFO) << "Current execution order algo is GPTO (" << mode << " mode)";
      GPTO(kernel_graph, &mock_exec_order);
    } else if (mode == "off") {
      MS_LOG(INFO) << "GPTO disabled by configuration";
    } else {
      MS_LOG(WARNING) << "Invalid value for 'mode': '" << mode << "'. Disabling GPTO by default";
    }
  }
}
}  // namespace gpto
}  // namespace mindspore
