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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_COMP_COMM_SCHEDULING_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_COMP_COMM_SCHEDULING_H_

#include <list>
#include <vector>
#include <utility>
#include <memory>
#include <unordered_map>
#include <string>

#include "mindspore/core/ir/anf.h"
#include "mindspore/core/ir/manager.h"
#include "mindspore/core/mindapi/base/shape_vector.h"

namespace mindspore {
namespace opt {
// Preliminary definitions
using Time = size_t;
using TaskId = size_t;
using PeId = size_t;
enum TaskType { kNone, kComp, kComm };

struct ProcessingElement {
  PeId id;
  TaskType type;
  Time load;
  std::list<std::pair<Time, Time>> idle;
};

struct Interval {  // Information extracted by scheduling
  TaskId id;
  TaskType type;
  Time start;
  Time end;
};

enum TaskSort {
  kSortByWeightMax = 0,
  kSortByWeightMin,
  kSortBySuccDiff,
  kSortByBottomLevelMax,
  kSortByBottomLevelMin,
  kSortByTopLevelMax,
  kSortByTopLevelMin,
  kSortByBottomTopLevelMaxSum,
  kSortByBottomTopLevelMinSum,
  kSortByBottomTopLevelComposite,
  kSortByWeightedLength,
  kSortByDepthMax,
  kSortByDepthMin,
  kSortByPredComm,
  kNumTaskSort
};

// Class to define tasks for scheduling
class Task {
 public:
  // Constructors
  Task(const TaskId &id, const TaskType &type) {
    this->id_ = id;
    this->type_ = type;
    // scheduling related
    this->weight_ = 1;
    this->parallel_weight_ = 1;
    this->bottom_level_ = 0;
    this->top_level_ = 0;
    this->depth_ = 0;
    this->succ_diff_type_ = 0;
    this->weighted_length_ = 0.0;
    this->start_ = SIZE_MAX;
    this->end_ = 0;
    this->pred_comm_ = 0;
  }

  // Accessors
  TaskId id() const { return this->id_; }
  TaskType type() const { return this->type_; }

  Time weight() const { return this->weight_; }
  Time parallel_weight() const { return this->parallel_weight_; }
  Time bottom_level() const { return this->bottom_level_; }
  Time top_level() const { return this->top_level_; }
  size_t depth() const { return this->depth_; }
  size_t succ_diff_type() const { return this->succ_diff_type_; }
  double weighted_length() const { return this->weighted_length_; }
  Time start() const { return this->start_; }
  Time end() const { return this->end_; }
  size_t pred_comm() const { return this->pred_comm_; }

  std::vector<std::weak_ptr<Task>> &parents() { return this->parents_; }
  std::vector<std::shared_ptr<Task>> &children() { return this->children_; }
  std::vector<std::shared_ptr<Task>> &no_dep_grandchildren() { return this->no_dep_grandchildren_; }

  // Mutators
  void set_id(TaskId id) { this->id_ = id; }
  void set_type(TaskType type) { this->type_ = type; }

  void set_weight(Time weight) { this->weight_ = weight; }
  void set_parallel_weight(Time parallel_weight) { this->parallel_weight_ = parallel_weight; }
  void set_bottom_level(Time bottom_level) { this->bottom_level_ = bottom_level; }
  void set_top_level(Time top_level) { this->top_level_ = top_level; }
  void set_depth(size_t depth) { this->depth_ = depth; }
  void set_succ_diff_type(size_t succ_diff_type) { this->succ_diff_type_ = succ_diff_type; }
  void set_weighted_length(double weighted_length) { this->weighted_length_ = weighted_length; }
  void set_start(Time start) { this->start_ = start; }
  void set_end(Time end) { this->end_ = end; }
  void set_pred_comm(size_t pred_comm) { this->pred_comm_ = pred_comm; }

  // Maintaining Graph Topology
  void AddParent(std::weak_ptr<Task> parent) { this->parents_.push_back(parent); }
  void AddChild(std::shared_ptr<Task> child) { this->children_.push_back(child); }
  void AddNoDepGrandchild(std::shared_ptr<Task> grandchild) { this->no_dep_grandchildren_.push_back(grandchild); }

  bool HasChild(std::shared_ptr<Task> child) {
    return std::find(children_.begin(), children_.end(), child) != children_.end();
  }
  bool HasNoDepGrandchild(std::shared_ptr<Task> grandchild) {
    return std::find(no_dep_grandchildren_.begin(), no_dep_grandchildren_.end(), grandchild) !=
           no_dep_grandchildren_.end();
  }

  // Weighting (Evaluation)
  void AssignWeight(size_t weight) {
    if (weight == 0) {
      this->weight_ = 1;
    } else if (weight < 0) {
      this->weight_ = SIZE_MAX;
    } else {
      this->weight_ = weight;
    }
    this->parallel_weight_ = this->weight_;
  }

  // Other
  void ResetStartEnd() {
    this->set_start(SIZE_MAX);
    this->set_end(0);
  }

 private:
  TaskId id_;
  TaskType type_;

  // Attributes used to select task during scheduling
  Time weight_;
  Time parallel_weight_;
  Time bottom_level_;
  Time top_level_;
  size_t depth_;
  size_t succ_diff_type_;
  double weighted_length_;
  Time start_;
  Time end_;
  size_t pred_comm_;

  // Attributes to maintain graph info

  std::vector<std::weak_ptr<Task>> parents_;
  std::vector<std::shared_ptr<Task>> children_;
  std::vector<std::shared_ptr<Task>> no_dep_grandchildren_;
};

using TaskPtr = std::shared_ptr<Task>;
using TaskSortFunction = bool (*)(std::shared_ptr<Task> const &, std::shared_ptr<Task> const &);

// Class for scheduling algorithms
struct SchedulingInput {
  std::vector<std::shared_ptr<Task>> tasks;
};

struct SchedulingOutput {
  std::vector<Interval> task_times;
  Time makespan;
};

namespace FastGreedyScheduler {
// Main functionality
SchedulingOutput Process(SchedulingInput &, const std::string &);
SchedulingOutput ProcessCore(std::vector<std::shared_ptr<Task>> &, std::unordered_map<TaskType, int32_t> &,
                             const TaskSortFunction &, bool);
SchedulingOutput ProcessSingle(const SchedulingInput &, const TaskSortFunction &, bool, const std::string &);

// Compute Auxiliary Values for Task Sorting
void ComputeBottomLevelAndWeightedLength(std::vector<std::shared_ptr<Task>> &);
void ComputeDepthAndTopLevel(std::vector<std::shared_ptr<Task>> &);
void ComputePredComm(std::vector<std::shared_ptr<Task>> &);

// Lower Bounds
Time LowerBoundBottomLevel(std::vector<std::shared_ptr<Task>> &);
Time LowerBoundPEs(std::vector<std::shared_ptr<Task>> &, std::unordered_map<TaskType, int32_t> &);

// Dependency Generation
std::vector<std::pair<TaskId, TaskId>> ScheduleToDependencies(const SchedulingOutput &);

// Verification
bool VerifyDAG(std::vector<std::shared_ptr<Task>> &);
bool VerifyScheduling(std::vector<std::shared_ptr<Task>> &);
bool VerifyDependencies(std::vector<std::shared_ptr<Task>> &, std::vector<std::pair<TaskId, TaskId>> &);

// Log
void PrintLog(const SchedulingOutput &, const std::vector<std::pair<TaskId, TaskId>> &, const std::string &);
}  // namespace FastGreedyScheduler

SchedulingInput ExtractSchedulingInput(const FuncGraphManagerPtr &, const std::vector<CNodePtr> &,
                                       std::unordered_map<CNodePtr, TaskPtr> *);
void AddRealDependencies(const FuncGraphManagerPtr &, const std::vector<CNodePtr> &,
                         const std::vector<std::pair<TaskId, TaskId>> &, std::unordered_map<CNodePtr, TaskPtr> &);

// Functions for integration
void CompCommScheduling(const FuncGraphPtr &);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_COMP_COMM_SCHEDULING_H_
