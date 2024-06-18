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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_GPTO_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_GPTO_H_

#include <list>
#include <vector>
#include <utility>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <set>

#include "mindspore/core/ir/anf.h"
#include "mindspore/core/ir/manager.h"
#include "mindspore/core/mindapi/base/shape_vector.h"
#include "mindspore/ccsrc/backend/common/somas/somas_solver_pre.h"

namespace mindspore {
namespace gpto {  // Graph Parallel Topology Optimizer
// Preliminary definitions
using Time = uint64_t;   // size_t;
using Memory = int64_t;  // maintain memory as signed integer, since memory impact of some operators may be negative
using GptoTaskId = size_t;
using PeId = size_t;
enum GptoTaskType { kNone = 0, kComp, kComm, kCube };
enum GptoTensorType { kSimple = 0, kWorkspace, kGraphOutput, kGraphInput };
enum class PEsSort { kSortByLoad = 0, kSortByValidStart, kNumPEsSort };
enum GPTO_MODE { kSingle = 1, kCompComm = 2, kMulti = 3 };
enum TaskSort {
  kSortByCostMax = 0,
  kSortByCostMin,
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
  kSortByPredCommDepth,
  kSortByPredCube,
  kSortByGreedyHeight,
  kNumTaskSort
};

// Namespace variables
inline Memory SOFT_MEMORY_LIMIT;
inline Memory HARD_MEMORY_LIMIT;  // preset value to capture max device size
inline Memory PARAMETER_SIZE = 0;
inline GPTO_MODE gpto_mode = kCompComm;

// Structs for scheduling
struct ProcessingElement {
  PeId id;
  GptoTaskType gpto_type;
  Time load;
  std::list<std::pair<Time, Time>> idle;
};

struct SortByLoad {
  bool operator()(const ProcessingElement &pe1, const ProcessingElement &pe2) const {
    return pe1.load < pe2.load || (pe1.load == pe2.load && pe1.id < pe2.id);
  }
};

// GPTO Task definitions
class GptoTensor;
class GptoTask {
 public:
  struct SortByIdWeak {
    bool operator()(const std::weak_ptr<GptoTask> &task1, const std::weak_ptr<GptoTask> &task2) const {
      return task1.lock()->id() < task2.lock()->id();
    }
  };

  struct SortByIdShared {
    bool operator()(const std::shared_ptr<GptoTask> &task1, const std::shared_ptr<GptoTask> &task2) const {
      return task1->id() < task2->id();
    }
  };

  GptoTask(const GptoTaskId &id, const GptoTaskType &real_type, const GptoTaskType &gpto_type,
           const std::string &name) {
    id_ = id;
    real_type_ = real_type;
    gpto_type_ = gpto_type;
    cnode_ = nullptr;
    cost_ = 1;
    bottom_level_ = 0;
    top_level_ = 0;
    depth_ = 0;
    succ_diff_type_ = 0;
    weighted_length_ = 0.0;
    start_ = SIZE_MAX;
    end_ = 0;
    pred_comm_ = 0;
    pred_cube_ = 0;
    name_ = name;
    initial_mem_impact_ = 0;
    current_mem_impact_ = 0;
    workspace_memory_ = 0;
    lower_bound_ = 0;
    subgraph_id_ = SIZE_MAX;
    condition_switch_ = false;
    condition_gather_ = false;
  }

  GptoTask(const GptoTask &t) {
    id_ = t.id_;
    real_type_ = t.real_type_;
    gpto_type_ = t.gpto_type_;
    cnode_ = t.cnode_;
    cost_ = t.cost_;
    bottom_level_ = t.bottom_level_;
    top_level_ = t.top_level_;
    depth_ = t.depth_;
    succ_diff_type_ = t.succ_diff_type_;
    weighted_length_ = t.weighted_length_;
    start_ = t.start_;
    end_ = t.end_;
    pred_comm_ = t.pred_comm_;
    pred_cube_ = t.pred_cube_;
    name_ = t.name_;
    initial_mem_impact_ = t.initial_mem_impact_;
    current_mem_impact_ = t.current_mem_impact_;
    workspace_memory_ = t.workspace_memory_;
    lower_bound_ = t.lower_bound_;
    subgraph_id_ = t.subgraph_id_;
    condition_switch_ = t.condition_switch_;
    condition_gather_ = t.condition_gather_;

    parents_ = t.parents_;
    children_ = t.children_;
    in_tensors_ = t.in_tensors_;
    out_tensors_ = t.out_tensors_;
    workspace_tensors_ = t.workspace_tensors_;
  }

  GptoTaskId id() const { return id_; }
  GptoTaskType real_type() const { return real_type_; }
  GptoTaskType gpto_type() const { return gpto_type_; }
  CNodePtr cnode() const { return cnode_; }
  Time cost() const { return cost_; }
  Time bottom_level() const { return bottom_level_; }
  Time top_level() const { return top_level_; }
  size_t depth() const { return depth_; }
  size_t succ_diff_type() const { return succ_diff_type_; }
  double weighted_length() const { return weighted_length_; }
  Time start() const { return start_; }
  Time end() const { return end_; }
  size_t pred_comm() const { return pred_comm_; }
  size_t pred_cube() const { return pred_cube_; }
  std::string name() const { return name_; }
  Memory initial_mem_impact() const { return initial_mem_impact_; }
  Memory current_mem_impact() const { return current_mem_impact_; }
  Memory workspace_memory() const { return workspace_memory_; }
  Time lower_bound() const { return lower_bound_; }
  size_t subgraph_id() const { return subgraph_id_; }
  bool condition_switch() const { return condition_switch_; }
  bool condition_gather() const { return condition_gather_; }

  std::set<std::weak_ptr<GptoTask>, SortByIdWeak> &parents() { return parents_; }
  std::set<std::shared_ptr<GptoTask>, SortByIdShared> &children() { return children_; }
  std::vector<std::shared_ptr<GptoTensor>> &in_tensors() { return in_tensors_; }
  std::vector<std::shared_ptr<GptoTensor>> &out_tensors() { return out_tensors_; }
  std::vector<std::shared_ptr<GptoTensor>> &workspace_tensors() { return workspace_tensors_; }

  void set_id(GptoTaskId id) { id_ = id; }
  void set_real_type(GptoTaskType real_type) { real_type_ = real_type; }
  void set_gpto_type(GptoTaskType gpto_type) { gpto_type_ = gpto_type; }
  void set_cnode(CNodePtr cnode) { cnode_ = cnode; }
  void set_cost(Time cost) { cost_ = cost; }
  void set_bottom_level(Time bottom_level) { bottom_level_ = bottom_level; }
  void set_top_level(Time top_level) { top_level_ = top_level; }
  void set_depth(size_t depth) { depth_ = depth; }
  void set_succ_diff_type(size_t succ_diff_type) { succ_diff_type_ = succ_diff_type; }
  void set_weighted_length(double weighted_length) { weighted_length_ = weighted_length; }
  void set_start(Time start) { start_ = start; }
  void set_end(Time end) { end_ = end; }
  void set_pred_comm(size_t pred_comm) { pred_comm_ = pred_comm; }
  void set_pred_cube(size_t pred_cube) { pred_cube_ = pred_cube; }
  void set_name(std::string name) { name_ = name; }
  void set_initial_mem_impact(Memory mem_add) { initial_mem_impact_ = mem_add; }
  void set_current_mem_impact(Memory mem_add) { current_mem_impact_ = mem_add; }
  void set_workspace_memory(Memory workspace_memory) { workspace_memory_ = workspace_memory; }
  void set_lower_bound(Time lb) { lower_bound_ = lb; }
  void set_subgraph_id(size_t id) { subgraph_id_ = id; }
  void set_condition_switch(bool cond) { condition_switch_ = cond; }
  void set_condition_gather(bool cond) { condition_gather_ = cond; }

  void AddParent(std::weak_ptr<GptoTask> parent) { parents_.insert(parent); }
  void RemoveParent(std::weak_ptr<GptoTask> parent) { parents_.erase(parent); }
  void ClearParents() { parents_.clear(); }

  void AddChild(std::shared_ptr<GptoTask> child) { children_.insert(child); }

  void RemoveChild(std::shared_ptr<GptoTask> child) { children_.erase(child); }
  void ClearChildren() { children_.clear(); }

  void AssignCost(Time cost) {
    if (cost == 0) {
      cost_ = 1;
    } else {
      cost_ = cost;
    }
    bottom_level_ = cost_;
    weighted_length_ = cost_;
  }

  void ResetStartEnd() {
    start_ = SIZE_MAX;
    end_ = 0;
  }

 private:
  GptoTaskId id_;
  GptoTaskType real_type_;
  GptoTaskType gpto_type_;
  CNodePtr cnode_;

  Time cost_;
  Time bottom_level_;
  Time top_level_;
  size_t depth_;
  size_t succ_diff_type_;
  double weighted_length_;
  Time start_;
  Time end_;
  size_t pred_comm_;
  size_t pred_cube_;
  std::string name_;
  Memory initial_mem_impact_;
  Memory current_mem_impact_;
  Memory workspace_memory_;
  Time lower_bound_;

  size_t subgraph_id_;
  bool condition_switch_;
  bool condition_gather_;

  std::set<std::weak_ptr<GptoTask>, SortByIdWeak> parents_;
  std::set<std::shared_ptr<GptoTask>, SortByIdShared> children_;
  std::vector<std::shared_ptr<GptoTensor>> in_tensors_;
  std::vector<std::shared_ptr<GptoTensor>> out_tensors_;
  std::vector<std::shared_ptr<GptoTensor>> workspace_tensors_;
};
using GptoTaskPtr = std::shared_ptr<GptoTask>;
using TaskSortFunction = bool (*)(GptoTaskPtr const &, GptoTaskPtr const &);
using KernelWithIndex = session::KernelWithIndex;

struct TaskDepthSort {
  bool operator()(const GptoTaskPtr t1, const GptoTaskPtr t2) const {
    return t1->depth() < t2->depth() || (t1->depth() == t2->depth() && t1->id() < t2->id());
  }
};

// GptoTensor definitions
class GptoTensor {
 private:
  size_t id_;
  Memory original_weight_;
  Memory weight_;
  std::weak_ptr<GptoTask> source_;
  GptoTensorType type_;
  std::set<std::weak_ptr<GptoTask>, GptoTask::SortByIdWeak> consumers_;

 public:
  GptoTensor(const size_t id, const Memory original_weight, const Memory weight, const std::weak_ptr<GptoTask> source,
             const GptoTensorType type) {
    id_ = id;
    original_weight_ = original_weight;
    weight_ = weight;
    source_ = source;
    type_ = type;
  }

  GptoTensor(const GptoTensor &t) {
    id_ = t.id_;
    original_weight_ = t.original_weight_;
    weight_ = t.weight_;
    source_ = t.source_;
    type_ = t.type_;
    consumers_ = t.consumers_;
  }

  ~GptoTensor() { consumers_.clear(); }

  const size_t &id() const { return id_; }
  const Memory &original_weight() const { return original_weight_; }
  const Memory &weight() const { return weight_; }
  const std::weak_ptr<GptoTask> &source() { return source_; }
  const GptoTensorType &type() const { return type_; }
  std::set<std::weak_ptr<GptoTask>, GptoTask::SortByIdWeak> &consumers() { return consumers_; }

  void set_type(GptoTensorType type) { type_ = type; }
  void set_original_weight(Memory original_weight) { original_weight_ = original_weight; }
  void set_weight(Memory weight) { weight_ = weight; }
};
using GptoTensorPtr = std::shared_ptr<GptoTensor>;

struct GptoTensorIdSort {
  bool operator()(const GptoTensorPtr t1, const GptoTensorPtr t2) const { return t1->id() < t2->id(); }
};

struct Interval {  // Information extracted by scheduling
  GptoTaskPtr task;
  Time start;
  Time end;
};

// Sorting for scheduling to dependencies (events)
struct SortByStart {
  bool operator()(const Interval &interval1, const Interval &interval2) const {
    const auto &id1 = interval1.task->id();
    const auto &start1 = interval1.start;
    const auto &end1 = interval1.end;
    const auto &id2 = interval2.task->id();
    const auto &start2 = interval2.start;
    const auto &end2 = interval2.end;
    return start1 < start2 || (start1 == start2 && end1 < end2) || (start1 == start2 && end1 == end2 && id1 < id2);
  }
};

struct SortByEnd {
  bool operator()(const Interval &interval1, const Interval &interval2) const {
    const auto &id1 = interval1.task->id();
    const auto &start1 = interval1.start;
    const auto &end1 = interval1.end;
    const auto &id2 = interval2.task->id();
    const auto &start2 = interval2.start;
    const auto &end2 = interval2.end;
    return end1 < end2 || (end1 == end2 && start1 < start2) || (end1 == end2 && start1 == start2 && id1 < id2);
  }
};

// GPTO Scheduling definitions
struct SchedulingInput {
  std::vector<GptoTaskPtr> tasks;
};

struct SchedulingOutput {
  std::vector<Interval> task_times;
  Time makespan;
  Memory memory_peak;
};

// Sorting functions
bool SortByCostMax(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByCostMin(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortBySuccDiff(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByBottomLevelMax(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByBottomLevelMin(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByTopLevelMax(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByTopLevelMin(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByBottomTopLevelMaxSum(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByBottomTopLevelMinSum(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByBottomTopLevelComposite(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByWeightedLength(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByDepthMax(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByDepthMin(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByPredComm(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByPredCommDepth(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByPredCube(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByGreedyHeight(const GptoTaskPtr &, const GptoTaskPtr &);

// Scheduling to dependencies (events) functions
bool Overlap(const Time &, const Time &, const Time &, const Time &);
std::vector<std::pair<CNodePtr, CNodePtr>> ScheduleToEvents(const SchedulingOutput &);

// Task-related functions
size_t CalculateCommCost(const CNodePtr &);
size_t CalculateCubeCost(const CNodePtr &);
size_t CalculateVectorCost(const CNodePtr &);
GptoTaskType GetRealType(const CNodePtr cnode);
GptoTaskType GetType(const CNodePtr cnode);
bool IsCubeKernel(const CNodePtr &node);

// Tensor-related functions
size_t GetAlignedSize(size_t original_size);
void ExtractRealTensors(const SchedulingInput &scheduling_input,
                        std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_gpto_map_ptr);
void StandardInputCase(const GptoTaskPtr &GptoTask, std::unordered_set<void *> *parameter_set, const CNodePtr &kernel,
                       std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_gpto_map_ptr);
void CleanOutput(size_t index, CNodePtr pre_node, GptoTaskPtr pre_task, const GptoTaskPtr &GptoTask);
void CleanWorkspace(CNodePtr pre_node, const GptoTaskPtr &pre_task, const GptoTaskPtr &task);
void ExtractOutputWorkspaceTensors(const SchedulingInput &scheduling_input, const std::vector<GptoTaskPtr> &tasks);
KernelWithIndex GetVisitKernelWithReturnType(const AnfNodePtr &ori_node, size_t ori_index,
                                             std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr);
void GraphOutputProcess(const KernelGraphPtr &, std::unordered_map<CNodePtr, GptoTaskPtr> *);
void RefNodeProcess(const KernelGraphPtr &, std::unordered_map<CNodePtr, GptoTaskPtr> *);

// Scheduling main functions
SchedulingOutput Process(const SchedulingInput &);
SchedulingInput ExtractSchedulingInput(const KernelGraphPtr &, std::unordered_map<CNodePtr, GptoTaskPtr> *);
SchedulingOutput ProcessCore(const std::vector<GptoTaskPtr> &, const std::unordered_map<GptoTaskType, int32_t> &,
                             const TaskSortFunction &, bool);
GptoTaskPtr SelectTaskToSchedule(Memory, std::set<GptoTaskPtr, TaskSortFunction> *,
                                 std::unordered_map<GptoTaskType, Memory> *);
std::pair<PeId, Time> SelectPEandTime(const GptoTask &, Time, std::set<ProcessingElement, SortByLoad> *);
std::pair<PeId, Time> SelectPEandTimeAvailableStart(const GptoTask &, Time, std::vector<ProcessingElement> *);
void UpdateMemoryImpactAndCandidates(
  std::set<GptoTaskPtr, TaskSortFunction> *, const GptoTaskPtr &,
  std::unordered_map<size_t, std::set<std::weak_ptr<GptoTask>>, GptoTask::SortByIdWeak> *,
  std::unordered_map<GptoTaskId, size_t> *, std::unordered_map<GptoTaskId, Time> *, Time *, Time *,
  std::unordered_set<GptoTaskPtr> *);

// Scheduling auxiliary functions
void InitializeTasks(const std::vector<GptoTaskPtr> &, std::unordered_map<GptoTaskId, Time> *,
                     std::unordered_map<GptoTaskId, size_t> *, std::set<GptoTaskPtr, TaskSortFunction> *,
                     std::unordered_set<GptoTaskPtr> *);
void InsertEdges(const KernelGraphPtr &, std::unordered_map<CNodePtr, GptoTaskPtr> *);
void InitializeProcessingElement(const std::unordered_map<GptoTaskType, int32_t> &, size_t *,
                                 std::unordered_map<GptoTaskType, const std::set<ProcessingElement, SortByLoad>> *,
                                 std::unordered_map<GptoTaskType, const std::vector<ProcessingElement>> *, bool);
void InitializeTaskInlineCondition(const CNodePtr &, GptoTaskPtr *,
                                   std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> *,
                                   std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> *);
void UpdateTasksInlineCondition(std::unordered_map<CNodePtr, GptoTaskPtr> *,
                                std::map<GptoTaskPtr, GptoTaskPtr, TaskDepthSort> *);

// Compute auxiliary values for task sorting criteria
void ComputeBottomLevelAndWeightedLength(const std::vector<GptoTaskPtr> &);
void ComputeDepthAndTopLevel(const std::vector<GptoTaskPtr> &);
void ComputePredComm(const std::vector<GptoTaskPtr> &);
void ComputePredCube(const std::vector<GptoTaskPtr> &);

// Memory-aware scheduling
void ComputeInitialMemoryImpact(const std::vector<GptoTaskPtr> &);
[[maybe_unused]] void ExtractTensors(const std::vector<GptoTaskPtr> &, std::set<GptoTensorPtr, GptoTensorIdSort> *);
[[maybe_unused]] void ComputeAncestorsDescendants(
  const std::vector<GptoTaskPtr> &,
  std::vector<mindspore::somas::DynamicBitSet> *);  // only needed for memory lower bound
[[maybe_unused]] Memory MemoryLowerBound(const std::vector<GptoTaskPtr> &,
                                         const std::vector<mindspore::somas::DynamicBitSet> &,
                                         const std::set<GptoTensorPtr, GptoTensorIdSort> &);

// Makespan lower bounds
Time LowerBoundBottomLevel(const std::vector<GptoTaskPtr> &);
Time LowerBoundPEs(const std::vector<GptoTaskPtr> &, const std::unordered_map<GptoTaskType, int32_t> &);

// Verification functions
bool VerifyDAG(const std::vector<GptoTaskPtr> &);
bool VerifyScheduling(const std::vector<GptoTaskPtr> &);

// Printing log files
[[maybe_unused]] void LogSchedulingOutput(const SchedulingOutput &, const std::unordered_map<CNodePtr, GptoTaskPtr> &,
                                          const std::vector<std::pair<CNodePtr, CNodePtr>> &, const KernelGraphPtr &,
                                          const std::set<GptoTensorPtr, GptoTensorIdSort> &, const Time, const Memory,
                                          const std::string &);
[[maybe_unused]] void LogBaseline(const SchedulingInput &, std::unordered_map<CNodePtr, GptoTaskPtr> *,
                                  const KernelGraphPtr &, const std::string &);

// Debug context function
std::pair<bool, std::string> GetDebugConfig();

// Integration function
void GPTO(const KernelGraphPtr &, std::vector<std::pair<CNodePtr, CNodePtr>> *);
}  // namespace gpto
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_GPTO_H_
