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
#include <queue>
#include <tuple>

#include "ir/anf.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/utils/contract.h"
#include "backend/common/somas/somas_solver_pre.h"

namespace mindspore {
namespace gpto {  // Graph Parallel Topology Optimizer
// Preliminary definitions
using Time = uint64_t;  // size_t;
using Memory = uint64_t;
using GptoTaskId = size_t;
using PeId = size_t;
using GptoTaskType = size_t;  // kVec, kCube, kComm, kComm2, ..., kCommN (arbitrary number of comm streams)
enum GptoTensorType { kSimple = 0, kWorkspace, kGraphOutput, kGraphInput, kNumTensorTypes };
enum class PEsSort { kSortByLoad = 0, kSortByValidStart, kNumPEsSort };
enum GPTO_MODE { kComp = 1, kCompComm = 2, kCompCubeComm = 3, kCompCommGroup = 4, kNumModes = 5 };
enum GPTO_OPTIONS_MODE { kBasic = 1, kAdvance = 2 };  // Profile Cost, TopoSort
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
  kSortByTypePriority,
  kSortByPredComm,
  kSortByPredCommDepth,
  kSortByPredCube,
  kSortByGreedyHeight,
  kSortBySValue,
  kSortByAValue,
  kSortByMValue,
  kSortByWeightedSValue,
  kSortByWeightedAValue,
  kSortByWeightedMValue,
  kSortByCostSValue,
  kSortByCostAValue,
  kSortByCostMValue,
  kSortByReversePostOrder,
  kNumTaskSort
};

// Namespace variables
inline Memory MEMORY_LIMIT;
inline Memory PARAMETER_SIZE;
inline GPTO_MODE gpto_mode;
inline size_t MAX_TENSOR_ID;
inline GptoTaskType kVec = 0;
inline GptoTaskType kCube = 1;
inline GptoTaskType kCommTrue = 1;
inline GptoTaskType kCommGroup = 2;

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

  struct SortByIdSharedInverse {
    bool operator()(const std::shared_ptr<GptoTask> &task1, const std::shared_ptr<GptoTask> &task2) const {
      return task1->id() > task2->id();
    }
  };

  struct SortForEventRecv {
    bool operator()(const std::weak_ptr<GptoTask> &task1, const std::weak_ptr<GptoTask> &task2) const {
      return task1.lock()->final_start() > task2.lock()->final_start() ||
             (task1.lock()->final_start() == task2.lock()->final_start() &&
              task1.lock()->gpto_type() < task2.lock()->gpto_type());
    }
  };

  GptoTask(const GptoTaskId &id, const GptoTaskType &gpto_type, const std::string &name) {
    id_ = id;
    gpto_type_ = gpto_type;
    gpto_original_type_ = gpto_type;
    cnode_ = nullptr;
    cost_ = 1;
    bottom_level_ = 0;
    top_level_ = 0;
    depth_ = 0;
    succ_diff_type_ = 0;
    weighted_length_ = 0.0;
    type_priority_ = 0;
    pred_comm_ = 0;
    pred_cube_ = 0;
    name_ = name;
    initial_mem_impact_ = 0;
    workspace_memory_ = 0;
    subgraph_id_ = SIZE_MAX;
    subgraph_id_parent_ = SIZE_MAX;
    condition_switch_ = false;
    condition_gather_ = false;
    s_value_ = 0;
    a_value_ = 0.0;
    m_value_ = 0;
    weighted_s_value_ = 0;
    weighted_a_value_ = 0.0;
    weighted_m_value_ = 0;
    cost_s_value_ = 0;
    cost_a_value_ = 0.0;
    cost_m_value_ = 0;
    post_order_time_ = 0;
    final_start_ = 0;
    final_end_ = 0;
  }

  GptoTask(const GptoTask &t) {
    id_ = t.id_;
    gpto_type_ = t.gpto_type_;
    gpto_original_type_ = t.gpto_original_type_;
    cnode_ = t.cnode_;
    cost_ = t.cost_;
    bottom_level_ = t.bottom_level_;
    top_level_ = t.top_level_;
    depth_ = t.depth_;
    succ_diff_type_ = t.succ_diff_type_;
    weighted_length_ = t.weighted_length_;
    start_ = t.start_;
    end_ = t.end_;
    type_priority_ = t.type_priority_;
    pred_comm_ = t.pred_comm_;
    pred_cube_ = t.pred_cube_;
    name_ = t.name_;
    initial_mem_impact_ = t.initial_mem_impact_;
    minus_mem_impact_ = t.minus_mem_impact_;
    workspace_memory_ = t.workspace_memory_;
    subgraph_id_ = t.subgraph_id_;
    subgraph_id_parent_ = t.subgraph_id_parent_;
    condition_switch_ = t.condition_switch_;
    condition_gather_ = t.condition_gather_;
    position_ = t.position_;
    s_value_ = t.s_value_;
    a_value_ = t.a_value_;
    m_value_ = t.m_value_;
    weighted_s_value_ = t.weighted_s_value_;
    weighted_a_value_ = t.weighted_a_value_;
    weighted_m_value_ = t.weighted_m_value_;
    cost_s_value_ = t.cost_s_value_;
    cost_a_value_ = t.cost_a_value_;
    cost_m_value_ = t.cost_m_value_;
    post_order_time_ = t.post_order_time_;
    final_start_ = t.final_start_;
    final_end_ = t.final_end_;
    parents_ = t.parents_;
    mw_parents_ = t.mw_parents_;
    children_ = t.children_;
    in_tensors_ = t.in_tensors_;
    in_params_ = t.in_params_;
    out_tensors_ = t.out_tensors_;
    workspace_tensors_ = t.workspace_tensors_;
    recv_events_ = t.recv_events_;
  }

  ~GptoTask() {
    parents_.clear();
    mw_parents_.clear();
    children_.clear();
    in_tensors_.clear();
    in_params_.clear();
    out_tensors_.clear();
    workspace_tensors_.clear();
    in_weights_.clear();
    recv_events_.clear();
  }

  GptoTaskId id() const { return id_; }
  GptoTaskType gpto_type() const { return gpto_type_; }
  GptoTaskType gpto_original_type() const { return gpto_original_type_; }
  CNodePtr cnode() const { return cnode_; }
  Time cost() const { return cost_; }
  Time bottom_level() const { return bottom_level_; }
  Time top_level() const { return top_level_; }
  size_t depth() const { return depth_; }
  size_t succ_diff_type() const { return succ_diff_type_; }
  double weighted_length() const { return weighted_length_; }
  std::vector<std::vector<Time>> &start() { return start_; }
  std::vector<std::vector<Time>> &end() { return end_; }
  Time start(const size_t task_sort, const size_t &pes_sort) { return start_[task_sort][pes_sort]; }
  Time end(const size_t task_sort, const size_t &pes_sort) { return end_[task_sort][pes_sort]; }
  size_t type_priority() const { return type_priority_; }
  size_t pred_comm() const { return pred_comm_; }
  size_t pred_cube() const { return pred_cube_; }
  std::string name() const { return name_; }
  Memory initial_mem_impact() const { return initial_mem_impact_; }
  std::vector<std::vector<Memory>> &minus_mem_impact() { return minus_mem_impact_; }
  Memory minus_mem_impact(const size_t &task_sort, const size_t &pes_sort) {
    return minus_mem_impact_[task_sort][pes_sort];
  }
  Memory minus_mem_impact_greedy_height() const { return minus_mem_impact_greedy_height_; }
  Memory workspace_memory() const { return workspace_memory_; }
  size_t subgraph_id() const { return subgraph_id_; }
  size_t subgraph_id_parent() const { return subgraph_id_parent_; }
  bool condition_switch() const { return condition_switch_; }
  bool condition_gather() const { return condition_gather_; }
  size_t post_order_time() const { return post_order_time_; }

  Time final_start() const { return final_start_; }
  Time final_end() const { return final_end_; }

  std::vector<std::vector<size_t>> &position() { return position_; }
  size_t position(const size_t &task_sort, const size_t &pes_sort) { return position_[task_sort][pes_sort]; }
  size_t s_value() const { return s_value_; }
  double a_value() const { return a_value_; }
  size_t m_value() const { return m_value_; }
  Memory sw_value() const { return weighted_s_value_; }
  double aw_value() const { return weighted_a_value_; }
  Memory mw_value() const { return weighted_m_value_; }
  Time sc_value() const { return cost_s_value_; }
  double ac_value() const { return cost_a_value_; }
  Time mc_value() const { return cost_m_value_; }

  std::set<std::weak_ptr<GptoTask>, SortByIdWeak> &parents() { return parents_; }
  std::set<std::shared_ptr<GptoTask>, SortByIdShared> &children() { return children_; }
  std::vector<std::weak_ptr<GptoTask>> &mw_parents() { return mw_parents_; }
  std::set<std::shared_ptr<GptoTensor>> &in_tensors() { return in_tensors_; }
  std::unordered_map<AnfNodePtr, size_t> &in_params() { return in_params_; }
  std::vector<std::shared_ptr<GptoTensor>> &out_tensors() { return out_tensors_; }
  std::vector<std::shared_ptr<GptoTensor>> &workspace_tensors() { return workspace_tensors_; }
  std::set<std::weak_ptr<GptoTask>, SortForEventRecv> &recv_events() { return recv_events_; }

  void init_start(const size_t &num_task_sort, const size_t &num_pes_sort) {
    start_.assign(num_task_sort, std::vector<Time>(num_pes_sort, static_cast<Time>(0)));
  }
  void init_end(const size_t &num_task_sort, const size_t &num_pes_sort) {
    end_.assign(num_task_sort, std::vector<Time>(num_pes_sort, static_cast<Time>(0)));
  }
  void init_minus_mem_impact(const size_t &num_task_sort, const size_t &num_pes_sort) {
    minus_mem_impact_.assign(num_task_sort, std::vector<Memory>(num_pes_sort, static_cast<Memory>(0)));
  }
  void init_position(const size_t &num_task_sort, const size_t &num_pes_sort) {
    position_.assign(num_task_sort, std::vector<size_t>(num_pes_sort, static_cast<size_t>(0)));
  }

  void set_id(const GptoTaskId id) { id_ = id; }
  void set_gpto_type(const GptoTaskType &gpto_type) { gpto_type_ = gpto_type; }
  void set_gpto_original_type(const GptoTaskType &gpto_original_type) { gpto_original_type_ = gpto_original_type; }
  void set_cnode(const CNodePtr &cnode) { cnode_ = cnode; }
  void set_cost(const Time &cost) { cost_ = cost; }
  void set_bottom_level(const Time &bottom_level) { bottom_level_ = bottom_level; }
  void set_top_level(const Time &top_level) { top_level_ = top_level; }
  void set_depth(const size_t &depth) { depth_ = depth; }
  void set_succ_diff_type(const size_t &succ_diff_type) { succ_diff_type_ = succ_diff_type; }
  void set_weighted_length(const double &weighted_length) { weighted_length_ = weighted_length; }
  void set_start(const Time &start, const size_t &task_sort, const size_t &pes_sort) {
    start_[task_sort][pes_sort] = start;
  }
  void set_end(const Time &end, const size_t &task_sort, const size_t &pes_sort) { end_[task_sort][pes_sort] = end; }
  void set_type_priority(const size_t &type_pr) { type_priority_ = type_pr; }
  void set_pred_comm(const size_t &pred_comm) { pred_comm_ = pred_comm; }
  void set_pred_cube(const size_t &pred_cube) { pred_cube_ = pred_cube; }
  void set_name(const std::string &name) { name_ = name; }
  void set_initial_mem_impact(const Memory &mem_add) { initial_mem_impact_ = mem_add; }
  void set_minus_mem_impact(const Memory &mem_add, const size_t &task_sort, const size_t &pes_sort) {
    minus_mem_impact_[task_sort][pes_sort] = mem_add;
  }
  void set_minus_mem_impact_greedy_height(const Memory &mem) { minus_mem_impact_greedy_height_ = mem; }
  void set_workspace_memory(const Memory &workspace_memory) { workspace_memory_ = workspace_memory; }
  void set_subgraph_id(const size_t &id) { subgraph_id_ = id; }
  void set_subgraph_id_parent(const size_t &id_parent) { subgraph_id_parent_ = id_parent; }
  void set_condition_switch(const bool &cond) { condition_switch_ = cond; }
  void set_condition_gather(const bool &cond) { condition_gather_ = cond; }
  void set_post_order_time(const size_t &post) { post_order_time_ = post; }
  void set_position(const size_t &pos, const size_t &task_sort, const size_t &pes_sort) {
    position_[task_sort][pes_sort] = pos;
  }
  void set_s_value(const size_t &s) { s_value_ = s; }
  void set_a_value(const double &a) { a_value_ = a; }
  void set_m_value(const size_t &m) { m_value_ = m; }
  void set_sw_value(const Memory &sw) { weighted_s_value_ = sw; }
  void set_aw_value(const double &aw) { weighted_a_value_ = aw; }
  void set_mw_value(const Memory &mw) { weighted_m_value_ = mw; }
  void set_sc_value(const Time &sc) { cost_s_value_ = sc; }
  void set_ac_value(const double &ac) { cost_a_value_ = ac; }
  void set_mc_value(const Time &mc) { cost_m_value_ = mc; }

  void set_final_start(const Time &fs) { final_start_ = fs; }
  void set_final_end(const Time &fe) { final_end_ = fe; }

  void AddParent(const std::weak_ptr<GptoTask> &parent) { parents_.insert(parent); }
  void AddMWParent(const std::weak_ptr<GptoTask> &mw_parent) { mw_parents_.push_back(mw_parent); }
  void AddChild(const std::shared_ptr<GptoTask> &child) { children_.insert(child); }

  void AssignCost(const Time cost) {
    if (cost == 0) {
      cost_ = 1;
    } else {
      cost_ = cost;
    }
    bottom_level_ = cost_;
    weighted_length_ = cost_;
  }

  std::unordered_map<std::shared_ptr<GptoTask>, Memory> &in_weights() { return in_weights_; }
  Memory &in_weights_sum() { return in_weights_sum_; }
  void set_in_weights(const std::unordered_map<std::shared_ptr<GptoTask>, Memory> in_weights) {
    in_weights_ = in_weights;
  }
  void set_in_weights_sum(const Memory in_weights_sum) { in_weights_sum_ = in_weights_sum; }
  bool child_is_data_dep(const std::shared_ptr<GptoTask> &child) { return child_is_data_dep_[child]; }
  void set_child_is_data_dep(const std::shared_ptr<GptoTask> &child, const bool &is_data_dep) {
    child_is_data_dep_[child] = is_data_dep;
  }

 private:
  GptoTaskId id_;
  GptoTaskType gpto_type_;
  GptoTaskType gpto_original_type_;
  CNodePtr cnode_;

  Time cost_;
  Time bottom_level_;
  Time top_level_;
  size_t depth_;
  size_t succ_diff_type_;
  double weighted_length_;
  std::vector<std::vector<Time>> start_;  // for multi-threading
  std::vector<std::vector<Time>> end_;    // for multi-threading
  size_t type_priority_;
  size_t pred_comm_;
  size_t pred_cube_;
  std::string name_;
  Memory initial_mem_impact_;
  std::vector<std::vector<Memory>> minus_mem_impact_;  // for multi-threading
  Memory minus_mem_impact_greedy_height_;              // for simple multi-threading dynamic greedy height task sorting
  Memory workspace_memory_;

  size_t subgraph_id_;
  size_t subgraph_id_parent_;
  bool condition_switch_;
  bool condition_gather_;
  std::vector<std::vector<size_t>> position_;  // for multi-threading
  size_t s_value_;
  double a_value_;
  size_t m_value_;
  Memory weighted_s_value_;
  double weighted_a_value_;
  Memory weighted_m_value_;
  Time cost_s_value_;
  double cost_a_value_;
  Time cost_m_value_;
  size_t post_order_time_;
  Time final_start_;
  Time final_end_;

  std::set<std::weak_ptr<GptoTask>, SortByIdWeak> parents_;
  std::vector<std::weak_ptr<GptoTask>> mw_parents_;
  std::set<std::shared_ptr<GptoTask>, SortByIdShared> children_;
  std::set<std::shared_ptr<GptoTensor>>
    in_tensors_;  // keep ordered set for fusion and communication node process support
  std::unordered_map<AnfNodePtr, size_t> in_params_;
  std::vector<std::shared_ptr<GptoTensor>> out_tensors_;
  std::vector<std::shared_ptr<GptoTensor>> workspace_tensors_;
  std::unordered_map<std::shared_ptr<GptoTask>, Memory> in_weights_;
  Memory in_weights_sum_;
  std::set<std::weak_ptr<GptoTask>, SortForEventRecv> recv_events_;
  std::unordered_map<std::shared_ptr<GptoTask>, bool> child_is_data_dep_;
};
using GptoTaskPtr = std::shared_ptr<GptoTask>;
using TaskSortFunction = bool (*)(GptoTaskPtr const &, GptoTaskPtr const &);
using KernelWithIndex = session::KernelWithIndex;

struct TaskDepthSort {
  inline bool operator()(const GptoTaskPtr &t1, const GptoTaskPtr &t2) const {
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
  ShapeVector shape_;
  TypeId dtype_;
  std::set<std::weak_ptr<GptoTask>, GptoTask::SortByIdWeak> consumers_;
  std::map<std::weak_ptr<GptoTask>, size_t, std::owner_less<>> consumers_inputs_;
  std::vector<std::vector<Time>> lifetime_end_;                      // for multi-threading
  std::vector<std::vector<std::weak_ptr<GptoTask>>> last_consumer_;  // for multi-threading
  bool contiguous_;

 public:
  GptoTensor(const size_t &id, const Memory &original_weight, const Memory &weight,
             const std::weak_ptr<GptoTask> &source, const GptoTensorType &type) {
    id_ = id;
    original_weight_ = original_weight;
    weight_ = weight;
    source_ = source;
    type_ = type;
    contiguous_ = false;
  }

  GptoTensor(const size_t &id, const Memory &original_weight, const Memory &weight,
             const std::weak_ptr<GptoTask> &source, const GptoTensorType &type, const ShapeVector &shape,
             const TypeId &dtype) {
    id_ = id;
    original_weight_ = original_weight;
    weight_ = weight;
    source_ = source;
    type_ = type;
    shape_ = shape;
    dtype_ = dtype;
    contiguous_ = false;
  }

  GptoTensor(const GptoTensor &t) {
    id_ = t.id_;
    original_weight_ = t.original_weight_;
    weight_ = t.weight_;
    source_ = t.source_;
    type_ = t.type_;
    consumers_ = t.consumers_;
    consumers_inputs_ = t.consumers_inputs_;
    shape_ = t.shape_;
    dtype_ = t.dtype_;
    lifetime_end_ = t.lifetime_end_;
    last_consumer_ = t.last_consumer_;
    contiguous_ = t.contiguous_;
  }

  ~GptoTensor() {
    consumers_.clear();
    consumers_inputs_.clear();
  }

  const size_t &id() const { return id_; }
  const Memory &original_weight() const { return original_weight_; }
  const Memory &weight() const { return weight_; }
  const std::weak_ptr<GptoTask> &source() { return source_; }
  const GptoTensorType &type() const { return type_; }
  const ShapeVector &shape() const { return shape_; }
  const TypeId &dtype() const { return dtype_; }
  std::set<std::weak_ptr<GptoTask>, GptoTask::SortByIdWeak> &consumers() { return consumers_; }
  std::map<std::weak_ptr<GptoTask>, size_t, std::owner_less<>> &consumers_inputs() { return consumers_inputs_; }
  std::vector<std::vector<Time>> &lifetime_end() { return lifetime_end_; }
  const Time &lifetime_end(const size_t &t, const size_t &p) { return lifetime_end_[t][p]; }
  std::vector<std::vector<std::weak_ptr<GptoTask>>> &last_consumer() { return last_consumer_; }
  const std::weak_ptr<GptoTask> &last_consumer(const size_t &t, const size_t &p) { return last_consumer_[t][p]; }
  const bool &contiguous() { return contiguous_; }

  void init_lifetime_end(const size_t &num_task_sort, const size_t &num_pes_sort) {
    lifetime_end_.assign(num_task_sort, std::vector<Time>(num_pes_sort, static_cast<Time>(0)));
  }
  void init_last_consumer(const size_t &num_task_sort, const size_t &num_pes_sort) {
    last_consumer_.assign(num_task_sort, std::vector<std::weak_ptr<GptoTask>>(num_pes_sort, std::weak_ptr<GptoTask>()));
  }

  void set_type(const GptoTensorType &type) { type_ = type; }
  void set_shape(const ShapeVector &shape) { shape_ = shape; }
  void set_dtype(const TypeId &dtype) { dtype_ = dtype; }
  void set_original_weight(const Memory &original_weight) { original_weight_ = original_weight; }
  void set_weight(const Memory &weight) { weight_ = weight; }
  void set_lifetime_end(const Time &le, const size_t &t, const size_t &p) { lifetime_end_[t][p] = le; }
  void set_last_consumer(const std::weak_ptr<GptoTask> &lc, const size_t &t, const size_t &p) {
    last_consumer_[t][p] = lc;
  }
  void set_contiguous(const bool &cont) { contiguous_ = cont; }
};
using GptoTensorPtr = std::shared_ptr<GptoTensor>;

struct GptoTensorIdSort {
  bool operator()(const GptoTensorPtr &t1, const GptoTensorPtr &t2) const { return t1->id() < t2->id(); }
};

struct Interval {  // Information extracted by scheduling
  GptoTaskPtr task;
  Time start;
  Time end;
  PeId pid;
};

// Sorting for scheduling to dependencies (events)
struct SortByStartMin {
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

struct SortByEndMin {
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

struct SortByEndMax {
  bool operator()(const Interval &interval1, const Interval &interval2) const {
    const auto &id1 = interval1.task->id();
    const auto &start1 = interval1.start;
    const auto &end1 = interval1.end;
    const auto &id2 = interval2.task->id();
    const auto &start2 = interval2.start;
    const auto &end2 = interval2.end;
    return end1 > end2 || (end1 == end2 && start1 > start2) || (end1 == end2 && start1 == start2 && id1 > id2);
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
bool SortByTypePriority(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByPredComm(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByPredCommDepth(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByPredCube(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByGreedyHeight(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortBySValue(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByAValue(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByMValue(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByWeightedSValue(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByWeightedAValue(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByWeightedMValue(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByCostSValue(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByCostAValue(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByCostMValue(const GptoTaskPtr &, const GptoTaskPtr &);
bool SortByReversePostOrder(const GptoTaskPtr &, const GptoTaskPtr &);

inline constexpr TaskSortFunction TASK_SORT[] = {SortByCostMax,
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
                                                 SortByTypePriority,
                                                 SortByPredComm,
                                                 SortByPredCommDepth,
                                                 SortByPredCube,
                                                 SortByGreedyHeight,
                                                 SortBySValue,
                                                 SortByAValue,
                                                 SortByMValue,
                                                 SortByWeightedSValue,
                                                 SortByWeightedAValue,
                                                 SortByWeightedMValue,
                                                 SortByCostSValue,
                                                 SortByCostAValue,
                                                 SortByCostMValue,
                                                 SortByReversePostOrder};

inline constexpr std::string_view TASK_SORT_NAMES[] = {"SortByCostMax",
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
                                                       "SortByTypePriority",
                                                       "SortByPredComm",
                                                       "SortByPredCommDepth",
                                                       "SortByPredCube",
                                                       "SortByGreedyHeight",
                                                       "SortBySValue",
                                                       "SortByAValue",
                                                       "SortByMValue",
                                                       "SortByWeightedSValue",
                                                       "SortByWeightedAValue",
                                                       "SortByWeightedMValue",
                                                       "SortByCostSValue",
                                                       "SortByCostAValue",
                                                       "SortByCostMValue",
                                                       "SortByReversePostOrder"};

inline constexpr std::string_view PE_NAME_SORT[] = {"SortByLoad", "SortByValidStart"};

// Scheduling to dependencies (events) functions
bool Overlap(const Time &, const Time &, const Time &, const Time &);
size_t ScheduleToEvents(const SchedulingOutput &);
size_t MemorySafetyEvents(const SchedulingOutput &);
size_t LiftEvents(const SchedulingOutput &, const std::unordered_set<size_t> &);
void MockExecutionOrder(const SchedulingOutput &,
                        std::vector<std::pair<CNodePtr, std::tuple<char, size_t, size_t, size_t>>> *, const size_t &);
void UpdateExecutionOrder(const KernelGraphPtr &, const SchedulingOutput &);

// Task-related functions
GptoTaskType GetType(const CNodePtr, std::unordered_set<size_t> *);
std::unordered_map<std::string, Time> GetProfCost(const std::string &);

// Tensor-related functions
std::tuple<std::unordered_map<std::string, std::vector<Memory>>, std::unordered_map<std::string, std::vector<Memory>>>
GetTensorsWeight();
size_t GetAlignedSize(size_t);
void ExtractRealTensors(const SchedulingInput &, std::unordered_map<CNodePtr, GptoTaskPtr> *);
void StandardInputCase(const GptoTaskPtr &, std::unordered_set<void *> *, const CNodePtr &,
                       std::unordered_map<CNodePtr, GptoTaskPtr> *);
void CleanOutput(size_t, CNodePtr, GptoTaskPtr, const GptoTaskPtr &);
void CleanWorkspace(CNodePtr, const GptoTaskPtr &, const GptoTaskPtr &);
void ExtractOutputWorkspaceTensors(const std::vector<GptoTaskPtr> &);
KernelWithIndex GetVisitKernelWithReturnType(const AnfNodePtr &, size_t, std::unordered_map<CNodePtr, GptoTaskPtr> *);
void GraphOutputProcess(const KernelGraphPtr &, std::unordered_map<CNodePtr, GptoTaskPtr> *);
std::vector<std::pair<GptoTensorPtr, GptoTensorPtr>> RefNodeProcess(const KernelGraphPtr &,
                                                                    std::unordered_map<CNodePtr, GptoTaskPtr> *);
std::vector<std::vector<GptoTensorPtr>> CommunicationNodeProcess(std::unordered_map<CNodePtr, GptoTaskPtr> *);
std::map<GptoTensorPtr, GptoTensorPtr> GetRefTensorsInContiguousList(const std::vector<std::vector<GptoTensorPtr>> &);
std::map<size_t, size_t> GetContiguousRefIndexMap(const std::vector<std::vector<GptoTensorPtr>> &,
                                                  const std::vector<std::vector<GptoTensorPtr>> &);
void UpdateRefNodeGpto(const KernelGraphPtr &, std::unordered_map<CNodePtr, GptoTaskPtr> *);

// Scheduling main functions
SchedulingInput ExtractSchedulingInput(const KernelGraphPtr &, std::unordered_map<CNodePtr, GptoTaskPtr> *, bool *);
SchedulingOutput MemAwareScheduler(const SchedulingInput &, const std::unordered_set<size_t> &, size_t *);
void MemAwareSchedulerPre(const std::vector<GptoTaskPtr> &);
void MemAwareSchedulerLockFreeThreadSafe(std::unordered_map<size_t, std::unordered_map<size_t, SchedulingOutput>> *,
                                         const std::vector<GptoTaskPtr> &, const bool &);
void MemAwareSchedulerReleaseMem(std::unordered_map<size_t, std::unordered_map<size_t, SchedulingOutput>> *,
                                 const std::vector<GptoTaskPtr> &, const bool &, const std::string &, const size_t &,
                                 const size_t &);
SchedulingOutput MemAwareSchedulerCore(const std::vector<GptoTaskPtr> &, const std::map<GptoTaskType, size_t> &,
                                       const size_t &, const size_t &);
std::tuple<GptoTaskPtr, Time, PeId> ScheduleTaskLoad(
  std::set<GptoTaskPtr, TaskSortFunction> *,
  std::unordered_map<GptoTaskType, std::set<ProcessingElement, SortByLoad>> *, std::vector<Time> *,
  std::map<Time, Memory> *, size_t *, const size_t &, const size_t &);
std::tuple<GptoTaskPtr, Time, PeId> ScheduleTaskStart(
  std::set<GptoTaskPtr, TaskSortFunction> *, std::unordered_map<GptoTaskType, std::vector<ProcessingElement>> *,
  std::vector<Time> *, std::map<Time, Memory> *, size_t *, const size_t &, const size_t &);
void AddMemory(const GptoTaskPtr &, const Time &, std::map<Time, Memory> *);
void SubtractMemory(std::set<GptoTaskPtr, TaskSortFunction> *, const GptoTaskPtr &,
                    std::unordered_map<size_t, std::set<std::weak_ptr<GptoTask>, GptoTask::SortByIdWeak>> *,
                    std::map<Time, Memory> *, const size_t &, const size_t &);
bool MemoryViolated(const GptoTaskPtr &, const Time &, const std::map<Time, Memory> &, const size_t &, bool *,
                    const size_t &, const size_t &);
bool MemoryViolatedCore(const GptoTaskPtr &, const Time &, const std::map<Time, Memory> &, const size_t &,
                        const size_t &);
void UpdateCandidates(std::set<GptoTaskPtr, TaskSortFunction> *, const GptoTaskPtr &, std::vector<size_t> *,
                      std::vector<Time> *, Time *, std::map<GptoTaskType, Time> *, Time *,
                      std::unordered_set<GptoTaskPtr> *, const size_t &, const size_t &, const size_t &);

std::tuple<size_t, size_t, size_t, size_t> GetBestSolutions(
  std::unordered_map<size_t, std::unordered_map<size_t, SchedulingOutput>> *, std::string *,
  std::pair<std::string, Memory> *);
void PrintBestSolutionStats(const SchedulingOutput &, const std::vector<GptoTaskPtr> &,
                            const std::map<GptoTaskType, size_t> &, const std::string &,
                            const std::pair<std::string, Memory> &, size_t *);

// Scheduling: auxiliary functions
void SetGPTOMode();
void InitializeTasks(const std::vector<GptoTaskPtr> &, std::vector<Time> *, std::vector<size_t> *,
                     std::set<GptoTaskPtr, TaskSortFunction> *, std::unordered_set<GptoTaskPtr> *,
                     std::unordered_map<size_t, std::set<std::weak_ptr<GptoTask>, GptoTask::SortByIdWeak>> *,
                     const size_t &, const size_t &);
void InsertEdges(const KernelGraphPtr &, std::unordered_map<CNodePtr, GptoTaskPtr> *);
std::map<GptoTaskType, size_t> GetPEs(const std::unordered_set<size_t> &);
void MakeGraphRoot(const GptoTaskPtr &, const GptoTaskPtr &, std::unordered_map<CNodePtr, GptoTaskPtr> *);
void InitializeProcessingElements(const std::map<GptoTaskType, size_t> &,
                                  std::unordered_map<GptoTaskType, std::set<ProcessingElement, SortByLoad>> *,
                                  std::unordered_map<GptoTaskType, std::vector<ProcessingElement>> *, const size_t &);
void PushTasksToVisit(std::queue<GptoTaskPtr> *, std::unordered_map<size_t, size_t> *, std::weak_ptr<GptoTask> &,
                      GptoTaskPtr, const size_t &);
void InitializeTaskInlineCondition(const CNodePtr &, GptoTaskPtr *,
                                   std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> *,
                                   std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> *);
void ExtractSwitchGather(std::map<GptoTaskPtr, GptoTaskPtr, TaskDepthSort> *,
                         std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> *,
                         std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> *);
void UpdateTasksInlineCondition(std::unordered_map<CNodePtr, GptoTaskPtr> *,
                                std::map<GptoTaskPtr, GptoTaskPtr, TaskDepthSort> *);
void UpdateExecutionOrder(const KernelGraphPtr &, const SchedulingOutput &);
bool SkipAlgorithm(const size_t &, const size_t &, const bool &);

// Compute auxiliary values for task sorting criteria
void ComputeBottomLevelAndWeightedLength(const std::vector<GptoTaskPtr> &);
void ComputeDepthAndTopLevel(const std::vector<GptoTaskPtr> &);
void ComputePredComm(const std::vector<GptoTaskPtr> &);
void ComputePredCube(const std::vector<GptoTaskPtr> &);
void ComputeSuccDiffType(const std::vector<GptoTaskPtr> &);
void ComputePostOrder(const std::vector<GptoTaskPtr> &);

// Memory-aware scheduling
void CalculateMemoryLimit(const KernelGraphPtr &, const std::pair<Time, Memory> &,
                          const std::unordered_map<CNodePtr, GptoTaskPtr> &);
void ComputeInitialMemoryImpact(const std::vector<GptoTaskPtr> &);
[[maybe_unused]] void ExtractTensors(const std::vector<GptoTaskPtr> &, std::set<GptoTensorPtr, GptoTensorIdSort> *);
[[maybe_unused]] void ComputeAncestorsDescendants(
  const std::vector<GptoTaskPtr> &,
  std::vector<mindspore::somas::DynamicBitSet> *);  // only needed for memory lower bound
Memory CalculateTaskLowerBound(const GptoTaskPtr &, const std::set<GptoTensorPtr, GptoTensorIdSort> &,
                               const std::vector<mindspore::somas::DynamicBitSet> &);

[[maybe_unused]] Memory MemoryLowerBound(const std::vector<GptoTaskPtr> &,
                                         const std::vector<mindspore::somas::DynamicBitSet> &,
                                         const std::set<GptoTensorPtr, GptoTensorIdSort> &);

// Makespan lower bounds
Time LowerBoundBottomLevel(const std::vector<GptoTaskPtr> &);
Time LowerBoundPEs(const std::vector<GptoTaskPtr> &, const std::map<GptoTaskType, size_t> &);
Time TotalTime(const std::vector<GptoTaskPtr> &);

// Verification functions
bool VerifyAll(const std::vector<GptoTaskPtr> &, std::map<Time, Memory> *, const size_t &, const size_t &);
bool VerifyDAG(const std::vector<GptoTaskPtr> &);
bool VerifyScheduling(const std::vector<GptoTaskPtr> &, const size_t &, const size_t &);
bool VerifyMemory(const std::vector<GptoTaskPtr> &, std::map<Time, Memory> *, const size_t &, const size_t &);

// Printing log files
[[maybe_unused]] void LogSchedulingOutput(const SchedulingOutput &, const KernelGraphPtr &,
                                          const std::set<GptoTensorPtr, GptoTensorIdSort> &, const Memory,
                                          std::unordered_set<size_t> *, const size_t &, const std::string &);
void PrintParameterInfo(const std::vector<Interval> &, std::ostringstream *);
[[maybe_unused]] std::pair<Time, Memory> LogBaseline(const std::unordered_map<CNodePtr, GptoTaskPtr> &,
                                                     const KernelGraphPtr &, bool);
[[maybe_unused]] Memory MemoryEstimateBaseline(const std::vector<CNodePtr> &,
                                               const std::unordered_map<CNodePtr, GptoTaskPtr> &,
                                               std::unordered_map<GptoTaskId, Time> *,
                                               std::unordered_map<GptoTaskId, Time> *);

// Debug context function
bool IsGptoDebug();

// SAM sorting
void InitializeInTensorsWeight(const GptoTaskPtr &);

void InitializeS(const GptoTaskPtr &, const size_t &, const size_t &, const size_t &);
void InitializeSW(const GptoTaskPtr &, const size_t &, const size_t &, const size_t &);
void InitializeSC(const GptoTaskPtr &, const size_t &, const size_t &, const size_t &);

void InitializeA(const GptoTaskPtr &, const size_t &, const size_t &, const size_t &);
void InitializeAW(const GptoTaskPtr &, const size_t &, const size_t &, const size_t &);
void InitializeAC(const GptoTaskPtr &, const size_t &, const size_t &, const size_t &);

void InitializeM(const GptoTaskPtr &, const size_t &, const size_t &, const size_t &);
void InitializeMW(const GptoTaskPtr &, const size_t &, const size_t &, const size_t &);
void InitializeMC(const GptoTaskPtr &, const size_t &, const size_t &, const size_t &);

void UpdateS(const GptoTaskPtr &, [[maybe_unused]] const size_t &, [[maybe_unused]] const size_t &,
             [[maybe_unused]] const size_t &);
void UpdateSW(const GptoTaskPtr &, [[maybe_unused]] const size_t &, [[maybe_unused]] const size_t &,
              [[maybe_unused]] const size_t &);
void UpdateSC(const GptoTaskPtr &, [[maybe_unused]] const size_t &, [[maybe_unused]] const size_t &,
              [[maybe_unused]] const size_t &);

void UpdateA(const GptoTaskPtr &, [[maybe_unused]] const size_t &, [[maybe_unused]] const size_t &,
             [[maybe_unused]] const size_t &);
void UpdateAW(const GptoTaskPtr &, [[maybe_unused]] const size_t &, [[maybe_unused]] const size_t &,
              [[maybe_unused]] const size_t &);
void UpdateAC(const GptoTaskPtr &, [[maybe_unused]] const size_t &, [[maybe_unused]] const size_t &,
              [[maybe_unused]] const size_t &);

void UpdateM(const GptoTaskPtr &, [[maybe_unused]] const size_t &, [[maybe_unused]] const size_t &,
             [[maybe_unused]] const size_t &);
void UpdateMW(const GptoTaskPtr &, [[maybe_unused]] const size_t &, [[maybe_unused]] const size_t &,
              [[maybe_unused]] const size_t &);
void UpdateMC(const GptoTaskPtr &, [[maybe_unused]] const size_t &, [[maybe_unused]] const size_t &,
              [[maybe_unused]] const size_t &);

bool VerifyS(const std::vector<GptoTaskPtr> &, const size_t &, const size_t &);
bool VerifySW(const std::vector<GptoTaskPtr> &, const size_t &, const size_t &);
bool VerifySC(const std::vector<GptoTaskPtr> &, const size_t &, const size_t &);

bool VerifyA(const std::vector<GptoTaskPtr> &, const size_t &, const size_t &);
bool VerifyAW(const std::vector<GptoTaskPtr> &, const size_t &, const size_t &);
bool VerifyAC(const std::vector<GptoTaskPtr> &, const size_t &, const size_t &);

bool VerifyM(const std::vector<GptoTaskPtr> &, const size_t &, const size_t &);
bool VerifyMW(const std::vector<GptoTaskPtr> &, const size_t &, const size_t &);
bool VerifyMC(const std::vector<GptoTaskPtr> &, const size_t &, const size_t &);

using UpdateFunctionSAM = void (*)(const GptoTaskPtr &, [[maybe_unused]] const size_t &,
                                   [[maybe_unused]] const size_t &, [[maybe_unused]] const size_t &);
using InitializeFunctionSAM = void (*)(const GptoTaskPtr &, const size_t &, const size_t &, const size_t &);
using VerifyFunctionSAM = bool (*)(const std::vector<GptoTaskPtr> &, const size_t &, const size_t &);

inline std::unordered_map<std::string_view, UpdateFunctionSAM> UPDATE_SAM = {
  {"SortBySValue", UpdateS},          {"SortByAValue", UpdateA},          {"SortByMValue", UpdateM},
  {"SortByWeightedSValue", UpdateSW}, {"SortByWeightedAValue", UpdateAW}, {"SortByWeightedMValue", UpdateMW},
  {"SortByCostSValue", UpdateSC},     {"SortByCostAValue", UpdateAC},     {"SortByCostMValue", UpdateMC},
};

inline std::unordered_map<std::string_view, InitializeFunctionSAM> INIT_SAM = {
  {"SortBySValue", InitializeS},          {"SortByAValue", InitializeA},
  {"SortByMValue", InitializeM},          {"SortByWeightedSValue", InitializeSW},
  {"SortByWeightedAValue", InitializeAW}, {"SortByWeightedMValue", InitializeMW},
  {"SortByCostSValue", InitializeSC},     {"SortByCostAValue", InitializeAC},
  {"SortByCostMValue", InitializeMC},
};

inline std::unordered_map<std::string_view, VerifyFunctionSAM> VERIFY_SAM = {
  {"SortBySValue", VerifyS},          {"SortByAValue", VerifyA},          {"SortByMValue", VerifyM},
  {"SortByWeightedSValue", VerifySW}, {"SortByWeightedAValue", VerifyAW}, {"SortByWeightedMValue", VerifyMW},
  {"SortByCostSValue", VerifySC},     {"SortByCostAValue", VerifyAC},     {"SortByCostMValue", VerifyMC},
};

// Integration function
void GPTO(const KernelGraphPtr &, std::vector<std::pair<CNodePtr, std::tuple<char, size_t, size_t, size_t>>> *);
void RunGPTO(const NotNull<KernelGraphPtr> &,
             std::vector<std::pair<CNodePtr, std::tuple<char, size_t, size_t, size_t>>> &);
}  // namespace gpto
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_GPTO_H_
