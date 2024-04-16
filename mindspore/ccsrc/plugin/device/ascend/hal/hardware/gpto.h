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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_GPTO_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_GPTO_H_

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
// DynamicBitSet data structure definition: copied from somas - only used for optional memory lower bound calculation
constexpr auto kHalfByteSize = 4;
class DynamicBitSet {
  const size_t bit_width_ = 64;

  inline size_t GetIndex(size_t index) const { return index / bit_width_; }

  inline uint64_t GetBitMask(size_t index) const {
    return ((static_cast<uint64_t>(0x1)) << ((bit_width_ - 1) - (index % bit_width_)));
  }

  inline void Reset(uint64_t val) {
    bit_.clear();
    for (size_t i = 0; i < bit_size_; i++) {
      bit_.push_back(val);
    }
  }

 public:
  size_t bit_size_;
  std::vector<uint64_t> bit_;
  explicit DynamicBitSet(size_t count) : bit_size_((count + bit_width_ - 1) / bit_width_) { Reset(0x0); }
  ~DynamicBitSet() = default;

  void SetBitTrue(size_t index, bool log = false) {
    if (log) {
      MS_LOG(INFO) << GetIndex(index) << " " << GetBitMask(index);
    }
    bit_[GetIndex(index)] |= GetBitMask(index);
  }

  void SetBitFalse(size_t index) { bit_[GetIndex(index)] &= (~GetBitMask(index)); }
  bool IsBitTrue(size_t index) const { return (bit_[GetIndex(index)] & GetBitMask(index)) != 0x0; }
	bool IsBitFalse(size_t index) const { return !IsBitTrue(index); }

  size_t CountOnesNum() const {
    size_t ret = 0;
    static unsigned char ones_num_in_hex[] = "\0\1\1\2\1\2\2\3\1\2\2\3\2\3\3\4";
    for (size_t i = 0; i < bit_size_; i++) {
      auto value = bit_[i];
      if (value == 0) {
        continue;
      }
      auto *char_value = reinterpret_cast<unsigned char *>(&value);
      for (size_t j = 0; j < bit_width_ / CHAR_BIT; j++) {
        ret += ones_num_in_hex[static_cast<int>(char_value[j] & 0xF)];
        char_value[j] >>= kHalfByteSize;
        ret += ones_num_in_hex[static_cast<int>(char_value[j] & 0xF)];
      }
    }
    return ret;
  }

  void Log() {
    std::cout << "Start Print Bitset ";
    for (size_t i = 0; i < bit_size_; i++) {
      std::cout << " bit [" << std::dec << i << "] = " << std::hex << bit_[i] << std::dec;
    }
    std::cout << std::endl;
  }

  friend void Union(DynamicBitSet *a, DynamicBitSet *b) {
    for (size_t i = 0; i < (*a).bit_size_; i++) {
      (*a).bit_[i] |= (*b).bit_[i];
    }
  }
};

// Preliminary definitions
using Time = uint64_t;	// size_t;
using Memory = int64_t;	// maintain memory as signed integer, since memory impact of some operators may be negative
using TaskId = size_t;
using PeId = size_t;
enum TaskType { kNone = 0, kComp, kComm, kCube };
enum TensorType { kOutput = 0, kWorkspace };	// kOutput: from one task to another, kWorkspace: workspace or to other subgraphs

struct ProcessingElement {
  PeId id;
  TaskType gpto_type;
  Time load;
  std::list<std::pair<Time, Time>> idle;
};

struct Interval {  // Information extracted by scheduling
  TaskId id;
  std::string name;
  TaskType gpto_type;
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
  kSortByPredCommDepth,
  kSortByPredCube,
  kSortByGreedyHeight,
  kNumTaskSort
};

class Task;

// GPTO Tensor definitions
class Tensor {
	private:
		size_t id_;
		Memory weight_;
		std::shared_ptr<Task> source_;
		TensorType type_;
		std::set<std::shared_ptr<Task>> consumers_;
		
	public:
		Tensor(const size_t id, const Memory weight, const std::shared_ptr<Task> source, const TensorType type){
			id_ = id;
			weight_ = weight;
			source_ = source;
			type_ = type;
		}
		
		Tensor(const Tensor &t){
			id_ = t.id_;
			weight_ = t.weight_;
			source_ = t.source_;
			type_ = t.type_;
			consumers_ = t.consumers_;
		}
		
		~Tensor() { consumers_.clear(); };
		
		const size_t& id() const { return id_; }
		const Memory& weight() const { return weight_; }
		const std::shared_ptr<Task>& source() { return source_; }
		const TensorType& type() const { return type_; }
		std::set<std::shared_ptr<Task>>& consumers() { return consumers_; }	

    void set_type(TensorType type) { type_ = type; }
};
using TensorPtr = std::shared_ptr<Tensor>;

// GPTO Task definitions
class Task {
 public:
	struct SortByIdWeak {
		bool operator()(const std::weak_ptr<Task> &task1, const std::weak_ptr<Task> &task2) const {
			return task1.lock()->id() < task2.lock()->id();
		}
	};

	struct SortByIdShared {
		bool operator()(const std::shared_ptr<Task> &task1, const std::shared_ptr<Task> &task2) const {
			return task1->id() < task2->id();
		}
	}; 
 
  Task(const TaskId &id, const TaskType &real_type, const TaskType &gpto_type, const std::string &name) {
    id_ = id;
    real_type_ = real_type;
    gpto_type_ = gpto_type;
		cnode_ = nullptr;
    weight_ = 1;
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
    mem_impact_ = 0;
		workspace_memory_ = 0;
		lower_bound_ = 0;
		subgraph_id_ = SIZE_MAX;
		condition_switch_ = false;
		condition_gather_ = false;
  }
	
	Task(const Task &t){
		id_ = t.id_;
    real_type_ = t.real_type_;
    gpto_type_ = t.gpto_type_;
		cnode_ = t.cnode_;
    weight_ = t.weight_;
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
    mem_impact_ = t.mem_impact_;
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

  TaskId id() const { return id_; }
  TaskType real_type() const { return real_type_; }
  TaskType gpto_type() const { return gpto_type_; }
  CNodePtr cnode() const { return cnode_; }
  Time weight() const { return weight_; }
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
  Memory mem_impact() const { return mem_impact_; }
	Memory workspace_memory() const { return workspace_memory_; }
	Time lower_bound() const { return lower_bound_; }
	size_t subgraph_id() const { return subgraph_id_; }
	bool condition_switch() const { return condition_switch_; }
	bool condition_gather() const { return condition_gather_; }
	size_t original_order() const { return original_order_; }

  std::set<std::weak_ptr<Task>,SortByIdWeak>& parents() { return parents_; }
  std::set<std::shared_ptr<Task>,SortByIdShared>& children() { return children_; }
  std::vector<TensorPtr>& in_tensors() { return in_tensors_; }
  std::vector<TensorPtr>& out_tensors() { return out_tensors_; }
	std::vector<TensorPtr>& workspace_tensors() { return workspace_tensors_; }

  void set_id(TaskId id) { id_ = id; }
  void set_real_type(TaskType real_type) { real_type_ = real_type; }
  void set_gpto_type(TaskType gpto_type) { gpto_type_ = gpto_type; }
  void set_cnode(CNodePtr cnode) { cnode_ = cnode; }
  void set_weight(Time weight) { weight_ = weight; }
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
  void set_mem_impact(Memory mem_add) { mem_impact_ = mem_add; }
	void set_workspace_memory(Memory workspace_memory) { workspace_memory_ = workspace_memory; }
  void set_lower_bound(Time lb) { lower_bound_ = lb; }
	void set_subgraph_id(size_t id) { subgraph_id_ = id; }
	void set_condition_switch(bool cond) { condition_switch_ = cond; }
	void set_condition_gather(bool cond) { condition_gather_ = cond; }
	void set_original_order(size_t order) { original_order_ = order; }

  void AddParent(std::weak_ptr<Task> parent) {
    parents_.insert(parent);
  }
  void RemoveParent(std::weak_ptr<Task> parent) {
    parents_.erase(parent);
  }
  void ClearParents() { 
		parents_.clear(); 
	}
	
  void AddChild(std::shared_ptr<Task> child) {
    children_.insert(child);
  }
	 
  void RemoveChild(std::shared_ptr<Task> child){
      children_.erase(child);
  }
  void ClearChildren() { 
		children_.clear(); 
	}

  bool HasChild(std::shared_ptr<Task> child) {
    return std::find(children_.begin(), children_.end(), child) != children_.end();
  }

  void AssignWeight(Time weight) {
    if (weight == 0) {
      weight_ = 1;
    } else {
      weight_ = weight;
    }
  }

  void ResetStartEnd() {
    start_ = SIZE_MAX;
    end_ = 0;
  }

 private:
  TaskId id_;
  TaskType real_type_;
  TaskType gpto_type_;
  CNodePtr cnode_;

  Time weight_;
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
	Memory mem_impact_;
	Memory workspace_memory_;
	Time lower_bound_;
	
	size_t subgraph_id_;
	bool condition_switch_;
	bool condition_gather_;
	size_t original_order_;

  std::set<std::weak_ptr<Task>,SortByIdWeak> parents_;
  std::set<std::shared_ptr<Task>, SortByIdShared> children_;
  std::vector<std::shared_ptr<Tensor>> in_tensors_;
  std::vector<std::shared_ptr<Tensor>> out_tensors_;
	std::vector<std::shared_ptr<Tensor>> workspace_tensors_;
};
using TaskPtr = std::shared_ptr<Task>;
using TaskSortFunction = bool (*)(std::shared_ptr<Task> const &, std::shared_ptr<Task> const &);

// GPTO Scheduling definitions
struct SchedulingInput {
  std::vector<std::shared_ptr<Task>> tasks;
};

struct SchedulingOutput {
  std::vector<Interval> task_times;
  Time makespan;
	Memory memory_peak;
};

namespace gpto {	// Graph Parallel Topology Optimizer
// Main functionality
SchedulingInput ExtractSchedulingInput(const std::vector<CNodePtr> &, std::unordered_map<CNodePtr, TaskPtr> *, std::set<std::shared_ptr<Tensor>> &);
SchedulingOutput Process(SchedulingInput &, const size_t, const FuncGraphPtr &, const std::set<TensorPtr> &);
SchedulingOutput ProcessCore(std::vector<std::shared_ptr<Task>> &, std::unordered_map<TaskType, int32_t> &,
                             const TaskSortFunction &, bool);

// Compute auxiliary values for task sorting criteria
void ComputeBottomLevelAndWeightedLength(std::vector<std::shared_ptr<Task>> &);
void ComputeDepthAndTopLevel(std::vector<std::shared_ptr<Task>> &);
void ComputePredComm(std::vector<std::shared_ptr<Task>> &);
void ComputePredCube(std::vector<std::shared_ptr<Task>> &);

// Functions for memory-aware scheduling
void InitializeMemoryImpact(std::vector<std::shared_ptr<Task>> &);
void ComputeAncestorsDescendants(const std::vector<std::shared_ptr<Task>>&, std::vector<DynamicBitSet>&);	// only needed for memory lower bound (optional)

// Makespan lower bounds
Time LowerBoundBottomLevel(std::vector<std::shared_ptr<Task>> &);
Time LowerBoundPEs(std::vector<std::shared_ptr<Task>> &, std::unordered_map<TaskType, int32_t> &);

// Dependency generation
std::vector<std::pair<TaskId, TaskId>> ScheduleToDependencies(const SchedulingOutput &);								// guide for real dependency generation
void AddRealDependencies(const FuncGraphManagerPtr &, const std::vector<CNodePtr> &, const std::vector<std::pair<TaskId, TaskId>> &, std::unordered_map<CNodePtr, TaskPtr> *);
std::vector<std::pair<TaskId, TaskId>> ScheduleToDependenciesDifferentTypes(const SchedulingOutput &);	// kbk event generation

// Verification functions
bool VerifyDAG(std::vector<std::shared_ptr<Task>> &);
bool VerifyScheduling(std::vector<std::shared_ptr<Task>> &);
bool VerifyDependencies(std::vector<std::shared_ptr<Task>> &, std::vector<std::pair<TaskId, TaskId>> &);

// Printing log files
void PrintLog(const SchedulingOutput &, const std::vector<std::pair<TaskId, TaskId>> &, const FuncGraphPtr &, size_t, std::set<std::shared_ptr<Tensor>> &tensors);
void PrintLogBaseline(const SchedulingInput &, const std::vector<std::shared_ptr<CNode>> &, std::unordered_map<std::shared_ptr<CNode>, std::shared_ptr<Task> >*, const FuncGraphPtr &, size_t);
void PrintLogForILP(const SchedulingInput &, const SchedulingOutput &, size_t, const FuncGraphPtr &, const Time, const std::set<TensorPtr> &);
} // namespace GPTO
// Integration function
std::vector<std::pair<CNodePtr,CNodePtr>> GPTO(const FuncGraphPtr &);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_GPTO_H_

