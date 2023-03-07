/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SPLIT_MODEL_AREA_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SPLIT_MODEL_AREA_H_

#include <memory>
#include <vector>
#include <utility>
#include <string>
#include "utils/hash_map.h"
#include "backend/common/graph_kernel/model/op_node.h"

namespace mindspore::graphkernel::inner {
using NodePattern = PrimOp::ComputeType;
// EdgeRelation indicates the pattern of node's input edges.
// the INJECTIVE means the input is directly sent into the kernel,
// the BROADCAST means the input is implicit broadcasted.
//
// Note, it should be distinguished from the PrimOp::ComputeType,
// which indicates the INNER logic of kernels.
enum class EdgeRelation : int { INJECTIVE = 0, BROADCAST = 1 };

// AreaMode indicates the finally mode of kernels.
// the BASIC means the node(s) of area will be inlined into the main graph
// the COMPOSITE means the node(s) of area will be kept as a GraphKernel node.
enum class AreaMode { BASIC, COMPOSITE };

class Area;
using AreaPtr = std::shared_ptr<Area>;
using AreaWithRelation = std::pair<AreaPtr, EdgeRelation>;

// Area is used to maintain the operator set that was fused.
class Area : public std::enable_shared_from_this<Area> {
  // NodeHandle is used to maintain the input and user edges of areas.
  // The handle's inputs should be other areas' handle.
  //
  // This class is derived from PrimOp, to reuse the compute_type field
  // and to avoid overriding pure virtual functions (if exists).
  //
  // This class is not visible outside the class Area.
  class NodeHandle : public PrimOp {
   public:
    NodeHandle(Area *area, const PrimOpPtr &p) : PrimOp("", p->compute_type()), area_(area) {}
    ~NodeHandle() = default;
    using PrimOp::compute_type_;
    AreaPtr area() const { return area_->shared_from_this(); }

   private:
    Area *const area_;
  };  // class Area::NodeHandle

 public:
  Area(size_t id, const PrimOpPtr &prim_op, bool is_output, const HashMap<NodePtr, AreaPtr> &node_area_map);
  ~Area() = default;

  size_t id() const { return unique_id_; }
  const AreaPtr &input(size_t i) const { return inputs_with_relation_[i].first; }
  std::vector<AreaPtr> inputs() const;
  EdgeRelation input_relation(size_t i) const { return inputs_with_relation_[i].second; }
  const std::vector<AreaWithRelation> &inputs_with_relation() const { return inputs_with_relation_; }
  size_t input_num() const { return inputs_with_relation_.size(); }
  // get the number of operators in the area
  size_t size() const { return ops_.size(); }
  std::vector<AreaPtr> users() const;
  std::vector<AreaWithRelation> users_with_relation() const;
  size_t user_num() const { return hd_->users().size(); }
  AreaMode mode() const { return mode_; }
  // get the dominant op node
  PrimOpPtr dom() const { return IsAlive() ? ops_[0] : nullptr; }
  NodePattern pattern() const { return hd_->compute_type(); }
  const std::vector<PrimOpPtr> &ops() const { return ops_; }
  bool is_output() const { return is_output_; }
  int64_t compute_size() const;

  // check whether the area is alive(true) or is fused(false)
  bool IsAlive() const { return !ops_.empty(); }
  std::string ToString() const;
  void SetOps(const std::vector<PrimOpPtr> &ops) { ops_ = ops; }
  void SetMode(AreaMode mode) { mode_ = mode; }
  // fuse `input_area` into `this` area. after that, the `input_area` will be discarded.
  // the `input_area` node should be in the input list of `this` area.
  void FuseInput(const AreaPtr &input_area);

 protected:
  // Make the inputs unique, and sync the inputs to NodeHandle
  void MakeUniqueAndSyncInputs();
  // Relink the `input_area`'s users to `this` area
  void UpdateUsersRelation(const AreaPtr &input_area);

  std::shared_ptr<NodeHandle> hd_;
  const size_t unique_id_;
  bool is_output_;
  std::vector<PrimOpPtr> ops_;
  AreaMode mode_{AreaMode::BASIC};
  // The `inputs_with_relation_.first` stores the input area of `this` area.
  // The `hd_->inputs` stores the NodeHandle of `this` area, to maintain the user edges.
  // They should always be in sync.
  std::vector<AreaWithRelation> inputs_with_relation_;
};
}  // namespace mindspore::graphkernel::inner
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SPLIT_MODEL_AREA_H_
