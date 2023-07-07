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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either AnfNodePtress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_OP_CONCATENATE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_OP_CONCATENATE_H_

#include <map>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include "include/backend/optimizer/pass.h"
#include "ir/func_graph.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/graph_kernel/core/parallel_op_combine.h"

namespace mindspore::graphkernel {
struct ConcatenatePlan {
  int concat_in_idx{0};
  int split_out_idx{0};
  ShapeVector in_shape;
  ShapeVector out_shape;
};

class ParallelOpConcatenater : public ParallelOpCombiner {
 public:
  explicit ParallelOpConcatenater(const std::string &op_name, uint64_t min_num_branches, const std::string &layout);
  virtual ~ParallelOpConcatenater() = default;

 protected:
  virtual bool IsSupportedOp(const AnfNodePtr n) = 0;
  virtual bool CanOpsBeCombined(const AnfNodePtr a, const AnfNodePtr b) = 0;
  virtual AnfNodePtr MakeCombinedOp(const Group &branches) = 0;
  bool IsArgCompatible(const AnfNodePtr a, const AnfNodePtr b);
  AnfNodePtr MakeCombinedAnfNodePtrFromFollowingOps(const AnfNodePtr &data, const Group &branches, size_t depth) final;
  void UpdateGroupOutput(const AnfNodePtr &data, const Group &branches, size_t depth) final;

  std::map<size_t, AnfNodePtr> ConcatUniqueInputs(std::map<size_t, AnfNodePtrList> unique_inputs, size_t concat_idx);
  ConcatenatePlan GetElemWiseFollowingPlan(const Group &branches, size_t depth);
  AnfNodePtrList ReloadInputs(const Group &branches, size_t depth, AnfNodePtr shared_input);
  std::vector<ConcatenatePlan> plans_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_OP_CONCATENATE_H_
