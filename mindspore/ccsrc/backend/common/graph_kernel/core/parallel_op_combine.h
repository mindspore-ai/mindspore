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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_OP_COMBINE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_OP_COMBINE_H_

#include <map>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include "include/backend/optimizer/pass.h"
#include "ir/func_graph.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "ops/array_op_name.h"

namespace mindspore::graphkernel {
struct Branch {
  Branch(AnfNodePtrList lst, int pos) : ops(lst), target_op_pos(pos) {}
  AnfNodePtrList ops;
  int target_op_pos;  // -1 means no target op in this branch
  AnfNodePtr root_data{nullptr};
  size_t size() { return ops.size(); }
  AnfNodePtr GetTargetOp() { return GetOp(target_op_pos); }
  AnfNodePtr GetOp(int depth) {
    if (depth < 0 || depth >= static_cast<int>(ops.size())) {
      return nullptr;
    }
    return ops[depth];
  }

  AnfNodePtr GetRootData() { return root_data; }
  void SetDataRoot(AnfNodePtr data) { root_data = data; }
  std::string ToString() {
    std::string res;
    res += "RootData: ";
    res += root_data->fullname_with_scope();
    res += "; Ops: [";
    for (size_t i = 0; i < ops.size(); ++i) {
      auto op = ops[i];
      res += op->fullname_with_scope();
      if (static_cast<int>(i) == target_op_pos) {
        res += "(LEAD OP)";
      }
      res += ", ";
    }
    res += "]";
    return res;
  }
};
using Group = std::vector<Branch>;
using FIsSupportedOp = std::function<bool(const AnfNodePtr &n)>;
using FAreCompatibleOps = std::function<bool(const AnfNodePtr &a, const AnfNodePtr &b)>;
using AnfNodePtrSubstMap = std::unordered_map<AnfNodePtr, AnfNodePtr>;
using AnfNodePtrSet = std::unordered_set<AnfNodePtr>;
class BranchGroupFinder {
 public:
  BranchGroupFinder(const std::string &op_name, FIsSupportedOp fis_supported_op, FAreCompatibleOps fare_compatible_ops);
  std::vector<Group> Find(const AnfNodePtr &start_node, const FuncGraphPtr &func_graph = nullptr);
  std::unordered_map<AnfNodePtr, AnfNodePtrSet> children_map_;

 private:
  std::string op_name_;
  AnfNodePtrSet op_roots_;
  FIsSupportedOp fis_supported_op_;
  FAreCompatibleOps fare_compatible_ops_;
  Branch CreateBranch(AnfNodePtr lead_op);
  AnfNodeIndexSet GetConsumers(FuncGraphManagerPtr mng, const AnfNodePtr &producer);
};

class ParallelOpCombiner {
 public:
  explicit ParallelOpCombiner(const std::string &op_name, uint64_t min_num_branches, const std::string &layout);
  AnfNodePtr Combine(const AnfNodePtr &root, const FuncGraphPtr &func_graph = nullptr);
  virtual ~ParallelOpCombiner() = default;

 protected:
  virtual bool IsSupportedOp(const AnfNodePtr n) = 0;
  virtual bool CanOpsBeCombined(const AnfNodePtr a, const AnfNodePtr b) = 0;
  virtual AnfNodePtr MakeCombinedOp(const Group &branches) = 0;
  virtual bool IsArgCompatible(const AnfNodePtr a, const AnfNodePtr b) = 0;
  virtual AnfNodePtr MakeCombinedAnfNodePtrFromFollowingOps(const AnfNodePtr &data, const Group &branches,
                                                            size_t depth) = 0;
  virtual void UpdateGroupOutput(const AnfNodePtr &data, const Group &branches, size_t depth) = 0;
  bool AutoUpdateInfo(const CNodePtr &to_update);

  std::map<size_t, AnfNodePtrList> GetUniqueInputs(const Group &branches, size_t depth) const;

  FuncGraphPtr main_graph_;
  AnfNodePtr combined_;
  std::unordered_map<AnfNodePtr, AnfNodePtrSet> children_map_;
  std::unordered_set<std::string> unsupported_ops_{mindspore::kTransposeOpName, mindspore::kReshapeOpName};

 private:
  void CombineBranches(const Group &branches);
  bool CheckLevel(const Group &branches, size_t depth);

  std::string op_name_;
  uint64_t min_num_branches_{2};
  std::string layout_;
};

class GraphBuilder {
 public:
  static CNodePtr NewTupleNode(const FuncGraphPtr &func_graph, AnfNodePtrList shared_inputs);
  static CNodePtr NewSplitNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, size_t split_dim,
                               size_t split_num);
  static CNodePtr NewConcatNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &input_node, size_t concat_dim,
                                size_t input_num);
  static CNodePtr NewElemwiseNoAttrNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &matmul_inputs);
  static CNodePtr NewReshapeNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &matmul_inputs,
                                 const AnfNodePtr &orig_node);
  static CNodePtr NewTransposeNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &matmul_inputs);

  static ShapeVector InferReshapeOut(const ShapeVector &orig_reshape_in, const ShapeVector &orig_reshape_out,
                                     const ShapeVector &new_reshape_in);
  static ShapeVector InferConcatReshapeOut(const ShapeVector &orig_reshape_in, const ShapeVector &orig_reshape_out,
                                           const ShapeVector &new_reshape_in);
  static ShapeVector InferTransposeOut(const ShapeVector &in_shape, const std::vector<int64_t> &perm);
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_OP_COMBINE_H_
