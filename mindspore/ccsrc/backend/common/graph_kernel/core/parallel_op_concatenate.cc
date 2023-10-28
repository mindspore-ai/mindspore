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
#include "backend/common/graph_kernel/core/parallel_op_concatenate.h"

#include <vector>
#include <string>
#include <set>
#include <unordered_set>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/graph_kernel/adapter/callback_impl.h"

namespace mindspore::graphkernel {
ParallelOpConcatenater::ParallelOpConcatenater(const std::string &op_name, uint64_t min_num_branches,
                                               const std::string &layout)
    : ParallelOpCombiner(op_name, min_num_branches, layout) {}

bool ParallelOpConcatenater::IsArgCompatible(const AnfNodePtr a, const AnfNodePtr b) {
  auto cnode_a = a->cast<CNodePtr>();
  auto cnode_b = b->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode_a);
  MS_EXCEPTION_IF_NULL(cnode_b);
  auto arg_size = cnode_a->size();
  if (arg_size != cnode_b->size()) {
    MS_LOG(DEBUG) << "Args size not compatible: " << arg_size << " vs " << cnode_b->size();
    return false;
  }
  auto cb = Callback::Instance();
  for (size_t i = 1; i < arg_size; ++i) {
    auto shape_a = cb->GetInputInferShape(a, i);
    auto shape_b = cb->GetInputInferShape(b, i);
    if (shape_a != shape_b) {
      MS_LOG(ERROR) << "Args shape not compatible:" << shape_a << " vs " << shape_b;
      return false;
    }
  }
  return true;
}

AnfNodePtr ParallelOpConcatenater::MakeCombinedAnfNodePtrFromFollowingOps(const AnfNodePtr &data, const Group &branches,
                                                                          size_t depth) {
  auto ew_plan = GetElemWiseFollowingPlan(branches, depth);
  plans_.push_back(ew_plan);
  auto overall_inputs = ReloadInputs(branches, depth, data);
  if (branches.empty()) {
    MS_LOG(EXCEPTION) << "Fail to sample ops in a empty group.";
  }
  // Since all the ops of same depth in group should be the same, we just sample op in first branch.
  Branch b0 = branches[0];
  auto orig_node = b0.GetOp(static_cast<int>(depth));
  MS_EXCEPTION_IF_NULL(orig_node);
  CNodePtr new_node;
  if (GetCNodePrimitive(orig_node)->name() == kReshapeOpName) {
    new_node = GraphBuilder::NewReshapeNode(main_graph_, overall_inputs, orig_node);
  } else if (GetCNodePrimitive(orig_node)->name() == kTransposeOpName) {
    new_node = GraphBuilder::NewTransposeNode(main_graph_, overall_inputs);
  } else {
    new_node = GraphBuilder::NewElemwiseNoAttrNode(main_graph_, overall_inputs);
  }
  MS_EXCEPTION_IF_CHECK_FAIL(AutoUpdateInfo(new_node), "AutoUpdateInfo fail");
  return new_node;
}

std::map<size_t, AnfNodePtr> ParallelOpConcatenater::ConcatUniqueInputs(std::map<size_t, AnfNodePtrList> unique_inputs,
                                                                        size_t concat_idx) {
  std::map<size_t, AnfNodePtr> concated_inputs;
  for (auto it : unique_inputs) {
    size_t input_idx = it.first;
    auto local_inputs = it.second;
    if (local_inputs.size() < kDim2) {
      MS_LOG(WARNING) << "Concat Op needs at least 2 inputs, while got " << local_inputs.size();
      continue;
    }
    auto concat_node = GraphBuilder::NewConcatNode(main_graph_, local_inputs, concat_idx, local_inputs.size());
    MS_EXCEPTION_IF_NULL(concat_node);
    MS_EXCEPTION_IF_CHECK_FAIL(AutoUpdateInfo(concat_node), "AutoUpdateInfo fail");
    concated_inputs[input_idx] = concat_node;
  }
  return concated_inputs;
}

void ParallelOpConcatenater::UpdateGroupOutput(const AnfNodePtr &data, const Group &branches, size_t depth) {
  if (depth >= plans_.size()) {
    MS_LOG(EXCEPTION) << "Cannot get plan at depth " << depth << " vs " << plans_.size();
  }
  auto ew_plan = plans_[depth];
  auto split_node = GraphBuilder::NewSplitNode(main_graph_, data, ew_plan.split_out_idx, branches.size());
  MS_EXCEPTION_IF_CHECK_FAIL(AutoUpdateInfo(split_node), "AutoUpdateInfo fail");
  main_graph_->AddNode(split_node);
  auto mng = main_graph_->manager();
  for (size_t i = 0; i < branches.size(); ++i) {
    auto br = branches[i];
    auto target = br.ops[depth];
    auto idx_val = MakeValue(SizeToLong(i));
    auto gt_idx = NewValueNode(idx_val);
    gt_idx->set_abstract(idx_val->ToAbstract());
    AnfNodePtrList gt_inputs{NewValueNode(prim::kPrimTupleGetItem), split_node, gt_idx};
    auto new_out = main_graph_->NewCNode(gt_inputs);
    new_out->set_abstract(target->abstract()->Clone());
    (void)mng->Replace(target, new_out);
  }
  return;
}

ConcatenatePlan ParallelOpConcatenater::GetElemWiseFollowingPlan(const Group &branches, size_t depth) {
  if (depth - 1 >= plans_.size()) {
    MS_LOG(EXCEPTION) << "Should get " << (depth - 1) << " plan first, current plan size = " << plans_.size();
  }
  auto last_plan = plans_[depth - 1];
  ConcatenatePlan ew_plan;
  auto unique_inputs = GetUniqueInputs(branches, depth);
  auto cb = Callback::Instance();
  for (auto it : unique_inputs) {
    for (auto in : it.second) {
      if (!ew_plan.in_shape.empty()) {
        break;
      }
      ew_plan.in_shape = cb->GetOutputInferShape(in, 0);
    }
  }
  auto UpdateIdx = [](ShapeVector &base_shape, ShapeVector &new_shape, int base_idx) -> int {
    if (new_shape.empty()) {
      return base_idx;
    }
    auto rank_diff = static_cast<int>(base_shape.size()) - static_cast<int>(new_shape.size());
    if (rank_diff > base_idx) {
      return base_idx;
    }
    return base_idx - rank_diff;
  };
  ew_plan.concat_in_idx = UpdateIdx(last_plan.in_shape, ew_plan.in_shape, last_plan.concat_in_idx);
  Branch b0 = branches[0];
  auto op = b0.ops[depth];
  ew_plan.out_shape = cb->GetOutputInferShape(op, 0);
  ew_plan.split_out_idx = UpdateIdx(last_plan.out_shape, ew_plan.out_shape, last_plan.split_out_idx);
  MS_LOG(DEBUG) << "EW plan: " << ew_plan.concat_in_idx << ", " << ew_plan.split_out_idx << ", " << ew_plan.out_shape;
  return ew_plan;
}

AnfNodePtrList ParallelOpConcatenater::ReloadInputs(const Group &branches, size_t depth, AnfNodePtr shared_input) {
  Branch b1 = branches[0];
  auto cnode = b1.ops[depth]->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_size = cnode->size();
  auto plan = plans_[depth];
  auto unique_inputs = GetUniqueInputs(branches, depth);
  AnfNodePtrList overall_inputs{cnode->input(0)};  // prim
  auto concated_inputs = ConcatUniqueInputs(unique_inputs, plan.concat_in_idx);
  for (size_t i = 1; i < input_size; ++i) {
    if (concated_inputs.find(i) != concated_inputs.end()) {
      overall_inputs.push_back(concated_inputs[i]);
    } else {
      overall_inputs.push_back(shared_input);
    }
  }
  return overall_inputs;
}

ShapeVector GraphBuilder::InferConcatReshapeOut(const ShapeVector &orig_reshape_in, const ShapeVector &orig_reshape_out,
                                                const ShapeVector &new_reshape_in) {
  std::map<int, int> idx_map_rev;
  std::map<int, int> mul_map;
  int oidx = static_cast<int>(orig_reshape_out.size()) - 1;
  for (int ridx = static_cast<int>(orig_reshape_in.size()) - 1; ridx >= 0; --ridx) {
    auto cur_size = orig_reshape_in[ridx];
    mul_map[ridx] = new_reshape_in[ridx] / orig_reshape_in[ridx];
    while (oidx >= 0 && cur_size >= orig_reshape_out[oidx] && cur_size % orig_reshape_out[oidx] == 0) {
      idx_map_rev[oidx] = ridx;
      cur_size = cur_size / orig_reshape_out[oidx];
      oidx--;
    }
  }
  ShapeVector new_shape_out;
  for (int i = 0; i < static_cast<int>(orig_reshape_out.size()); ++i) {
    auto in_idx = idx_map_rev[i];
    auto mul = mul_map[in_idx];
    new_shape_out.push_back(orig_reshape_out[i] * mul);
    mul_map[in_idx] = 1;
  }
  return new_shape_out;
}
}  // namespace mindspore::graphkernel
