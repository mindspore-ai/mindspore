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
#include "backend/common/graph_kernel/core/parallel_op_combine.h"

#include <vector>
#include <string>
#include <set>
#include <deque>
#include <utility>
#include <algorithm>
#include <unordered_set>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/framework_utils.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "include/backend/kernel_graph.h"
#include "utils/anf_utils.h"
#include "include/common/utils/utils.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "utils/ms_context.h"
#include "ops/array_ops.h"
#include "backend/common/graph_kernel/adapter/callback_impl.h"

namespace mindspore::graphkernel {
namespace {
constexpr auto kPerm = "perm";
constexpr auto kShape = "shape";
const int kMinUpdateSize = 2;
std::vector<int64_t> GetTransposePerm(const PrimitivePtr &primitive) {
  ValuePtr perm = primitive->GetAttr(kPerm);
  MS_EXCEPTION_IF_NULL(perm);
  auto perm_val = perm->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(perm_val);
  auto perm_val_data = perm_val->value();
  std::vector<int64_t> perm_int;
  (void)std::transform(perm_val_data.begin(), perm_val_data.end(), std::back_inserter(perm_int),
                       [=](const ValuePtr &e) -> int64_t {
                         if (e->isa<Int64Imm>()) {
                           return GetValue<int64_t>(e);
                         } else if (e->isa<Int32Imm>()) {
                           return GetValue<int>(e);
                         } else {
                           MS_LOG(EXCEPTION) << "Perm must be int";
                           return -1;
                         }
                       });
  return perm_int;
}
}  // namespace
BranchGroupFinder::BranchGroupFinder(const std::string &op_name, FIsSupportedOp fis_supported_op,
                                     FAreCompatibleOps fare_compatible_ops)
    : op_name_(op_name), fis_supported_op_(fis_supported_op), fare_compatible_ops_(fare_compatible_ops) {}

AnfNodeIndexSet BranchGroupFinder::GetConsumers(FuncGraphManagerPtr mng, const AnfNodePtr &producer) {
  AnfNodeIndexSet consumers;
  auto users = mng->node_users()[producer];
  for (auto it : users) {
    auto user = it.first;
    if (user && user->cast<CNodePtr>() && AnfUtils::IsRealKernel(user) && fis_supported_op_(user)) {
      consumers.add(CNodeIndexPair(it.first, it.second));
      (void)children_map_[producer].insert(user);
    }
  }
  return consumers;
}

std::vector<Group> BranchGroupFinder::Find(const AnfNodePtr &start_node, const FuncGraphPtr &func_graph) {
  auto graph_kernel_fg = func_graph == nullptr ? common::AnfAlgo::GetCNodeFuncGraphPtr(start_node) : func_graph;
  MS_EXCEPTION_IF_NULL(graph_kernel_fg);
  auto mng = graph_kernel_fg->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto cnode = start_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::deque<AnfNodePtr> init_consumer;
  (void)std::transform(graph_kernel_fg->parameters().begin(), graph_kernel_fg->parameters().end(),
                       std::back_inserter(init_consumer), [](const AnfNodePtr &global_in) { return global_in; });
  for (size_t i = 1; i < cnode->size(); ++i) {
    init_consumer.push_back(cnode->input(i));
  }
  while (!init_consumer.empty()) {
    auto new_node = init_consumer.front();
    init_consumer.pop_front();
    auto new_consumer = GetConsumers(mng, new_node);
    (void)std::transform(new_consumer.begin(), new_consumer.end(), std::back_inserter(init_consumer),
                         [](const CNodeIndexPair &index_pair) { return index_pair.first; });
  }
  for (auto it : children_map_) {
    if (it.second.size() > 1) {
      (void)op_roots_.insert(it.first);
    }
  }
  std::vector<Group> groups;
  for (const auto &root : op_roots_) {
    size_t ngroups = groups.size();
    auto childrens = children_map_.at(root);
    for (auto child : childrens) {
      auto prim = GetCNodePrimitive(child);
      if (!prim) {
        continue;
      }
      auto prim_name = prim->name();
      // Branch should start with target node that specified by `op_name_`
      if (prim_name != op_name_) {
        continue;
      }
      auto branch = CreateBranch(child);
      branch.SetDataRoot(root);
      auto it = std::find_if(groups.begin() + ngroups, groups.end(), [this, &branch](const Group &group) {
        MS_EXCEPTION_IF_CHECK_FAIL(!group.empty() && !group[0].ops.empty(), "group empty or group[0] empty");
        auto top_branch = group[0];
        return (branch.target_op_pos == top_branch.target_op_pos) &&
               fare_compatible_ops_(branch.GetTargetOp(), top_branch.GetTargetOp());
      });
      if (it != groups.end()) {
        it->push_back(branch);
      } else {
        (void)groups.emplace_back();
        groups.back().push_back(branch);
      }
    }
  }
  return groups;
}

Branch BranchGroupFinder::CreateBranch(AnfNodePtr lead_op) {
  AnfNodePtrList ops{lead_op};
  int root_idx = GetCNodePrimitive(lead_op)->name() == op_name_ ? 0 : -1;
  auto it = children_map_.find(lead_op);
  while (it != children_map_.end() && it->second.size() == 1) {
    auto node = *(it->second).begin();
    ops.push_back(node);
    auto prim_name = GetCNodePrimitive(node)->name();
    if (prim_name == op_name_) {
      root_idx = static_cast<int>(ops.size());
    }
    it = children_map_.find(node);
  }
  return Branch(ops, root_idx);
}

ParallelOpCombiner::ParallelOpCombiner(const std::string &op_name, uint64_t min_num_branches, const std::string &layout)
    : op_name_(op_name), min_num_branches_(min_num_branches), layout_(layout) {}

AnfNodePtr ParallelOpCombiner::Combine(const AnfNodePtr &root, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(root);
  if (func_graph) {
    main_graph_ = func_graph;
  } else {
    main_graph_ = common::AnfAlgo::GetCNodeFuncGraphPtr(root);
  }
  MS_EXCEPTION_IF_NULL(main_graph_);
  auto finder = BranchGroupFinder(
    op_name_, [&](const AnfNodePtr n) { return IsSupportedOp(n); },
    [&](const AnfNodePtr a, const AnfNodePtr b) { return CanOpsBeCombined(a, b); });
  auto groups = finder.Find(root, main_graph_);
  children_map_ = std::move(finder.children_map_);
  for (const Group &group : groups) {
    if (group.size() < min_num_branches_) {
      MS_LOG(INFO) << "group size = " << group.size() << " < " << min_num_branches_ << ", skip.";
      continue;
    }
    CombineBranches(group);
  }
  return combined_;
}

void ParallelOpCombiner::CombineBranches(const Group &branches) {
  auto combined = MakeCombinedOp(branches);
  auto it = std::min_element(branches.begin(), branches.end(), [](const Branch &branch_a, const Branch &branch_b) {
    return branch_a.ops.size() < branch_b.ops.size();
  });
  size_t depth = it->ops.size();
  size_t pos;
  for (pos = 0; pos < depth; ++pos) {
    if (static_cast<int>(pos) == it->target_op_pos) {
      continue;
    }
    if (!CheckLevel(branches, pos)) {
      break;
    }
    combined = MakeCombinedAnfNodePtrFromFollowingOps(combined, branches, pos);
  }
  if (pos > 0) {
    UpdateGroupOutput(combined, branches, pos - 1);
  }
  combined_ = combined;
}

bool ParallelOpCombiner::CheckLevel(const Group &branches, size_t depth) {
  auto repr = branches[0].ops[depth];
  auto repr_prim_name = GetCNodePrimitive(repr)->name();
  // check if all branches in current depth can be combined
  for (auto it = branches.begin() + 1; it != branches.end(); it++) {
    const Branch &branch = *it;
    auto node = branch.ops[depth];
    auto prim_name = GetCNodePrimitive(node)->name();
    if (prim_name != repr_prim_name) {
      MS_LOG(INFO) << "Prim not compatible!" << prim_name << " vs " << repr_prim_name;
      return false;
    }
    if (unsupported_ops_.find(prim_name) != unsupported_ops_.end()) {
      MS_LOG(INFO) << "Op " << prim_name << " not supported for combination for now, stop.";
      return false;
    }
    if (!IsArgCompatible(repr, node)) {
      return false;
    }
  }
  MS_LOG(DEBUG) << "Op " << repr_prim_name << " can be combined at depth " << depth;
  return true;
}

bool ParallelOpCombiner::AutoUpdateInfo(const CNodePtr &to_update) {
  if (to_update->size() < kMinUpdateSize) {
    MS_LOG(ERROR) << "Cannot auto update for " << to_update->fullname_with_scope() << " with input size "
                  << to_update->size();
    return false;
  }
#ifndef MSLITE_ENABLE_GRAPH_KERNEL
  Callback::Instance()->ResetKernelInfo(to_update);
#else
  auto rep_input = to_update->input(1);
  // NOTE: We assume the inputs' formats and types are consistent with outputs'.
  std::string input_format = Callback::Instance()->GetTargetFromContext() == kAscendDevice ? "" : kOpFormat_NCHW;
  auto GetPrevOutFormat = [&input_format](const CNodePtr &cnode) -> bool {
    if (cnode == nullptr || !cnode->HasAttr(kOutputsFormat)) {
      return false;
    }
    auto prev_of = GetValue<std::vector<std::string> >(cnode->GetAttr(kOutputsFormat));
    if (prev_of.size() > 0) {
      input_format = prev_of[0];
      return true;
    }
    return false;
  };
  if (AnfUtils::IsRealKernel(rep_input)) {
    (void)GetPrevOutFormat(rep_input->cast<CNodePtr>());
  }
  if (input_format.empty()) {
    auto it = children_map_.find(rep_input);
    if (it != children_map_.end()) {
      for (auto orig_user : it->second) {
        if (GetPrevOutFormat(orig_user->cast<CNodePtr>())) {
          break;
        }
      }
    }
  }
  if (input_format.empty()) {
    MS_LOG(WARNING) << "Cannot find prev node's input format, use " << layout_
                    << " by default and that may cause error.";
    input_format = layout_;
  }
  std::vector<std::string> outputs_formats(AnfUtils::GetOutputTensorNum(to_update), input_format);
  to_update->AddAttr(kOutputsFormat, MakeValue(outputs_formats));
#endif
  return true;
}

std::map<size_t, AnfNodePtrList> ParallelOpCombiner::GetUniqueInputs(const Group &branches, size_t depth) const {
  std::map<size_t, AnfNodePtrList> unique_inputs;
  AnfNodePtrList parent_in_branch;
  if (depth >= 1) {
    (void)std::transform(branches.begin(), branches.end(), std::back_inserter(parent_in_branch),
                         [&depth](const Branch &br) { return br.ops[depth - 1]; });
  } else {
    Branch b1 = branches[0];
    parent_in_branch.push_back(b1.GetRootData());
  }

  for (auto br : branches) {
    auto op = br.ops[depth];
    auto cnode = op->cast<CNodePtr>();
    // Here we can know for sure that op's arg length are the same (check before)
    for (size_t i = 1; i < cnode->size(); ++i) {
      auto in = cnode->input(i);
      if (std::any_of(parent_in_branch.begin(), parent_in_branch.end(),
                      [&in](const AnfNodePtr &p) { return in == p; })) {
        continue;
      }
      unique_inputs[i].push_back(in);
    }
  }
  return unique_inputs;
}

CNodePtr GraphBuilder::NewConcatNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &input_node,
                                     size_t concat_dim, size_t input_num) {
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name()))};
  if (Callback::Instance()->GetTargetFromContext() == kAscendDevice) {
    auto maketuple = NewTupleNode(func_graph, input_node);
    concat_inputs.push_back(maketuple);
  } else {
    for (size_t i = 0; i < input_node.size(); ++i) {
      auto n = input_node[i];
      concat_inputs.push_back(n);
    }
  }
  auto concat = func_graph->NewCNode(concat_inputs);
  MS_EXCEPTION_IF_NULL(concat);
  func_graph->AddNode(concat);
  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(input_node[0], 0)};
  auto shape = common::AnfAlgo::GetOutputInferShape(input_node[0], 0);
  shape[concat_dim] *= SizeToLong(input_num);
  std::vector<ShapeVector> shapes(1, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, concat.get());
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(static_cast<int64_t>(concat_dim)), concat);
  common::AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue(static_cast<int64_t>(input_num)), concat);
  common::AnfAlgo::SetNodeAttr(kAttrN, MakeValue(static_cast<int64_t>(input_num)), concat);
  return concat;
}

CNodePtr GraphBuilder::NewTupleNode(const FuncGraphPtr &func_graph, AnfNodePtrList shared_inputs) {
  auto mk_inputs = AnfNodePtrList{NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name()))};
  AbstractBasePtrList abs_list;
  for (auto in : shared_inputs) {
    mk_inputs.push_back(in);
    abs_list.push_back(in->abstract());
  }
  auto make_tuple_node = func_graph->NewCNode(mk_inputs);
  func_graph->AddNode(make_tuple_node);
  make_tuple_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
  return make_tuple_node;
}

CNodePtr GraphBuilder::NewSplitNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, size_t split_dim,
                                    size_t split_num) {
  if (split_num == 0) {
    MS_LOG(EXCEPTION) << "split_num should not be zero.";
  }
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> split_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplit->name())),
                                          input_node};
  auto split = func_graph->NewCNode(split_inputs);
  func_graph->AddNode(split);
  MS_EXCEPTION_IF_NULL(split);
  auto dtype = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
  std::vector<TypeId> dtypes(split_num, dtype);
  auto shape = common::AnfAlgo::GetOutputInferShape(input_node, 0);
  shape[split_dim] /= SizeToLong(split_num);
  std::vector<ShapeVector> shapes(split_num, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split.get());
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue<int64_t>(split_dim), split);
  common::AnfAlgo::SetNodeAttr(kAttrOutputNum, MakeValue<int64_t>(split_num), split);
  return split;
}

CNodePtr GraphBuilder::NewElemwiseNoAttrNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &inputs) {
  auto node = func_graph->NewCNode(inputs);
  func_graph->AddNode(node);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() > kIndex1, "Input size should be larger than 1");
  MS_EXCEPTION_IF_NULL(inputs[kIndex1]);
  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(inputs[kIndex1], 0)};
  std::vector<ShapeVector> shapes = {common::AnfAlgo::GetOutputInferShape(inputs[kIndex1], 0)};
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, node.get());
  return node;
}

CNodePtr GraphBuilder::NewReshapeNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &inputs,
                                      const AnfNodePtr &orig_node) {
  auto node = func_graph->NewCNode(inputs);
  func_graph->AddNode(node);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() > kIndex1, "Input size should be larger than 1");
  MS_EXCEPTION_IF_NULL(inputs[kIndex1]);
  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(inputs[kIndex1], 0)};
  auto new_shape_in = common::AnfAlgo::GetOutputInferShape(inputs[kIndex1], 0);
  auto orig_shape_in = common::AnfAlgo::GetPrevNodeOutputInferShape(orig_node, 0);
  auto orig_shape_out = common::AnfAlgo::GetOutputInferShape(orig_node, 0);
  auto new_out_shape = InferReshapeOut(orig_shape_in, orig_shape_out, new_shape_in);
  GetCNodePrimitive(node)->set_attr(kShape, MakeValue(new_out_shape));
  std::vector<ShapeVector> shapes = {new_out_shape};
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, node.get());
  return node;
}

CNodePtr GraphBuilder::NewTransposeNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &inputs) {
  auto node = func_graph->NewCNode(inputs);
  func_graph->AddNode(node);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() > kIndex1, "Input size should be larger than 1");
  MS_EXCEPTION_IF_NULL(inputs[kIndex1]);
  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(inputs[kIndex1], 0)};
  auto new_shape_in = common::AnfAlgo::GetOutputInferShape(inputs[kIndex1], 0);
  auto perm_int = GetTransposePerm(GetCNodePrimitive(node));
  auto new_out_shape = InferTransposeOut(new_shape_in, perm_int);
  std::vector<ShapeVector> shapes = {new_out_shape};
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, node.get());
  return node;
}

ShapeVector GraphBuilder::InferReshapeOut(const ShapeVector &orig_reshape_in, const ShapeVector &orig_reshape_out,
                                          const ShapeVector &new_reshape_in) {
  ShapeVector new_shape_out;
  if (orig_reshape_in.size() == new_reshape_in.size()) {
    return InferConcatReshapeOut(orig_reshape_in, orig_reshape_out, new_reshape_in);
  } else {
    MS_LOG(EXCEPTION) << "Stack combiner infer for reshape not impl yet";
  }
  return new_shape_out;
}

ShapeVector GraphBuilder::InferTransposeOut(const ShapeVector &in_shape, const std::vector<int64_t> &perm) {
  ShapeVector out_shape;
  for (int64_t i : perm) {
    auto idx = LongToSize(i);
    out_shape.push_back(in_shape[idx]);
  }
  return out_shape;
}
}  // namespace mindspore::graphkernel
