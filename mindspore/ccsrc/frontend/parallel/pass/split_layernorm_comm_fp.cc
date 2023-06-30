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

#include <utility>
#include <vector>
#include <memory>

#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "frontend/parallel/pass/split_layernorm_comm_fp.h"
#include "frontend/parallel/step_parallel.h"
#include "include/common/utils/utils.h"
#include "ir/pattern_matcher.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr int64_t kLongZero = 0;
constexpr int64_t kLongOne = 1;
constexpr int64_t kLongTwo = 2;
using PrimitiveIndex = std::pair<PrimitivePtr, int>;

bool IsAnyMatMulInputTranspose(const CNodePtr &matmul_cnode) {
  MS_EXCEPTION_IF_NULL(matmul_cnode);
  if (!IsPrimitiveCNode(matmul_cnode)) {
    return false;
  }
  return GetValue<bool>(GetCNodePrimitive(matmul_cnode)->GetAttr("transpose_a")) ||
         GetValue<bool>(GetCNodePrimitive(matmul_cnode)->GetAttr("transpose_b"));
}

void CopyAllAttrs(const CNodePtr &dst_cnode, const CNodePtr &src_cnode) {
  MS_EXCEPTION_IF_NULL(dst_cnode);
  MS_EXCEPTION_IF_NULL(src_cnode);
  dst_cnode->set_attrs(src_cnode->attrs());
  auto dst_prim_node = GetCNodePrimitive(dst_cnode);
  auto src_prim_node = GetCNodePrimitive(src_cnode);
  auto src_attrs = src_prim_node->attrs();
  for (const auto &attr : src_attrs) {
    dst_prim_node->set_attr(attr.first, attr.second);
  }
}

ShapeVector GetSliceShape(const ShapeVector &origin_shape, size_t slice_axis = 0, int64_t slice_size = 2) {
  if (slice_axis >= origin_shape.size()) {
    MS_LOG(EXCEPTION) << "The slice_axis must be less than origin_shape.size(), but got " << slice_axis << " and "
                      << origin_shape.size();
  }
  if (slice_size == 0) {
    MS_LOG(EXCEPTION) << "The input 'slice_size' must be a positive integer, but got " << slice_size;
  }
  if (origin_shape[slice_axis] % slice_size != 0) {
    MS_LOG(EXCEPTION) << "The slice_size must be divisible int origin_shape[" << slice_axis << "], but got "
                      << slice_size << " and " << origin_shape[slice_axis];
  }
  auto slice_shape = origin_shape;
  slice_shape[slice_axis] /= slice_size;
  return slice_shape;
}

// Only for single outputs cnode
CNodePtr NewCNodeAndCloneAttrsSetSliceAbstract(const FuncGraphPtr &func_graph, const CNodePtr &src_cnode,
                                               std::vector<AnfNodePtr> &&inputs, size_t slice_axis = 0) {
  auto cnode = func_graph->NewCNode(inputs);
  CopyAllAttrs(cnode, src_cnode);

  // set abstract
  ShapeVector src_cnode_shape = common::AnfAlgo::GetOutputInferShape(src_cnode, slice_axis);
  ShapeVector slice_shape = GetSliceShape(src_cnode_shape, slice_axis);
  common::AnfAlgo::SetOutputTypeAndDetailShape({common::AnfAlgo::GetOutputInferDataType(src_cnode, slice_axis)},
                                               {std::make_shared<abstract::Shape>(slice_shape)}, cnode.get());
  return cnode;
}

CNodePtr NewTupleGetItemCNodeAndSetAbstract(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node,
                                            const int64_t index) {
  MS_EXCEPTION_IF_NULL(input_node);
  auto tuple_get_item_cnode =
    func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem->Clone()), input_node, NewValueNode(MakeValue(index))});

  auto dtype = common::AnfAlgo::GetOutputInferDataType(input_node, LongToSize(index));
  auto slice_shape = common::AnfAlgo::GetOutputInferShape(input_node, LongToSize(index));
  auto slice_shape_abstract = std::make_shared<abstract::Shape>(slice_shape);
  common::AnfAlgo::SetOutputTypeAndDetailShape({dtype}, {slice_shape_abstract}, tuple_get_item_cnode.get());
  return tuple_get_item_cnode;
}

CNodePtr NewLayerNormCNodeAndCloneAttrsSetSliceAbstract(const FuncGraphPtr &func_graph, const CNodePtr &src_cnode,
                                                        std::vector<AnfNodePtr> &&inputs) {
  MS_EXCEPTION_IF_NULL(src_cnode);
  auto layernorm_cnode = func_graph->NewCNode(inputs);
  CopyAllAttrs(layernorm_cnode, src_cnode);

  // set abstract
  auto slice_shape = GetSliceShape(common::AnfAlgo::GetOutputInferShape(src_cnode, kIndex0));
  auto slice_shape_abstract = std::make_shared<abstract::Shape>(slice_shape);
  common::AnfAlgo::SetOutputTypeAndDetailShape(
    {common::AnfAlgo::GetOutputInferDataType(src_cnode, kIndex0),
     common::AnfAlgo::GetOutputInferDataType(src_cnode, kIndex1),
     common::AnfAlgo::GetOutputInferDataType(src_cnode, kIndex2)},
    {slice_shape_abstract, std::make_shared<abstract::Shape>(ShapeVector{slice_shape[kIndex0], kLongOne}),
     std::make_shared<abstract::Shape>(ShapeVector{slice_shape[kIndex0], kLongOne})},
    layernorm_cnode.get());

  return layernorm_cnode;
}

CNodePtr NewSplitCNodeAndSetAbstract(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, int64_t axis,
                                     int64_t output_num) {
  MS_EXCEPTION_IF_NULL(input_node);
  auto input_node_shape = BaseShapeToShape(AnfAlgo::GetOutputDetailShape(input_node, 0));
  if (output_num == 0) {
    MS_LOG(EXCEPTION) << "The input 'output_num' must be a positive integer, but got " << output_num;
  }
  if (LongToSize(axis) >= input_node_shape.size() || input_node_shape[axis] % output_num != 0) {
    return nullptr;
  }
  auto split_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimSplit->Clone()), input_node});
  auto input_shape = common::AnfAlgo::GetOutputInferShape(input_node, kIndex0);
  int64_t slice_size = input_shape[kIndex0] / kLongTwo;
  AddCNodePrimAttr(split_cnode, kAttrAxis, MakeValue(axis));
  AddCNodePrimAttr(split_cnode, kAttrOutputNum, MakeValue(output_num));
  AddCNodePrimAttr(split_cnode, kAttrSizeSplits, MakeValue(ShapeVector{slice_size, slice_size}));
  AddCNodePrimAttr(split_cnode, kAttrNumSplit, MakeValue(output_num));

  auto dtype = common::AnfAlgo::GetOutputInferDataType(input_node, kIndex0);
  ShapeVector slice_shape = GetSliceShape(input_shape);
  auto slice_shape_abstract = std::make_shared<abstract::Shape>(slice_shape);
  common::AnfAlgo::SetOutputTypeAndDetailShape({dtype, dtype}, {slice_shape_abstract, slice_shape_abstract},
                                               split_cnode.get());
  return split_cnode;
}

CNodePtr NewConcatCNodeAndSetAbstract(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node0,
                                      const AnfNodePtr &input_node1, int64_t axis, int64_t input_num) {
  auto make_tuple_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimMakeTuple->Clone()), input_node0, input_node1});
  auto concat_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimConcat->Clone()), make_tuple_cnode});
  AddCNodePrimAttr(concat_cnode, kAttrAxis, MakeValue(axis));
  AddCNodePrimAttr(concat_cnode, "N", MakeValue(input_num));
  AddCNodePrimAttr(concat_cnode, kAttrInputNums, MakeValue(input_num));

  auto input_node0_dtype = common::AnfAlgo::GetOutputInferDataType(input_node0, kIndex0);
  auto input_node0_shape = common::AnfAlgo::GetOutputInferShape(input_node0, kIndex0);
  auto input_node1_dtype = common::AnfAlgo::GetOutputInferDataType(input_node1, kIndex0);
  auto input_node1_shape = common::AnfAlgo::GetOutputInferShape(input_node1, kIndex0);
  common::AnfAlgo::SetOutputTypeAndDetailShape(
    {input_node0_dtype, input_node1_dtype},
    {std::make_shared<abstract::Shape>(input_node0_shape), std::make_shared<abstract::Shape>(input_node1_shape)},
    make_tuple_cnode.get());
  auto concat_shape = input_node0_shape;
  concat_shape[axis] += input_node1_shape[axis];
  common::AnfAlgo::SetOutputTypeAndDetailShape({input_node0_dtype}, {std::make_shared<abstract::Shape>(concat_shape)},
                                               concat_cnode.get());
  return concat_cnode;
}

void InsertDependAndSetAbstract(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager,
                                const AnfNodePtr &prior_node, const AnfNodePtr &post_node) {
  MS_EXCEPTION_IF_NULL(prior_node);
  MS_EXCEPTION_IF_NULL(post_node);
  auto post_cnode = post_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(post_cnode);
  std::vector<AnfNodePtr> depend_inputs = {NewValueNode(prim::kPrimDepend), post_cnode->input(kIndex1), prior_node};
  auto depend_cnode = func_graph->NewCNode(depend_inputs);
  manager->SetEdge(post_node, kIndex1, depend_cnode);
  depend_cnode->set_abstract(depend_cnode->input(kIndex1)->abstract());
}
}  // namespace

static bool IsForwardCNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  return !(cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId) || cnode->HasAttr(kAttrDuplicated));
}

static bool IsCareNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!IsOneOfPrimitiveCNode(cnode, {prim::kPrimCast, prim::kPrimGeLU, prim::kPrimFastGeLU})) {
    return false;
  }
  const auto &node_user = cnode->func_graph()->manager()->node_users()[cnode];
  return node_user.size() == kSizeOne;
}

// LayerNorm->TupleGetItem->Cast->AllGather->Matmul->Add->Activation->MatMul->ReduceScatter
static bool PatternFilter(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr || !IsForwardCNode(cnode)) {
    return true;
  }
  if (!IsPrimitiveCNode(cnode, prim::kPrimLayerNorm)) {
    return true;
  }

  static std::vector<PrimitiveIndex> expect_primitive_list = {
    {prim::kPrimTupleGetItem, 1}, {prim::kPrimCast, 1},     {prim::kPrimAllGather, 1}, {prim::kPrimMatMul, 1},
    {prim::kPrimAdd, 1},          {prim::kPrimFastGeLU, 1}, {prim::kPrimMatMul, 1},    {prim::kPrimReduceScatter, 1}};
  AnfNodePtr cur_node = node;
  for (const auto &expect_prim : expect_primitive_list) {
    auto cur_cnode = cur_node->cast<CNodePtr>();
    auto output_node_set = cur_cnode->func_graph()->manager()->node_users()[cur_cnode];
    if (output_node_set.size() != kSizeOne) {
      return true;
    }
    auto index = output_node_set.front().second;
    if (index != expect_prim.second) {
      return true;
    }
    auto next_node = output_node_set.front().first;
    auto next_cnode = next_node->cast<CNodePtr>();
    if (next_cnode == nullptr || !IsForwardCNode(next_cnode)) {
      return true;
    }
    if (!IsPrimitiveCNode(next_cnode, expect_prim.first)) {
      return true;
    }
    cur_node = next_node;
  }
  return false;
}

static void ExpandSliceRangeToLeft(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager,
                                   const CNodePtr &split_cnode) {
  MS_EXCEPTION_IF_NULL(split_cnode);
  auto pre_node = split_cnode->input(kIndex1);
  std::shared_ptr<abstract::Shape> slice_pre_cnode_shape_abstract;
  while (IsCareNode(pre_node)) {
    auto pre_cnode = pre_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(pre_cnode);
    auto pre_cnode_dtype = common::AnfAlgo::GetOutputInferDataType(pre_cnode, kIndex0);
    auto pre_cnode_shape = common::AnfAlgo::GetOutputInferShape(pre_cnode, kIndex0);
    auto slice_pre_cnode_shape = {pre_cnode_shape[kIndex0] / kLongTwo, pre_cnode_shape[kIndex1]};
    slice_pre_cnode_shape_abstract = std::make_shared<abstract::Shape>(slice_pre_cnode_shape);
    auto pre_node_prim = GetCNodePrimitive(pre_cnode);
    auto node_users = manager->node_users()[split_cnode];
    for (const auto &node_user : node_users) {
      auto tuple_get_item_node = node_user.first;
      auto tuple_get_item_node_users = manager->node_users()[tuple_get_item_node];
      auto pre_cnode_sub = func_graph->NewCNode({NewValueNode(pre_node_prim->Clone()), tuple_get_item_node});
      common::AnfAlgo::SetOutputTypeAndDetailShape({pre_cnode_dtype}, {slice_pre_cnode_shape_abstract},
                                                   pre_cnode_sub.get());
      for (const auto &tuple_get_item_node_user : tuple_get_item_node_users) {
        manager->SetEdge(tuple_get_item_node_user.first, tuple_get_item_node_user.second, pre_cnode_sub);
      }
    }

    manager->SetEdge(split_cnode, kIndex1, pre_cnode->input(kIndex1));
    pre_node = split_cnode->input(kIndex1);
  }
  if (slice_pre_cnode_shape_abstract == nullptr) {
    return;
  }
  // Refresh abstract for split and tuple_get_item
  auto new_split_cnode_dtype = common::AnfAlgo::GetOutputInferDataType(split_cnode->input(kIndex1), kIndex0);
  common::AnfAlgo::SetOutputTypeAndDetailShape({new_split_cnode_dtype, new_split_cnode_dtype},
                                               {slice_pre_cnode_shape_abstract, slice_pre_cnode_shape_abstract},
                                               split_cnode.get());
  auto node_users = manager->node_users()[split_cnode];
  for (const auto &node_user : node_users) {
    common::AnfAlgo::SetOutputTypeAndDetailShape({new_split_cnode_dtype}, {slice_pre_cnode_shape_abstract},
                                                 node_user.first.get());
  }
}

static void ExpandSliceRangeToRight(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager,
                                    const CNodePtr &concat_cnode) {
  while (true) {
    const auto &node_users = manager->node_users()[concat_cnode];
    if (node_users.size() != kSizeOne) {
      return;
    }
    auto next_cnode = node_users.front().first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(next_cnode);
    auto next_cnode_prim = GetCNodePrimitive(next_cnode);
    if (!IsOneOfPrimitiveCNode(next_cnode, {prim::kPrimCast, prim::kPrimGeLU, prim::kPrimFastGeLU})) {
      return;
    }

    auto make_tuple_cnode = concat_cnode->input(kIndex1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple_cnode);

    for (size_t i = 1; i < make_tuple_cnode->inputs().size(); ++i) {
      auto input_node = make_tuple_cnode->input(i);
      auto next_cnode_sub = func_graph->NewCNode({NewValueNode(next_cnode_prim->Clone()), input_node});
      next_cnode_sub->set_abstract(input_node->abstract());
      manager->SetEdge(make_tuple_cnode, SizeToInt(i), next_cnode_sub);
    }
    auto next_cnode_users = manager->node_users()[next_cnode];
    for (const auto &next_cnode_pair : next_cnode_users) {
      manager->SetEdge(next_cnode_pair.first, next_cnode_pair.second, concat_cnode);
    }
  }
}

static void SplitIntoInterleaved(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager,
                                 const AnfNodePtr &layernorm_node) {
  auto layernorm_cnode = layernorm_node->cast<CNodePtr>();
  auto tuple_get_item_cnode = manager->node_users()[layernorm_cnode].front().first->cast<CNodePtr>();
  auto cast_cnode = manager->node_users()[tuple_get_item_cnode].front().first->cast<CNodePtr>();
  auto allgather_cnode = manager->node_users()[cast_cnode].front().first->cast<CNodePtr>();
  auto matmul1_cnode = manager->node_users()[allgather_cnode].front().first->cast<CNodePtr>();
  if (IsAnyMatMulInputTranspose(matmul1_cnode)) {
    return;
  }
  auto add_cnode = manager->node_users()[matmul1_cnode].front().first->cast<CNodePtr>();
  auto fast_gelu_cnode = manager->node_users()[add_cnode].front().first->cast<CNodePtr>();
  auto matmul2_cnode = manager->node_users()[fast_gelu_cnode].front().first->cast<CNodePtr>();
  if (IsAnyMatMulInputTranspose(matmul2_cnode)) {
    return;
  }
  auto reduce_scatter_cnode = manager->node_users()[matmul2_cnode].front().first->cast<CNodePtr>();

  // New split(layernorm_input1, 0, 2)
  auto split_cnode = NewSplitCNodeAndSetAbstract(func_graph, layernorm_cnode->input(kIndex1), kLongZero, kLongTwo);
  if (split_cnode == nullptr) {
    return;
  }

  // branch_a: split_cnode->TupleGetItem(0)->LayerNorm->...->ReduceScatter->MakeTuple->Concat
  auto get_slice_a = NewTupleGetItemCNodeAndSetAbstract(func_graph, split_cnode, 0);
  auto layernorm_a =
    NewLayerNormCNodeAndCloneAttrsSetSliceAbstract(func_graph, layernorm_cnode,
                                                   {NewValueNode(prim::kPrimLayerNorm->Clone()), get_slice_a,
                                                    layernorm_cnode->input(kIndex2), layernorm_cnode->input(kIndex3)});
  auto tuple_get_item_a = NewCNodeAndCloneAttrsSetSliceAbstract(
    func_graph, tuple_get_item_cnode,
    {NewValueNode(prim::kPrimTupleGetItem->Clone()), layernorm_a, tuple_get_item_cnode->input(kIndex2)});
  auto cast_a = NewCNodeAndCloneAttrsSetSliceAbstract(
    func_graph, cast_cnode, {NewValueNode(prim::kPrimCast->Clone()), tuple_get_item_a, cast_cnode->input(kIndex2)});
  auto allgather_a = NewCNodeAndCloneAttrsSetSliceAbstract(func_graph, allgather_cnode,
                                                           {NewValueNode(prim::kPrimAllGather->Clone()), cast_a});
  auto matmul1_a = NewCNodeAndCloneAttrsSetSliceAbstract(
    func_graph, matmul1_cnode, {NewValueNode(prim::kPrimMatMul->Clone()), allgather_a, matmul1_cnode->input(kIndex2)});
  auto add_a = NewCNodeAndCloneAttrsSetSliceAbstract(
    func_graph, add_cnode, {NewValueNode(prim::kPrimAdd->Clone()), matmul1_a, add_cnode->input(kIndex2)});
  auto fast_gelu_a = NewCNodeAndCloneAttrsSetSliceAbstract(func_graph, fast_gelu_cnode,
                                                           {NewValueNode(prim::kPrimFastGeLU->Clone()), add_a});
  auto matmul2_a = NewCNodeAndCloneAttrsSetSliceAbstract(
    func_graph, matmul2_cnode, {NewValueNode(prim::kPrimMatMul->Clone()), fast_gelu_a, matmul2_cnode->input(kIndex2)});
  auto reduce_scatter_a = NewCNodeAndCloneAttrsSetSliceAbstract(
    func_graph, reduce_scatter_cnode, {NewValueNode(prim::kPrimReduceScatter->Clone()), matmul2_a});

  // branch_b: split_cnode->TupleGetItem(0)->LayerNorm->...->ReduceScatter->MakeTuple->Concat
  auto get_slice_b = NewTupleGetItemCNodeAndSetAbstract(func_graph, split_cnode, 1);
  auto layernorm_b =
    NewLayerNormCNodeAndCloneAttrsSetSliceAbstract(func_graph, layernorm_cnode,
                                                   {NewValueNode(prim::kPrimLayerNorm->Clone()), get_slice_b,
                                                    layernorm_cnode->input(kIndex2), layernorm_cnode->input(kIndex3)});
  auto tuple_get_item_b = NewCNodeAndCloneAttrsSetSliceAbstract(
    func_graph, tuple_get_item_cnode,
    {NewValueNode(prim::kPrimTupleGetItem->Clone()), layernorm_b, tuple_get_item_cnode->input(kIndex2)});
  auto cast_b = NewCNodeAndCloneAttrsSetSliceAbstract(
    func_graph, cast_cnode, {NewValueNode(prim::kPrimCast->Clone()), tuple_get_item_b, cast_cnode->input(kIndex2)});
  auto allgather_b = NewCNodeAndCloneAttrsSetSliceAbstract(func_graph, allgather_cnode,
                                                           {NewValueNode(prim::kPrimAllGather->Clone()), cast_b});
  auto matmul1_b = NewCNodeAndCloneAttrsSetSliceAbstract(
    func_graph, matmul1_cnode, {NewValueNode(prim::kPrimMatMul->Clone()), allgather_b, matmul1_cnode->input(kIndex2)});
  auto add_b = NewCNodeAndCloneAttrsSetSliceAbstract(
    func_graph, add_cnode, {NewValueNode(prim::kPrimAdd->Clone()), matmul1_b, add_cnode->input(kIndex2)});
  auto fast_gelu_b = NewCNodeAndCloneAttrsSetSliceAbstract(func_graph, fast_gelu_cnode,
                                                           {NewValueNode(prim::kPrimFastGeLU->Clone()), add_b});
  auto matmul2_b = NewCNodeAndCloneAttrsSetSliceAbstract(
    func_graph, matmul2_cnode, {NewValueNode(prim::kPrimMatMul->Clone()), fast_gelu_b, matmul2_cnode->input(kIndex2)});
  auto reduce_scatter_b = NewCNodeAndCloneAttrsSetSliceAbstract(
    func_graph, reduce_scatter_cnode, {NewValueNode(prim::kPrimReduceScatter->Clone()), matmul2_b});

  // Insert depend node
  InsertDependAndSetAbstract(func_graph, manager, allgather_a, allgather_b);
  InsertDependAndSetAbstract(func_graph, manager, reduce_scatter_a, reduce_scatter_b);
  InsertDependAndSetAbstract(func_graph, manager, allgather_b, reduce_scatter_a);

  // New concat(MakeTuple(reduce_scatter_a, reduce_scatter_b))
  auto concat_cnode = NewConcatCNodeAndSetAbstract(func_graph, reduce_scatter_a, reduce_scatter_b, kLongZero, kLongTwo);

  // Replace graph
  auto prev_cnode = layernorm_cnode->input(kIndex1);
  manager->SetEdge(split_cnode, kIndex1, prev_cnode);
  auto next_cnode_users = manager->node_users()[reduce_scatter_cnode];
  for (const auto &next_cnode_pair : next_cnode_users) {
    manager->SetEdge(next_cnode_pair.first, next_cnode_pair.second, concat_cnode);
  }

  // Expand slice range by white list
  ExpandSliceRangeToLeft(func_graph, manager, split_cnode);
  ExpandSliceRangeToRight(func_graph, manager, concat_cnode);
}

void SplitLayerNormCommFp(const FuncGraphPtr &func_graph) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    return;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_enable = ms_context->get_param<bool>(MS_CTX_INTERLEAVED_LAYERNORM_COMM);
  if (!is_enable) {
    return;
  }

  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto todo = DeepScopedGraphSearchWithFilter(func_graph->get_return(), AlwaysInclude, PatternFilter);
  for (const auto &node : todo) {
    SplitIntoInterleaved(func_graph, manager, node);
  }
}
}  // namespace parallel
}  // namespace mindspore
