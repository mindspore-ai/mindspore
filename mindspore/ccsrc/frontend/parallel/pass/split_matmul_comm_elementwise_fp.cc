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

#include "frontend/parallel/pass/split_matmul_comm_elementwise_fp.h"

#include <memory>

#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "include/common/utils/utils.h"
#include "ir/pattern_matcher.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr int64_t kInt64Num0 = 0;
constexpr int64_t kInt64Num1 = 1;
constexpr int64_t kInt64Num2 = 2;
}  // namespace

static bool IsForwardCNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  return !(cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId) || cnode->HasAttr(kAttrDuplicated));
}

// MatMul -> AllReduce -> Add
static bool PatternFilter(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr || !IsForwardCNode(cnode)) {
    return true;
  }

  static PrimitiveSet expect_prim_type = {prim::kPrimAllReduce};
  if (!IsOneOfPrimitiveCNode(cnode, expect_prim_type)) {
    return true;
  }
  auto input_node = cnode->input(kIndex1);
  if (input_node == nullptr || !IsPrimitiveCNode(input_node, prim::kPrimMatMul)) {
    return true;
  }
  const auto &input_node_set = cnode->func_graph()->manager()->node_users()[input_node];
  if (input_node_set.size() != kSizeOne) {
    return true;
  }
  const auto &output_node_set = cnode->func_graph()->manager()->node_users()[cnode];
  if (output_node_set.size() != kSizeOne) {
    return true;
  }

  auto output_node = output_node_set.front().first;
  auto index = output_node_set.front().second;
  if (!IsPrimitiveCNode(output_node, prim::kPrimAdd) || index != kIndex1) {
    return true;
  }
  return false;
}

static void CopyAllAttrs(const CNodePtr &dst_cnode, const CNodePtr &src_cnode) {
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

static void SplitIntoInterleaved(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager,
                                 const AnfNodePtr &comm_node) {
  auto comm_cnode = comm_node->cast<CNodePtr>();
  auto matmul_cnode = comm_cnode->input(kIndex1)->cast<CNodePtr>();
  auto add_cnode = manager->node_users()[comm_cnode].front().first->cast<CNodePtr>();

  bool transpose_a = GetValue<bool>(GetCNodePrimitive(matmul_cnode)->GetAttr("transpose_a"));
  bool transpose_b = GetValue<bool>(GetCNodePrimitive(matmul_cnode)->GetAttr("transpose_b"));
  const int64_t axis_a_0 = transpose_a ? 1 : 0;
  const int64_t axis_b_1 = transpose_b ? 0 : 1;

  auto comm_primtive = GetCNodePrimitive(comm_cnode);
  auto matmul_input1 = matmul_cnode->input(kIndex1);
  auto matmul_input1_shape = BaseShapeToShape(AnfAlgo::GetOutputDetailShape(matmul_input1, 0));
  if (matmul_input1_shape[axis_a_0] % kInt64Num2 != 0) {
    return;
  }
  auto matmul_input2 = matmul_cnode->input(kIndex2);
  auto matmul_input2_shape = BaseShapeToShape(AnfAlgo::GetOutputDetailShape(matmul_input2, 0));
  auto add_input2 = add_cnode->input(kIndex2);

  // Create const value
  auto value0 = NewValueNode(MakeValue(kInt64Num0));
  auto value1 = NewValueNode(MakeValue(kInt64Num1));

  // New split(matmul_input1, axis_a_0, 2)
  auto split_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimSplit->Clone()), matmul_input1});
  int64_t slice_size = matmul_input1_shape[axis_a_0] / kInt64Num2;
  AddCNodePrimAttr(split_cnode, kAttrAxis, MakeValue(axis_a_0));
  AddCNodePrimAttr(split_cnode, kAttrOutputNum, MakeValue(kInt64Num2));
  AddCNodePrimAttr(split_cnode, kAttrSizeSplits, MakeValue(ShapeVector{slice_size, slice_size}));
  AddCNodePrimAttr(split_cnode, kAttrNumSplit, MakeValue(kInt64Num2));

  // branch_a: split_cnode->TupleGetItem(0)->Matmul_a->AllReduce_a->Add_a
  auto tuple_get_item_a = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem->Clone()), split_cnode, value0});
  auto matmul_a = func_graph->NewCNode({NewValueNode(prim::kPrimMatMul->Clone()), tuple_get_item_a, matmul_input2});
  CopyAllAttrs(matmul_a, matmul_cnode);
  auto comm_a = func_graph->NewCNode({NewValueNode(comm_primtive->Clone()), matmul_a});
  CopyAllAttrs(comm_a, comm_cnode);
  CNodePtr add_a = func_graph->NewCNode({NewValueNode(prim::kPrimAdd->Clone()), comm_a, add_input2});

  // branch_b: split_cnode->TupleGetItem(1)->Matmul_b->AllReduce_b(depend AllReduce_a) ->Add_b
  auto tuple_get_item_b = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem->Clone()), split_cnode, value1});
  auto matmul_b = func_graph->NewCNode({NewValueNode(prim::kPrimMatMul->Clone()), tuple_get_item_b, matmul_input2});
  CopyAllAttrs(matmul_b, matmul_cnode);
  auto depend = func_graph->NewCNode({NewValueNode(prim::kPrimDepend->Clone()), matmul_b, comm_a});
  auto comm_b = func_graph->NewCNode({NewValueNode(comm_primtive->Clone()), depend});
  CopyAllAttrs(comm_b, comm_cnode);
  auto add_b = func_graph->NewCNode({NewValueNode(prim::kPrimAdd->Clone()), comm_b, add_input2});

  // New concat(MakeTuple(add_a, add_b))
  auto make_tuple_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimMakeTuple->Clone()), add_a, add_b});
  auto concat_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimConcat->Clone()), make_tuple_cnode});
  AddCNodePrimAttr(concat_cnode, kAttrAxis, MakeValue(axis_a_0));
  AddCNodePrimAttr(concat_cnode, "N", MakeValue(kInt64Num2));
  AddCNodePrimAttr(concat_cnode, kAttrInputNums, MakeValue(kInt64Num2));

  // Infer path_a abstract
  auto dtype = common::AnfAlgo::GetOutputInferDataType(matmul_input1, 0);
  ShapeVector split_single_shape = matmul_input1_shape;
  split_single_shape[axis_a_0] /= kInt64Num2;
  auto split_shape_abstract = std::make_shared<abstract::Shape>(split_single_shape);
  common::AnfAlgo::SetOutputTypeAndDetailShape({dtype, dtype}, {split_shape_abstract, split_shape_abstract},
                                               split_cnode.get());
  common::AnfAlgo::SetOutputTypeAndDetailShape({dtype}, {split_shape_abstract}, tuple_get_item_a.get());
  ShapeVector matmul_ab_shape{split_single_shape[axis_a_0], matmul_input2_shape[axis_b_1]};
  auto matmul_ab_abstract = std::make_shared<abstract::Shape>(matmul_ab_shape);
  common::AnfAlgo::SetOutputTypeAndDetailShape({dtype}, {matmul_ab_abstract}, matmul_a.get());
  comm_a->set_abstract(matmul_a->abstract());
  add_a->set_abstract(matmul_a->abstract());

  // set path_b from path_a
  tuple_get_item_b->set_abstract(tuple_get_item_a->abstract());
  matmul_b->set_abstract(matmul_a->abstract());
  comm_b->set_abstract(matmul_b->abstract());
  depend->set_abstract(comm_b->abstract());
  add_b->set_abstract(add_a->abstract());

  // set abstract for make_tuple and concat
  common::AnfAlgo::SetOutputTypeAndDetailShape({dtype, dtype}, {matmul_ab_abstract, matmul_ab_abstract},
                                               make_tuple_cnode.get());
  concat_cnode->set_abstract(add_cnode->abstract());

  // Replace graph
  auto prev_cnode = matmul_cnode->input(kIndex1);
  manager->SetEdge(split_cnode, kIndex1, prev_cnode);
  auto next_cnode_users = manager->node_users()[add_cnode];
  for (const auto &next_cnode_pair : next_cnode_users) {
    manager->SetEdge(next_cnode_pair.first, next_cnode_pair.second, concat_cnode);
  }
}

// From:
// MatMul -> AllReduce -> Add
// To:
//        --> TupleGetItem(0) -> MatMul_a ->                        AllReduce_a -> Add_a
// Split                                                                                 -> Concat
//        --> TupleGetItem(1) -> MatMul_b -> Depend(AllReduce_a) -> AllReduce_b -> Add_b
void SplitMatmulCommElementwiseFp(const FuncGraphPtr &func_graph) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    MS_LOG(INFO) << "SplitMatmulCommElementwiseFp is only support under [semi_]auto_parallel, skip it.";
    return;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_enable = ms_context->get_param<bool>(MS_CTX_INTERLEAVED_MATMUL_COMM);
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
