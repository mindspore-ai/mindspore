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

#include "frontend/parallel/pass/allreduce_slice_to_reducescatter.h"
#include <memory>
#include <list>
#include <vector>
#include <string>
#include <utility>
#include "include/utils/utils.h"
#include "include/utils/anfalgo.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "primitive/other_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace parallel {
const size_t kBatchMatMulStrategyNum = 2;
const size_t kBatchMatMulTransposeAIndex = 3;
const size_t kBatchMatMulTransposeBIndex = 4;
const size_t kReshapeShapeIndex = 2;
const size_t kStridedSliceBeginIndex = 2;
const size_t kStridedSliceEndIndex = 3;
const size_t kStridedSliceStrideIndex = 4;
const size_t k4DTransposeDim = 4;
const size_t k5DTransposeDim = 5;
const size_t k6DTransposeDim = 6;
const size_t kEpIndexFromLeft = 2;
const size_t kMinInputDimNum = 3;
const size_t kMpIndexFromRight = 2;
const size_t kEpIndexFromRight = 3;
const size_t kBiasAddRightIndex = 2;

std::vector<int64_t> GetTransposePerm(size_t dim) {
  if (dim == k4DTransposeDim) {
    return {2, 1, 0, 3};
  } else if (dim == k5DTransposeDim) {
    return {3, 1, 2, 0, 4};
  } else if (dim == k6DTransposeDim) {
    return {4, 1, 2, 3, 0, 5};
  }
  return {};
}

CNodePtr CreateReshapeNode(const FuncGraphPtr &graph, const AnfNodePtr &input_node,
                           const std::vector<int64_t> &target_shape) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input_node);
  auto input_shape_node = CreateTuple(target_shape);
  std::vector<AnfNodePtr> reshape_inputs = {NewValueNode(prim::kPrimReshape->Clone()), input_node, input_shape_node};
  auto reshape_node = graph->NewCNode(reshape_inputs);
  reshape_node->set_scope(input_node->scope());
  return reshape_node;
}

CNodePtr CreateReshapeNode(const FuncGraphPtr &graph, const AnfNodePtr &input_node,
                           const AnfNodePtr &input_shape_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(input_shape_node);
  std::vector<AnfNodePtr> reshape_inputs = {NewValueNode(prim::kPrimReshape->Clone()), input_node, input_shape_node};
  auto reshape_node = graph->NewCNode(reshape_inputs);
  MS_EXCEPTION_IF_NULL(reshape_node);
  reshape_node->set_scope(input_node->scope());
  return reshape_node;
}

CNodePtr CreateTransposeNode(const FuncGraphPtr &graph, const AnfNodePtr &input_node,
                             const std::vector<int64_t> &transpose_perm) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input_node);
  auto transpose_perm_node = CreateTuple(transpose_perm);
  std::vector<AnfNodePtr> transpose_inputs = {NewValueNode(prim::kPrimTranspose->Clone()), input_node,
                                              transpose_perm_node};
  auto transpose_node = graph->NewCNode(transpose_inputs);
  transpose_node->set_scope(input_node->scope());
  return transpose_node;
}

CNodePtr CreateReduceScatterNode(const FuncGraphPtr &graph, const AnfNodePtr &input_node,
                                 const AnfNodePtr &allreduce_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(allreduce_node);
  auto allreduce_prim = GetCNodePrimitive(allreduce_node);
  MS_EXCEPTION_IF_NULL(allreduce_prim);
  if (!allreduce_prim->HasAttr(OP) || !allreduce_prim->HasAttr(GROUP)) {
    return nullptr;
  }
  Attr attr_op = std::make_pair(OP, allreduce_prim->GetAttr(OP));
  Attr attr_group = std::make_pair(GROUP, allreduce_prim->GetAttr(GROUP));
  OperatorAttrs attrs = {attr_op, attr_group};
  auto reduce_scatter_prim = CreateOpInstance(attrs, prim::kPrimReduceScatter->name(), allreduce_prim->instance_name());
  std::vector<AnfNodePtr> reduce_scatter_inputs = {NewValueNode(reduce_scatter_prim), input_node};
  auto reduce_scatter_node = graph->NewCNode(reduce_scatter_inputs);
  reduce_scatter_node->set_scope(input_node->scope());
  return reduce_scatter_node;
}

AnfNodePtr GetSingleUserPrimNode(const FuncGraphManagerPtr &manager, const AnfNodePtr &node, const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(prim);
  auto node_users = manager->node_users()[node];
  if (node_users.size() != 1) {
    return nullptr;
  }

  if (!IsPrimitiveCNode(node_users.front().first, prim)) {
    return nullptr;
  }

  return node_users.front().first;
}

struct AllReduceSliceToReduceScatterParams {
  const FuncGraphManagerPtr &manager;
  const FuncGraphPtr &graph;
  AnfNodePtr batch_matmul{nullptr};
  AnfNodePtr allreduce{nullptr};
  AnfNodePtr reshape{nullptr};
  AnfNodePtr bias_add{nullptr};
  AnfNodePtr stridedslice{nullptr};
  AnfNodePtr anchor{nullptr};
  AnfNodePtr final_reshape{nullptr};
  AnfNodePtr strategy_stridedslice{nullptr};

  size_t expert_parallel{1};
  size_t model_parallel{1};
  size_t bias_add_replace_index{1};

  AllReduceSliceToReduceScatterParams(const FuncGraphManagerPtr &m, const FuncGraphPtr &g) : manager(m), graph(g) {}
};

std::vector<Shape> GetNodeStrategy(const AnfNodePtr &node) {
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  if (!prim->HasAttr(IN_STRATEGY)) {
    return {};
  }

  auto input_strategy = prim->GetAttr(IN_STRATEGY);
  if (input_strategy == nullptr) {
    return {};
  }

  return GetValue<std::vector<Shape>>(input_strategy);
}

std::vector<Shape> GetBatchMatMulStrategy(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return {};
  }

  auto transpose_a = cnode->input(kBatchMatMulTransposeAIndex);
  auto transpose_b = cnode->input(kBatchMatMulTransposeBIndex);
  if (transpose_a == nullptr || transpose_b == nullptr) {
    return {};
  }

  auto transpose_a_value_node = transpose_a->cast<ValueNodePtr>();
  auto transpose_b_value_node = transpose_b->cast<ValueNodePtr>();
  if (transpose_a_value_node == nullptr || transpose_b_value_node == nullptr) {
    return {};
  }

  auto transpose_a_value = transpose_a_value_node->value();
  auto transpose_b_value = transpose_b_value_node->value();
  if (transpose_a_value == nullptr || transpose_b_value == nullptr) {
    return {};
  }

  auto transpose_a_bool = GetValue<bool>(transpose_a_value);
  auto transpose_b_bool = GetValue<bool>(transpose_b_value);

  auto origin_strategy = GetNodeStrategy(node);
  if (origin_strategy.size() < kBatchMatMulStrategyNum) {
    return {};
  }

  // (x, ep, 1, mp) * (ep, mp, 1)
  auto &bmm_input1_strategy = origin_strategy.at(0);
  auto &bmm_input2_strategy = origin_strategy.at(1);
  if (transpose_a_bool && bmm_input1_strategy.size() >= kMinInputDimNum) {
    auto tmp = bmm_input1_strategy[bmm_input1_strategy.size() - 1];
    bmm_input1_strategy[bmm_input1_strategy.size() - 1] =
      bmm_input1_strategy[bmm_input1_strategy.size() - kMpIndexFromRight];
    bmm_input1_strategy[bmm_input1_strategy.size() - kMpIndexFromRight] = tmp;
  }

  if (transpose_b_bool && bmm_input2_strategy.size() >= kMinInputDimNum) {
    auto tmp = bmm_input2_strategy[bmm_input2_strategy.size() - 1];
    bmm_input2_strategy[bmm_input2_strategy.size() - 1] =
      bmm_input2_strategy[bmm_input2_strategy.size() - kMpIndexFromRight];
    bmm_input2_strategy[bmm_input2_strategy.size() - kMpIndexFromRight] = tmp;
  }
  return origin_strategy;
}

bool CheckBatchMatmulStrategy(const std::vector<Shape> &bmm_input_strategy) {
  if (bmm_input_strategy.size() < kBatchMatMulStrategyNum) {
    return false;
  }

  // (x, ep, 1, mp) * (ep, mp, 1)
  auto &bmm_input1_strategy = bmm_input_strategy.at(0);
  auto &bmm_input2_strategy = bmm_input_strategy.at(1);
  if (bmm_input1_strategy.size() < kMinInputDimNum || bmm_input2_strategy.size() < kMinInputDimNum) {
    return false;
  }

  auto input1_mp = bmm_input1_strategy[bmm_input1_strategy.size() - 1];
  auto input2_mp = bmm_input2_strategy[bmm_input2_strategy.size() - kMpIndexFromRight];
  if (input1_mp == 1 || input1_mp != input2_mp) {
    return false;
  }

  auto input1_no_split = bmm_input1_strategy[bmm_input1_strategy.size() - kMpIndexFromRight];
  auto input2_no_split = bmm_input2_strategy[bmm_input2_strategy.size() - 1];
  if (input1_no_split != 1 || input1_no_split != input2_no_split) {
    return false;
  }

  auto input1_ep = bmm_input1_strategy[bmm_input1_strategy.size() - kEpIndexFromRight];
  auto input2_ep = bmm_input2_strategy[bmm_input2_strategy.size() - kEpIndexFromRight];
  if (input1_ep != input2_ep) {
    return false;
  }

  return true;
}

bool CheckBatchMatmulAndStridedSliceStrategy(const std::vector<Shape> &bmm_input_strategy,
                                             const std::vector<Shape> &stridedslice_input_strategy) {
  // (x, ep, 1, mp) * (ep, mp, 1)
  if (!CheckBatchMatmulStrategy(bmm_input_strategy)) {
    return false;
  }

  if (stridedslice_input_strategy.empty()) {
    return false;
  }

  auto &bmm_input1_strategy = bmm_input_strategy.at(0);
  auto input1_mp = bmm_input1_strategy[bmm_input1_strategy.size() - 1];
  auto input1_ep = bmm_input1_strategy[bmm_input1_strategy.size() - kEpIndexFromRight];
  // (ep, 1, mp, 1) or (1, ep, mp, 1)
  auto &stridedslice_input1_strategy = stridedslice_input_strategy.at(0);
  if (stridedslice_input1_strategy.size() < kEpIndexFromRight + 1) {
    return false;
  }

  auto ss_no_split1 = stridedslice_input1_strategy[stridedslice_input1_strategy.size() - 1];
  auto ss_mp = stridedslice_input1_strategy[stridedslice_input1_strategy.size() - kMpIndexFromRight];
  auto ss_ep1 = stridedslice_input1_strategy[stridedslice_input1_strategy.size() - kEpIndexFromRight];
  auto ss_ep2 = stridedslice_input1_strategy[stridedslice_input1_strategy.size() - kEpIndexFromRight - 1];
  if (ss_mp != input1_mp || ss_no_split1 != 1) {
    return false;
  }

  if (ss_ep1 != input1_ep && ss_ep2 != input1_ep) {
    return false;
  }

  return true;
}

bool CheckCase1Strategy(const AllReduceSliceToReduceScatterParams &params) {
  auto bmm_input_strategy = GetBatchMatMulStrategy(params.batch_matmul);
  auto stridedslice_input_strategy = GetNodeStrategy(params.strategy_stridedslice);
  if (!CheckBatchMatmulAndStridedSliceStrategy(bmm_input_strategy, stridedslice_input_strategy)) {
    return false;
  }
  return true;
}

size_t GetReshapeDim(const CNodePtr &reshape_cnode) {
  auto default_dim = k4DTransposeDim;
  if (reshape_cnode == nullptr) {
    return default_dim;
  }

  auto shape_node = reshape_cnode->input(kReshapeShapeIndex);
  if (shape_node == nullptr) {
    return default_dim;
  }

  auto shape_value_node = shape_node->cast<ValueNodePtr>();
  if (shape_value_node == nullptr) {
    return default_dim;
  }

  auto shape_index_value = shape_value_node->value();
  if (shape_index_value == nullptr) {
    return default_dim;
  }

  auto shape_vector = GetValue<std::vector<int64_t>>(shape_index_value);
  if (shape_vector.size() < k4DTransposeDim) {
    return default_dim;
  }

  return shape_vector.size();
}

/*
 * AllReduceSliceToReduceScatter Case1:
 * BatchMatMul -> AllReduce （-> Reshape）-> StridedSlice change to
 * BatchMatMul -> Reshape -> Transpose -> ReduceScatter -> Transpose
 */
void AllReduceSliceToReduceScatterCase1(const AllReduceSliceToReduceScatterParams &params) {
  auto transpose_dim = k4DTransposeDim;
  auto current_node = params.batch_matmul;
  if (params.reshape != nullptr) {
    auto origin_reshape_cnode = params.reshape->cast<CNodePtr>();
    if (origin_reshape_cnode == nullptr) {
      return;
    }
    current_node =
      CreateReshapeNode(params.graph, params.batch_matmul, origin_reshape_cnode->input(kReshapeShapeIndex));
    transpose_dim = GetReshapeDim(origin_reshape_cnode);
  }

  auto perm = GetTransposePerm(transpose_dim);
  auto transpose_out = CreateTransposeNode(params.graph, current_node, perm);
  auto reduce_scatter_node = CreateReduceScatterNode(params.graph, transpose_out, params.allreduce);
  if (reduce_scatter_node == nullptr) {
    return;
  }
  auto transpose_in = CreateTransposeNode(params.graph, reduce_scatter_node, perm);
  params.manager->Replace(params.anchor, transpose_in);
}

bool CheckBiasAddAndStridedSliceStrategy(const std::vector<Shape> &bias_add_input_strategy,
                                         const std::vector<Shape> &stridedslice_input_strategy) {
  if (bias_add_input_strategy.empty() || stridedslice_input_strategy.empty()) {
    return false;
  }

  // (x, ep, mp, 1)+(1, ep, 1, 1)
  auto &bias_add_input1_strategy = bias_add_input_strategy.at(0);
  auto &bias_add_input2_strategy = bias_add_input_strategy.at(1);

  // (ep, 1, mp, 1)
  auto &stridedslice_input1_strategy = stridedslice_input_strategy.at(0);
  if (bias_add_input1_strategy.size() < kMinInputDimNum || bias_add_input2_strategy.size() < kMinInputDimNum ||
      stridedslice_input1_strategy.size() < kMinInputDimNum + 1) {
    return false;
  }

  auto input1_no_split = bias_add_input1_strategy[bias_add_input1_strategy.size() - 1];
  auto input1_mp = bias_add_input1_strategy[bias_add_input1_strategy.size() - kMpIndexFromRight];
  auto input1_ep = bias_add_input1_strategy[bias_add_input1_strategy.size() - kEpIndexFromRight];

  auto input2_no_split1 = bias_add_input2_strategy[bias_add_input2_strategy.size() - 1];
  auto input2_no_split2 = bias_add_input2_strategy[bias_add_input2_strategy.size() - kMpIndexFromRight];
  auto input2_ep = bias_add_input2_strategy[bias_add_input2_strategy.size() - kEpIndexFromRight];

  auto ss_no_split1 = stridedslice_input1_strategy[stridedslice_input1_strategy.size() - 1];
  auto ss_mp = stridedslice_input1_strategy[stridedslice_input1_strategy.size() - kMpIndexFromRight];
  auto ss_no_split2 = stridedslice_input1_strategy[stridedslice_input1_strategy.size() - kEpIndexFromRight];
  auto ss_ep = stridedslice_input1_strategy[stridedslice_input1_strategy.size() - kEpIndexFromRight - 1];

  if (input1_no_split != ss_no_split1 || input2_no_split1 != ss_no_split1 || input2_no_split2 != ss_no_split1 ||
      ss_no_split2 != ss_no_split1) {
    return false;
  }
  if (input1_mp != ss_mp || input1_ep != ss_ep || input2_ep != ss_ep) {
    return false;
  }

  return true;
}

bool CheckCase2Strategy(const AllReduceSliceToReduceScatterParams &params) {
  auto bmm_input_strategy = GetBatchMatMulStrategy(params.batch_matmul);
  auto stridedslice_input_strategy = GetNodeStrategy(params.strategy_stridedslice);
  if (!CheckBatchMatmulAndStridedSliceStrategy(bmm_input_strategy, stridedslice_input_strategy)) {
    return false;
  }

  auto bias_add_input_strategy = GetNodeStrategy(params.bias_add);
  if (!CheckBiasAddAndStridedSliceStrategy(bias_add_input_strategy, stridedslice_input_strategy)) {
    return false;
  }

  return true;
}

std::vector<int64_t> GetShapeFromStridedSliceNode(const AnfNodePtr &stridedslice) {
  std::vector<int64_t> target_shape{};
  if (stridedslice == nullptr) {
    return target_shape;
  }
  auto origin_cnode = stridedslice->cast<CNodePtr>();
  if (origin_cnode == nullptr) {
    return target_shape;
  }

  auto begin_node = origin_cnode->input(kStridedSliceBeginIndex);
  auto end_node = origin_cnode->input(kStridedSliceEndIndex);
  auto stride_node = origin_cnode->input(kStridedSliceStrideIndex);
  if (begin_node == nullptr || end_node == nullptr || stride_node == nullptr) {
    return target_shape;
  }

  auto begin_value_node = begin_node->cast<ValueNodePtr>();
  auto end_value_node = end_node->cast<ValueNodePtr>();
  auto stride_value_node = stride_node->cast<ValueNodePtr>();
  if (begin_value_node == nullptr || end_value_node == nullptr || stride_value_node == nullptr) {
    return target_shape;
  }

  auto begin_index_value = begin_value_node->value();
  auto end_index_value = end_value_node->value();
  auto stride_value = stride_value_node->value();
  if (begin_index_value == nullptr || end_index_value == nullptr || stride_value == nullptr) {
    return target_shape;
  }

  auto begin_vector = GetValue<std::vector<int64_t>>(begin_index_value);
  auto end_vector = GetValue<std::vector<int64_t>>(end_index_value);
  auto stride_vector = GetValue<std::vector<int64_t>>(stride_value);
  if (begin_vector.size() != end_vector.size() || begin_vector.size() != stride_vector.size()) {
    return target_shape;
  }

  for (size_t i = 0; i < begin_vector.size(); ++i) {
    if (stride_vector[i] <= 0) {
      return target_shape;
    }
    end_vector[i] = (end_vector[i] - begin_vector[i]) / stride_vector[i];
  }
  return end_vector;
}

std::vector<int64_t> GetCase2TargetShapeFromStridedSlice(const AllReduceSliceToReduceScatterParams &params) {
  std::vector<int64_t> target_shape{};
  if (params.stridedslice == nullptr) {
    return target_shape;
  }
  auto origin_shape = GetShapeFromStridedSliceNode(params.stridedslice);
  if (origin_shape.size() <= kMpIndexFromRight) {
    return target_shape;
  }

  size_t reshape_dim = origin_shape.size() - kMpIndexFromRight;
  auto expert_parallel = params.expert_parallel;
  auto group_size = params.model_parallel;
  if (expert_parallel < 1 || group_size == 0 || LongToSize(origin_shape[reshape_dim]) % expert_parallel != 0) {
    return target_shape;
  }

  size_t pad_dim = 0;
  size_t final_dim = origin_shape.size();
  if (origin_shape.size() < k4DTransposeDim) {
    pad_dim = k4DTransposeDim - origin_shape.size();
    for (size_t i = 0; i < pad_dim; ++i) {
      target_shape.push_back(1);
    }
    final_dim = k4DTransposeDim;
  }

  for (size_t i = pad_dim; i < final_dim; ++i) {
    auto origin_shape_idx = i - pad_dim;
    if (origin_shape_idx == reshape_dim - 1) {
      target_shape.push_back(origin_shape[origin_shape_idx] * expert_parallel);
    } else if (origin_shape_idx == reshape_dim) {
      target_shape.push_back(origin_shape[origin_shape_idx] * group_size / expert_parallel);
    } else {
      target_shape.push_back(origin_shape[origin_shape_idx]);
    }
  }

  return target_shape;
}

/*
 * AllReduceSliceToReduceScatter Case2:
 * BatchMatMul -> AllReduce (-> Reshape) -> StridedSlice (-> Reshape) -> BiasAdd -> Reshape ->
 * AlltoAll（-> AlltoAll -> Reshape）change to BatchMatMul -> Reshape -> Transpose -> ReduceScatter -> Transpose ->
 * BiasAdd
 */
void AllReduceSliceToReduceScatterCase2(const AllReduceSliceToReduceScatterParams &params) {
  auto bias_add_cnode = params.bias_add->cast<CNodePtr>();
  if (bias_add_cnode == nullptr) {
    return;
  }

  if (!CheckCase2Strategy(params)) {
    return;
  }

  auto target_shape = GetCase2TargetShapeFromStridedSlice(params);
  if (target_shape.size() != k4DTransposeDim) {
    return;
  }
  auto perm = GetTransposePerm(target_shape.size());
  auto reshape_node = CreateReshapeNode(params.graph, params.batch_matmul, target_shape);
  auto transpose_out = CreateTransposeNode(params.graph, reshape_node, perm);
  auto reduce_scatter_node = CreateReduceScatterNode(params.graph, transpose_out, params.allreduce);
  if (reduce_scatter_node == nullptr) {
    return;
  }

  auto transpose_in = CreateTransposeNode(params.graph, reduce_scatter_node, perm);

  auto reshape_before_biasadd = target_shape;
  reshape_before_biasadd[1] = reshape_before_biasadd[1] / params.expert_parallel;
  reshape_before_biasadd[kEpIndexFromLeft] =
    reshape_before_biasadd[kEpIndexFromLeft] * params.expert_parallel / params.model_parallel;
  auto reshape_before_biasadd_node = CreateReshapeNode(params.graph, transpose_in, reshape_before_biasadd);

  if (params.final_reshape != nullptr) {
    params.manager->Replace(bias_add_cnode->input(params.bias_add_replace_index), reshape_before_biasadd_node);
    params.manager->Replace(params.anchor, params.bias_add);
  } else {
    auto final_shape = GetShapeFromStridedSliceNode(params.strategy_stridedslice);
    if (final_shape.size() < k4DTransposeDim) {
      return;
    }

    auto current_idx = final_shape.size() - kMpIndexFromRight;
    final_shape[current_idx] = final_shape[current_idx] / params.model_parallel;
    current_idx = final_shape.size() - k4DTransposeDim;
    final_shape[current_idx] = final_shape[current_idx] / params.expert_parallel;
    auto final_reshape_node = CreateReshapeNode(params.graph, params.bias_add, final_shape);

    params.manager->Replace(bias_add_cnode->input(params.bias_add_replace_index), reshape_before_biasadd_node);
    params.manager->Replace(params.anchor, final_reshape_node);
  }
}

bool CheckCase3Strategy(const AllReduceSliceToReduceScatterParams &params) {
  auto bmm_input_strategy = GetBatchMatMulStrategy(params.batch_matmul);
  auto bias_add_input_strategy = GetNodeStrategy(params.bias_add);
  // (x, ep, 1, mp) * (ep, mp, 1)
  if (!CheckBatchMatmulStrategy(bmm_input_strategy)) {
    return false;
  }

  // (x, ep, mp, 1)+(1, ep, 1, 1)
  if (bias_add_input_strategy.empty()) {
    return false;
  }

  auto &bmm_input1_strategy = bmm_input_strategy.at(0);
  auto input1_mp = bmm_input1_strategy[bmm_input1_strategy.size() - 1];
  auto input1_ep = bmm_input1_strategy[bmm_input1_strategy.size() - kEpIndexFromRight];
  auto &bias_add_input1_strategy = bias_add_input_strategy.at(0);
  auto &bias_add_input2_strategy = bias_add_input_strategy.at(1);
  if (bias_add_input1_strategy.size() < kMinInputDimNum || bias_add_input2_strategy.size() < kMinInputDimNum) {
    return false;
  }

  auto bias_input1_mp = bias_add_input1_strategy[bias_add_input1_strategy.size() - kMpIndexFromRight];
  auto bias_input1_ep = bias_add_input1_strategy[bias_add_input1_strategy.size() - kEpIndexFromRight];
  auto bias_input2_ep = bias_add_input2_strategy[bias_add_input2_strategy.size() - kEpIndexFromRight];
  if (input1_mp != bias_input1_mp || input1_ep != bias_input1_ep || input1_ep != bias_input2_ep) {
    return false;
  }
  return true;
}

/*
 * AllReduceSliceToReduceScatter Case3:
 * BatchMatMul -> AllReduce -> StridedSlice -> BiasAdd
 * change to BatchMatMul -> Transpose -> ReduceScatter -> Transpose -> BiasAdd
 */
void AllReduceSliceToReduceScatterCase3(const AllReduceSliceToReduceScatterParams &params) {
  auto bias_add_cnode = params.bias_add->cast<CNodePtr>();
  if (bias_add_cnode == nullptr) {
    return;
  }

  if (!CheckCase3Strategy(params)) {
    return;
  }

  auto origin_shape = GetShapeFromStridedSliceNode(params.stridedslice);
  if (origin_shape.size() != k4DTransposeDim) {
    return;
  }
  auto perm = GetTransposePerm(origin_shape.size());
  auto transpose_out = CreateTransposeNode(params.graph, params.batch_matmul, perm);
  auto reduce_scatter_node = CreateReduceScatterNode(params.graph, transpose_out, params.allreduce);
  if (reduce_scatter_node == nullptr) {
    return;
  }
  auto transpose_in = CreateTransposeNode(params.graph, reduce_scatter_node, perm);
  params.manager->Replace(bias_add_cnode->input(params.bias_add_replace_index), transpose_in);
}

bool CheckCase4Strategy(const AllReduceSliceToReduceScatterParams &params) {
  auto bmm_input_strategy = GetBatchMatMulStrategy(params.batch_matmul);
  auto stridedslice_input_strategy = GetNodeStrategy(params.stridedslice);
  if (!CheckBatchMatmulAndStridedSliceStrategy(bmm_input_strategy, stridedslice_input_strategy)) {
    return false;
  }
  return true;
}

bool CheckAndFillParallelParams(AllReduceSliceToReduceScatterParams *params) {
  MS_EXCEPTION_IF_NULL(params);
  auto bmm_input_strategy = GetBatchMatMulStrategy(params->batch_matmul);

  // (x, ep, 1, mp) * (ep, mp, 1)
  auto &bmm_input1_strategy = bmm_input_strategy.at(0);
  auto &bmm_input2_strategy = bmm_input_strategy.at(1);
  if (bmm_input1_strategy.size() < kMinInputDimNum || bmm_input2_strategy.size() < kMinInputDimNum) {
    return false;
  }

  auto input1_mp = bmm_input1_strategy[bmm_input1_strategy.size() - 1];
  auto input1_ep = bmm_input1_strategy[bmm_input1_strategy.size() - kEpIndexFromRight];

  params->model_parallel = LongToSize(input1_mp);
  params->expert_parallel = LongToSize(input1_ep);
  return true;
}

bool FillBiasAddParams(AllReduceSliceToReduceScatterParams *params) {
  MS_EXCEPTION_IF_NULL(params);
  if (!CheckAndFillParallelParams(params)) {
    return false;
  }

  auto current_node = GetSingleUserPrimNode(params->manager, params->anchor, prim::kPrimReshape);
  if (current_node == nullptr) {
    current_node = params->anchor;
  }
  auto bias_add_node = GetSingleUserPrimNode(params->manager, current_node, prim::kPrimAdd);
  if (bias_add_node == nullptr) {
    return false;
  }

  params->bias_add = bias_add_node;
  auto bias_add_cnode = bias_add_node->cast<CNodePtr>();
  if (bias_add_cnode == nullptr) {
    return false;
  }

  if (bias_add_cnode->input(kBiasAddRightIndex) == current_node) {
    params->bias_add_replace_index = kBiasAddRightIndex;
  }
  return true;
}

bool FillCase2Params(AllReduceSliceToReduceScatterParams *params) {
  MS_EXCEPTION_IF_NULL(params);

  auto current_node = GetSingleUserPrimNode(params->manager, params->bias_add, prim::kPrimReshape);
  if (current_node == nullptr) {
    current_node = params->bias_add;
  }

  current_node = GetSingleUserPrimNode(params->manager, current_node, prim::kPrimAlltoAll);
  if (current_node == nullptr) {
    return false;
  }

  auto anchor_node = current_node;
  auto final_reshape = GetSingleUserPrimNode(params->manager, current_node, prim::kPrimReshape);
  if (final_reshape != nullptr) {
    current_node = final_reshape;
  }

  auto strategy_stridedslice_node = GetSingleUserPrimNode(params->manager, current_node, prim::kPrimStridedSlice);
  if (strategy_stridedslice_node == nullptr) {
    current_node = GetSingleUserPrimNode(params->manager, current_node, prim::kPrimAlltoAll);
    if (current_node == nullptr) {
      return false;
    }
    anchor_node = current_node;
    final_reshape = GetSingleUserPrimNode(params->manager, current_node, prim::kPrimReshape);
    if (final_reshape != nullptr) {
      current_node = final_reshape;
    }
    strategy_stridedslice_node = GetSingleUserPrimNode(params->manager, current_node, prim::kPrimStridedSlice);
  }

  params->anchor = anchor_node;
  params->final_reshape = final_reshape;
  params->strategy_stridedslice = strategy_stridedslice_node;
  return true;
}

/*
 * For Structure as following:
 * AllReduceSliceToReduceScatter case1:
 * BatchMatMul -> AllReduce -> Reshape -> StridedSlice change to
 * BatchMatMul -> Reshape -> Transpose -> ReduceScatter -> Transpose
 *
 * AllReduceSliceToReduceScatter case2:
 * BatchMatMul -> AllReduce (-> Reshape) -> StridedSlice (-> Reshape) -> BiasAdd -> Reshape ->
 * AlltoAll（-> AlltoAll -> Reshape）change to BatchMatMul -> Reshape -> Transpose -> ReduceScatter -> Transpose ->
 * (Reshape ->) BiasAdd (-> Reshape)
 *
 * AllReduceSliceToReduceScatter case3:
 * BatchMatMul -> AllReduce -> StridedSlice -> BiasAdd
 * change to BatchMatMul -> Transpose -> ReduceScatter -> Transpose -> BiasAdd
 *
 * thus it can reduce half of the communication traffic.
 */
bool AllReduceSliceToReduceScatter(const FuncGraphPtr &graph, const opt::OptimizerPtr &) {
  MS_EXCEPTION_IF_NULL(graph);
  // assume no change to graph
  constexpr bool kChanges = false;

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_enable = ms_context->get_param<bool>(MS_CTX_ENABLE_ALLREDUCE_SLICE_TO_REDUCESCATTER);
  if (!is_enable) {
    return kChanges;
  }
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  constexpr size_t kAllreduceInputSize = 2;
  for (const auto &child_graph : manager->func_graphs()) {
    std::list<CNodePtr> graph_orders = child_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    for (const auto &node : origin_nodes_topological) {
      // allreduce node is the target node
      if (!IsPrimitiveCNode(node, prim::kPrimAllReduce)) {
        continue;
      }

      auto input_nodes = node->inputs();
      if (input_nodes.size() != kAllreduceInputSize) {
        continue;
      }

      if (!IsPrimitiveCNode(input_nodes[1], prim::kPrimBatchMatMul)) {
        continue;
      }

      AnfNodePtr stridedslice_node = nullptr;
      auto reshape_node = GetSingleUserPrimNode(manager, node, prim::kPrimReshape);
      if (reshape_node != nullptr) {
        stridedslice_node = GetSingleUserPrimNode(manager, reshape_node, prim::kPrimStridedSlice);
      } else {
        stridedslice_node = GetSingleUserPrimNode(manager, node, prim::kPrimStridedSlice);
      }

      if (stridedslice_node == nullptr) {
        continue;
      }

      auto current_node = GetSingleUserPrimNode(manager, stridedslice_node, prim::kPrimReshape);
      if (current_node == nullptr) {
        current_node = stridedslice_node;
      }

      AllReduceSliceToReduceScatterParams params{manager, child_graph};
      params.batch_matmul = input_nodes[1];
      params.allreduce = node;
      params.reshape = reshape_node;
      params.stridedslice = stridedslice_node;
      params.anchor = stridedslice_node;
      params.strategy_stridedslice = GetSingleUserPrimNode(manager, current_node, prim::kPrimStridedSlice);
      if (params.strategy_stridedslice != nullptr) {
        if (CheckCase1Strategy(params)) {
          AllReduceSliceToReduceScatterCase1(params);
        }
      } else if (FillBiasAddParams(&params)) {
        if (FillCase2Params(&params)) {
          AllReduceSliceToReduceScatterCase2(params);
        } else {
          AllReduceSliceToReduceScatterCase3(params);
        }
      } else if (CheckCase4Strategy(params)) {
        AllReduceSliceToReduceScatterCase1(params);
      }
    }
  }
  return kChanges;
}
}  // namespace parallel
}  // namespace mindspore
