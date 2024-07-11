/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/graph_util/flops_collection.h"
#include <memory>
#include <list>
#include <set>
#include <algorithm>
#include <functional>
#include <vector>
#include <string>
#include <unordered_map>
#include "mindspore/core/ops/framework_ops.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "abstract/abstract_function.h"
#include "ir/func_graph_cloner.h"
#include "frontend/parallel/pass/pass_utils.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/conv_pool_ops.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace parallel {
namespace {

typedef struct {
  int64_t shard_flops;
  int64_t full_flops;
  std::vector<ShapeVector> origin_input_shapes;
  std::vector<ShapeVector> origin_output_shapes;
  std::vector<ShapeVector> shard_input_shapes;
  std::vector<ShapeVector> shard_output_shapes;
} OpInfo;

using OpInfoPtr = std::shared_ptr<OpInfo>;

std::set<std::string> fp_primitive_set = {prim::kPrimMatMul->name(), prim::kPrimFlashAttentionScore->name(),
                                          prim::kPrimBatchMatMul->name(), prim::kPrimConv2D->name()};
std::set<std::string> bp_primitive_set = {prim::kPrimMatMul->name(), prim::kPrimFlashAttentionScoreGrad->name(),
                                          prim::kPrimBatchMatMul->name(), prim::kPrimConv2DBackpropInput->name(),
                                          prim::kPrimConv2DBackpropFilter->name()};
constexpr size_t k_bsh_shape_size = 3;
constexpr size_t k_double_expand_ratio = 2;

int64_t CalFAFlops(const std::vector<ShapeVector> &input_shapes) {
  (void)CheckAndConvertUtils::CheckInteger("rank of 'flash attention inputs'", input_shapes.size(), kGreaterEqual,
                                           kIndex2, "FA");
  auto q_shape = input_shapes[0];
  auto k_shape = input_shapes[1];
  auto b = q_shape[0];
  (void)CheckAndConvertUtils::CheckInRange("rank of 'flash attention input[0]'", q_shape.size(), kIncludeBoth,
                                           {kIndex3, kIndex4}, "FA");
  (void)CheckAndConvertUtils::CheckInRange("rank of 'flash attention input[1]'", k_shape.size(), kIncludeBoth,
                                           {kIndex3, kIndex4}, "FA");
  auto s1 = q_shape.size() == k_bsh_shape_size ? q_shape[1] : q_shape[kIndex2];
  auto s2 = k_shape.size() == k_bsh_shape_size ? k_shape[1] : k_shape[kIndex2];
  auto h1 = q_shape.size() == k_bsh_shape_size ? q_shape[kIndex2] : q_shape[1] * q_shape[kIndex3];
  return k_double_expand_ratio * k_double_expand_ratio * b * s1 * s2 * h1;
}

OpInfoPtr GetFAInfo(const CNodePtr &node) {
  OpInfoPtr op_info = std::make_shared<OpInfo>();
  for (size_t i = 1; i < kIndex3; i++) {
    op_info->shard_input_shapes.emplace_back(node->input(i)->abstract()->GetShapeTrack()->GetShapeVector());
  }
  op_info->origin_input_shapes = op_info->shard_input_shapes;
  if (node->HasPrimalAttr("origin_input_shapes")) {
    op_info->origin_input_shapes = GetValue<std::vector<ShapeVector>>(node->GetPrimalAttr("origin_input_shapes"));
  }
  op_info->full_flops = CalFAFlops(op_info->origin_input_shapes);
  op_info->shard_flops = CalFAFlops(op_info->shard_input_shapes);
  return op_info;
}

int64_t CalBMMFlops(const std::vector<ShapeVector> &input_shapes, bool transpose_b) {
  (void)CheckAndConvertUtils::CheckInteger("rank of 'batchmatmul inputs'", input_shapes.size(), kGreaterEqual, kIndex2,
                                           "BMM");
  auto a_shape = input_shapes[0];
  auto b_shape = input_shapes[1];
  auto a_dim_index = a_shape.size() - kIndex2;
  auto b_dim_index = a_shape.size() - 1;
  int64_t flops = 1;
  auto pre_shape = a_shape.size() > b_shape.size() ? a_shape : b_shape;
  for (size_t i = 0; i < pre_shape.size() - kIndex2; i++) {
    flops *= pre_shape[i];
  }
  // [N, C]*[C, M]
  auto M = transpose_b ? *(b_shape.end() - 2) : *(b_shape.end() - 1);
  flops *= k_double_expand_ratio * a_shape[a_dim_index] * a_shape[b_dim_index] * M;
  return flops;
}

OpInfoPtr GetBMMInfo(const CNodePtr &node) {
  OpInfoPtr op_info = std::make_shared<OpInfo>();
  for (size_t i = 1; i < kIndex3; i++) {
    op_info->shard_input_shapes.emplace_back(node->input(i)->abstract()->GetShapeTrack()->GetShapeVector());
  }
  op_info->origin_input_shapes = op_info->shard_input_shapes;
  if (node->HasPrimalAttr("origin_input_shapes")) {
    op_info->origin_input_shapes = GetValue<std::vector<ShapeVector>>(node->GetPrimalAttr("origin_input_shapes"));
  }
  MS_EXCEPTION_IF_NULL(node);
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  auto tb = GetValue<bool>(GetValueNode(node->input(kIndex4)));
  op_info->full_flops = CalBMMFlops(op_info->origin_input_shapes, tb);
  op_info->shard_flops = CalBMMFlops(op_info->shard_input_shapes, tb);
  return op_info;
}

int64_t CalConv2DFlops(const std::vector<ShapeVector> &input_shapes, const ShapeVector &out_shape, bool nchw) {
  (void)CheckAndConvertUtils::CheckInteger("rank of 'conv2d inputs'", input_shapes.size(), kGreaterEqual, kIndex2,
                                           "Conv2D");
  auto a_shape = input_shapes[0];
  auto b_shape = input_shapes[1];
  (void)CheckAndConvertUtils::CheckInteger("rank of 'conv2d input0'", a_shape.size(), kGreaterEqual, kIndex4, "Conv2D");
  (void)CheckAndConvertUtils::CheckInteger("rank of 'conv2d input1'", b_shape.size(), kGreaterEqual, kIndex4, "Conv2D");
  auto N = a_shape[0];
  auto C_IN = nchw ? a_shape[1] : a_shape[kIndex3];
  auto H_OUT = nchw ? out_shape[kIndex2] : out_shape[1];
  auto W_OUT = nchw ? out_shape[kIndex3] : out_shape[kIndex2];
  auto C_OUT = b_shape[0];
  auto K0 = b_shape[kIndex2];
  auto K1 = b_shape[kIndex3];
  return N * k_double_expand_ratio * C_IN * K0 * K1 * H_OUT * W_OUT * C_OUT;
}

OpInfoPtr GetConv2DInfo(const CNodePtr &node) {
  OpInfoPtr op_info = std::make_shared<OpInfo>();
  for (size_t i = 1; i < kIndex3; i++) {
    op_info->shard_input_shapes.emplace_back(node->input(i)->abstract()->GetShapeTrack()->GetShapeVector());
  }
  op_info->origin_input_shapes = op_info->shard_input_shapes;
  op_info->shard_output_shapes = std::vector<ShapeVector>{node->abstract()->GetShapeTrack()->GetShapeVector()};
  op_info->origin_output_shapes = op_info->shard_output_shapes;
  if (node->HasPrimalAttr("origin_input_shapes")) {
    op_info->origin_output_shapes = {GetValue<ShapeVector>(node->GetPrimalAttr("origin_output_shape"))};
    op_info->origin_input_shapes = GetValue<std::vector<ShapeVector>>(node->GetPrimalAttr("origin_input_shapes"));
  }
  MS_EXCEPTION_IF_NULL(node);
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  auto data_format = GetValue<std::string>(prim->GetAttr(FORMAT));
  bool nchw = data_format == "NCHW";
  op_info->full_flops = CalConv2DFlops(op_info->origin_input_shapes, op_info->origin_output_shapes[0], nchw);
  op_info->shard_flops = CalConv2DFlops(op_info->shard_input_shapes, op_info->shard_output_shapes[0], nchw);
  return op_info;
}

int64_t CalMatMulFlops(const std::vector<ShapeVector> &input_shapes, bool transpose_b) {
  auto full_a_shape = input_shapes[0];
  auto full_b_shape = input_shapes[1];
  int64_t full_flops = transpose_b ? k_double_expand_ratio * full_a_shape[0] * full_a_shape[1] * full_b_shape[0]
                                   : k_double_expand_ratio * full_a_shape[0] * full_a_shape[1] * full_b_shape[1];
  return full_flops;
}

OpInfoPtr GetMatMulInfo(const CNodePtr &node) {
  OpInfoPtr op_info = std::make_shared<OpInfo>();
  for (size_t i = 1; i < kIndex3; i++) {
    op_info->shard_input_shapes.emplace_back(node->input(i)->abstract()->GetShapeTrack()->GetShapeVector());
  }
  op_info->origin_input_shapes = op_info->shard_input_shapes;
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  if (node->HasPrimalAttr("origin_input_shapes")) {
    op_info->origin_input_shapes = GetValue<std::vector<ShapeVector>>(node->GetPrimalAttr("origin_input_shapes"));
  }
  auto tb = GetValue<bool>(GetValueNode(node->input(kIndex4)));
  op_info->shard_flops = CalMatMulFlops(op_info->shard_input_shapes, tb);
  op_info->full_flops = CalMatMulFlops(op_info->origin_input_shapes, tb);
  return op_info;
}

OpInfoPtr GetFwdNodeInfo(const CNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScore)) {
    return GetFAInfo(node);
  }
  if (IsPrimitiveCNode(node, prim::kPrimBatchMatMul)) {
    return GetBMMInfo(node);
  }
  if (IsPrimitiveCNode(node, prim::kPrimConv2D)) {
    return GetConv2DInfo(node);
  }
  return GetMatMulInfo(node);
}

OpInfoPtr GetBpNodeInfo(const CNodePtr &node, std::unordered_map<std::string, OpInfoPtr> op_info_map_dx_dw_map) {
  MS_EXCEPTION_IF_NULL(node);
  std::string fwd_unique_id = GetValue<std::string>(node->GetPrimalAttr(kPrimalAttrForwardUniqueId));
  auto fwd_info = op_info_map_dx_dw_map[fwd_unique_id];
  OpInfoPtr op_info = std::make_shared<OpInfo>();
  if (IsPrimitiveCNode(node, prim::kPrimBatchMatMul) || IsPrimitiveCNode(node, prim::kPrimConv2DBackpropInput) ||
      IsPrimitiveCNode(node, prim::kPrimConv2DBackpropFilter)) {
    return fwd_info;
  }
  if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScoreGrad)) {
    op_info->origin_input_shapes = fwd_info->origin_input_shapes;
    op_info->shard_input_shapes = fwd_info->shard_input_shapes;
    op_info->shard_flops = fwd_info->shard_flops * k_double_expand_ratio;
    op_info->full_flops = fwd_info->full_flops * k_double_expand_ratio;
  } else if (IsPrimitiveCNode(node, prim::kPrimMatMul)) {
    auto forward_unique_id_list_ptr = node->GetPrimalAttr(FORWARD_UNIQUE_ID_LIST);
    if (!forward_unique_id_list_ptr) {
      return fwd_info;
    }
    auto forward_unique_id_list = GetValue<std::vector<std::string>>(forward_unique_id_list_ptr);
    op_info->origin_input_shapes = fwd_info->origin_input_shapes;
    op_info->shard_input_shapes = fwd_info->shard_input_shapes;
    op_info->shard_flops = fwd_info->shard_flops * SizeToLong(forward_unique_id_list.size());
    op_info->full_flops = fwd_info->full_flops * SizeToLong(forward_unique_id_list.size());
  }
  return op_info;
}

void GetFwdNodeInfos(std::unordered_map<std::string, OpInfoPtr> *op_info_map_dx_dw_map, const FuncGraphPtr &graph) {
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (const auto &each_graph : manager->func_graphs()) {
    std::list<CNodePtr> graph_orders = each_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    for (const auto &node : origin_nodes_topological) {
      if (node->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
        // only support forward node
        continue;
      }
      if (GetCNodePrimitive(node) && fp_primitive_set.find(GetCNodePrimitive(node)->name()) != fp_primitive_set.end()) {
        auto forward_unique_id = node->UniqueId();
        if (node->HasPrimalAttr(kPrimalAttrUniqueId)) {
          forward_unique_id = GetValue<std::string>(node->GetPrimalAttr(kPrimalAttrUniqueId));
        }
        (*op_info_map_dx_dw_map)[forward_unique_id] = GetFwdNodeInfo(node);
      }
    }
  }
}

void GetBpNodeInfos(std::unordered_map<CNodePtr, OpInfoPtr> *op_info_map_dx_dw_map,
                    const std::unordered_map<std::string, OpInfoPtr> &fwd_op_info_map_dx_dw_map,
                    const FuncGraphPtr &graph) {
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (const auto &each_graph : manager->func_graphs()) {
    std::list<CNodePtr> graph_orders = each_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    for (const auto &node : origin_nodes_topological) {
      if (!node->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
        // only support bprop node
        continue;
      }
      if (GetCNodePrimitive(node) && bp_primitive_set.find(GetCNodePrimitive(node)->name()) != bp_primitive_set.end()) {
        (*op_info_map_dx_dw_map)[node] = GetBpNodeInfo(node, fwd_op_info_map_dx_dw_map);
      }
    }
  }
}

nlohmann::ordered_json ToJson(const CNodePtr &para_node,
                              std::unordered_map<std::string, OpInfoPtr> op_info_map_dx_dw_map,
                              std::unordered_map<CNodePtr, OpInfoPtr> bprop_op_info_map_dx_dw_map) {
  nlohmann::ordered_json args;
  MS_EXCEPTION_IF_NULL(para_node);
  auto prim = GetCNodePrimitive(para_node);
  MS_EXCEPTION_IF_NULL(prim);
  OpInfoPtr op_info;
  if (para_node->HasPrimalAttr(kPrimalAttrUniqueId)) {
    op_info = op_info_map_dx_dw_map[GetValue<std::string>(para_node->GetPrimalAttr(kPrimalAttrUniqueId))];
  } else if (para_node->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
    op_info = bprop_op_info_map_dx_dw_map[para_node];
  } else {
    op_info = op_info_map_dx_dw_map[para_node->UniqueId()];
  }
  MS_EXCEPTION_IF_NULL(op_info);
  args["op_name"] = para_node->UniqueName();
  args["op_type"] = prim->name();
  args["intput_origin_shape"] = op_info->origin_input_shapes;
  args["intput_shard_shape"] = op_info->shard_input_shapes;
  args["origin_output_shape"] = op_info->origin_output_shapes;
  args["output_shard_shape"] = op_info->shard_output_shapes;
  args["is_recompute"] = common::AnfAlgo::IsRecompute(para_node);
  args["shard_flops"] = op_info->shard_flops;
  args["full_flops"] = op_info->full_flops;
  return args;
}

void GetCalOps(const FuncGraphPtr &graph, size_t *op_id, nlohmann::ordered_json *args,
               const std::unordered_map<std::string, OpInfoPtr> &op_info_map_dx_dw_map,
               const std::unordered_map<CNodePtr, OpInfoPtr> &bprop_op_info_map_dx_dw_map) {
  MS_EXCEPTION_IF_NULL(graph);
  std::list<CNodePtr> graph_orders = graph->GetOrderedCnodes();
  for (auto &node : graph_orders) {
    MS_EXCEPTION_IF_NULL(node);
    if (IsValueNode<FuncGraph>(node->input(0))) {
      FuncGraphPtr sub_graph = node->input(0)->cast<ValueNodePtr>()->value()->cast<FuncGraphPtr>();
      GetCalOps(sub_graph, op_id, args, op_info_map_dx_dw_map, bprop_op_info_map_dx_dw_map);
    } else if (GetCNodePrimitive(node) &&
               (fp_primitive_set.find(GetCNodePrimitive(node)->name()) != fp_primitive_set.end() ||
                bp_primitive_set.find(GetCNodePrimitive(node)->name()) != bp_primitive_set.end())) {
      (*args)[std::to_string(*op_id)] = ToJson(node, op_info_map_dx_dw_map, bprop_op_info_map_dx_dw_map);
      *op_id = *op_id + 1;
    } else if (node->input(0)->isa<CNode>() && node->input(0)->abstract() != nullptr) {
      auto abs = node->input(0)->abstract();
      if (abs->isa<abstract::FuncGraphAbstractClosure>()) {
        const auto &abstract_func_graph = abs->cast<abstract::FuncGraphAbstractClosurePtr>();
        MS_EXCEPTION_IF_NULL(abstract_func_graph->func_graph());
        GetCalOps(abstract_func_graph->func_graph(), op_id, args, op_info_map_dx_dw_map, bprop_op_info_map_dx_dw_map);
      } else if (abs->isa<abstract::PartialAbstractClosure>()) {
        const auto &abstract_partial_func = abs->cast<abstract::PartialAbstractClosurePtr>();
        const auto &abstract_fn = abstract_partial_func->fn();
        if (abstract_fn->isa<abstract::FuncGraphAbstractClosure>()) {
          const auto &abstract_func_graph = abstract_fn->cast<abstract::FuncGraphAbstractClosurePtr>();
          MS_EXCEPTION_IF_NULL(abstract_func_graph->func_graph());
          GetCalOps(abstract_func_graph->func_graph(), op_id, args, op_info_map_dx_dw_map, bprop_op_info_map_dx_dw_map);
        }
      }
    }
  }
}
}  // namespace

py::tuple FlopsCollection(const FuncGraphPtr &graph) {
  MS_LOG(INFO) << "cal model flops.";
  size_t op_id = 0;
  nlohmann::ordered_json args;
  int64_t full_mfu = 0;
  int64_t full_hfu = 0;
  int64_t shard_mfu = 0;
  int64_t shard_hfu = 0;
  bool is_dynamic_shape = false;
  std::unordered_map<std::string, OpInfoPtr> op_info_map_dx_dw_map;
  std::unordered_map<CNodePtr, OpInfoPtr> bprop_info_map_dx_dw_map;
  GetFwdNodeInfos(&op_info_map_dx_dw_map, graph);
  GetBpNodeInfos(&bprop_info_map_dx_dw_map, op_info_map_dx_dw_map, graph);
  GetCalOps(graph, &op_id, &args, op_info_map_dx_dw_map, bprop_info_map_dx_dw_map);
  constexpr size_t json_dump_mode = 2;
  for (auto arg : args) {
    auto full_op_flops = static_cast<int64_t>(arg["full_flops"]);
    auto shard_op_flops = static_cast<int64_t>(arg["shard_flops"]);
    if (arg["op_type"] == prim::kPrimFlashAttentionScoreGrad->name()) {
      const double expand_ratio_fa_bp = 1.5;
      if (shard_op_flops < 0) {
        is_dynamic_shape = true;
        break;
      }
      full_hfu += full_op_flops * expand_ratio_fa_bp;
      shard_hfu += shard_op_flops * expand_ratio_fa_bp;
    } else {
      if (shard_op_flops < 0) {
        is_dynamic_shape = true;
        break;
      }
      full_hfu += full_op_flops;
      shard_hfu += shard_op_flops;
    }
    if (!arg["is_recompute"]) {
      full_mfu += full_op_flops;
      shard_mfu += shard_op_flops;
    }
  }
  MS_LOG(DEBUG) << args.dump(json_dump_mode);
  return py::make_tuple(full_mfu, full_hfu, shard_mfu, shard_hfu, is_dynamic_shape);
}
}  // namespace parallel
}  // namespace mindspore
