/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <set>
#include <string>
#include "ir/func_graph.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/nn_ops.h"

#include "utils/parallel_node_check.h"

namespace mindspore {
// clang-format off
#ifndef ENABLE_SECURITY
static const std::set<std::string> PARALLEL_BLACK_LIST_ = {mindspore::kTupleGetItemOpName, "J", "list_getitem",
  "array_getitem", "tuple_setitem", "Depend", "list_setitem", "array_setitem", "dict_getitem",
  "list_append", "list_map", "list_reduce", "tuple_reversed", "tile_shape", "tuple_div", "tuple_to_array",
  "make_dict", "make_slice", "string_eq", "VirtualLoss", "Return", "env_getitem", "TensorShape", "ScalarToTensor",
  "partial", "env_setitem", "env_getitem", "env_add", "Shape",
  "dot", "im2col", "col2im", "im2col_v1", "state_setitem", "TensorDump", "_MirrorSilentCheck",
  "ImageSummary", "TensorSummary", "Debug", "HistogramSummary", "col2im_v1", "resolve", "BroadcastGradientArgs",
  "InvertPermutation", "DropoutGenMask", "StatelessDropOutGenMask", "embed", "create_instance", "RefToEmbed",
  "StopGradient", "UpdateState", "Load", "Switch", "Print", "call_instance", "TensorMove", "DType",
  "ScalarAdd", "ScalarSub", "ScalarMul", "ScalarDiv", "ScalarFloorDiv", "ScalarPow", "ScalarSummary", "ScalarCast",
  "ScalarMod", "ScalarGt", "ScalarGe", "ScalarLt", "ScalarLe", "ScalarEq"};
#else
static const std::set<std::string> PARALLEL_BLACK_LIST_ = {mindspore::kTupleGetItemOpName, "J", "list_getitem",
  "array_getitem", "tuple_setitem", "Depend", "list_setitem", "array_setitem", "dict_getitem",
  "list_append", "list_map", "list_reduce", "tuple_reversed", "tile_shape", "tuple_div", "tuple_to_array",
  "make_dict", "make_slice", "string_eq", "VirtualLoss", "Return", "env_getitem", "TensorShape", "ScalarToTensor",
  "identity", "partial", "env_setitem", "env_getitem", "env_add", "Shape", "FillV2", "_MirrorSilentCheck",
  "dot", "im2col", "col2im", "im2col_v1", "state_setitem", "Debug", "col2im_v1", "resolve", "BroadcastGradientArgs",
  "InvertPermutation", "DropoutGenMask", "StatelessDropOutGenMask", "embed", "create_instance", "RefToEmbed",
  "StopGradient", "UpdateState", "Load", "Switch", "Print", "call_instance", "TensorMove", "DType",
  "ScalarAdd", "ScalarSub", "ScalarMul", "ScalarDiv", "ScalarFloorDiv", "ScalarPow", "ScalarSummary", "ScalarCast",
  "ScalarMod", "ScalarGt", "ScalarGe", "ScalarLt", "ScalarLe", "ScalarEq"};
#endif
static const std::set<PrimitivePtr> ALLGATHER_NODE_LIST_ = {prim::kPrimAllGather, prim::kPrimMiniStepAllGather,
                                                            prim::kPrimMicroStepAllGather};
static const std::set<PrimitivePtr> TRIVIAL_NODE_LIST_ = {prim::kPrimCast, prim::kPrimDepend, prim::kPrimQuant,
                                                          prim::kPrimMirrorSilentCheck,
                                                            std::make_shared<Primitive>("AscendAntiQuant")};
// clang-format on

bool IsInParallelBlackList(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  return (PARALLEL_BLACK_LIST_.find(prim->name()) != PARALLEL_BLACK_LIST_.end());
}

bool IsInAllGatherNodeList(const CNodePtr &cnode) {
  for (auto &value : ALLGATHER_NODE_LIST_) {
    if (IsPrimitiveCNode(cnode, value)) {
      return true;
    }
  }
  return false;
}

bool IsInTrivialNodeList(const CNodePtr &cnode) {
  for (auto &value : TRIVIAL_NODE_LIST_) {
    if (IsPrimitiveCNode(cnode, value)) {
      return true;
    }
  }
  return false;
}

// Return true if cnode is ReShape and match pattern DropoutGenMask -> ReShape -> FlashAttentionScore
bool IsReshapeBetweenDropoutGenMaskAndFlashAttentionScore(const CNodePtr &cnode) {
  if (!IsPrimitiveCNode(cnode, prim::kPrimReshape)) {
    return false;
  }
  auto input1 = cnode->input(kIndex1);
  if (!IsPrimitiveCNode(input1, prim::kPrimDropoutGenMask)) {
    return false;
  }
  auto func_graph = cnode->func_graph();
  auto manager = func_graph->manager();
  auto node_users = manager->node_users()[cnode];
  if (node_users.size() != 1 || !IsPrimitiveCNode(node_users.begin()->first, prim::kPrimFlashAttentionScore)) {
    return false;
  }
  return true;
}

bool IsParallelConsiderCNode(const CNodePtr &cnode) {
  if (cnode == nullptr || cnode->size() == 0) {
    return false;
  }
  const auto &prim_node = cnode->input(0)->cast<ValueNodePtr>();
  if (prim_node == nullptr) {
    return false;
  }
  const auto &prim = prim_node->value()->cast<PrimitivePtr>();
  if (prim == nullptr) {
    return false;
  }
  // If match pattern DropoutGenMask -> ReShape -> FlashAttentionScore, skip ReShape
  if (IsReshapeBetweenDropoutGenMaskAndFlashAttentionScore(cnode)) {
    return false;
  }
  return !IsInParallelBlackList(prim);
}
}  // namespace mindspore
