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
#include "backend/common/pass/ir_fusion/batchmatmul_reducescatter_alltoall_fusion.h"
#include <set>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <vector>
#include "base/base.h"
#include "backend/common/pass/ir_fusion/mc2_fusion.h"
#include "include/utils/anfalgo.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "primitive/math_ops.h"
#include "primitive/other_ops.h"
#include "primitive/lite_ops.h"
#include "primitive/array_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore::opt {
namespace {
using AnfNodeIndex = std::pair<AnfNodePtr, int>;
using AnfNodeIndexList = std::vector<AnfNodeIndex>;
constexpr int64_t kSplitTwo = 2;
constexpr int64_t kSplitFour = 4;
constexpr int64_t kSplitEight = 8;
constexpr int64_t kSplitSixteen = 16;
bool IsForwardNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  return !(cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId) || cnode->HasAttr(kAttrDuplicated));
}

bool IsRecomputeNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  return cnode->HasAttr(kAttrDuplicated);
}

bool IsBpropNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  return node->fullname_with_scope().find("Gradients") == 0;
}

namespace {
bool EnableLccl() {
  auto ascend_soc_version = MsContext::GetInstance()->ascend_soc_version();
  if (ascend_soc_version != "ascend910b" && ascend_soc_version != "ascend910_93") {
    return false;
  }
  auto enable_infer_boost = MsContext::GetInstance()->IsEnableInferBoost();
  if (enable_infer_boost) {
    static bool disable_lccl = common::GetEnv("MS_ENABLE_LCCL") == "off";
    if (disable_lccl) {
      return false;
    }
    return true;
  } else {
    static bool enable_lccl = common::GetEnv("MS_ENABLE_LCCL") == "on";
    if (enable_lccl) {
      MS_LOG(EXCEPTION)
        << "Currently, the LCCL communication library is only supported in the inference boost scenario. "
           "Please remove the environment variable MS_ENABLE_LCCL=on.";
    }
    return false;
  }
}
}  // namespace

bool IsKbkAclnnMode() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  bool is_k_by_k_mode = ms_context->IsKByKExecutorMode();
  bool enable_lccl = EnableLccl();
  //  When lccl communication is not enabled in the kbk scenario
  return is_k_by_k_mode && !enable_lccl;
}

bool IsAscendVerison93() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &soc_version = ms_context->ascend_soc_version();
  return soc_version == "ascend910_93";
}

bool GetExpectedInput(const AnfNodePtr &node, const PrimitivePtr expect_prim,
                      const std::vector<PrimitivePtr> &skip_prim_vector, AnfNodePtr *expect_in,
                      AnfNodePtr *expect_bias = nullptr) {
  MS_EXCEPTION_IF_NULL(node);
  size_t idx0 = 0;
  auto prenode = common::AnfAlgo::GetInputNode(node->cast<CNodePtr>(), idx0);
  if (prenode == nullptr || !prenode->isa<CNode>()) {
    return false;
  }
  for (auto skip_prim : skip_prim_vector) {
    if (!IsPrimitiveCNode(prenode, skip_prim)) {
      MS_LOG(DEBUG) << "IsPrimitiveCNode for prenode: " << prenode->fullname_with_scope();
      return false;
    }
    auto cnode = prenode->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto from_shape = common::AnfAlgo::GetOutputInferShape(cnode->input(kIndex1), kIndex0);
    auto to_shape = common::AnfAlgo::GetOutputInferShape(cnode, kIndex0);
    if (skip_prim == prim::kPrimStridedSlice || skip_prim == prim::kPrimStridedSliceGrad) {
      if (from_shape.size() != to_shape.size() || !std::equal(from_shape.begin(), from_shape.end(), to_shape.begin())) {
        MS_LOG(DEBUG) << "StridedSlice or StridedSliceGrad input shape is not equal to output shape.";
        return false;
      }
    }
    if (skip_prim == prim::kPrimReshape) {
      auto in = common::AnfAlgo::GetInputNode(prenode->cast<CNodePtr>(), idx0);
      if (!IsPrimitiveCNode(in, prim::kPrimBatchMatMul)) {
        auto from = from_shape.rbegin();
        auto to = to_shape.rbegin();
        while (from != from_shape.rend() && to != to_shape.rend()) {
          if (*from != *to) {
            MS_LOG(DEBUG) << "Reshape shape mismatch rules.";
            return false;
          }
          ++from;
          ++to;
        }
      }
    }
    if (skip_prim == prim::kPrimBiasAdd && expect_bias != nullptr) {
      *expect_bias = prenode;
    }
    prenode = common::AnfAlgo::GetInputNode(prenode->cast<CNodePtr>(), idx0);
    if (prenode == nullptr || !prenode->isa<CNode>()) {
      return false;
    }
  }
  if (IsPrimitiveCNode(prenode, expect_prim)) {
    *expect_in = prenode;
    return true;
  }
  MS_LOG(DEBUG) << "Is expect_prim for prenode: " << prenode->fullname_with_scope();
  return false;
}

CNodePtr CreateReshapeCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, const ShapeVector &to_shape) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input_node);
  auto from_shape = common::AnfAlgo::GetOutputInferShape(input_node, kIndex0);
  if (std::accumulate(from_shape.begin(), from_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()) !=
      std::accumulate(to_shape.begin(), to_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>())) {
    MS_LOG(EXCEPTION) << "Failed to insert reshape behind " << input_node->fullname_with_scope() << ". From shape is "
                      << from_shape << " and to_shape is " << to_shape;
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto to_shape_node = kernel_graph->NewValueNode(MakeValue<std::vector<int64_t>>(to_shape));
  kernel_graph->AddValueNodeToGraph(to_shape_node);

  auto reshape_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimReshape), input_node, to_shape_node});
  MS_EXCEPTION_IF_NULL(reshape_cnode);
  auto dtype = common::AnfAlgo::GetOutputInferDataType(input_node, kIndex0);
  common::AnfAlgo::SetOutputInferTypeAndShape({dtype}, {to_shape}, reshape_cnode.get());
  return reshape_cnode;
}

ValuePtr GetAttrFromCNodePrimitive(const CNodePtr &node, const std::string &attr_name) {
  MS_EXCEPTION_IF_NULL(node);
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  auto attr = prim->GetAttr(attr_name);
  if (attr == nullptr) {
    MS_LOG(EXCEPTION) << "Get attribute " << attr_name << " from node " << node->DebugString() << " failed.";
  }
  return attr;
}
}  // namespace

bool BatchMatMulReduceScatterAllToAllFusion::IsValidAllToAll(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsPrimitiveCNode(node, prim::kPrimAlltoAll)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  if (!prim->HasAttr(kAttrSplitDim) || !prim->HasAttr(kAttrConcatDim)) {
    return false;
  }

  if (mc2_fusion_level_ == kMC2FusionForward && !IsForwardNode(node)) {
    MS_LOG(DEBUG) << "MC2 fusion level is " << kMC2FusionForward << ", only apply to forward node. Skip node "
                  << node->fullname_with_scope();
    return false;
  }
  if (mc2_fusion_level_ == kMC2FusionBackward && !(IsBpropNode(node) || IsRecomputeNode(node))) {
    MS_LOG(DEBUG) << "MC2 fusion level is " << kMC2FusionBackward << ", only apply to backward node. Skip node "
                  << node->fullname_with_scope();
    return false;
  }

  auto input_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0);
  if (input_dtype != kNumberTypeFloat16 && input_dtype != kNumberTypeBFloat16) {
    return false;
  }

  auto split_dim_attr = prim->GetAttr(kAttrSplitDim);
  auto concat_dim_attr = prim->GetAttr(kAttrConcatDim);
  MS_EXCEPTION_IF_NULL(split_dim_attr);
  MS_EXCEPTION_IF_NULL(concat_dim_attr);
  return GetValue<int64_t>(split_dim_attr) == GetValue<int64_t>(concat_dim_attr) + 1;
}

void BatchMatMulReduceScatterAllToAllFusion::ClearAttr() {
  alltoall_cnode_ = nullptr;
  reducescatter_cnode_ = nullptr;
  batch_matmul_cnode_ = nullptr;
  bias_add_cnode_ = nullptr;
  split_dim_ = -1;
  concat_dim_ = -1;
  hidden_size_ = -1;
  ep_world_size_ = -1;
  tp_world_size_ = -1;
  y_shard_type_ = 0;
}

bool BatchMatMulReduceScatterAllToAllFusion::InferInputDimAndSplitConcatDimByAllToAllCNode() {
  split_dim_ = GetValue<int64_t>(GetAttrFromCNodePrimitive(alltoall_cnode_, kAttrSplitDim));
  concat_dim_ = GetValue<int64_t>(GetAttrFromCNodePrimitive(alltoall_cnode_, kAttrConcatDim));
  ep_world_size_ = GetValue<int64_t>(GetAttrFromCNodePrimitive(alltoall_cnode_, kAttrSplitCount));
  if (ep_world_size_ != kSplitTwo && ep_world_size_ != kSplitFour && ep_world_size_ != kSplitEight &&
      ep_world_size_ != kSplitSixteen) {
    return false;
  }
  auto input_shape = common::AnfAlgo::GetOutputInferShape(alltoall_cnode_->input(kIndex1), kIndex0);
  hidden_size_ = input_shape.back();
  return true;
}
// Create node and set abstract by call op's infer
CNodePtr BatchMatMulReduceScatterAllToAllFusion::PreProcessAndCreateBatchMatMulReduceScatterAllToAllCNode(
  const FuncGraphPtr func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto x = batch_matmul_cnode_->input(kIndex1);
  // Reshape.
  auto x_type_id1 = common::AnfAlgo::GetPrevNodeOutputInferDataType(batch_matmul_cnode_, kIndex0);
  auto shape_vec1 = common::AnfAlgo::GetOutputInferShape(batch_matmul_cnode_->input(kIndex1), kIndex0);
  auto AdjustShaperTo3D = [](ShapeVector shape) -> ShapeVector {
    if (shape.size() > kSizeThree) {
      int64_t first_dim_product = 1;
      for (size_t i = 0; i < shape.size() - kSizeTwo; ++i) {
        first_dim_product *= shape[i];
      }
      return {first_dim_product, shape[shape.size() - kSizeTwo], shape[shape.size() - kSizeOne]};
    }
    return shape;
  };

  shape_vec1 = AdjustShaperTo3D(shape_vec1);
  auto reshape_node1 = CreateReshapeCNode(func_graph, x, shape_vec1);
  MS_LOG(DEBUG) << "Process succeed. The new reshape node is " << reshape_node1->fullname_with_scope();

  // BatchMatMulReduceScatterAlltoAll
  auto prim = prim::kPrimBatchMatMulReduceScatterAlltoAll->Clone();
  MS_EXCEPTION_IF_NULL(prim);
  auto alltoall_prim = GetCNodePrimitive(alltoall_cnode_);
  MS_EXCEPTION_IF_NULL(alltoall_prim);
  auto reducescatter_prim = GetCNodePrimitive(reducescatter_cnode_);
  MS_EXCEPTION_IF_NULL(reducescatter_prim);
  tp_world_size_ = GetValue<int64_t>(reducescatter_prim->GetAttr(kAttrRankSize));
  if (tp_world_size_ != kSplitTwo && tp_world_size_ != kSplitFour && tp_world_size_ != kSplitEight &&
      tp_world_size_ != kSplitSixteen) {
    return nullptr;
  }

  prim->AddAttr(kAttrGroupEp, alltoall_prim->GetAttr(kAttrGroup));
  prim->AddAttr(kAttrGroupTp, reducescatter_prim->GetAttr(kAttrGroup));
  prim->AddAttr(kAttrEpWorldSize, MakeValue<int64_t>(ep_world_size_));
  prim->AddAttr(kAttrTpWorldSize, MakeValue<int64_t>(tp_world_size_));
  prim->AddAttr(kAttrYShardType, MakeValue<int64_t>(y_shard_type_));
  auto transpose_weight = GetValue<bool>(GetValueNode(batch_matmul_cnode_->input(kIndex4)));
  prim->AddAttr(kAttrTransposeWeight, MakeValue<bool>(transpose_weight));

  auto weight = batch_matmul_cnode_->input(kIndex2);
  AnfNodePtrList inputs = {NewValueNode(prim), reshape_node1, weight};
  if (bias_add_cnode_) {
    inputs.push_back(bias_add_cnode_->input(kIndex2));
  }
  auto batchmatmul_reducescatter_alltoall_cnode = func_graph->NewCNode(inputs);
  auto shape_vec = common::AnfAlgo::GetOutputInferShape(alltoall_cnode_, kIndex0);
  shape_vec = AdjustShaperTo3D(shape_vec);
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(x_type_id1), shape_vec);
  batchmatmul_reducescatter_alltoall_cnode->set_abstract(abs);
  MS_LOG(DEBUG) << "Process succeed. The new fusion node is "
                << batchmatmul_reducescatter_alltoall_cnode->fullname_with_scope();

  // Reshape.
  auto shape_vec2 = common::AnfAlgo::GetOutputInferShape(alltoall_cnode_, kIndex0);
  auto reshape_node2 = CreateReshapeCNode(func_graph, batchmatmul_reducescatter_alltoall_cnode, shape_vec2);
  MS_LOG(DEBUG) << "Process succeed. The new fusion node is " << reshape_node2->fullname_with_scope();
  return reshape_node2;
}

// 1、BatchMatMul -> Reshape -> Transpose -> ReduceScatter -> Transpose ->  StridedSlice -> AllToAll
// 2、BatchMatMul -> Reshape -> Transpose -> ReduceScatter -> Transpose -> Reshape ->  StridedSlice -> Reshape ->
//    AllToAll
// 3、BatchMatMul -> Reshape -> Transpose -> ReduceScatter -> Transpose -> Reshape ->  StridedSlice -> AllToAll
// 4、BatchMatMul -> Reshape -> Transpose -> ReduceScatter -> Transpose -> Reshape ->  StridedSlice -> StridedSlice ->
//    AllToAll
// 5、BatchMatMul -> Reshape -> Transpose -> ReduceScatter -> Transpose ->  Reshape -> BiasAdd -> Reshape ->
//    StridedSlice -> AllToAll
// 6、BatchMatMul -> StridedSliceGrad -> Split -> TupleGetItem -> MakeTuple ->  Concat ->
//                                             -> TupleGetItem ->
//    ReduceScatter -> StridedSliceGrad -> AllToAll
AnfNodePtr BatchMatMulReduceScatterAllToAllFusion::Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  // 1. Match fusion pattern and get attribute
  // 1.1 AllToAll
  if (!IsValidAllToAll(node)) {
    return nullptr;
  }
  MS_LOG(DEBUG) << "Try to fusion for alltoall node: " << node->fullname_with_scope();
  ClearAttr();
  alltoall_cnode_ = node->cast<CNodePtr>();
  if (!InferInputDimAndSplitConcatDimByAllToAllCNode()) {
    MS_LOG(DEBUG) << "InferInputDimAndSplitConcatDimByAllToAllCNode failed, skip fusion.";
    return nullptr;
  }

  // 1.2 ReduceScatter
  AnfNodePtr reducescatter;
  AnfNodePtr biasadd = nullptr;
  if (!GetExpectedInput(alltoall_cnode_, prim::kPrimReduceScatter,
                        {prim::kPrimStridedSlice, prim::kPrimTranspose, prim::kPrimTransposeView}, &reducescatter) &&
      !GetExpectedInput(alltoall_cnode_, prim::kPrimReduceScatter,
                        {prim::kPrimReshape, prim::kPrimStridedSlice, prim::kPrimReshape, prim::kPrimTranspose,
                         prim::kPrimTransposeView},
                        &reducescatter) &&
      !GetExpectedInput(alltoall_cnode_, prim::kPrimReduceScatter,
                        {prim::kPrimStridedSlice, prim::kPrimReshape, prim::kPrimTranspose, prim::kPrimTransposeView},
                        &reducescatter) &&
      !GetExpectedInput(alltoall_cnode_, prim::kPrimReduceScatter,
                        {prim::kPrimStridedSlice, prim::kPrimStridedSlice, prim::kPrimReshape, prim::kPrimTranspose,
                         prim::kPrimTransposeView},
                        &reducescatter) &&
      !GetExpectedInput(alltoall_cnode_, prim::kPrimReduceScatter, {prim::kPrimStridedSliceGrad},
                        &reducescatter)  // grad
      &&
      !GetExpectedInput(alltoall_cnode_, prim::kPrimReduceScatter,
                        {prim::kPrimReshape, prim::kPrimStridedSliceGrad, prim::kPrimReshape}, &reducescatter)  // grad
      && !GetExpectedInput(alltoall_cnode_, prim::kPrimReduceScatter,
                           {prim::kPrimReshape, prim::kPrimBiasAdd, prim::kPrimReshape, prim::kPrimTranspose,
                            prim::kPrimTransposeView, prim::kPrimStridedSlice},
                           &reducescatter, &biasadd)) {
    MS_LOG(DEBUG) << "Cannot find expected ReduceScatter input.";
    return nullptr;
  }
  reducescatter_cnode_ = reducescatter->cast<CNodePtr>();
  auto in = common::AnfAlgo::GetOutputInferShape(reducescatter_cnode_->input(kIndex1), kIndex0);
  auto out = common::AnfAlgo::GetOutputInferShape(reducescatter_cnode_, kIndex0);
  if (in.back() == out.back() && in.back() == hidden_size_) {
    y_shard_type_ = 1;
  } else {
    MS_LOG(WARNING) << "Currently, The shard_type of 0 is not support";
    return nullptr;
  }
  if (biasadd) {
    bias_add_cnode_ = biasadd->cast<CNodePtr>();
  }

  // 1.3 BatchMatMul
  AnfNodePtr batchmatmul;
  if (!GetExpectedInput(reducescatter, prim::kPrimBatchMatMul,
                        {prim::kPrimTranspose, prim::kPrimTransposeView, prim::kPrimReshape}, &batchmatmul) &&
      !GetExpectedInput(reducescatter, prim::kPrimBatchMatMul,
                        {prim::kPrimConcat, prim::kPrimMakeTuple, prim::kPrimTupleGetItem, prim::kPrimSplit,
                         prim::kPrimStridedSliceGrad, prim::kPrimReshape},
                        &batchmatmul)  // new
      && !GetExpectedInput(reducescatter, prim::kPrimBatchMatMul,
                           {prim::kPrimConcat, prim::kPrimMakeTuple, prim::kPrimTupleGetItem, prim::kPrimSplit,
                            prim::kPrimReshape, prim::kPrimStridedSliceGrad, prim::kPrimReshape},
                           &batchmatmul)  // new
  ) {
    MS_LOG(DEBUG) << "Cannot find expected BatchMatMul input.";
    return nullptr;
  }
  batch_matmul_cnode_ = batchmatmul->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(batch_matmul_cnode_);
  auto inputs = batch_matmul_cnode_->inputs();
  if (inputs.size() != kIndex5) {
    MS_LOG(DEBUG) << "The input size of BatchMatMul should be 5, but got " << inputs.size();
    return nullptr;
  }
  auto transpose_x = GetValue<bool>(GetValueNode(batch_matmul_cnode_->input(kIndex3)));
  if (transpose_x) {
    MS_LOG(DEBUG) << "Currently, The transpose_x of BatchMatMul only support 0, but got " << transpose_x;
    return nullptr;
  }
  // 2. Create Fusion Node
  auto ret_node = PreProcessAndCreateBatchMatMulReduceScatterAllToAllCNode(func_graph);
  return ret_node;
}

bool BatchMatMulReduceScatterAllToAllFusion::Run(const FuncGraphPtr &func_graph) {
  if (IsKbkAclnnMode() && IsAscendVerison93()) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    mc2_fusion_level_ = ms_context->get_param<int>(MS_CTX_COMPUTE_COMMUNICATE_FUSION_LEVEL);
    if (mc2_fusion_level_ != kMC2NotFusion && mc2_fusion_level_ != kMC2FusionForward &&
        mc2_fusion_level_ != kMC2FusionBackward && mc2_fusion_level_ != kMC2FusionFull) {
      MS_LOG(DEBUG) << "In KBK mode MC2 fusion level is " << mc2_fusion_level_ << ", only support 0, 1, 2, 3.";
      return false;
    }
    if (mc2_fusion_level_ == kMC2NotFusion) {
      MS_LOG(DEBUG) << "MC2 fusion level is 0, not enable fusion.";
      return false;
    }

    if (common::IsExecuteSimulation()) {
      MS_LOG(EXCEPTION) << "Not support compute_communication_fusion_level when MS_SIMULATION_LEVEL=3.";
    }
    return NodePass::Run(func_graph);
  }
  return false;
}
}  // namespace mindspore::opt
