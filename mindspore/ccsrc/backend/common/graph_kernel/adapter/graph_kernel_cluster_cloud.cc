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
#include "backend/common/graph_kernel/adapter/graph_kernel_cluster_cloud.h"

#include <algorithm>
#include <utility>
#include <string>
#include <vector>

#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/lite_ops.h"
#include "mindspore/core/ops/comparison_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "ir/graph_utils.h"
#include "include/common/utils/anfalgo.h"
#include "utils/anf_utils.h"
#include "utils/ms_context.h"
#include "utils/file_utils.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/core/value_depend_op_utils.h"

namespace mindspore::graphkernel {
std::vector<PrimitivePtr> StaticShapeCluster::GetClusterOps() {
  std::vector<OpWithLevel> clusterable_ops_with_level = {
    // all target
    {kAllTarget, OpLevel_0, prim::kPrimAbs},
    {kAllTarget, OpLevel_0, prim::kPrimAdd},
    {kAllTarget, OpLevel_0, prim::kPrimCast},
    {kAllTarget, OpLevel_0, prim::kPrimEqual},
    {kAllTarget, OpLevel_0, prim::kPrimExp},
    {kAllTarget, OpLevel_0, prim::kPrimLog},
    {kAllTarget, OpLevel_0, prim::kPrimMaximum},
    {kAllTarget, OpLevel_0, prim::kPrimMinimum},
    {kAllTarget, OpLevel_0, prim::kPrimMul},
    {kAllTarget, OpLevel_0, prim::kPrimNeg},
    {kAllTarget, OpLevel_0, prim::kPrimPow},
    {kAllTarget, OpLevel_0, prim::kPrimRealDiv},
    {kAllTarget, OpLevel_0, prim::kPrimReciprocal},
    {kAllTarget, OpLevel_1, prim::kPrimReduceSum},
    {kAllTarget, OpLevel_1, prim::kPrimReshape},
    {kAllTarget, OpLevel_0, prim::kPrimRound},
    {kAllTarget, OpLevel_0, prim::kPrimRsqrt},
    {kAllTarget, OpLevel_0, prim::kPrimSqrt},
    {kAllTarget, OpLevel_0, prim::kPrimSub},
    {kAllTarget, OpLevel_0, prim::kPrimTanh},
    {kAllTarget, OpLevel_1, prim::kPrimTranspose},
    // ascend
    {kAscendDevice, OpLevel_1, prim::kPrimMatMul},
    {kAscendDevice, OpLevel_1, prim::kPrimTransData},
    {kAscendDevice, OpLevel_1, prim::kPrimBatchMatMul},
    // gpu
    {kGPUDevice, OpLevel_0, prim::kPrimACos},
    {kGPUDevice, OpLevel_0, prim::kPrimAcosh},
    {kGPUDevice, OpLevel_2, prim::kPrimArgMax},
    {kGPUDevice, OpLevel_2, prim::kPrimArgmin},
    {kGPUDevice, OpLevel_0, prim::kPrimAsin},
    {kGPUDevice, OpLevel_0, prim::kPrimAsinh},
    {kGPUDevice, OpLevel_0, prim::kPrimAssign},
    {kGPUDevice, OpLevel_0, prim::kPrimAtan},
    {kGPUDevice, OpLevel_0, prim::kPrimAtan2},
    {kGPUDevice, OpLevel_0, prim::kPrimCos},
    {kGPUDevice, OpLevel_0, prim::kPrimDiv},
    {kGPUDevice, OpLevel_0, prim::kPrimErf},
    {kGPUDevice, OpLevel_0, prim::kPrimExpm1},
    {kGPUDevice, OpLevel_0, prim::kPrimFloor},
    {kGPUDevice, OpLevel_0, prim::kPrimFloorDiv},
    {kGPUDevice, OpLevel_0, prim::kPrimFloorMod},
    {kGPUDevice, OpLevel_0, prim::kPrimGreater},
    {kGPUDevice, OpLevel_0, prim::kPrimGreaterEqual},
    {kGPUDevice, OpLevel_0, prim::kPrimIsFinite},
    {kGPUDevice, OpLevel_0, prim::kPrimIsInf},
    {kGPUDevice, OpLevel_0, prim::kPrimIsNan},
    {kGPUDevice, OpLevel_0, prim::kPrimLess},
    {kGPUDevice, OpLevel_0, prim::kPrimLessEqual},
    {kGPUDevice, OpLevel_0, prim::kPrimLogicalAnd},
    {kGPUDevice, OpLevel_0, prim::kPrimLogicalOr},
    {kGPUDevice, OpLevel_0, prim::kPrimLogicalNot},
    {kGPUDevice, OpLevel_0, prim::kPrimMod},
    {kGPUDevice, OpLevel_0, prim::kPrimNotEqual},
    {kGPUDevice, OpLevel_1, prim::kPrimReduceMax},
    {kGPUDevice, OpLevel_1, prim::kPrimReduceMin},
    {kGPUDevice, OpLevel_0, prim::kPrimSelect},
    {kGPUDevice, OpLevel_0, prim::kPrimSign},
    {kGPUDevice, OpLevel_0, prim::kPrimSin},
    {kGPUDevice, OpLevel_0, prim::kPrimStridedSlice},
    {kGPUDevice, OpLevel_1, prim::kPrimCumSum},
    {kGPUDevice, OpLevel_1, prim::kPrimOneHot},
    // cpu
    {kCPUDevice, OpLevel_0, prim::kPrimLogicalNot},
    {kCPUDevice, OpLevel_0, prim::kPrimMod},
    {kCPUDevice, OpLevel_1, prim::kPrimReduceMax},
    {kCPUDevice, OpLevel_0, prim::kPrimSelect},
    {kCPUDevice, OpLevel_0, prim::kPrimLess},
    {kCPUDevice, OpLevel_0, prim::kPrimLessEqual},
  };
  const auto &flags = GraphKernelFlags::GetInstance();
  auto ops = GkUtils::GetValidOps(clusterable_ops_with_level, flags.fusion_ops_level, flags.enable_cluster_ops_only,
                                  flags.enable_cluster_ops, flags.disable_cluster_ops);
  return GkUtils::FilterExcludedOps(ops);
}

std::vector<PrimitivePtr> StaticShapeCluster::GetClusterableOpList() { return StaticShapeCluster::GetClusterOps(); }

bool StaticShapeCluster::IsClusterableOp(const AnfNodePtr &node) {
  if (AnfUtils::IsGraphKernel(node)) {
    auto sub_graph = GetCNodeFuncGraph(node);
    if (auto type = sub_graph->get_attr("composite_type")) {
      if (GetValue<std::string>(type) == "inplace_assign_builder") {
        return false;
      }
    }
    return true;
  }
  if (GkUtils::IsKeepBasicNode(node)) {
    return false;
  }
  if (common::AnfAlgo::IsDynamicShape(node)) {
    return false;
  }
  bool node_in_oplist = std::any_of(op_list_.begin(), op_list_.end(),
                                    [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
  if (!node_in_oplist) {
    return false;
  }

  auto cb = Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  // if node's output type is complex64 or complex128, cannot be added to the cluster list.
  auto node_output_type = cb->GetOutputType(node, 0);
  if (node_output_type == kNumberTypeComplex64 || node_output_type == kNumberTypeComplex128) {
    return false;
  }
  if (IsPrimitiveCNode(node, prim::kPrimCast)) {
    auto node_input_type = cb->GetInputType(node, 0);
    if ((node_input_type == kNumberTypeComplex64) || (node_input_type == kNumberTypeComplex128)) {
      return false;
    }
  }
  // For AICPU operators, only the Reshape can be clustered.
  if (cb->GetTargetFromContext() == kAscendDevice) {
    if (cb->GetProcessor(node) != "aicore" && !IsPrimitiveCNode(node, prim::kPrimReshape)) {
      return false;
    }
  }
  if (!ValueDependOpUtils::IsConstInput(node)) {
    return false;
  }
  return true;
}

std::vector<PrimitivePtr> DynamicShapeCluster::GetClusterableOpList() {
  std::vector<PrimitivePtr> clusterable_ops_with_level = {
    prim::kPrimAbs,       prim::kPrimAdd,       prim::kPrimCast,    prim::kPrimExp,     prim::kPrimLog,
    prim::kPrimMaximum,   prim::kPrimMinimum,   prim::kPrimMul,     prim::kPrimNeg,     prim::kPrimPow,
    prim::kPrimRealDiv,   prim::kPrimSqrt,      prim::kPrimSub,     prim::kPrimReshape, prim::kPrimReduceSum,
    prim::kPrimReduceMin, prim::kPrimReduceMax, prim::kPrimBiasAdd, prim::kPrimMatMul,  prim::kPrimBatchMatMul,
    prim::kPrimTranspose};
  return clusterable_ops_with_level;
}

bool DynamicShapeCluster::IsClusterableOp(const AnfNodePtr &node) {
  bool node_in_oplist = std::any_of(op_list_.begin(), op_list_.end(),
                                    [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
  if (!node_in_oplist || !common::AnfAlgo::IsDynamicShape(node) || common::AnfAlgo::IsDynamicRankNode(node)) {
    return false;
  }
  if (GkUtils::IsKeepBasicNode(node)) {
    return false;
  }
  if (!ValueDependOpUtils::IsConstInput(node)) {
    return false;
  }
  return true;
}

bool DynamicShapeCluster::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  Init(func_graph);
  bool changed = Process(func_graph);
  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  Clean();
  return changed;
}
}  // namespace mindspore::graphkernel
