/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/adapter/graph_kernel_expander_cloud.h"

#include "mindspore/core/ops/random_ops.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/lite_ops.h"
#include "mindspore/core/ops/comparison_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
namespace mindspore::graphkernel {
namespace {
bool DvmSupported(const AnfNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimAddN)) {
    constexpr auto max_input_num = 10;
    auto input_num = common::AnfAlgo::GetInputTensorNum(node);
    if (input_num > max_input_num) {
      return false;
    }
  }
  auto cb = Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  auto node_output_type = cb->GetOutputType(node, 0);
  return (node_output_type == kNumberTypeFloat16 || node_output_type == kNumberTypeFloat32);
}
}  // namespace

std::vector<PrimitivePtr> GraphKernelExpanderCloud::GetExpanderOps() {
  std::vector<OpWithLevel> expand_ops_with_level = {
    {kAllTarget, OpLevel_0, prim::kPrimAddN},
    {kAllTarget, OpLevel_0, prim::kPrimAssignAdd},
    {kAllTarget, OpLevel_1, prim::kPrimExpandDims},
    {kAllTarget, OpLevel_0, prim::kPrimGeLU},
    {kAllTarget, OpLevel_0, prim::kPrimGelu},
    {kAllTarget, OpLevel_0, prim::kPrimGeLUGrad},
    {kAllTarget, OpLevel_0, prim::kPrimSqrtGrad},
    {kAllTarget, OpLevel_0, prim::kPrimSquare},
    {kAllTarget, OpLevel_0, prim::kPrimTile},
    {kAscendDevice, OpLevel_0, prim::kLambApplyOptimizerAssign},
    {kAscendDevice, OpLevel_0, prim::kLambApplyWeightAssign},
    {kAscendDevice, OpLevel_0, prim::kPrimClipByNormNoDivSum},
    {kAscendDevice, OpLevel_1, prim::kSoftmaxGradExt},
    {kAscendDevice, OpLevel_0, prim::kFusedMulAdd},
    {kGPUDevice, OpLevel_0, prim::kPrimErfc},
    {kGPUDevice, OpLevel_1, prim::kPrimAdamWeightDecay},
    {kGPUDevice, OpLevel_1, prim::kPrimBatchMatMul},
    {kGPUDevice, OpLevel_0, prim::kPrimBiasAdd},
    {kGPUDevice, OpLevel_1, prim::kPrimBiasAddGrad},
    {kGPUDevice, OpLevel_0, prim::kPrimDropout},
    {kGPUDevice, OpLevel_0, prim::kPrimDropoutGrad},
    {kGPUDevice, OpLevel_1, prim::kPrimMaximumGrad},
    {kGPUDevice, OpLevel_1, prim::kPrimMinimumGrad},
    {kGPUDevice, OpLevel_1, prim::kPrimLayerNorm},
    {kGPUDevice, OpLevel_1, prim::kPrimLayerNormGrad},
    {kGPUDevice, OpLevel_0, prim::kPrimLogSoftmax},
    {kGPUDevice, OpLevel_0, prim::kPrimLogSoftmaxGrad},
    {kGPUDevice, OpLevel_1, prim::kPrimMatMul},
    {kGPUDevice, OpLevel_1, prim::kPrimReduceMean},
    {kGPUDevice, OpLevel_1, prim::kPrimArgMaxWithValue},
    {kGPUDevice, OpLevel_1, prim::kPrimArgMinWithValue},
    {kGPUDevice, OpLevel_0, prim::kPrimReLU},
    {kGPUDevice, OpLevel_0, prim::kPrimReluGrad},
    {kGPUDevice, OpLevel_0, prim::kPrimSigmoid},
    {kGPUDevice, OpLevel_0, prim::kPrimSigmoidGrad},
    {kGPUDevice, OpLevel_0, prim::kPrimSigmoidCrossEntropyWithLogits},
    {kGPUDevice, OpLevel_0, prim::kPrimSigmoidCrossEntropyWithLogitsGrad},
    {kGPUDevice, OpLevel_0, prim::kPrimSlice},
    {kGPUDevice, OpLevel_1, prim::kPrimSoftmax},
    {kGPUDevice, OpLevel_1, prim::kPrimSoftmaxCrossEntropyWithLogits},
    {kGPUDevice, OpLevel_0, prim::kPrimSquaredDifference},
    {kGPUDevice, OpLevel_0, prim::kPrimSqueeze},
    {kGPUDevice, OpLevel_0, prim::kPrimEqualCount},
    {kGPUDevice, OpLevel_0, prim::kPrimSquareSumAll},
    {kGPUDevice, OpLevel_0, prim::kPrimIdentityMath},
    {kGPUDevice, OpLevel_0, prim::kPrimOnesLike},
    {kGPUDevice, OpLevel_0, prim::kPrimStandardNormal},
    {kCPUDevice, OpLevel_0, prim::kPrimOnesLike},
    {kCPUDevice, OpLevel_0, prim::kPrimBiasAdd},
    {kCPUDevice, OpLevel_1, prim::kPrimBiasAddGrad},
    {kCPUDevice, OpLevel_0, prim::kPrimReLU},
    {kCPUDevice, OpLevel_1, prim::kPrimMaximumGrad},
    {kCPUDevice, OpLevel_1, prim::kPrimMinimumGrad},
    {kCPUDevice, OpLevel_1, prim::kPrimAdam},
    {kCPUDevice, OpLevel_1, prim::kPrimTanhGrad},
    {kCPUDevice, OpLevel_1, prim::kPrimSoftplus},
    {kCPUDevice, OpLevel_1, prim::kPrimSoftplusGrad},
  };
  std::vector<OpWithLevel> expand_ops_with_level_dvm = {
    {kAscendDevice, OpLevel_0, prim::kPrimAddN},
    {kAscendDevice, OpLevel_0, prim::kPrimGeLU},
    {kAscendDevice, OpLevel_0, prim::kPrimGelu},
    {kAscendDevice, OpLevel_0, prim::kPrimGeLUGrad},
    {kAscendDevice, OpLevel_0, prim::kPrimRsqrtGrad},
    {kAscendDevice, OpLevel_0, prim::kPrimSqrtGrad},
    {kAscendDevice, OpLevel_0, prim::kPrimSquare},
    {kAscendDevice, OpLevel_0, prim::kPrimTile},
    {kAscendDevice, OpLevel_0, prim::kPrimClipByNormNoDivSum},
    {kAscendDevice, OpLevel_0, prim::kFusedMulAdd},
    {kAscendDevice, OpLevel_0, prim::kPrimSigmoid},
    {kAscendDevice, OpLevel_0, prim::kPrimSigmoidGrad},
    {kAscendDevice, OpLevel_0, prim::kPrimOnesLike},
    {kAscendDevice, OpLevel_0, prim::kPrimZerosLike},
    {kAscendDevice, OpLevel_0, prim::kPrimReduceMean},
    {kAscendDevice, OpLevel_0, prim::kPrimSoftmaxBackward},
    {kAscendDevice, OpLevel_0, prim::kPrimReLU},
    {kAscendDevice, OpLevel_0, prim::kPrimReluGrad},
    {kAscendDevice, OpLevel_1, prim::kPrimAssignAdd},
    {kAscendDevice, OpLevel_1, prim::kPrimExpandDims},
    {kAscendDevice, OpLevel_1, prim::kLambApplyOptimizerAssign},
    {kAscendDevice, OpLevel_1, prim::kLambApplyWeightAssign},
    {kAscendDevice, OpLevel_1, prim::kSoftmaxGradExt},
  };
  const auto &flags = GraphKernelFlags::GetInstance();
  auto ops_with_level = GraphKernelFlags::GetInstance().kernel_generator == "DVM" ? std::move(expand_ops_with_level_dvm)
                                                                                  : std::move(expand_ops_with_level);
  auto ops = GkUtils::GetValidOps(ops_with_level, flags.fusion_ops_level, flags.enable_expand_ops_only,
                                  flags.enable_expand_ops, flags.disable_expand_ops);
  return GkUtils::FilterExcludedOps(ops);
}

std::vector<PrimitivePtr> GraphKernelExpanderCloud::InitOpList() { return GraphKernelExpanderCloud::GetExpanderOps(); }

bool GraphKernelExpanderCloud::CanExpand(const CNodePtr &node) const {
  bool is_dvm = (GraphKernelFlags::GetInstance().kernel_generator == "DVM");
  if (IsComplexOp(node) && !is_dvm) {
    return true;
  }
  if (!GraphKernelExpander::CanExpand(node)) {
    return false;
  }
  if (is_dvm && !DvmSupported(node)) {
    return false;
  }
  bool enable_dynshape_expander = GraphKernelFlags::GetInstance().enable_dynamic_shape_fusion;
  if (enable_dynshape_expander) {
    if (common::AnfAlgo::IsDynamicRankNode(node)) {
      return false;
    }
  } else if (common::AnfAlgo::IsDynamicShape(node)) {
    return false;
  }
  return true;
}

ExpanderPtr GraphKernelExpanderCloud::InitExpander(const AnfNodePtr &node) {
  auto e = GetExpander(node, std::make_shared<LitegraphExpander>(Callback::Instance()));
  return e;
}
}  // namespace mindspore::graphkernel
