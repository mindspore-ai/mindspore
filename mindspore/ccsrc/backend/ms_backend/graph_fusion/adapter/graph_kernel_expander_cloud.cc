/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_expander_cloud.h"
#include <set>
#include "primitive/random_ops.h"
#include "primitive/nn_optimizer_ops.h"
#include "primitive/nn_ops.h"
#include "primitive/math_ops.h"
#include "primitive/comparison_ops.h"
#include "primitive/array_ops.h"
#include "primitive/framework_ops.h"
#include "include/utils/anfalgo.h"
#include "utils/ms_context.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_utils.h"
#include "backend/ms_backend/graph_fusion/graph_kernel_helper.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_h.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_o.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_z.h"
namespace mindspore::graphkernel {
namespace {
bool IsFloat(const TypeId &type_id) {
  switch (type_id) {
    case kNumberTypeFloat16:
    case kNumberTypeBFloat16:
    case kNumberTypeFloat32:
      return true;
    default:
      return false;
  }
}

bool IsFloatInt(const TypeId &type_id) { return IsFloat(type_id) || type_id == kNumberTypeInt32; }

bool FormatCheck(const AnfNodePtr &node) {
  if (common::AnfAlgo::IsDynamicShape(node) && !CheckDefaultFormat(node)) {
    // dvm kernel infer shape use inputs device shape, but the output abstract shape inferred from device shape is
    // not unique if some shape value are not a multiple of 16
    MS_LOG(DEBUG) << "skip node: " << node->fullname_with_scope()
                  << " because only default format is supported in dynamic shape";
    return false;
  }
  return true;
}

bool CommonCheck(const AnfNodePtr &node, const std::function<bool(const TypeId &)> &check) {
  if (!FormatCheck(node)) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto cb = Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  for (size_t i = 1; i < cnode->size(); i++) {
    auto input_abstract = cnode->input(i)->abstract();
    if (input_abstract != nullptr && input_abstract->isa<abstract::AbstractTensor>() &&
        !check(cb->GetInputType(node, i - 1))) {
      return false;
    }
  }
  auto node_output_type = cb->GetOutputType(node, 0);
  return check(node_output_type);
}

bool OutputCheck(const AnfNodePtr &node, const std::function<bool(const TypeId &)> &check) {
  if (!FormatCheck(node)) {
    return false;
  }
  auto cb = Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  auto node_output_type = cb->GetOutputType(node, 0);
  return check(node_output_type);
}

bool FloatCheck(const AnfNodePtr &node) { return CommonCheck(node, IsFloat); }
bool FloatIntCheck(const AnfNodePtr &node) { return CommonCheck(node, IsFloatInt); }
bool OutputFloatCheck(const AnfNodePtr &node) { return OutputCheck(node, IsFloat); }
bool OutputFloatIntCheck(const AnfNodePtr &node) { return OutputCheck(node, IsFloatInt); }

bool OptimizerCheck(const AnfNodePtr &node) {
  if (!FormatCheck(node)) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(node);
  auto cb = Callback::Instance();
  auto node_output_type = cb->GetOutputType(node, 0);
  if (!IsFloat(node_output_type)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  for (size_t i = 1; i < cnode->size(); i++) {
    auto input_abstract = cnode->input(i)->abstract();
    if (input_abstract != nullptr && input_abstract->isa<abstract::AbstractTensor>() &&
        cb->GetInputType(node, i - 1) != node_output_type) {
      return false;
    }
  }
  return true;
}

bool AddNCheck(const AnfNodePtr &node) {
  constexpr auto max_input_num = 10;
  auto input_num = common::AnfAlgo::GetInputTensorNum(node);
  if (input_num > max_input_num) {
    return false;
  }
  return CommonCheck(node, IsFloatInt);
}

const std::vector<OpWithLevel> expand_ops_with_level = {
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

const std::vector<OpWithLevel> expand_ops_with_level_v2 = {
  // CPU
  {kCPUDevice, OpLevel_0, prim::kPrimIdentityMath},
  {kCPUDevice, OpLevel_0, prim::kPrimSqueeze},
  {kCPUDevice, OpLevel_0, prim::kPrimSlice},

  // GPU
  {kGPUDevice, OpLevel_0, prim::kPrimBiasAdd},
  {kGPUDevice, OpLevel_0, prim::kPrimDropout},
  {kGPUDevice, OpLevel_0, prim::kPrimDropoutGrad},
  {kGPUDevice, OpLevel_0, prim::kPrimLayerNorm},
  {kGPUDevice, OpLevel_0, prim::kPrimLayerNormGrad},
  {kGPUDevice, OpLevel_0, prim::kPrimRelu},
  {kGPUDevice, OpLevel_0, prim::kPrimReluGrad},
  {kGPUDevice, OpLevel_0, prim::kPrimClipByNorm},
};

// note: inplace op can not be fused by default, as view + inplace case may have precision error
const std::vector<OpWithLevel> expand_ops_with_level_dvm = {
  {kAscendDevice, OpLevel_0, prim::kPrimAdam, OptimizerCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimAdamApplyOneWithDecayAssign, OptimizerCheck},
  {kAscendDevice, OpLevel_0, prim::kLambApplyOptimizerAssign, OptimizerCheck},
  {kAscendDevice, OpLevel_0, prim::kLambApplyWeightAssign, OptimizerCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimAdamApplyOneWithDecay, OptimizerCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimAddcmul, FloatIntCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimAddN, AddNCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimBiasAdd, FloatIntCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimBiasAddGrad, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimFillV2},
  {kAscendDevice, OpLevel_0, prim::kPrimGeLU, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimFastGelu, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimFastGeluGrad, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimFastGeLU, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimFastGeLUGrad, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimSiLU, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimSiLUGrad, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimGeLUGrad, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimMuls, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimRsqrtGrad, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimSqrtGrad, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimSquare, FloatIntCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimTile, OutputFloatIntCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimClipByNormNoDivSum, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimSigmoid, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimSigmoidGrad, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimSigmoidCrossEntropyWithLogits, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimSigmoidCrossEntropyWithLogitsGrad, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimSquaredDifference, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimTanhGrad, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimOnesLike, FloatIntCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimZerosLike, FloatIntCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimReduceMean, OutputFloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimReLU, FloatIntCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimReluGrad, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimAssignAdd, FloatIntCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimRepeatInterleaveInt, FloatIntCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimHShrink, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimHSigmoid, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimHSwish, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimErf, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimTanh, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimCosh, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimSinh, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimDivMod, FloatCheck},
  // mint ops
  {kAscendDevice, OpLevel_0, prim::kPrimAddExt, FloatIntCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimSubExt, FloatIntCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimSumExt, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimMeanExt, OutputFloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimAcoshExt, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimAsinhExt, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimEluExt, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimSoftplusExt, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimLeakyReLUExt, FloatCheck},
  {kAscendDevice, OpLevel_0, prim::kPrimZerosLikeExt, FloatIntCheck},
};

bool DvmSupported(const AnfNodePtr &node) {
  auto iter = std::find_if(expand_ops_with_level_dvm.begin(), expand_ops_with_level_dvm.end(),
                           [&node](const OpWithLevel &item) { return IsPrimitiveCNode(node, item.prim); });
  if (iter != expand_ops_with_level_dvm.end() && iter->check_func != nullptr) {
    return iter->check_func(node);
  }
  return true;
}
}  // namespace

std::vector<PrimitivePtr> GraphKernelExpanderCloud::GetExpanderOps() {
  const auto &flags = GraphKernelFlags::GetInstance();
  std::vector<std::string> disable_expand_ops = flags.disable_expand_ops;
  auto cb = Callback::Instance();

  std::vector<OpWithLevel> expand_ops;
  std::vector<std::string> disable_expand_op_list_v2 = {
    "OnesLike", "OneHot", "StridedSlice", "CumSum", "Transpose", "BatchMatMul", "MatMul", "ExpandDims", "BroadcastTo"};
  if (flags.kernel_generator == "AKG_V2") {
    expand_ops = expand_ops_with_level;
    expand_ops.insert(expand_ops.end(), expand_ops_with_level_v2.begin(), expand_ops_with_level_v2.end());
    if (cb->GetTargetFromContext() == kGPUDevice) {
      for (const std::string &item : disable_expand_op_list_v2) {
        if (std::find(flags.enable_expand_ops.begin(), flags.enable_expand_ops.end(), item) ==
            flags.enable_expand_ops.end()) {
          disable_expand_ops.push_back(item);
        }
      }
    }
  } else if (flags.kernel_generator == "DVM") {
    expand_ops = expand_ops_with_level_dvm;
  } else {
    expand_ops = expand_ops_with_level;
  }
  auto ops = GkUtils::GetValidOps(expand_ops, flags.fusion_ops_level, flags.enable_expand_ops_only,
                                  flags.enable_expand_ops, disable_expand_ops);
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
  if (GkUtils::InplaceWithViewInputs(node)) {
    return false;
  }
  if (GkUtils::IsShapeZero(node)) {
    return false;
  }
  if (is_dvm) {
    GkUtils::CheckOpLevel(node, expand_ops_with_level_dvm, OpLevel_1);
  }
  if (!common::AnfAlgo::IsDynamicShape(node)) {
    // for static cases, the node can be expanded if this is complex op
    // or in the list
    return true;
  }

  // deal wich dynamic cases
  // the node with dyn rank will not be expand
  if (common::AnfAlgo::IsDynamicRankNode(node)) {
    return false;
  }

  auto enable_dynamic_shape = GraphKernelFlags::GetInstance().enable_dynamic_shape_fusion;
  if (is_dvm) {
    return enable_dynamic_shape;
  }

  std::vector<PrimitivePtr> expand_ops_dyn = {prim::kPrimReLU, prim::kPrimReluGrad, prim::kPrimBiasAdd,
                                              prim::kPrimBiasAddGrad, prim::kPrimDropout};

  bool dyn_can_expand_op = std::any_of(expand_ops_dyn.begin(), expand_ops_dyn.end(),
                                       [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
  // the dyn shape node can be expanded
  return (enable_dynamic_shape && dyn_can_expand_op);
}

ExpanderPtr GraphKernelExpanderCloud::InitExpander(const AnfNodePtr &node) {
  auto e = GetExpander(node, std::make_shared<LitegraphExpander>(Callback::Instance()));
  return e;
}
}  // namespace mindspore::graphkernel
