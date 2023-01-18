/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "backend/operator/ops_backend_infer_function.h"
#include "abstract/abstract_function.h"
#include "abstract/ops/infer_functions.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/core_ops.h"
#include "utils/ms_context.h"
#include "ops/export_infer.h"
namespace mindspore {
namespace abstract {
using R = PrimitiveEvalImplMap::mapped_type;
static PrimitiveEvalImplMap prim_backend_eval_implement_map{
  // backend infer functions.
  // Do not add anything in this initializer anymore since it will be removed soon, backend will register their infer
  // function in backend plugin.
  {prim::kPrimApplyAdadeltaD, R{ops::ApplyAdadeltaInfer, nullptr, true}},
  {prim::kPrimApplyAdagradD, R{ops::ApplyAdagradInferFunc, nullptr, true}},
  {prim::kPrimApplyAdagradV2D, R{ops::ApplyAdagradV2Infer, nullptr, true}},
  {prim::kPrimApplyAdaMaxD, R{ops::ApplyAdaMaxInfer, nullptr, true}},
  {prim::kPrimApplyAddSignD, R{ops::ApplyAddSignInfer, nullptr, true}},
  {prim::kPrimApplyPowerSignD, R{ops::ApplyPowerSignInfer, nullptr, true}},
  {prim::kPrimFastGelu, R{ops::FastGeLUInfer, nullptr, true}},
  {prim::kPrimFastGeluGrad, R{ops::FastGeLUGradInfer, nullptr, true}},
  {prim::kPrimGelu, R{ops::GeLUInfer, nullptr, true}},
  {prim::kPrimHardSwish, R{ops::HSwishInfer, nullptr, true}},
  {prim::kPrimLarsV2Update, R{ops::LARSUpdateInfer, nullptr, true}},
  {prim::kPrimLogSoftmaxV2, R{ops::LogSoftmaxInfer, nullptr, true}},
  {prim::kPrimRelu6Grad, R{ops::ReLU6GradInferFunc, nullptr, true}},
  {prim::kPrimSelu, R{ops::SeLUInfer, nullptr, true}},
  {prim::kPrimGeluGrad, R{ops::GeLUGradInfer, nullptr, true}},
  {prim::kPrimIou, R{ops::IouInferFunc, nullptr, true}},
  {prim::kPrimSplitD, R{ops::SplitInfer, nullptr, true}},
  {prim::kPrimArgMin, R{ops::ArgminV2Infer, nullptr, true}},
  {prim::kPrimCeluV2, R{ops::CeLUInfer, nullptr, true}},
  {prim::kPrimCumsum, R{ops::CumSumInfer, nullptr, true}},
  {prim::kPrimDropOutDoMask, R{ops::DropoutDoMaskInfer, nullptr, true}},
  {prim::kPrimGatherV2, R{ops::GatherInfer, nullptr, true}},
  {prim::kPrimHardSwishGrad, R{ops::HSwishGradInfer, nullptr, true}},
  {prim::kPrimPRelu, R{ops::PReLUInfer, nullptr, true}},
  {prim::kPrimReduceSumD, R{ops::ReduceArithmeticInferFunc, nullptr, true}},
  {prim::kPrimReduceMeanD, R{ops::ReduceArithmeticInferFunc, nullptr, true}},
  {prim::kPrimRelu, R{ops::ReLUInferFunc, nullptr, true}},
  {prim::kPrimResizeBilinearV2Grad, R{ops::ResizeBilinearGradInfer, nullptr, true}},
  {prim::kPrimSigmoidCrossEntropyWithLogitsV2, R{ops::BCEWithLogitsLossInfer, nullptr, true}},
  {prim::kPrimSplitVD, R{ops::SplitVInfer, nullptr, true}},
  {prim::kPrimSparseApplyFtrlD, R{ops::SparseApplyFtrlInfer, nullptr, true}},
  {prim::kPrimSoftmaxV2, R{ops::SoftmaxInfer, nullptr, true}},
  {prim::kPrimPadD, R{InferImplPad, nullptr, true}},
  {prim::kPrimConcatD, R{InferImplConcat, nullptr, true}},
  {prim::kPrimApplyAdamD, R{ops::ApplyAdamInferFunc, nullptr, true}},
  {prim::kPrimSparseApplyProximalAdagradD, R{ops::SparseApplyProximalAdagradInfer, nullptr, true}},
  {prim::kPrimCast, R{InferImplCast, nullptr, true}},                  // remove when Cast core/ops infer ready
  {prim::kPrimBroadcast, R{InferImplBroadcast, nullptr, true}},        // remove when Broadcast core/ops infer ready
  {prim::kPrimAllGather, R{InferImplAllGather, nullptr, true}},        // remove when AllGather core/ops infer ready
  {prim::kPrimConcatOffset, R{InferImplConcatOffset, nullptr, true}},  // remove when ConcatOffset core/ops infer ready
  {prim::kPrimTransData, R{InferImplTransData, nullptr, true}},
  {prim::kPrimAdamApplyOne, R{InferImplAdamApplyOne, nullptr, true}},
  {prim::kPrimAdamApplyOneWithDecay, R{InferImplAdamApplyOneWithDecay, nullptr, true}},
};
PrimitiveEvalImplMap *GetBackendPrimitiveInferMapPtr() { return &prim_backend_eval_implement_map; }
const PrimitiveEvalImplMap &GetBackendPrimitiveInferMap() { return prim_backend_eval_implement_map; }

std::optional<StandardPrimitiveImplReg> GetBackendPrimitiveInferImpl(const PrimitivePtr &primitive) {
  auto found = abstract::GetPrimitiveInferImpl(primitive);
  if (found.has_value()) {
    return found.value();
  }
  auto iter = GetBackendPrimitiveInferMap().find(primitive);
  if (iter != GetBackendPrimitiveInferMap().end()) {
    return iter->second;
  }
  return std::optional<StandardPrimitiveImplReg>();
}
}  // namespace abstract
}  // namespace mindspore
