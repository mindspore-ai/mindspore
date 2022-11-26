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
  {prim::kPrimAcosGrad, R{ops::ACosGradInfer, nullptr, true}},
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
  {prim::kPrimInplaceIndexAdd, R{ops::IndexAddInfer, nullptr, true}},
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
  {prim::kPrimPack, R{ops::StackInfer, nullptr, true}},
  {prim::kPrimApplyAdamD, R{ops::ApplyAdamDInfer, nullptr, true}},
  {prim::kPrimSparseApplyProximalAdagradD, R{ops::SparseApplyProximalAdagradInfer, nullptr, true}},
  {prim::kPrimMul, R{ops::MulInfer, nullptr, true}},
  {prim::kPrimMod, R{ops::ModInfer, nullptr, true}},
  {prim::kPrimAdd, R{ops::AddInfer, nullptr, false}},
  {prim::kPrimArgmin, R{ops::ArgMinInfer, nullptr, true}},
  // {prim::kPrimSqrtGrad, R{InferImplSqrtGrad, nullptr, true}},
  {prim::kPrimSub, R{ops::SubInfer, nullptr, false}},
  {prim::kPrimNeg, R{ops::NegInfer, nullptr, false}},
  {prim::kPrimTile, R{ops::TileInfer, nullptr, true}},
  {prim::kPrimEqual, R{ops::EqualInfer, nullptr, true}},
  {prim::kPrimGreater, R{ops::GreaterInfer, nullptr, true}},
  {prim::kPrimGreaterEqual, R{ops::GreaterEqualInfer, nullptr, true}},
  {prim::kPrimNotEqual, R{ops::NotEqualInfer, nullptr, true}},
  {prim::kPrimLog, R{ops::LogInfer, nullptr, true}},
  {prim::kPrimReciprocal, R{ops::ReciprocalInfer, nullptr, true}},
  // {prim::kPrimBiasAddGrad, R{InferImplBiasAddGrad, nullptr, true}},
  // {prim::kPrimReduceScatter, R{InferImplReduceScatter, nullptr, true}},
  {prim::kPrimCast, R{InferImplCast, nullptr, true}},
  // {prim::kPrimExp, R{ops::ExpInfer, nullptr, true}},
  // {prim::kPrimAllReduce, R{InferImplAllReduce, nullptr, true}},
  {prim::kPrimBroadcast, R{InferImplBroadcast, nullptr, true}},
  {prim::kPrimAllGather, R{InferImplAllGather, nullptr, true}},
  // {prim::kPrimMinimum, R{InferImplMinimum, nullptr, true}},
  // {prim::kPrimDivNoNan, R{InferImplDivNoNan, nullptr, true}},
  // {prim::kPrimLinSpace, R{InferImplLinSpace, nullptr, true}},
  // {prim::kPrimLess, R{InferImplLess, nullptr, true}},
  // {prim::kPrimPad, R{InferImplPad, nullptr, true}},
  // {prim::kPrimUnsortedSegmentSum, R{InferImplUnsortedSegmentSum, nullptr, true}},
  {prim::kPrimDiv, R{InferImplDiv, nullptr, true}},
  {prim::kPrimRealDiv, R{ops::RealDivInfer, nullptr, false}},
  // {prim::kPrimTranspose, R{InferImplTranspose, nullptr, true}},
  {prim::kPrimStridedSlice, R{ops::StridedSliceInfer, nullptr, true}},
  {prim::kPrimSlice, R{ops::SliceInfer, nullptr, true}},
  {prim::kPrimSliceGrad, R{ops::SliceGradInfer, nullptr, true}},
  // {prim::kPrimConcat, R{InferImplConcat, nullptr, true}},
  {prim::kPrimConcatOffset, R{InferImplConcatOffset, nullptr, true}},
  {prim::kPrimTransData, R{InferImplTransData, nullptr, true}},
  // {prim::kPrimTensorMove, R{InferImplTensorMove, nullptr, true}},
  {prim::kPrimLstm, R{ops::LstmInfer, nullptr, true}},
  {prim::kPrimStack, R{ops::StackInfer, nullptr, true}},
  {prim::kPrimRpcRecv, R{ops::RpcRecvInfer, nullptr, true}},
  {prim::kPrimRpcSend, R{ops::RpcSendInfer, nullptr, true}},
  {prim::kPrimAdamApplyOne, R{InferImplAdamApplyOne, nullptr, true}},
  {prim::kPrimAdamApplyOneWithDecay, R{InferImplAdamApplyOneWithDecay, nullptr, true}},
  {prim::kPrimTensorScatterUpdate, R{ops::TensorScatterArithmeticInfer, nullptr, true}},
  {prim::kPrimMaxPool, R{ops::MaxPoolInfer, nullptr, true}},
  {
    prim::kPrimMaxPoolGrad,
    R{ops::MaxPoolGradInfer, nullptr, true},
  }};
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
