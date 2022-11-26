/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "expander/infer.h"

#include <algorithm>
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/base_operator.h"
#include "abstract/ops/infer_functions.h"
#include "ops/export_infer.h"

namespace mindspore {
namespace expander {
using R = abstract::PrimitiveEvalImplMap::mapped_type;
// this map will be removed soon since it's just a quick fix, some core/ops implemented the infer already but
// did not register them into core/ops infer map.
static abstract::PrimitiveEvalImplMap unreg_infer_map = {
  {prim::kPrimAcosGrad, R{ops::ACosGradInfer, nullptr, true}},
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
  {prim::kPrimArgMin, R{ops::ArgminV2Infer, nullptr, true}},
  {prim::kPrimCeluV2, R{ops::CeLUInfer, nullptr, true}},
  {prim::kPrimCumsum, R{ops::CumSumInfer, nullptr, true}},
  {prim::kPrimDropOutDoMask, R{ops::DropoutDoMaskInfer, nullptr, true}},
  {prim::kPrimGatherV2, R{ops::GatherInfer, nullptr, true}},
  {prim::kPrimHardSwishGrad, R{ops::HSwishGradInfer, nullptr, true}},
  {prim::kPrimInplaceIndexAdd, R{ops::IndexAddInfer, nullptr, true}},
  {prim::kPrimPRelu, R{ops::PReLUInfer, nullptr, true}},
  {prim::kPrimRelu, R{ops::ReLUInferFunc, nullptr, true}},
  {prim::kPrimResizeBilinearV2Grad, R{ops::ResizeBilinearGradInfer, nullptr, true}},
  {prim::kPrimSigmoidCrossEntropyWithLogitsV2, R{ops::BCEWithLogitsLossInfer, nullptr, true}},
  {prim::kPrimSoftmaxV2, R{ops::SoftmaxInfer, nullptr, true}},
  {prim::kPrimPack, R{ops::StackInfer, nullptr, true}},
  {prim::kPrimMul, R{ops::MulInfer, nullptr, true}},
  {prim::kPrimMod, R{ops::ModInfer, nullptr, true}},
  {prim::kPrimAdd, R{ops::AddInfer, nullptr, false}},
  {prim::kPrimArgmin, R{ops::ArgMinInfer, nullptr, true}},
  {prim::kPrimSub, R{ops::SubInfer, nullptr, false}},
  {prim::kPrimNeg, R{ops::NegInfer, nullptr, false}},
  {prim::kPrimTile, R{ops::TileInfer, nullptr, true}},
  {prim::kPrimEqual, R{ops::EqualInfer, nullptr, true}},
  {prim::kPrimGreater, R{ops::GreaterInfer, nullptr, true}},
  {prim::kPrimGreaterEqual, R{ops::GreaterEqualInfer, nullptr, true}},
  {prim::kPrimNotEqual, R{ops::NotEqualInfer, nullptr, true}},
  {prim::kPrimLog, R{ops::LogInfer, nullptr, true}},
  {prim::kPrimReciprocal, R{ops::ReciprocalInfer, nullptr, true}},
  {prim::kPrimCast, R{abstract::InferImplCast, nullptr, true}},
  {prim::kPrimBroadcast, R{abstract::InferImplBroadcast, nullptr, true}},
  {prim::kPrimAllGather, R{abstract::InferImplAllGather, nullptr, true}},
  {prim::kPrimDiv, R{abstract::InferImplDiv, nullptr, true}},
  {prim::kPrimRealDiv, R{ops::RealDivInfer, nullptr, false}},
  {prim::kPrimStridedSlice, R{ops::StridedSliceInfer, nullptr, true}},
  {prim::kPrimSlice, R{ops::SliceInfer, nullptr, true}},
  {prim::kPrimSliceGrad, R{ops::SliceGradInfer, nullptr, true}},
  {prim::kPrimConcatOffset, R{abstract::InferImplConcatOffset, nullptr, true}},
  {prim::kPrimTransData, R{abstract::InferImplTransData, nullptr, true}},
  {prim::kPrimLstm, R{ops::LstmInfer, nullptr, true}},
  {prim::kPrimStack, R{ops::StackInfer, nullptr, true}},
  {prim::kPrimRpcRecv, R{ops::RpcRecvInfer, nullptr, true}},
  {prim::kPrimRpcSend, R{ops::RpcSendInfer, nullptr, true}},
  {prim::kPrimAdamApplyOne, R{abstract::InferImplAdamApplyOne, nullptr, true}},
  {prim::kPrimAdamApplyOneWithDecay, R{abstract::InferImplAdamApplyOneWithDecay, nullptr, true}},
  {prim::kPrimTensorScatterUpdate, R{ops::TensorScatterArithmeticInfer, nullptr, true}},
  {prim::kPrimMaxPool, R{ops::MaxPoolInfer, nullptr, true}},
  {prim::kPrimMaxPoolGrad, R{ops::MaxPoolGradInfer, nullptr, true}}};

void CppInfer::Infer(const NodePtr &node) {
  auto anfnode = node->get();
  if (anfnode->isa<ValueNode>()) {
    anfnode->set_abstract(anfnode->cast<ValueNodePtr>()->value()->ToAbstract());
    return;
  }
  auto cnode = anfnode->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  AbstractBasePtrList abs_list;
  abs_list.reserve(cnode->size());
  (void)std::transform(cnode->inputs().cbegin() + 1, cnode->inputs().cend(), std::back_inserter(abs_list),
                       [](const AnfNodePtr &node) {
                         const auto &abs = node->abstract();
                         if (abs == nullptr) {
                           MS_EXCEPTION_IF_CHECK_FAIL(node->isa<ValueNode>(), node->ToString() + " has no abstract");
                           return node->cast<ValueNodePtr>()->value()->ToAbstract();
                         }
                         return abs;
                       });
  AbstractBasePtr result = nullptr;
  auto found = abstract::GetPrimitiveInferImpl(prim);
  if (found.has_value()) {
    auto infer = found.value();
    MS_EXCEPTION_IF_CHECK_FAIL(infer.IsImplInferShapeAndType(), "There is no infer-abstract implement!");
    result = infer.InferShapeAndType(nullptr, prim, abs_list);
  } else {
    auto iter = unreg_infer_map.find(prim);
    if (iter != unreg_infer_map.end()) {
      auto infer = iter->second;
      MS_EXCEPTION_IF_CHECK_FAIL(infer.IsImplInferShapeAndType(), "There is no infer-abstract implement!");
      result = infer.InferShapeAndType(nullptr, prim, abs_list);
    } else {
      MS_LOG(EXCEPTION) << "The infer function of [" << prim->name() << "] is not defined.";
    }
  }
  cnode->set_abstract(result);
}

BaseShapePtr CppInfer::GetShape(const NodePtr &node) {
  auto abs = node->get()->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  return abs->BuildShape();
}

TypePtr CppInfer::GetDtype(const NodePtr &node) {
  auto abs = node->get()->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  return abs->BuildType();
}
}  // namespace expander
}  // namespace mindspore
