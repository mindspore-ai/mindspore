/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <memory>
#include "src/ops/ops_utils.h"

#ifdef PRIMITIVE_WRITEABLE
#include "mindspore/core/ir/anf.h"

namespace mindspore {
namespace lite {
schema::PrimitiveT *GetPrimitiveT(const AnfNodePtr &node) {
  auto prim = GetValueNode<std::shared_ptr<Primitive>>(node);
  if (prim == nullptr) {
    MS_LOG(DEBUG) << "primitive is nullptr";
    return nullptr;
  }

  if (prim->name().empty()) {
    MS_LOG(ERROR) << "the name of primitive is null";
    return nullptr;
  }

  MS_LOG(INFO) << "export prim: " << prim->name();
  auto creator = MSOpsRegistry::GetInstance()->GetPrimitiveCreator(prim->name());
  if (creator != nullptr) {
    return creator(node);
  } else {
    MS_LOG(ERROR) << "can not find MSOpsRegistry for op: " << prim->name();
    return nullptr;
  }
}

schema::PrimitiveT *AbsPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Abs>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ActivationPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Activation>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ActivationGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::ActivationGrad>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *AdderFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::AdderFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *AddFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::AddFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *AddGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::AddGrad>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *AddNPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::AddN>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *AllPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::All>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ApplyMomentumPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::ApplyMomentum>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ArgMaxFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::ArgMaxFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ArgMinFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::ArgMinFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *AssertPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Assert>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *AssignPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Assign>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *AssignAddPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::AssignAdd>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *AvgPoolFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::AvgPoolFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *BatchNormPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::BatchNorm>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *BatchToSpacePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::BatchToSpace>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *BatchToSpaceNDPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::BatchToSpaceND>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *BiasAddPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::BiasAdd>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *BNGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::BatchNormGrad>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *BroadcastToPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::BroadcastTo>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *CastPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Cast>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *CeilPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Ceil>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ClipPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Clip>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ConcatPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Concat>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

schema::PrimitiveT *ConstantOfShapePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::ConstantOfShape>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

schema::PrimitiveT *Conv2DBackpropFilterFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Conv2DBackpropFilterFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *Conv2DBackpropInputFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Conv2DBackpropInputFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *Conv2DFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Conv2DFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *Conv2dTransposeFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Conv2dTransposeFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *CosPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Cos>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *CropPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Crop>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *CustomExtractFeaturesPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::CustomExtractFeatures>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *CustomNormalizePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::CustomNormalize>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *CustomPredictPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::CustomPredict>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *DependPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Depend>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *DepthToSpacePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::DepthToSpace>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *DetectionPostProcessPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::DetectionPostProcess>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *DivFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::DivFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *DivGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::DivGrad>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *DropoutPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Dropout>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *DropoutGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::DropoutGrad>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *EltwisePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Eltwise>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *EluPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Elu>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *EmbeddingLookupFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::EmbeddingLookupFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *EqualPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Equal>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ExpandDimsPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::ExpandDims>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ExpFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::ExpFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *FftImagPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::FftImag>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *FftRealPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::FftReal>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *FillPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Fill>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *FlattenPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Flatten>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *FlattenGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::FlattenGrad>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *FloorPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Floor>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *FloorDivPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::FloorDiv>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *FloorModPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::FloorMod>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *FullConnectionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::FullConnection>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *FusedBatchNormPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::FusedBatchNorm>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *GatherPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Gather>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *GatherNdPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::GatherNd>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *GreaterPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Greater>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *GreaterEqualPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::GreaterEqual>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *HashtableLookupPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::HashtableLookup>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *InstanceNormPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::InstanceNorm>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *LayerNormFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::LayerNormFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *LeakyReluPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::LeakyRelu>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *LessPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Less>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *LessEqualPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::LessEqual>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *LogPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Log>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *LogGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::LogGrad>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *LogicalAndPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::LogicalAnd>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *LogicalNotPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::LogicalNot>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *LogicalOrPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::LogicalOr>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *LrnPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Lrn>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *LpNormalizationPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::LpNormalization>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *LshProjectionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::LshProjection>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *LSTMPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::LSTM>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *L2NormalizeFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::L2NormalizeFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *MatMulPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::MatMul>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *MaximumPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Maximum>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *MaximumGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::MaximumGrad>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *MaxPoolFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::MaxPoolFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *MergePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Merge>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *MinimumPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Minimum>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *MinimumGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::MinimumGrad>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ModPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Mod>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *MulFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::MulFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *MulGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::MulGrad>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *NegPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Neg>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *NegGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::NegGrad>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *NotEqualPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::NotEqual>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *NonMaxSuppressionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::NonMaxSuppression>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *OneHotPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::OneHot>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *OnesLikePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::OnesLike>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *PadFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::PadFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *PartialFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::PartialFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *PowerGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::PowerGrad>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *PowFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::PowFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *PReLUFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::PReLUFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *QuantDTypeCastPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::QuantDTypeCast>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *RangePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Range>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *RankPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Rank>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *RealDivPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::RealDiv>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ReciprocalPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Reciprocal>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ReduceFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::ReduceFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ReshapePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Reshape>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ResizePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Resize>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ReverseV2PrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::ReverseV2>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ReverseSequencePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::ReverseSequence>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *RfftPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Rfft>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ROIPoolingPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::ROIPooling>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *RoundPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Round>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *RsqrtPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Rsqrt>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ScaleFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::ScaleFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ShapePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Shape>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SigmoidCrossEntropyWithLogitsPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::SigmoidCrossEntropyWithLogits>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SigmoidCrossEntropyWithLogitsGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::SigmoidCrossEntropyWithLogitsGrad>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SinPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Sin>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SkipGramPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::SkipGram>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SliceFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::SliceFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SmoothL1LossPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::SmoothL1Loss>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SmoothL1LossGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::SmoothL1LossGrad>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SoftmaxPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Softmax>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SpaceToBatchPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::SpaceToBatch>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SpaceToBatchNDPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::SpaceToBatchND>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SpaceToDepthPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::SpaceToDepth>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SparseToDensePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::SparseToDense>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SplitPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Split>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SqrtPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Sqrt>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SquarePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Square>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SquaredDifferencePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::SquaredDifference>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SqueezePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Squeeze>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *StackPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Stack>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *StridedSlicePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::StridedSlice>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SubFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::SubFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SubGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::SubGrad>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *SwitchPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Switch>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *TensorListFromTensorPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::TensorListFromTensor>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *TensorListGetItemPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::TensorListGetItem>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *TensorListReservePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::TensorListReserve>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *TensorListSetItemPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::TensorListSetItem>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *TensorListStackPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::TensorListStack>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *TileFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::TileFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *TopKFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::TopKFusion>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *TransposePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Transpose>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *UniquePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Unique>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *UnpackPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Unpack>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *UnsortedSegmentSumPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::UnsortedSegmentSum>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *UnsqueezePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Unsqueeze>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *WherePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::Where>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
schema::PrimitiveT *ZerosLikePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetValueNode<std::shared_ptr<mindspore::ops::ZerosLike>>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

RegistryMSOps g_AbsPrimitiveCreatorRegistry("Abs", AbsPrimitiveCreator);
RegistryMSOps g_ActivationPrimitiveCreatorRegistry("Activation", ActivationPrimitiveCreator);
RegistryMSOps g_ActivationGradPrimitiveCreatorRegistry("ActivationGrad", ActivationGradPrimitiveCreator);
RegistryMSOps g_AddPrimitiveCreatorRegistry("Add", AddFusionPrimitiveCreator);
RegistryMSOps g_AddFusionPrimitiveCreatorRegistry("AddFusion", AddFusionPrimitiveCreator);
RegistryMSOps g_AddGradPrimitiveCreatorRegistry("AddGrad", AddGradPrimitiveCreator);
RegistryMSOps g_AdderPrimitiveCreatorRegistry("Adder", AdderFusionPrimitiveCreator);
RegistryMSOps g_AdderFusionPrimitiveCreatorRegistry("AdderFusion", AdderFusionPrimitiveCreator);
RegistryMSOps g_AddNPrimitiveCreatorRegistry("AddN", AddNPrimitiveCreator);
RegistryMSOps g_AllPrimitiveCreatorRegistry("All", AllPrimitiveCreator);
RegistryMSOps g_ApplyMomentumPrimitiveCreatorRegistry("ApplyMomentum", ApplyMomentumPrimitiveCreator);
RegistryMSOps g_ArgMaxPrimitiveCreatorRegistry("ArgMax", ArgMaxFusionPrimitiveCreator);
RegistryMSOps g_ArgMaxFusionPrimitiveCreatorRegistry("ArgMaxFusion", ArgMaxFusionPrimitiveCreator);
RegistryMSOps g_ArgMinPrimitiveCreatorRegistry("ArgMin", ArgMinFusionPrimitiveCreator);
RegistryMSOps g_ArgMinFusionPrimitiveCreatorRegistry("ArgMinFusion", ArgMinFusionPrimitiveCreator);
RegistryMSOps g_AssertPrimitiveCreatorRegistry("Assert", AssertPrimitiveCreator);
RegistryMSOps g_AssignPrimitiveCreatorRegistry("Assign", AssignPrimitiveCreator);
RegistryMSOps g_AssignAddPrimitiveCreatorRegistry("AssignAdd", AssignAddPrimitiveCreator);
RegistryMSOps g_AvgPoolPrimitiveCreatorRegistry("AvgPool", AvgPoolFusionPrimitiveCreator);
RegistryMSOps g_AvgPoolFusionPrimitiveCreatorRegistry("AvgPoolFusion", AvgPoolFusionPrimitiveCreator);
RegistryMSOps g_BatchNormPrimitiveCreatorRegistry("BatchNorm", BatchNormPrimitiveCreator);
RegistryMSOps g_BatchToSpacePrimitiveCreatorRegistry("BatchToSpace", BatchToSpacePrimitiveCreator);
RegistryMSOps g_BatchToSpaceNDPrimitiveCreatorRegistry("BatchToSpaceND", BatchToSpaceNDPrimitiveCreator);
RegistryMSOps g_BiasAddPrimitiveCreatorRegistry("BiasAdd", BiasAddPrimitiveCreator);
RegistryMSOps g_BNGradPrimitiveCreatorRegistry("BNGrad", BNGradPrimitiveCreator);
RegistryMSOps g_BroadcastToPrimitiveCreatorRegistry("BroadcastTo", BroadcastToPrimitiveCreator);
RegistryMSOps g_CastPrimitiveCreatorRegistry("Cast", CastPrimitiveCreator);
RegistryMSOps g_CeilPrimitiveCreatorRegistry("Ceil", CeilPrimitiveCreator);
RegistryMSOps g_ClipPrimitiveCreatorRegistry("Clip", ClipPrimitiveCreator);
RegistryMSOps g_ConcatPrimitiveCreatorRegistry("Concat", ConcatPrimitiveCreator);
// RegistryMSOps g_ControlDependPrimitiveCreatorRegistry("ControlDepend", ControlDependPrimitiveCreator);
RegistryMSOps g_Conv2DBackpropFilterFusionPrimitiveCreatorRegistry("Conv2DBackpropFilterFusion",
                                                                   Conv2DBackpropFilterFusionPrimitiveCreator);
RegistryMSOps g_Conv2DBackpropInputFusionPrimitiveCreatorRegistry("Conv2DBackpropInputFusion",
                                                                  Conv2DBackpropInputFusionPrimitiveCreator);
RegistryMSOps g_Conv2DPrimitiveCreatorRegistry("Conv2D", Conv2DFusionPrimitiveCreator);
RegistryMSOps g_Conv2DFusionPrimitiveCreatorRegistry("Conv2DFusion", Conv2DFusionPrimitiveCreator);
RegistryMSOps g_Conv2dTransposePrimitiveCreatorRegistry("Conv2dTranspose", Conv2dTransposeFusionPrimitiveCreator);
RegistryMSOps g_Conv2dTransposeFusionPrimitiveCreatorRegistry("Conv2dTransposeFusion",
                                                              Conv2dTransposeFusionPrimitiveCreator);
RegistryMSOps g_ConstantOfShapePrimitiveCreatorRegistry("ConstantOfShape", ConstantOfShapePrimitiveCreator);
RegistryMSOps g_CosPrimitiveCreatorRegistry("Cos", CosPrimitiveCreator);
RegistryMSOps g_CropPrimitiveCreatorRegistry("Crop", CropPrimitiveCreator);
RegistryMSOps g_CustomExtractFeaturesPrimitiveCreatorRegistry("CustomExtractFeatures",
                                                              CustomExtractFeaturesPrimitiveCreator);
RegistryMSOps g_CustomNormalizePrimitiveCreatorRegistry("CustomNormalize", CustomNormalizePrimitiveCreator);
RegistryMSOps g_CustomPredictPrimitiveCreatorRegistry("CustomPredict", CustomPredictPrimitiveCreator);
RegistryMSOps g_DependPrimitiveCreatorRegistry("Depend", DependPrimitiveCreator);
RegistryMSOps g_DepthToSpacePrimitiveCreatorRegistry("DepthToSpace", DepthToSpacePrimitiveCreator);
RegistryMSOps g_DetectionPostProcessPrimitiveCreatorRegistry("DetectionPostProcess",
                                                             DetectionPostProcessPrimitiveCreator);
RegistryMSOps g_DivPrimitiveCreatorRegistry("Div", DivFusionPrimitiveCreator);
RegistryMSOps g_DivFusionPrimitiveCreatorRegistry("DivFusion", DivFusionPrimitiveCreator);
RegistryMSOps g_DivGradPrimitiveCreatorRegistry("DivGrad", DivGradPrimitiveCreator);
RegistryMSOps g_DropoutPrimitiveCreatorRegistry("Dropout", DropoutPrimitiveCreator);
RegistryMSOps g_DropoutGradPrimitiveCreatorRegistry("DropoutGrad", DropoutGradPrimitiveCreator);
RegistryMSOps g_EltwisePrimitiveCreatorRegistry("Eltwise", EltwisePrimitiveCreator);
RegistryMSOps g_EluPrimitiveCreatorRegistry("Elu", EluPrimitiveCreator);
RegistryMSOps g_EqualPrimitiveCreatorRegistry("Equal", EqualPrimitiveCreator);
RegistryMSOps g_EmbeddingLookupFusionPrimitiveCreatorRegistry("EmbeddingLookupFusion",
                                                              EmbeddingLookupFusionPrimitiveCreator);
RegistryMSOps g_ExpandDimsPrimitiveCreatorRegistry("ExpandDims", ExpandDimsPrimitiveCreator);
RegistryMSOps g_ExpPrimitiveCreatorRegistry("Exp", ExpFusionPrimitiveCreator);
RegistryMSOps g_ExpFusionPrimitiveCreatorRegistry("ExpFusion", ExpFusionPrimitiveCreator);
RegistryMSOps g_FftImagPrimitiveCreatorRegistry("FftImag", FftImagPrimitiveCreator);
RegistryMSOps g_FftRealPrimitiveCreatorRegistry("FftReal", FftRealPrimitiveCreator);
RegistryMSOps g_FillPrimitiveCreatorRegistry("Fill", FillPrimitiveCreator);
RegistryMSOps g_FlattenPrimitiveCreatorRegistry("Flatten", FlattenPrimitiveCreator);
RegistryMSOps g_FlattenGradPrimitiveCreatorRegistry("FlattenGrad", FlattenGradPrimitiveCreator);
RegistryMSOps g_FloorPrimitiveCreatorRegistry("Floor", FloorPrimitiveCreator);
RegistryMSOps g_FloorDivPrimitiveCreatorRegistry("FloorDiv", FloorDivPrimitiveCreator);
RegistryMSOps g_FloorModPrimitiveCreatorRegistry("FloorMod", FloorModPrimitiveCreator);
RegistryMSOps g_FullConnectionPrimitiveCreatorRegistry("FullConnection", FullConnectionPrimitiveCreator);
RegistryMSOps g_FusedBatchNormPrimitiveCreatorRegistry("FusedBatchNorm", FusedBatchNormPrimitiveCreator);
RegistryMSOps g_GatherPrimitiveCreatorRegistry("Gather", GatherPrimitiveCreator);
RegistryMSOps g_GatherNdPrimitiveCreatorRegistry("GatherNd", GatherNdPrimitiveCreator);
RegistryMSOps g_GreaterPrimitiveCreatorRegistry("Greater", GreaterPrimitiveCreator);
RegistryMSOps g_GreaterEqualPrimitiveCreatorRegistry("GreaterEqual", GreaterEqualPrimitiveCreator);
RegistryMSOps g_HashtableLookupPrimitiveCreatorRegistry("HashtableLookup", HashtableLookupPrimitiveCreator);
RegistryMSOps g_InstanceNormPrimitiveCreatorRegistry("InstanceNorm", InstanceNormPrimitiveCreator);
RegistryMSOps g_LayerNormPrimitiveCreatorRegistry("LayerNorm", LayerNormFusionPrimitiveCreator);
RegistryMSOps g_LayerNormFusionPrimitiveCreatorRegistry("LayerNormFusion", LayerNormFusionPrimitiveCreator);
RegistryMSOps g_LeakyReluPrimitiveCreatorRegistry("LeakyRelu", LeakyReluPrimitiveCreator);
RegistryMSOps g_LessPrimitiveCreatorRegistry("Less", LessPrimitiveCreator);
RegistryMSOps g_LessEqualPrimitiveCreatorRegistry("LessEqual", LessEqualPrimitiveCreator);
RegistryMSOps g_LogPrimitiveCreatorRegistry("Log", LogPrimitiveCreator);
RegistryMSOps g_LogGradPrimitiveCreatorRegistry("LogGrad", LogGradPrimitiveCreator);
RegistryMSOps g_LogicalAndPrimitiveCreatorRegistry("LogicalAnd", LogicalAndPrimitiveCreator);
RegistryMSOps g_LogicalNotPrimitiveCreatorRegistry("LogicalNot", LogicalNotPrimitiveCreator);
RegistryMSOps g_LogicalOrPrimitiveCreatorRegistry("LogicalOr", LogicalOrPrimitiveCreator);
RegistryMSOps g_LpNormalizationPrimitiveCreatorRegistry("LpNormalization", LpNormalizationPrimitiveCreator);
RegistryMSOps g_LrnPrimitiveCreatorRegistry("Lrn", LrnPrimitiveCreator);
RegistryMSOps g_LshProjectionPrimitiveCreatorRegistry("LshProjection", LshProjectionPrimitiveCreator);
RegistryMSOps g_LSTMPrimitiveCreatorRegistry("LSTM", LSTMPrimitiveCreator);
RegistryMSOps g_L2NormalizeFusionPrimitiveCreatorRegistry("L2NormalizeFusion", L2NormalizeFusionPrimitiveCreator);
RegistryMSOps g_MatMulPrimitiveCreatorRegistry("MatMul", MatMulPrimitiveCreator);
RegistryMSOps g_MaximumPrimitiveCreatorRegistry("Maximum", MaximumPrimitiveCreator);
RegistryMSOps g_MaximumGradPrimitiveCreatorRegistry("MaximumGrad", MaximumGradPrimitiveCreator);
RegistryMSOps g_MaxPoolPrimitiveCreatorRegistry("MaxPool", MaxPoolFusionPrimitiveCreator);
RegistryMSOps g_MaxPoolFusionPrimitiveCreatorRegistry("MaxPoolFusion", MaxPoolFusionPrimitiveCreator);
RegistryMSOps g_MergePrimitiveCreatorRegistry("Merge", MergePrimitiveCreator);
RegistryMSOps g_MinimumPrimitiveCreatorRegistry("Minimum", MinimumPrimitiveCreator);
RegistryMSOps g_MinimumGradPrimitiveCreatorRegistry("MinimumGrad", MinimumGradPrimitiveCreator);
RegistryMSOps g_ModPrimitiveCreatorRegistry("Mod", ModPrimitiveCreator);
RegistryMSOps g_MulPrimitiveCreatorRegistry("Mul", MulFusionPrimitiveCreator);
RegistryMSOps g_MulMulFusionPrimitiveCreatorRegistry("MulFusion", MulFusionPrimitiveCreator);
RegistryMSOps g_MulGradPrimitiveCreatorRegistry("MulGrad", MulGradPrimitiveCreator);
RegistryMSOps g_NegPrimitiveCreatorRegistry("Neg", NegPrimitiveCreator);
RegistryMSOps g_NegGradPrimitiveCreatorRegistry("NegGrad", NegGradPrimitiveCreator);
RegistryMSOps g_NonMaxSuppressionPrimitiveCreatorRegistry("NonMaxSuppression", NonMaxSuppressionPrimitiveCreator);
RegistryMSOps g_NotEqualPrimitiveCreatorRegistry("NotEqual", NotEqualPrimitiveCreator);
RegistryMSOps g_OneHotPrimitiveCreatorRegistry("OneHot", OneHotPrimitiveCreator);
RegistryMSOps g_OnesLikePrimitiveCreatorRegistry("OnesLike", OnesLikePrimitiveCreator);
RegistryMSOps g_PadPrimitiveCreatorRegistry("Pad", PadFusionPrimitiveCreator);
RegistryMSOps g_PadFusionPrimitiveCreatorRegistry("PadFusion", PadFusionPrimitiveCreator);
RegistryMSOps g_PartialFusionPrimitiveCreatorRegistry("PartialFusion", PartialFusionPrimitiveCreator);
RegistryMSOps g_PowerGradPrimitiveCreatorRegistry("PowerGrad", PowerGradPrimitiveCreator);
RegistryMSOps g_PowFusionPrimitiveCreatorRegistry("PowFusion", PowFusionPrimitiveCreator);
RegistryMSOps g_PReLUFusionPrimitiveCreatorRegistry("PReLUFusion", PReLUFusionPrimitiveCreator);
RegistryMSOps g_RangePrimitiveCreatorRegistry("Range", RangePrimitiveCreator);
RegistryMSOps g_RankPrimitiveCreatorRegistry("Rank", RankPrimitiveCreator);
RegistryMSOps g_ReciprocalPrimitiveCreatorRegistry("Reciprocal", ReciprocalPrimitiveCreator);
RegistryMSOps g_RealDivPrimitiveCreatorRegistry("RealDiv", RealDivPrimitiveCreator);
RegistryMSOps g_ReducePrimitiveCreatorRegistry("Reduce", ReduceFusionPrimitiveCreator);
RegistryMSOps g_ReduceFusionPrimitiveCreatorRegistry("ReduceFusion", ReduceFusionPrimitiveCreator);
RegistryMSOps g_ReshapePrimitiveCreatorRegistry("Reshape", ReshapePrimitiveCreator);
RegistryMSOps g_ResizePrimitiveCreatorRegistry("Resize", ResizePrimitiveCreator);
RegistryMSOps g_ReverseV2PrimitiveCreatorRegistry("ReverseV2", ReverseV2PrimitiveCreator);
RegistryMSOps g_ReverseSequencePrimitiveCreatorRegistry("ReverseSequence", ReverseSequencePrimitiveCreator);
RegistryMSOps g_RfftPrimitiveCreatorRegistry("Rfft", RfftPrimitiveCreator);
RegistryMSOps g_ROIPoolingPrimitiveCreatorRegistry("ROIPooling", ROIPoolingPrimitiveCreator);
RegistryMSOps g_RoundPrimitiveCreatorRegistry("Round", RoundPrimitiveCreator);
RegistryMSOps g_RsqrtPrimitiveCreatorRegistry("Rsqrt", RsqrtPrimitiveCreator);
RegistryMSOps g_QuantDTypeCastPrimitiveCreatorRegistry("QuantDTypeCast", QuantDTypeCastPrimitiveCreator);
RegistryMSOps g_ScalePrimitiveCreatorRegistry("Scale", ScaleFusionPrimitiveCreator);
RegistryMSOps g_ScaleFusionPrimitiveCreatorRegistry("ScaleFusion", ScaleFusionPrimitiveCreator);
RegistryMSOps g_ShapePrimitiveCreatorRegistry("Shape", ShapePrimitiveCreator);
RegistryMSOps g_SigmoidCrossEntropyWithLogitsPrimitiveCreatorRegistry("SigmoidCrossEntropyWithLogits",
                                                                      SigmoidCrossEntropyWithLogitsPrimitiveCreator);
RegistryMSOps g_SigmoidCrossEntropyWithLogitsGradPrimitiveCreatorRegistry(
  "SigmoidCrossEntropyWithLogitsGrad", SigmoidCrossEntropyWithLogitsGradPrimitiveCreator);
RegistryMSOps g_SinPrimitiveCreatorRegistry("Sin", SinPrimitiveCreator);
RegistryMSOps g_SkipGramPrimitiveCreatorRegistry("SkipGram", SkipGramPrimitiveCreator);
RegistryMSOps g_SliceFusionPrimitiveCreatorRegistry("SliceFusion", SliceFusionPrimitiveCreator);
RegistryMSOps g_SmoothL1LossPrimitiveCreatorRegistry("SmoothL1Loss", SmoothL1LossPrimitiveCreator);
RegistryMSOps g_SmoothL1LossGradPrimitiveCreatorRegistry("SmoothL1LossGrad", SmoothL1LossGradPrimitiveCreator);
RegistryMSOps g_SoftmaxPrimitiveCreatorRegistry("Softmax", SoftmaxPrimitiveCreator);
RegistryMSOps g_SpaceToBatchPrimitiveCreatorRegistry("SpaceToBatch", SpaceToBatchPrimitiveCreator);
RegistryMSOps g_SpaceToBatchNDPrimitiveCreatorRegistry("SpaceToBatchND", SpaceToBatchNDPrimitiveCreator);
RegistryMSOps g_SpaceToDepthPrimitiveCreatorRegistry("SpaceToDepth", SpaceToDepthPrimitiveCreator);
RegistryMSOps g_SparseToDensePrimitiveCreatorRegistry("SparseToDense", SparseToDensePrimitiveCreator);
RegistryMSOps g_SplitPrimitiveCreatorRegistry("Split", SplitPrimitiveCreator);
RegistryMSOps g_SqrtPrimitiveCreatorRegistry("Sqrt", SqrtPrimitiveCreator);
RegistryMSOps g_SqueezePrimitiveCreatorRegistry("Squeeze", SqueezePrimitiveCreator);
RegistryMSOps g_SquarePrimitiveCreatorRegistry("Square", SquarePrimitiveCreator);
RegistryMSOps g_SquaredDifferencePrimitiveCreatorRegistry("SquaredDifference", SquaredDifferencePrimitiveCreator);
RegistryMSOps g_StackPrimitiveCreatorRegistry("Stack", StackPrimitiveCreator);
RegistryMSOps g_StridedSlicePrimitiveCreatorRegistry("StridedSlice", StridedSlicePrimitiveCreator);
RegistryMSOps g_SubPrimitiveCreatorRegistry("Sub", SubFusionPrimitiveCreator);
RegistryMSOps g_SubFusionPrimitiveCreatorRegistry("SubFusion", SubFusionPrimitiveCreator);
RegistryMSOps g_SubGradPrimitiveCreatorRegistry("SubGrad", SubGradPrimitiveCreator);
RegistryMSOps g_SwitchPrimitiveCreatorRegistry("Switch", SwitchPrimitiveCreator);
RegistryMSOps g_TensorListFromTensorPrimitiveCreatorRegistry("TensorListFromTensor",
                                                             TensorListFromTensorPrimitiveCreator);
RegistryMSOps g_TensorListGetItemPrimitiveCreatorRegistry("TensorListGetItem", TensorListGetItemPrimitiveCreator);
RegistryMSOps g_TensorListReservePrimitiveCreatorRegistry("TensorListReserve", TensorListReservePrimitiveCreator);
RegistryMSOps g_TensorListSetItemPrimitiveCreatorRegistry("TensorListSetItem", TensorListSetItemPrimitiveCreator);
RegistryMSOps g_TensorListStackPrimitiveCreatorRegistry("TensorListStack", TensorListStackPrimitiveCreator);
RegistryMSOps g_TileFusionPrimitiveCreatorRegistry("TileFusion", TileFusionPrimitiveCreator);
RegistryMSOps g_TopKPrimitiveCreatorRegistry("TopK", TopKFusionPrimitiveCreator);
RegistryMSOps g_TopKFusionPrimitiveCreatorRegistry("TopKFusion", TopKFusionPrimitiveCreator);
RegistryMSOps g_TransposePrimitiveCreatorxRegistry("Transpose", TransposePrimitiveCreator);
RegistryMSOps g_UniquePrimitiveCreatorRegistry("Unique", UniquePrimitiveCreator);
RegistryMSOps g_UnpackPrimitiveCreatorRegistry("Unpack", UnpackPrimitiveCreator);
RegistryMSOps g_UnsortedSegmentSumPrimitiveCreatorRegistry("UnsortedSegmentSum", UnsortedSegmentSumPrimitiveCreator);
RegistryMSOps g_UnsqueezePrimitiveCreatorRegistry("Unsqueeze", UnsqueezePrimitiveCreator);
RegistryMSOps g_WherePrimitiveCreatorRegistry("Where", WherePrimitiveCreator);
RegistryMSOps g_ZerosLikePrimitiveCreatorRegistry("ZerosLike", ZerosLikePrimitiveCreator);
}  // namespace lite
}  // namespace mindspore

#endif
