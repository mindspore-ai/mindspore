/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_BASE_NN_OPTIMIZER_OP_NAME_H_
#define MINDSPORE_CORE_BASE_NN_OPTIMIZER_OP_NAME_H_

namespace mindspore {
// optimizer
constexpr auto kAssignOpName = "Assign";
constexpr auto kAssignAddOpName = "AssignAdd";
constexpr auto kAssignSubOpName = "AssignSub";
constexpr auto kApplyAddSignOpName = "ApplyAddSign";
constexpr auto kSparseApplyCenteredRMSPropOpName = "SparseApplyCenteredRMSProp";
constexpr auto kSparseApplyAdagradOpName = "SparseApplyAdagrad";
constexpr auto kSparseApplyAdagradV2OpName = "SparseApplyAdagradV2";
constexpr auto kSparseApplyRMSPropOpName = "SparseApplyRMSProp";
constexpr auto kSparseApplyAdadeltaOpName = "SparseApplyAdadelta";
constexpr auto kApplyRMSPropOpName = "ApplyRMSProp";
constexpr auto kSparseApplyAdagradDAOpName = "SparseApplyAdagradDA";
constexpr auto kSparseApplyMomentumOpName = "SparseApplyMomentum";
constexpr auto kSparseApplyProximalGradientDescentOpName = "SparseApplyProximalGradientDescent";

// activation
constexpr auto kGeLUGradOpName = "GeLUGrad";
constexpr auto kFastGeLUOpName = "FastGeLU";
constexpr auto kFastGeLUGradOpName = "FastGeLUGrad";
constexpr auto kReLUOpName = "ReLU";
constexpr auto kReLUGradOpName = "ReluGrad";
constexpr auto kReLU6OpName = "ReLU6";
constexpr auto kSiLUOpName = "SiLU";
constexpr auto kReLUV3OpName = "ReLUV3";
constexpr auto kReLUGradV2OpName = "ReluGradV2";
constexpr auto kGLUOpName = "GLU";
constexpr auto kGluGradOpName = "GluGrad";
constexpr auto kGeLUOpName = "GeLU";
constexpr auto kAdamApplyOneAssignOpName = "AdamApplyOneAssign";
constexpr auto kAdamApplyOneOpName = "AdamApplyOne";
constexpr auto kAdamApplyOneWithDecayAssignOpName = "AdamApplyOneWithDecayAssign";
constexpr auto kAdamApplyOneWithDecayOpName = "AdamApplyOneWithDecay";
constexpr auto kAdamOpName = "Adam";
constexpr auto kAdamWeightDecayOpName = "AdamWeightDecay";
constexpr auto kApplyAdadeltaDOpName = "ApplyAdadeltaD";
constexpr auto kApplyAdadeltaOpName = "ApplyAdadelta";
constexpr auto kApplyAdagradDADOpName = "ApplyAdagradDAD";
constexpr auto kApplyAdagradDAOpName = "ApplyAdagradDA";
constexpr auto kApplyAdagradDOpName = "ApplyAdagradD";
constexpr auto kApplyAdagradOpName = "ApplyAdagrad";
constexpr auto kApplyAdagradV2OpName = "ApplyAdagradV2";
constexpr auto kApplyAdagradV2DOpName = "ApplyAdagradV2D";
constexpr auto kApplyAdaMaxDOpName = "ApplyAdaMaxD";
constexpr auto kApplyAdaMaxOpName = "ApplyAdaMax";
constexpr auto kApplyAdamDOpName = "ApplyAdamD";
constexpr auto kApplyAdamOpName = "ApplyAdam";
constexpr auto kApplyAdamWithAmsgradOpName = "ApplyAdamWithAmsgrad";
constexpr auto kApplyAdamWithAmsgradDOpName = "ApplyAdamWithAmsgradD";
constexpr auto kApplyAdamWithAmsgradV2OpName = "ApplyAdamWithAmsgradV2";
constexpr auto kApplyAddSignDOpName = "ApplyAddSignD";
constexpr auto kApplyCenteredRMSPropDOpName = "ApplyCenteredRMSPropD";
constexpr auto kApplyCenteredRMSPropOpName = "ApplyCenteredRMSProp";
constexpr auto kApplyFtrlDOpName = "ApplyFtrlD";
constexpr auto kApplyFtrlOpName = "ApplyFtrl";
constexpr auto kApplyGradientDescentOpName = "ApplyGradientDescent";
constexpr auto kApplyKerasMomentumDOpName = "ApplyKerasMomentumD";
constexpr auto kApplyKerasMomentumOpName = "ApplyKerasMomentum";
constexpr auto kApplyMomentumDOpName = "ApplyMomentumD";
constexpr auto kApplyMomentumOpName = "ApplyMomentum";
constexpr auto kApplyPowerSignDOpName = "ApplyPowerSignD";
constexpr auto kApplyPowerSignOpName = "ApplyPowerSign";
constexpr auto kApplyProximalAdagradDOpName = "ApplyProximalAdagradD";
constexpr auto kApplyProximalAdagradOpName = "ApplyProximalAdagrad";
constexpr auto kApplyProximalGradientDescentOpName = "ApplyProximalGradientDescent";
constexpr auto kCeLUOpName = "CeLU";
constexpr auto kCeluV2OpName = "CeluV2";
constexpr auto kEluGradV2OpName = "EluGradV2";
constexpr auto kFastGeluOpName = "FastGelu";
constexpr auto kFastGeluGradOpName = "FastGeluGrad";
constexpr auto kFusedAdaFactorOpName = "FusedAdaFactor";
constexpr auto kFusedAdaFactorWithGlobalNormOpName = "FusedAdaFactorWithGlobalNorm";
constexpr auto kFusedAdamOpName = "FusedAdam";
constexpr auto kFusedAdamWeightDecayOpName = "FusedAdamWeightDecay";
constexpr auto kFusedSparseAdamOpName = "FusedSparseAdam";
constexpr auto kFusedSparseFtrlOpName = "FusedSparseFtrl";
constexpr auto kFusedSparseLazyAdamOpName = "FusedSparseLazyAdam";
constexpr auto kFusedSparseProximalAdagradOpName = "FusedSparseProximalAdagrad";
constexpr auto kGeluOpName = "Gelu";
constexpr auto kGeluGradOpName = "GeluGrad";
constexpr auto kMomentumOpName = "Momentum";
constexpr auto kPReLUOpName = "PReLU";
constexpr auto kPReluOpName = "PRelu";
constexpr auto kPReLUGradOpName = "PReLUGrad";
constexpr auto kReluOpName = "Relu";
constexpr auto kReLU6GradOpName = "ReLU6Grad";
constexpr auto kRelu6GradOpName = "Relu6Grad";
constexpr auto kSeLUOpName = "SeLU";
constexpr auto kSeluOpName = "Selu";
constexpr auto kSGDOpName = "SGD";
constexpr auto kSparseApplyAdagradDOpName = "SparseApplyAdagradD";
constexpr auto kSparseApplyAdagradV2DOpName = "SparseApplyAdagradV2D";
constexpr auto kSparseApplyFtrlOpName = "SparseApplyFtrl";
constexpr auto kSparseApplyFtrlDOpName = "SparseApplyFtrlD";
constexpr auto kSparseApplyFtrlV2DOpName = "SparseApplyFtrlV2D";
constexpr auto kSparseApplyProximalAdagradDOpName = "SparseApplyProximalAdagradD";
constexpr auto kSparseApplyProximalAdagradOpName = "SparseApplyProximalAdagrad";
constexpr auto kSparseApplyRMSPropDOpName = "SparseApplyRMSPropD";
constexpr auto kCombineMomentumOpName = "CombineMomentum";
constexpr auto kCombineScaleMomentumOpName = "CombineScaleMomentum";
constexpr auto kCombineWeightDecayScaleMomentumOpName = "CombineWeightDecayScaleMomentum";
constexpr auto kFusedMulApplyMomentumOpName = "FusedMulApplyMomentum";
constexpr auto kFusedScaleApplyMomentumOpName = "FusedScaleApplyMomentum";
constexpr auto kFusedWeightApplyMomentumOpName = "FusedWeightApplyMomentum";
constexpr auto kFusedWeightScaleApplyMomentumOpName = "FusedWeightScaleApplyMomentum";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_NN_OPTIMIZER_OP_NAME_H_
