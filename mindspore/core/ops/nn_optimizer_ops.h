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

#ifndef MINDSPORE_CORE_BASE_NN_OPTIMIZER_OPS_H_
#define MINDSPORE_CORE_BASE_NN_OPTIMIZER_OPS_H_

#include <iostream>
#include <memory>
#include <string>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "utils/flags.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace prim {
// optimizer
constexpr auto kAssign = "Assign";
constexpr auto kAssignAdd = "AssignAdd";
constexpr auto kAssignSub = "AssignSub";
constexpr auto kApplyAddSign = "ApplyAddSign";
constexpr auto kSparseApplyCenteredRMSProp = "SparseApplyCenteredRMSProp";
constexpr auto kSparseApplyAdagrad = "SparseApplyAdagrad";
constexpr auto kSparseApplyAdagradV2 = "SparseApplyAdagradV2";
constexpr auto kSparseApplyRMSProp = "SparseApplyRMSProp";
constexpr auto kSparseApplyAdadelta = "SparseApplyAdadelta";
constexpr auto kApplyRMSProp = "ApplyRMSProp";
constexpr auto kSparseApplyAdagradDA = "SparseApplyAdagradDA";
constexpr auto kSparseApplyMomentum = "SparseApplyMomentum";
constexpr auto kSparseApplyProximalGradientDescent = "SparseApplyProximalGradientDescent";

// activation
constexpr auto kGeLUGrad = "GeLUGrad";
constexpr auto kFastGeLU = "FastGeLU";
constexpr auto kFastGeLUGrad = "FastGeLUGrad";
constexpr auto kReLU = "ReLU";
constexpr auto kReLUGrad = "ReluGrad";
constexpr auto kReLU6 = "ReLU6";
constexpr auto kSiLU = "SiLU";
constexpr auto kReLUV2 = "ReLUV2";
constexpr auto kReLUV3 = "ReLUV3";
constexpr auto kReLUGradV2 = "ReluGradV2";
constexpr auto kGLU = "GLU";
constexpr auto kGluGrad = "GluGrad";
constexpr auto kGeLU = "GeLU";

// Activation
GVAR_DEF(PrimitivePtr, kPrimCeLU, std::make_shared<Primitive>("CeLU"));
GVAR_DEF(PrimitivePtr, kPrimCeluV2, std::make_shared<Primitive>("CeluV2"));
GVAR_DEF(PrimitivePtr, kPrimReluGrad, std::make_shared<Primitive>(kReLUGrad));
GVAR_DEF(PrimitivePtr, kPrimReluGradV2, std::make_shared<Primitive>("ReluGradV2"));
GVAR_DEF(PrimitivePtr, kPrimReLU6Grad, std::make_shared<Primitive>("ReLU6Grad"));
GVAR_DEF(PrimitivePtr, kPrimRelu6Grad, std::make_shared<Primitive>("Relu6Grad"));
GVAR_DEF(PrimitivePtr, kPrimGeLU, std::make_shared<Primitive>(kGeLU));
GVAR_DEF(PrimitivePtr, kPrimGelu, std::make_shared<Primitive>("Gelu"));
GVAR_DEF(PrimitivePtr, kPrimGeLUGrad, std::make_shared<Primitive>(kGeLUGrad));
GVAR_DEF(PrimitivePtr, kPrimGeluGrad, std::make_shared<Primitive>("GeluGrad"));
GVAR_DEF(PrimitivePtr, kPrimFastGeLU, std::make_shared<Primitive>(kFastGeLU));
GVAR_DEF(PrimitivePtr, kPrimFastGelu, std::make_shared<Primitive>("FastGelu"));
GVAR_DEF(PrimitivePtr, kPrimFastGeLUGrad, std::make_shared<Primitive>(kFastGeLUGrad));
GVAR_DEF(PrimitivePtr, kPrimFastGeluGrad, std::make_shared<Primitive>("FastGeluGrad"));
GVAR_DEF(PrimitivePtr, kPrimReLU, std::make_shared<Primitive>(kReLU));
GVAR_DEF(PrimitivePtr, kPrimRelu, std::make_shared<Primitive>("Relu"));
GVAR_DEF(PrimitivePtr, kPrimElu, std::make_shared<Primitive>("Elu"));
GVAR_DEF(PrimitivePtr, kPrimEluGrad, std::make_shared<Primitive>("EluGrad"));
GVAR_DEF(PrimitivePtr, kPrimReLU6, std::make_shared<Primitive>(kReLU6));
GVAR_DEF(PrimitivePtr, kPrimReLUV2, std::make_shared<Primitive>(kReLUV2));
GVAR_DEF(PrimitivePtr, kPrimReluV2, std::make_shared<Primitive>("ReluV2"));
GVAR_DEF(PrimitivePtr, kPrimReLUV3, std::make_shared<Primitive>(kReLUV3));
GVAR_DEF(PrimitivePtr, kPrimPReLU, std::make_shared<Primitive>("PReLU"));
GVAR_DEF(PrimitivePtr, kPrimPRelu, std::make_shared<Primitive>("PRelu"));
GVAR_DEF(PrimitivePtr, kPrimPReLUGrad, std::make_shared<Primitive>("PReLUGrad"));
GVAR_DEF(PrimitivePtr, kPrimGLU, std::make_shared<Primitive>(kGLU));
GVAR_DEF(PrimitivePtr, kPrimGluGrad, std::make_shared<Primitive>(kGluGrad));
GVAR_DEF(PrimitivePtr, kPrimSeLU, std::make_shared<Primitive>("SeLU"));
GVAR_DEF(PrimitivePtr, kPrimSelu, std::make_shared<Primitive>("Selu"));
GVAR_DEF(PrimitivePtr, kPrimSiLU, std::make_shared<Primitive>("SiLU"));
GVAR_DEF(PrimitivePtr, kPrimSiLUGrad, std::make_shared<Primitive>("SiLUGrad"));

// nn optimizer
GVAR_DEF(PrimitivePtr, kPrimDynamicAssign, std::make_shared<Primitive>("DynamicAssign"));
GVAR_DEF(PrimitivePtr, kPrimAssign, std::make_shared<Primitive>(kAssign));
GVAR_DEF(PrimitivePtr, kPrimAssignAdd, std::make_shared<Primitive>(kAssignAdd));
GVAR_DEF(PrimitivePtr, kPrimAssignSub, std::make_shared<Primitive>(kAssignSub));
GVAR_DEF(PrimitivePtr, kPrimFusedAdam, std::make_shared<Primitive>("FusedAdam"));
GVAR_DEF(PrimitivePtr, kPrimFusedAdaFactor, std::make_shared<Primitive>("FusedAdaFactor"));
GVAR_DEF(PrimitivePtr, kPrimFusedAdaFactorWithGlobalNorm, std::make_shared<Primitive>("FusedAdaFactorWithGlobalNorm"));
GVAR_DEF(PrimitivePtr, kPrimFusedAdamWeightDecay, std::make_shared<Primitive>("FusedAdamWeightDecay"));
GVAR_DEF(PrimitivePtr, kPrimSGD, std::make_shared<Primitive>("SGD"));
GVAR_DEF(PrimitivePtr, kPrimApplyProximalAdagrad, std::make_shared<Primitive>("ApplyProximalAdagrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyAdadelta, std::make_shared<Primitive>(kSparseApplyAdadelta));
GVAR_DEF(PrimitivePtr, kPrimApplyRMSProp, std::make_shared<Primitive>(kApplyRMSProp));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyCenteredRMSProp, std::make_shared<Primitive>(kSparseApplyCenteredRMSProp));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyAdagrad, std::make_shared<Primitive>("SparseApplyAdagrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyAdagradV2, std::make_shared<Primitive>("SparseApplyAdagradV2"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyAdagradDA, std::make_shared<Primitive>(kSparseApplyAdagradDA));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyMomentum, std::make_shared<Primitive>(kSparseApplyMomentum));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyProximalGradientDescent,
         std::make_shared<Primitive>(kSparseApplyProximalGradientDescent));
GVAR_DEF(PrimitivePtr, kPrimAdam, std::make_shared<Primitive>("Adam"));
GVAR_DEF(PrimitivePtr, kPrimAdamWeightDecay, std::make_shared<Primitive>("AdamWeightDecay"));
GVAR_DEF(PrimitivePtr, kPrimAdamNoUpdateParam, std::make_shared<Primitive>("AdamNoUpdateParam"));
GVAR_DEF(PrimitivePtr, kPrimLamb, std::make_shared<Primitive>("Lamb"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdaMax, std::make_shared<Primitive>("ApplyAdaMax"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdaMaxD, std::make_shared<Primitive>("ApplyAdaMaxD"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdam, std::make_shared<Primitive>("ApplyAdam"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdamD, std::make_shared<Primitive>("ApplyAdamD"));
GVAR_DEF(PrimitivePtr, kPrimApplyGradientDescent, std::make_shared<Primitive>("ApplyGradientDescent"));
GVAR_DEF(PrimitivePtr, kPrimApplyPowerSign, std::make_shared<Primitive>("ApplyPowerSign"));
GVAR_DEF(PrimitivePtr, kPrimApplyPowerSignD, std::make_shared<Primitive>("ApplyPowerSignD"));
GVAR_DEF(PrimitivePtr, kPrimFusedSparseAdam, std::make_shared<Primitive>("FusedSparseAdam"));
GVAR_DEF(PrimitivePtr, kPrimFusedSparseFtrl, std::make_shared<Primitive>("FusedSparseFtrl"));
GVAR_DEF(PrimitivePtr, kPrimFusedSparseLazyAdam, std::make_shared<Primitive>("FusedSparseLazyAdam"));
GVAR_DEF(PrimitivePtr, kPrimFusedSparseProximalAdagrad, std::make_shared<Primitive>("FusedSparseProximalAdagrad"));
GVAR_DEF(PrimitivePtr, kPrimApplyCenteredRMSProp, std::make_shared<Primitive>("ApplyCenteredRMSProp"));
GVAR_DEF(PrimitivePtr, kPrimMomentum, std::make_shared<Primitive>("Momentum"));
GVAR_DEF(PrimitivePtr, kPrimApplyMomentum, std::make_shared<Primitive>("ApplyMomentum"));
GVAR_DEF(PrimitivePtr, kPrimApplyMomentumD, std::make_shared<Primitive>("ApplyMomentumD"));
GVAR_DEF(PrimitivePtr, kPrimApplyFtrl, std::make_shared<Primitive>("ApplyFtrl"));
GVAR_DEF(PrimitivePtr, kPrimApplyFtrlD, std::make_shared<Primitive>("ApplyFtrlD"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyFtrl, std::make_shared<Primitive>("SparseApplyFtrl"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyFtrlD, std::make_shared<Primitive>("SparseApplyFtrlD"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyFtrlV2D, std::make_shared<Primitive>("SparseApplyFtrlV2D"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyProximalAdagrad, std::make_shared<Primitive>("SparseApplyProximalAdagrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyProximalAdagradD, std::make_shared<Primitive>("SparseApplyProximalAdagradD"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdagradDA, std::make_shared<Primitive>("ApplyAdagradDA"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdagradV2, std::make_shared<Primitive>("ApplyAdagradV2"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdagradV2D, std::make_shared<Primitive>("ApplyAdagradV2D"));
GVAR_DEF(PrimitivePtr, kPrimApplyProximalGradientDescent, std::make_shared<Primitive>("ApplyProximalGradientDescent"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyRMSProp, std::make_shared<Primitive>("SparseApplyRMSProp"));
GVAR_DEF(PrimitivePtr, kPrimApplyKerasMomentum, std::make_shared<Primitive>("ApplyKerasMomentum"));
GVAR_DEF(PrimitivePtr, kLambApplyOptimizerAssign, std::make_shared<Primitive>("LambApplyOptimizerAssign"));
GVAR_DEF(PrimitivePtr, kLambApplyWeightAssign, std::make_shared<Primitive>("LambApplyWeightAssign"));
GVAR_DEF(PrimitivePtr, kPrimApplyAddSign, std::make_shared<Primitive>("ApplyAddSign"));
GVAR_DEF(PrimitivePtr, kPrimApplyAddSignD, std::make_shared<Primitive>("ApplyAddSignD"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdagrad, std::make_shared<Primitive>("ApplyAdagrad"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdagradD, std::make_shared<Primitive>("ApplyAdagradD"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdadelta, std::make_shared<Primitive>("ApplyAdadelta"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdadeltaD, std::make_shared<Primitive>("ApplyAdadeltaD"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdamWithAmsgrad, std::make_shared<Primitive>("ApplyAdamWithAmsgrad"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdamWithAmsgradV2, std::make_shared<Primitive>("ApplyAdamWithAmsgradV2"));

// AdamApplyOne
GVAR_DEF(PrimitivePtr, kPrimAdamApplyOne, std::make_shared<Primitive>("AdamApplyOne"));
GVAR_DEF(PrimitivePtr, kPrimAdamApplyOneAssign, std::make_shared<Primitive>("AdamApplyOneAssign"));

// AdamApplyOneWithDecay
GVAR_DEF(PrimitivePtr, kPrimAdamApplyOneWithDecay, std::make_shared<Primitive>("AdamApplyOneWithDecay"));
GVAR_DEF(PrimitivePtr, kPrimAdamApplyOneWithDecayAssign, std::make_shared<Primitive>("AdamApplyOneWithDecayAssign"));
}  // namespace prim
}  // namespace mindspore
#endif  // MINDSPORE_CORE_BASE_NN_OPTIMIZER_OPS_H_
