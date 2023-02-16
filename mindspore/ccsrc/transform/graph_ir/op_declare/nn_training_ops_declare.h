/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_TRAINING_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_TRAINING_OPS_DECLARE_H_

#include "utils/hash_map.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/nn_training_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(ApplyAdam)
DECLARE_OP_USE_OUTPUT(ApplyAdam)

DECLARE_OP_ADAPTER(ApplyAdamD)
DECLARE_OP_USE_OUTPUT(ApplyAdamD)

DECLARE_OP_ADAPTER(ApplyAdagradD)
DECLARE_OP_USE_OUTPUT(ApplyAdagradD)

DECLARE_OP_ADAPTER(ApplyAdagradV2D)
DECLARE_OP_USE_OUTPUT(ApplyAdagradV2D)

DECLARE_OP_ADAPTER(ApplyAddSignD)
DECLARE_OP_USE_OUTPUT(ApplyAddSignD)

DECLARE_OP_ADAPTER(SparseApplyAdagradV2D)
DECLARE_OP_USE_OUTPUT(SparseApplyAdagradV2D)

DECLARE_OP_ADAPTER(DataFormatDimMap)
DECLARE_OP_USE_OUTPUT(DataFormatDimMap)

DECLARE_OP_ADAPTER(ApplyAdadeltaD)
DECLARE_OP_USE_OUTPUT(ApplyAdadeltaD)

DECLARE_OP_ADAPTER(ApplyAdaMaxD)
DECLARE_OP_USE_OUTPUT(ApplyAdaMaxD)

DECLARE_OP_ADAPTER(ApplyGradientDescent)
DECLARE_OP_USE_OUTPUT(ApplyGradientDescent)

DECLARE_OP_ADAPTER(ApplyPowerSignD)
DECLARE_OP_USE_OUTPUT(ApplyPowerSignD)

DECLARE_OP_ADAPTER(ApplyProximalGradientDescent)
DECLARE_OP_USE_OUTPUT(ApplyProximalGradientDescent)

DECLARE_OP_ADAPTER(SGD)
DECLARE_OP_USE_OUTPUT(SGD)

DECLARE_OP_ADAPTER(ApplyMomentum)
DECLARE_OP_USE_OUTPUT(ApplyMomentum)

DECLARE_OP_ADAPTER(SparseApplyAdagradD)
DECLARE_OP_USE_OUTPUT(SparseApplyAdagradD)

DECLARE_OP_ADAPTER(ApplyProximalAdagradD)
DECLARE_OP_USE_OUTPUT(ApplyProximalAdagradD)

DECLARE_OP_ADAPTER(SparseApplyProximalAdagradD)
DECLARE_OP_USE_OUTPUT(SparseApplyProximalAdagradD)

DECLARE_OP_ADAPTER(LarsV2Update)
DECLARE_OP_USE_OUTPUT(LarsV2Update)

DECLARE_OP_ADAPTER(ApplyFtrl)
DECLARE_OP_USE_OUTPUT(ApplyFtrl)

DECLARE_OP_ADAPTER(SparseApplyFtrlD)
DECLARE_OP_USE_OUTPUT(SparseApplyFtrlD)

DECLARE_OP_ADAPTER(SparseApplyFtrl)
DECLARE_OP_USE_OUTPUT(SparseApplyFtrl)

DECLARE_OP_ADAPTER(SparseApplyFtrlV2D)
DECLARE_OP_USE_OUTPUT(SparseApplyFtrlV2D)

DECLARE_OP_ADAPTER(SparseApplyFtrlV2)
DECLARE_OP_USE_OUTPUT(SparseApplyFtrlV2)

DECLARE_OP_ADAPTER(ApplyRMSPropD)
DECLARE_OP_USE_INPUT_ATTR(ApplyRMSPropD)
DECLARE_OP_USE_OUTPUT(ApplyRMSPropD)

DECLARE_OP_ADAPTER(ApplyCenteredRMSProp)
DECLARE_OP_USE_OUTPUT(ApplyCenteredRMSProp)

DECLARE_OP_ADAPTER(SparseApplyRMSProp)
DECLARE_OP_USE_OUTPUT(SparseApplyRMSProp)

DECLARE_OP_ADAPTER(ApplyAdaMax)
DECLARE_OP_USE_OUTPUT(ApplyAdaMax)

DECLARE_OP_ADAPTER(SparseApplyAdagrad)
DECLARE_OP_USE_OUTPUT(SparseApplyAdagrad)

DECLARE_OP_ADAPTER(SparseApplyAdagradV2)
DECLARE_OP_USE_OUTPUT(SparseApplyAdagradV2)

DECLARE_OP_ADAPTER(ApplyKerasMomentumD)
DECLARE_OP_USE_OUTPUT(ApplyKerasMomentumD)

DECLARE_OP_ADAPTER(ApplyKerasMomentum)
DECLARE_OP_USE_OUTPUT(ApplyKerasMomentum)

DECLARE_OP_ADAPTER(ApplyAdamWithAmsgrad)
DECLARE_OP_USE_OUTPUT(ApplyAdamWithAmsgrad)

DECLARE_OP_ADAPTER(ApplyPowerSign)
DECLARE_OP_USE_OUTPUT(ApplyPowerSign)

DECLARE_OP_ADAPTER(ApplyAddSign)
DECLARE_OP_USE_OUTPUT(ApplyAddSign)

DECLARE_OP_ADAPTER(ApplyAdagrad)
DECLARE_OP_USE_OUTPUT(ApplyAdagrad)

DECLARE_OP_ADAPTER(ApplyAdagradV2)
DECLARE_OP_USE_OUTPUT(ApplyAdagradV2)

DECLARE_OP_ADAPTER(ApplyAdagradDA)
DECLARE_OP_USE_OUTPUT(ApplyAdagradDA)

DECLARE_OP_ADAPTER(ApplyRMSProp)
DECLARE_OP_USE_OUTPUT(ApplyRMSProp)

DECLARE_OP_ADAPTER(ApplyProximalAdagrad)
DECLARE_OP_USE_OUTPUT(ApplyProximalAdagrad)

DECLARE_OP_ADAPTER(SparseApplyProximalAdagrad)
DECLARE_OP_USE_OUTPUT(SparseApplyProximalAdagrad)

DECLARE_OP_ADAPTER(ApplyAdadelta)
DECLARE_OP_USE_OUTPUT(ApplyAdadelta)

DECLARE_OP_ADAPTER(SparseApplyAdadelta)
DECLARE_OP_USE_OUTPUT(SparseApplyAdadelta)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_TRAINING_OPS_DECLARE_H_
