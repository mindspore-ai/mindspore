/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_IMAGE_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_IMAGE_OPS_DECLARE_H_

#include <string>
#include <unordered_map>
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/nn_norm_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(SmoothL1Loss)
DECLARE_OP_USE_OUTPUT(SmoothL1Loss)

DECLARE_OP_ADAPTER(SmoothL1LossGrad)
DECLARE_OP_USE_OUTPUT(SmoothL1LossGrad)

DECLARE_OP_ADAPTER(SigmoidCrossEntropyWithLogits)
DECLARE_OP_USE_OUTPUT(SigmoidCrossEntropyWithLogits)

DECLARE_OP_ADAPTER(SigmoidCrossEntropyWithLogitsGrad)
DECLARE_OP_USE_OUTPUT(SigmoidCrossEntropyWithLogitsGrad)

DECLARE_OP_ADAPTER(SigmoidCrossEntropyWithLogitsV2)
DECLARE_OP_USE_OUTPUT(SigmoidCrossEntropyWithLogitsV2)

DECLARE_OP_ADAPTER(LogSoftmaxGrad)
DECLARE_OP_USE_OUTPUT(LogSoftmaxGrad)

DECLARE_OP_ADAPTER(LogSoftmaxV2)
DECLARE_OP_USE_OUTPUT(LogSoftmaxV2)

DECLARE_OP_ADAPTER(LayerNorm)
DECLARE_OP_USE_OUTPUT(LayerNorm)

DECLARE_OP_ADAPTER(LayerNormGrad)
DECLARE_OP_USE_OUTPUT(LayerNormGrad)

DECLARE_OP_ADAPTER(DropOutDoMask)
DECLARE_OP_USE_OUTPUT(DropOutDoMask)

DECLARE_OP_ADAPTER(SoftmaxCrossEntropyWithLogits)
DECLARE_OP_USE_OUTPUT(SoftmaxCrossEntropyWithLogits)

DECLARE_OP_ADAPTER(SoftmaxV2)
DECLARE_OP_USE_OUTPUT(SoftmaxV2)

DECLARE_OP_ADAPTER(SoftmaxGrad)
DECLARE_OP_USE_OUTPUT(SoftmaxGrad)

DECLARE_OP_ADAPTER(BinaryCrossEntropy)
DECLARE_OP_USE_OUTPUT(BinaryCrossEntropy)

DECLARE_OP_ADAPTER(BinaryCrossEntropyGrad)
DECLARE_OP_USE_OUTPUT(BinaryCrossEntropyGrad)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_IMAGE_OPS_DECLARE_H_
