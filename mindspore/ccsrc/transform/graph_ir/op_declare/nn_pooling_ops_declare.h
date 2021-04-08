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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_POOLING_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_POOLING_OPS_DECLARE_H_

#include <string>
#include <unordered_map>
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/nn_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(MaxPoolWithArgmax)
DECLARE_OP_USE_OUTPUT(MaxPoolWithArgmax)

DECLARE_OP_ADAPTER(MaxPoolGradWithArgmax)
DECLARE_OP_USE_OUTPUT(MaxPoolGradWithArgmax)

DECLARE_OP_ADAPTER(MaxPool)
DECLARE_OP_USE_OUTPUT(MaxPool)

DECLARE_OP_ADAPTER(MaxPoolGrad)
DECLARE_OP_USE_OUTPUT(MaxPoolGrad)

DECLARE_OP_ADAPTER(MaxPool3D)
DECLARE_OP_USE_OUTPUT(MaxPool3D)

DECLARE_OP_ADAPTER(MaxPool3DGrad)
DECLARE_OP_USE_OUTPUT(MaxPool3DGrad)

DECLARE_OP_ADAPTER(AvgPool)
DECLARE_OP_USE_OUTPUT(AvgPool)

DECLARE_OP_ADAPTER(AvgPoolGrad)
DECLARE_OP_USE_OUTPUT(AvgPoolGrad)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_POOLING_OPS_DECLARE_H_
