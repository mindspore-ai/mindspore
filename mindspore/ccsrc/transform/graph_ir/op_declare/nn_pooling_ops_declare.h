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

#include "utils/hash_map.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "ops/nn_ops.h"
#include "ops/nn_pooling_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(MaxPoolWithArgmax)
DECLARE_OP_USE_OUTPUT(MaxPoolWithArgmax)

DECLARE_OP_ADAPTER(MaxPoolWithArgmaxV2)
DECLARE_OP_USE_OUTPUT(MaxPoolWithArgmaxV2)

DECLARE_OP_ADAPTER(MaxPoolGradWithArgmax)
DECLARE_OP_USE_OUTPUT(MaxPoolGradWithArgmax)

DECLARE_OP_ADAPTER(MaxPoolGradWithArgmaxV2)
DECLARE_OP_USE_OUTPUT(MaxPoolGradWithArgmaxV2)

DECLARE_OP_ADAPTER(MaxPoolGradGradWithArgmax)
DECLARE_OP_USE_OUTPUT(MaxPoolGradGradWithArgmax)

DECLARE_OP_ADAPTER(MaxPool)
DECLARE_OP_USE_OUTPUT(MaxPool)

DECLARE_OP_ADAPTER(MaxPoolGrad)
DECLARE_OP_USE_OUTPUT(MaxPoolGrad)

DECLARE_OP_ADAPTER(MaxPoolGradGrad)
DECLARE_OP_USE_OUTPUT(MaxPoolGradGrad)

DECLARE_OP_ADAPTER(MaxPool3D)
DECLARE_OP_USE_OUTPUT(MaxPool3D)

DECLARE_OP_ADAPTER(MaxPool3DGrad)
DECLARE_OP_USE_OUTPUT(MaxPool3DGrad)

DECLARE_OP_ADAPTER(MaxPool3DGradGrad)
DECLARE_OP_USE_OUTPUT(MaxPool3DGradGrad)

DECLARE_OP_ADAPTER(AvgPool)
DECLARE_OP_USE_OUTPUT(AvgPool)

DECLARE_OP_ADAPTER(AdaptiveMaxPool2d)
DECLARE_OP_USE_OUTPUT(AdaptiveMaxPool2d)

DECLARE_OP_ADAPTER(AvgPool3D)
DECLARE_OP_USE_OUTPUT(AvgPool3D)

DECLARE_OP_ADAPTER(AvgPool3DD)
DECLARE_OP_USE_OUTPUT(AvgPool3DD)

DECLARE_OP_ADAPTER(AvgPoolGrad)
DECLARE_OP_USE_OUTPUT(AvgPoolGrad)

DECLARE_OP_ADAPTER(Pooling)
DECLARE_OP_USE_OUTPUT(Pooling)

DECLARE_OP_ADAPTER(MaxPoolV3)
DECLARE_OP_USE_OUTPUT(MaxPoolV3)

DECLARE_OP_ADAPTER(AvgPoolV2)
DECLARE_OP_USE_OUTPUT(AvgPoolV2)

DECLARE_OP_ADAPTER(GlobalAveragePool)
DECLARE_OP_USE_OUTPUT(GlobalAveragePool)

DECLARE_OP_ADAPTER(Upsample)
DECLARE_OP_USE_OUTPUT(Upsample)

DECLARE_OP_ADAPTER(AvgPool3DGrad)
DECLARE_OP_USE_OUTPUT(AvgPool3DGrad)

DECLARE_OP_ADAPTER(Dilation2DBackpropFilter)
DECLARE_OP_USE_OUTPUT(Dilation2DBackpropFilter)

DECLARE_OP_ADAPTER(Dilation2DBackpropInput)
DECLARE_OP_USE_OUTPUT(Dilation2DBackpropInput)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_POOLING_OPS_DECLARE_H_
