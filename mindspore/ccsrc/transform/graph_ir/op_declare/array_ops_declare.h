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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_ARRAY_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_ARRAY_OPS_DECLARE_H_

#include "utils/hash_map.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/array_ops.h"
#include "ops/selection_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(Shape)
DECLARE_OP_USE_OUTPUT(Shape)

DECLARE_OP_ADAPTER(Reshape)
DECLARE_OP_USE_OUTPUT(Reshape)

DECLARE_OP_ADAPTER(GetShape)
DECLARE_OP_USE_DYN_INPUT(GetShape)
DECLARE_OP_USE_OUTPUT(GetShape)

DECLARE_OP_ADAPTER(TransShape)
DECLARE_OP_USE_INPUT_ATTR(TransShape)
DECLARE_OP_USE_OUTPUT(TransShape)

DECLARE_OP_ADAPTER(MirrorPad)
DECLARE_OP_USE_OUTPUT(MirrorPad)

DECLARE_OP_ADAPTER(MirrorPadGrad)
DECLARE_OP_USE_OUTPUT(MirrorPadGrad)

DECLARE_OP_ADAPTER(Expand)
DECLARE_OP_USE_OUTPUT(Expand)

DECLARE_OP_ADAPTER(ExpandDims)
DECLARE_OP_USE_OUTPUT(ExpandDims)

DECLARE_OP_ADAPTER(Squeeze)
DECLARE_OP_USE_OUTPUT(Squeeze)

DECLARE_OP_ADAPTER(SqueezeV3)
DECLARE_OP_USE_OUTPUT(SqueezeV3)

DECLARE_OP_ADAPTER(Constant)
DECLARE_OP_USE_OUTPUT(Constant)

DECLARE_OP_ADAPTER(Summary)

DECLARE_OP_ADAPTER(Const)
DECLARE_OP_USE_OUTPUT(Const)

DECLARE_OP_ADAPTER(Data)
DECLARE_OP_USE_OUTPUT(Data)

DECLARE_OP_ADAPTER(ReverseSequence)
DECLARE_OP_USE_OUTPUT(ReverseSequence)

DECLARE_OP_ADAPTER(EditDistance)
DECLARE_OP_USE_OUTPUT(EditDistance)

DECLARE_OP_ADAPTER(NonZeroWithValue)
DECLARE_OP_USE_OUTPUT(NonZeroWithValue)

DECLARE_OP_ADAPTER(NonZeroWithValueShape)
DECLARE_OP_USE_OUTPUT(NonZeroWithValueShape)

DECLARE_OP_ADAPTER(Unsqueeze)
DECLARE_OP_USE_OUTPUT(Unsqueeze)

DECLARE_OP_ADAPTER(Identity)
DECLARE_OP_USE_OUTPUT(Identity)

DECLARE_OP_ADAPTER(IdentityN)
DECLARE_OP_USE_DYN_OUTPUT(IdentityN)

DECLARE_OP_ADAPTER(SelectV2)
DECLARE_OP_USE_OUTPUT(SelectV2)

DECLARE_OP_ADAPTER(Unique)
DECLARE_OP_USE_OUTPUT(Unique)

DECLARE_OP_ADAPTER(BroadcastGradientArgs)
DECLARE_OP_USE_OUTPUT(BroadcastGradientArgs)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_ARRAY_OPS_DECLARE_H_
