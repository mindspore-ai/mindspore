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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_TRANSFORMATION_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_TRANSFORMATION_OPS_DECLARE_H_

#include "utils/hash_map.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/transformation_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(ExtractImagePatches)
DECLARE_OP_USE_OUTPUT(ExtractImagePatches)

DECLARE_OP_ADAPTER(Unpack)
DECLARE_OP_USE_DYN_OUTPUT(Unpack)

DECLARE_OP_ADAPTER(TransposeD)
DECLARE_OP_USE_INPUT_ATTR(TransposeD)

DECLARE_OP_ADAPTER(Transpose)

DECLARE_OP_ADAPTER(TransData)
DECLARE_OP_USE_OUTPUT(TransData)

DECLARE_OP_ADAPTER(Flatten)
DECLARE_OP_USE_OUTPUT(Flatten)

DECLARE_OP_ADAPTER(SpaceToDepth)
DECLARE_OP_USE_OUTPUT(SpaceToDepth)

DECLARE_OP_ADAPTER(DepthToSpace)
DECLARE_OP_USE_OUTPUT(DepthToSpace)

DECLARE_OP_ADAPTER(SpaceToBatchD)
DECLARE_OP_USE_OUTPUT(SpaceToBatchD)

DECLARE_OP_ADAPTER(SpaceToBatchND)
DECLARE_OP_USE_OUTPUT(SpaceToBatchND)

DECLARE_OP_ADAPTER(BatchToSpaceD)
DECLARE_OP_USE_OUTPUT(BatchToSpaceD)

DECLARE_OP_ADAPTER(BatchToSpaceND)
DECLARE_OP_USE_OUTPUT(BatchToSpaceND)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_TRANSFORMATION_OPS_DECLARE_H_
