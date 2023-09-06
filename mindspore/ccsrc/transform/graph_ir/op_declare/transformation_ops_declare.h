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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_TRANSFORMATION_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_TRANSFORMATION_OPS_DECLARE_H_

#include "inc/ops/transformation_ops.h"
#include "transform/graph_ir/custom_op_proto/cust_array_ops.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "utils/hash_map.h"

DECLARE_OP_ADAPTER(ExtractImagePatches)
DECLARE_OP_USE_OUTPUT(ExtractImagePatches)

DECLARE_OP_ADAPTER(Unpack)
DECLARE_OP_USE_DYN_OUTPUT(Unpack)

DECLARE_OP_ADAPTER(TransposeD)
DECLARE_OP_USE_INPUT_ATTR(TransposeD)

DECLARE_OP_ADAPTER(Transpose)
DECLARE_OP_USE_OUTPUT(Transpose)

DECLARE_OP_ADAPTER(TransData)
DECLARE_OP_USE_OUTPUT(TransData)

DECLARE_OP_ADAPTER(TransDataRNN)
DECLARE_OP_USE_OUTPUT(TransDataRNN)

DECLARE_OP_ADAPTER(Flatten)
DECLARE_OP_USE_OUTPUT(Flatten)

DECLARE_OP_ADAPTER(SpaceToDepth)
DECLARE_OP_USE_OUTPUT(SpaceToDepth)

DECLARE_OP_ADAPTER(DepthToSpace)
DECLARE_OP_USE_OUTPUT(DepthToSpace)

DECLARE_OP_ADAPTER(SpaceToBatchD)
DECLARE_OP_USE_OUTPUT(SpaceToBatchD)

DECLARE_OP_ADAPTER(SpaceToBatch)
DECLARE_OP_USE_OUTPUT(SpaceToBatch)

DECLARE_OP_ADAPTER(SpaceToBatchND)
DECLARE_OP_USE_OUTPUT(SpaceToBatchND)

DECLARE_OP_ADAPTER(BatchToSpaceD)
DECLARE_OP_USE_OUTPUT(BatchToSpaceD)

DECLARE_OP_ADAPTER(BatchToSpace)
DECLARE_OP_USE_OUTPUT(BatchToSpace)

DECLARE_OP_ADAPTER(ExtractVolumePatches)
DECLARE_OP_USE_OUTPUT(ExtractVolumePatches)

DECLARE_OP_ADAPTER(BatchToSpaceND)
DECLARE_OP_USE_OUTPUT(BatchToSpaceND)

DECLARE_OP_ADAPTER(TfIdfVectorizer)
DECLARE_OP_USE_OUTPUT(TfIdfVectorizer)

DECLARE_OP_ADAPTER(AffineGrid)
DECLARE_OP_USE_OUTPUT(AffineGrid)

DECLARE_CUST_OP_ADAPTER(AffineGridGrad)
DECLARE_CUST_OP_USE_OUTPUT(AffineGridGrad)

DECLARE_OP_ADAPTER(Im2col)
DECLARE_OP_USE_OUTPUT(Im2col)
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_TRANSFORMATION_OPS_DECLARE_H_
