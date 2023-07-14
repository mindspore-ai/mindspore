/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_LINALG_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_LINALG_OPS_DECLARE_H_

#include "inc/ops/linalg_ops.h"
#include "transform/graph_ir/custom_op_proto/cust_linalg_ops.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "utils/hash_map.h"

DECLARE_OP_ADAPTER(Ger)
DECLARE_OP_USE_OUTPUT(Ger)

DECLARE_OP_ADAPTER(Svd)
DECLARE_OP_USE_OUTPUT(Svd)

DECLARE_OP_ADAPTER(LogMatrixDeterminant)
DECLARE_OP_USE_OUTPUT(LogMatrixDeterminant)

DECLARE_OP_ADAPTER(MatrixInverse)
DECLARE_OP_USE_OUTPUT(MatrixInverse)

DECLARE_OP_ADAPTER(MatrixDeterminant)
DECLARE_OP_USE_OUTPUT(MatrixDeterminant)

DECLARE_OP_ADAPTER(MatrixSolve)
DECLARE_OP_USE_OUTPUT(MatrixSolve)

DECLARE_OP_ADAPTER(CholeskyGrad)
DECLARE_OP_USE_OUTPUT(CholeskyGrad)

DECLARE_OP_ADAPTER(Cholesky)
DECLARE_OP_USE_OUTPUT(Cholesky)

DECLARE_CUST_OP_ADAPTER(Geqrf)
DECLARE_CUST_OP_USE_OUTPUT(Geqrf)

DECLARE_OP_ADAPTER(MatrixTriangularSolve)
DECLARE_OP_USE_OUTPUT(MatrixTriangularSolve)

DECLARE_CUST_OP_ADAPTER(LuUnpack)
DECLARE_CUST_OP_USE_OUTPUT(LuUnpack)

DECLARE_CUST_OP_ADAPTER(LuUnpackGrad)
DECLARE_CUST_OP_USE_OUTPUT(LuUnpackGrad)

DECLARE_CUST_OP_ADAPTER(LuSolve)
DECLARE_CUST_OP_USE_OUTPUT(LuSolve)

DECLARE_OP_ADAPTER(Qr)
DECLARE_OP_USE_OUTPUT(Qr)
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_LINALG_OPS_DECLARE_H_
