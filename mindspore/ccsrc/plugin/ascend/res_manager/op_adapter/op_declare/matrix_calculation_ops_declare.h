/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MATRIX_CALCULATION_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MATRIX_CALCULATION_OPS_DECLARE_H_

#include "include/utils/utils.h"
#include "plugin/ascend/res_manager/op_adapter/custom_op_proto/cust_math_ops.h"
#include "plugin/ascend/res_manager/op_adapter/custom_op_proto/cust_matrix_ops.h"
#include "plugin/ascend/res_manager/op_adapter/op_declare/op_declare_macro.h"

DECLARE_OP_ADAPTER(TensorScatterUpdate)
DECLARE_OP_USE_OUTPUT(TensorScatterUpdate)

DECLARE_OP_ADAPTER(ScatterUpdate)
DECLARE_OP_USE_OUTPUT(ScatterUpdate)

DECLARE_OP_ADAPTER(ScatterNdUpdate)
DECLARE_OP_USE_OUTPUT(ScatterNdUpdate)

DECLARE_OP_ADAPTER(ScatterNdMax)
DECLARE_OP_USE_OUTPUT(ScatterNdMax)

DECLARE_OP_ADAPTER(ScatterNdMin)
DECLARE_OP_USE_OUTPUT(ScatterNdMin)

DECLARE_OP_ADAPTER(ScatterMax)
DECLARE_OP_USE_OUTPUT(ScatterMax)

DECLARE_OP_ADAPTER(ScatterMin)
DECLARE_OP_USE_OUTPUT(ScatterMin)

DECLARE_OP_ADAPTER(ScatterAdd)
DECLARE_OP_USE_OUTPUT(ScatterAdd)

DECLARE_OP_ADAPTER(ScatterSub)
DECLARE_OP_USE_OUTPUT(ScatterSub)

DECLARE_OP_ADAPTER(ScatterMul)
DECLARE_OP_USE_OUTPUT(ScatterMul)

DECLARE_OP_ADAPTER(ScatterDiv)
DECLARE_OP_USE_OUTPUT(ScatterDiv)

DECLARE_OP_ADAPTER(ScatterNdAdd)
DECLARE_OP_USE_OUTPUT(ScatterNdAdd)

DECLARE_OP_ADAPTER(ScatterNdSub)
DECLARE_OP_USE_OUTPUT(ScatterNdSub)

DECLARE_OP_ADAPTER(BatchMatMul)
DECLARE_OP_USE_OUTPUT(BatchMatMul)

DECLARE_OP_ADAPTER(BatchMatMulV2)
DECLARE_OP_USE_OUTPUT(BatchMatMulV2)

DECLARE_OP_ADAPTER(MatMul)
DECLARE_OP_USE_OUTPUT(MatMul)

DECLARE_OP_ADAPTER(MatMulV2)
DECLARE_OP_USE_OUTPUT(MatMulV2)

DECLARE_OP_ADAPTER(MatrixDiagD)
DECLARE_OP_USE_OUTPUT(MatrixDiagD)

DECLARE_OP_ADAPTER(MatrixDiagPartD)
DECLARE_OP_USE_OUTPUT(MatrixDiagPartD)

DECLARE_OP_ADAPTER(MatrixDiagPartV3)
DECLARE_OP_USE_OUTPUT(MatrixDiagPartV3)

DECLARE_OP_ADAPTER(MatrixSetDiagD)
DECLARE_OP_USE_OUTPUT(MatrixSetDiagD)

DECLARE_OP_ADAPTER(DiagPart)
DECLARE_OP_USE_OUTPUT(DiagPart)

DECLARE_OP_ADAPTER(DiagPartD)
DECLARE_OP_USE_OUTPUT(DiagPartD)

DECLARE_OP_ADAPTER(L2Loss)
DECLARE_OP_USE_OUTPUT(L2Loss)

DECLARE_OP_ADAPTER(ScatterElements)
DECLARE_OP_USE_OUTPUT(ScatterElements)

DECLARE_OP_ADAPTER(Scatter)
DECLARE_OP_USE_OUTPUT(Scatter)

DECLARE_OP_ADAPTER(FullyConnection)
DECLARE_OP_USE_OUTPUT(FullyConnection)

DECLARE_OP_ADAPTER(IndexAdd)
DECLARE_OP_USE_OUTPUT(IndexAdd)

DECLARE_OP_ADAPTER(ConfusionMatrix)
DECLARE_OP_USE_OUTPUT(ConfusionMatrix)

DECLARE_OP_ADAPTER(MatrixSetDiagV3)
DECLARE_OP_USE_OUTPUT(MatrixSetDiagV3)

DECLARE_OP_ADAPTER(TensorScatterAdd)
DECLARE_OP_USE_OUTPUT(TensorScatterAdd)

DECLARE_OP_ADAPTER(TensorScatterSub)
DECLARE_OP_USE_OUTPUT(TensorScatterSub)

DECLARE_OP_ADAPTER(TensorScatterMin)
DECLARE_OP_USE_OUTPUT(TensorScatterMin)

DECLARE_OP_ADAPTER(TensorScatterMax)
DECLARE_OP_USE_OUTPUT(TensorScatterMax)

DECLARE_OP_ADAPTER(Triu)
DECLARE_OP_USE_OUTPUT(Triu)

DECLARE_OP_ADAPTER(MatrixDiagV3)
DECLARE_OP_USE_OUTPUT(MatrixDiagV3)

DECLARE_OP_ADAPTER(Tril)
DECLARE_OP_USE_OUTPUT(Tril)

DECLARE_OP_ADAPTER(Eye)
DECLARE_OP_USE_OUTPUT(Eye)

DECLARE_OP_ADAPTER(FillDiagonal)
DECLARE_OP_USE_OUTPUT(FillDiagonal)

DECLARE_OP_ADAPTER(Trace)
DECLARE_OP_USE_OUTPUT(Trace)

DECLARE_CUST_OP_ADAPTER(TraceGrad)
DECLARE_CUST_OP_USE_OUTPUT(TraceGrad)

DECLARE_CUST_OP_ADAPTER(TraceV2)
DECLARE_CUST_OP_USE_OUTPUT(TraceV2)

DECLARE_CUST_OP_ADAPTER(TraceV2Grad)
DECLARE_CUST_OP_USE_OUTPUT(TraceV2Grad)

DECLARE_OP_ADAPTER(SwinAttentionFFN)
DECLARE_OP_USE_OUTPUT(SwinAttentionFFN)

DECLARE_OP_ADAPTER(SwinTransformerLnQKV)
DECLARE_OP_USE_OUTPUT(SwinTransformerLnQKV)

DECLARE_OP_ADAPTER(SwinAttentionScore)
DECLARE_OP_USE_OUTPUT(SwinAttentionScore)

DECLARE_OP_ADAPTER(SparseTensorDenseMatMul)
DECLARE_OP_USE_OUTPUT(SparseTensorDenseMatMul)

DECLARE_CUST_OP_ADAPTER(IndexPut)
DECLARE_CUST_OP_USE_DYN_INPUT(IndexPut)
DECLARE_CUST_OP_USE_OUTPUT(IndexPut)
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MATRIX_CALCULATION_OPS_DECLARE_H_
