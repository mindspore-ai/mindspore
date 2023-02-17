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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MATRIX_CALCULATION_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MATRIX_CALCULATION_OPS_DECLARE_H_

#include "utils/hash_map.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "ops/matrix_calculation_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(TensorScatterUpdate)
DECLARE_OP_USE_OUTPUT(TensorScatterUpdate)

DECLARE_OP_ADAPTER(ScatterUpdate)
DECLARE_OP_USE_OUTPUT(ScatterUpdate)

DECLARE_OP_ADAPTER(ScatterNdUpdate)
DECLARE_OP_USE_OUTPUT(ScatterNdUpdate)

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

DECLARE_OP_ADAPTER(MatrixDiag)
DECLARE_OP_USE_OUTPUT(MatrixDiag)

DECLARE_OP_ADAPTER(MatrixDiagPartD)
DECLARE_OP_USE_OUTPUT(MatrixDiagPartD)

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

DECLARE_OP_ADAPTER(FullyConnection)
DECLARE_OP_USE_OUTPUT(FullyConnection)

DECLARE_OP_ADAPTER(IndexAdd)
DECLARE_OP_USE_OUTPUT(IndexAdd)

DECLARE_OP_ADAPTER(ConfusionMatrix)
DECLARE_OP_USE_OUTPUT(ConfusionMatrix)

DECLARE_OP_ADAPTER(MatrixDiagPart)
DECLARE_OP_USE_OUTPUT(MatrixDiagPart)

DECLARE_OP_ADAPTER(MatrixSetDiag)
DECLARE_OP_USE_OUTPUT(MatrixSetDiag)

DECLARE_OP_ADAPTER(TensorScatterAdd)
DECLARE_OP_USE_OUTPUT(TensorScatterAdd)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MATRIX_CALCULATION_OPS_DECLARE_H_
