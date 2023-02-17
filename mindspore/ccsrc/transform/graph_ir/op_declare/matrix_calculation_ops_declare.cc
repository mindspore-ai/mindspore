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

#include "transform/graph_ir/op_declare/matrix_calculation_ops_declare.h"
#include <string>

namespace mindspore::transform {
// TensorScatterUpdate
INPUT_MAP(TensorScatterUpdate) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(TensorScatterUpdate) = EMPTY_ATTR_MAP;
OUTPUT_MAP(TensorScatterUpdate) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(TensorScatterUpdate, kNameTensorScatterUpdate, ADPT_DESC(TensorScatterUpdate))

// ScatterUpdate
INPUT_MAP(ScatterUpdate) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterUpdate) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ScatterUpdate) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(ScatterUpdate, kNameScatterUpdate, ADPT_DESC(ScatterUpdate))

// ScatterNdUpdate
INPUT_MAP(ScatterNdUpdate) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterNdUpdate) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ScatterNdUpdate) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(ScatterNdUpdate, kNameScatterNdUpdate, ADPT_DESC(ScatterNdUpdate))

// ScatterMax
INPUT_MAP(ScatterMax) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterMax) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ScatterMax) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(ScatterMax, kNameScatterMax, ADPT_DESC(ScatterMax))

// ScatterMin
INPUT_MAP(ScatterMin) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterMin) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ScatterMin) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(ScatterMin, kNameScatterMin, ADPT_DESC(ScatterMin))

// ScatterAdd
INPUT_MAP(ScatterAdd) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterAdd) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ScatterAdd) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(ScatterAdd, kNameScatterAdd, ADPT_DESC(ScatterAdd))

// ScatterSub
INPUT_MAP(ScatterSub) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterSub) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ScatterSub) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(ScatterSub, kNameScatterSub, ADPT_DESC(ScatterSub))

// ScatterMul
INPUT_MAP(ScatterMul) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterMul) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ScatterMul) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(ScatterMul, kNameScatterMul, ADPT_DESC(ScatterMul))

// ScatterDiv
INPUT_MAP(ScatterDiv) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterDiv) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ScatterDiv) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(ScatterDiv, kNameScatterDiv, ADPT_DESC(ScatterDiv))

// ScatterNdAdd
INPUT_MAP(ScatterNdAdd) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterNdAdd) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ScatterNdAdd) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(ScatterNdAdd, kNameScatterNdAdd, ADPT_DESC(ScatterNdAdd))

// ScatterNdSub
INPUT_MAP(ScatterNdSub) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterNdSub) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ScatterNdSub) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(ScatterNdSub, kNameScatterNdSub, ADPT_DESC(ScatterNdSub))

// MatMul
INPUT_MAP(MatMul) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(bias)}};
ATTR_MAP(MatMul) = {{"transpose_x1", ATTR_DESC(transpose_x1, AnyTraits<bool>())},
                    {"transpose_x2", ATTR_DESC(transpose_x2, AnyTraits<bool>())}};
OUTPUT_MAP(MatMul) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatMul, kNameMatMul, ADPT_DESC(MatMul))

// MatMulV2
INPUT_MAP(MatMulV2) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(bias)}};
ATTR_MAP(MatMulV2) = {{"transpose_a", ATTR_DESC(transpose_x1, AnyTraits<bool>())},
                      {"transpose_b", ATTR_DESC(transpose_x2, AnyTraits<bool>())}};
OUTPUT_MAP(MatMulV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatMulV2, prim::kPrimMatMul->name(), ADPT_DESC(MatMulV2))
REG_ADPT_DESC(MatMulV2Duplicate, prim::kPrimMatMulV2->name(), ADPT_DESC(MatMulV2))

// MatrixDiag
INPUT_MAP(MatrixDiag) = {{1, INPUT_DESC(x)}};
ATTR_MAP(MatrixDiag) = EMPTY_ATTR_MAP;
OUTPUT_MAP(MatrixDiag) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixDiag, kNameMatrixDiag, ADPT_DESC(MatrixDiag))
REG_ADPT_DESC(MatrixDiagD, kMatrixDiagDOpName, ADPT_DESC(MatrixDiag))

// MatrixDiagPartD
INPUT_MAP(MatrixDiagPartD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(assist)}};
ATTR_MAP(MatrixDiagPartD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(MatrixDiagPartD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixDiagPartD, kNameMatrixDiagPartD, ADPT_DESC(MatrixDiagPartD))

// MatrixSetDiagD
INPUT_MAP(MatrixSetDiagD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(diagonal)}, {3, INPUT_DESC(assist)}};
ATTR_MAP(MatrixSetDiagD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(MatrixSetDiagD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixSetDiagD, kNameMatrixSetDiagD, ADPT_DESC(MatrixSetDiagD))

// MatrixDiagPart
INPUT_MAP(MatrixDiagPart) = {{1, INPUT_DESC(x)}};
ATTR_MAP(MatrixDiagPart) = EMPTY_ATTR_MAP;
OUTPUT_MAP(MatrixDiagPart) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixDiagPart, kMatrixDiagPartDOpName, ADPT_DESC(MatrixDiagPart))

// MatrixSetDiag
INPUT_MAP(MatrixSetDiag) = {{1, INPUT_DESC(x)}};
ATTR_MAP(MatrixSetDiag) = EMPTY_ATTR_MAP;
OUTPUT_MAP(MatrixSetDiag) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixSetDiag, kMatrixSetDiagDOpName, ADPT_DESC(MatrixSetDiag))

// ConfusionMatrix
INPUT_MAP(ConfusionMatrix) = {{1, INPUT_DESC(labels)}, {2, INPUT_DESC(predictions)}, {3, INPUT_DESC(weights)}};
ATTR_MAP(ConfusionMatrix) = {{"num_classes", ATTR_DESC(num_classes, AnyTraits<int64_t>())},
                             {"dtype", ATTR_DESC(dtype, AnyTraits<std::string>())}};
OUTPUT_MAP(ConfusionMatrix) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ConfusionMatrix, kNameConfusionMatrix, ADPT_DESC(ConfusionMatrix))

// DiagPart
INPUT_MAP(DiagPart) = {{1, INPUT_DESC(x)}};
ATTR_MAP(DiagPart) = EMPTY_ATTR_MAP;
OUTPUT_MAP(DiagPart) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DiagPart, kNameDiagPart, ADPT_DESC(DiagPart))

// DiagPartD
INPUT_MAP(DiagPartD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(assist)}};
ATTR_MAP(DiagPartD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(DiagPartD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DiagPartD, kNameDiagPartD, ADPT_DESC(DiagPartD))

// BatchMatMul
INPUT_MAP(BatchMatMul) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(BatchMatMul) = {{"transpose_x1", ATTR_DESC(adj_x1, AnyTraits<bool>())},
                         {"transpose_x2", ATTR_DESC(adj_x2, AnyTraits<bool>())}};
OUTPUT_MAP(BatchMatMul) = {{0, OUTPUT_DESC(y)}};

// BatchMatMul->BatchMatMulV2
INPUT_MAP(BatchMatMulV2) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(BatchMatMulV2) = {{"transpose_x1", ATTR_DESC(adj_x1, AnyTraits<bool>())},
                           {"transpose_x2", ATTR_DESC(adj_x2, AnyTraits<bool>())}};
OUTPUT_MAP(BatchMatMulV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(BatchMatMul, kNameBatchMatMul, ADPT_DESC(BatchMatMul))
REG_ADPT_DESC(BatchMatMulV2, kNameBatchMatMulV2, ADPT_DESC(BatchMatMulV2))

// L2Loss
INPUT_MAP(L2Loss) = {{1, INPUT_DESC(x)}};
ATTR_MAP(L2Loss) = EMPTY_ATTR_MAP;
OUTPUT_MAP(L2Loss) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(L2Loss, kNameL2Loss, ADPT_DESC(L2Loss))

// ScatterElements
INPUT_MAP(ScatterElements) = {{1, INPUT_DESC(data)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterElements) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>())}};
OUTPUT_MAP(ScatterElements) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(TensorScatterElements, kNameTensorScatterElements, ADPT_DESC(ScatterElements))
REG_ADPT_DESC(ScatterElements, kNameScatterElements, ADPT_DESC(ScatterElements))

// FullyConnection
INPUT_MAP(FullyConnection) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(w)}, {3, INPUT_DESC(b)}, {4, INPUT_DESC(offset_w)}};

ATTR_MAP(FullyConnection) = {{"num_output", ATTR_DESC(num_output, AnyTraits<int64_t>())},
                             {"transpose", ATTR_DESC(transpose, AnyTraits<bool>())},
                             {"axis", ATTR_DESC(axis, AnyTraits<int64_t>())},
                             {"offset_x", ATTR_DESC(offset_x, AnyTraits<int64_t>())}};

OUTPUT_MAP(FullyConnection) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(FullyConnection, kNameFullConnection, ADPT_DESC(FullyConnection))

// IndexAdd
INPUT_MAP(IndexAdd) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(IndexAdd) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>())}};
OUTPUT_MAP(IndexAdd) = {{0, OUTPUT_DESC(var_out)}};

// TensorScatterAdd
INPUT_MAP(TensorScatterAdd) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(TensorScatterAdd) = EMPTY_ATTR_MAP;
OUTPUT_MAP(TensorScatterAdd) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(TensorScatterAdd, kNameTensorScatterAdd, ADPT_DESC(TensorScatterAdd))
}  // namespace mindspore::transform
