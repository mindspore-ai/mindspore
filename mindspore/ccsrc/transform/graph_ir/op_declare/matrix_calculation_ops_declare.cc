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
#include <vector>
#include "ops/array_op_name.h"
#include "ops/ascend_op_name.h"
#include "ops/math_op_name.h"
#include "ops/math_ops.h"

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

// ScatterNdMax
INPUT_MAP(ScatterNdMax) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterNdMax) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ScatterNdMax) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(ScatterNdMax, kScatterNdMaxOpName, ADPT_DESC(ScatterNdMax))

// ScatterNdMin
INPUT_MAP(ScatterNdMin) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterNdMin) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ScatterNdMin) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(ScatterNdMin, kScatterNdMinOpName, ADPT_DESC(ScatterNdMin))

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
INPUT_MAP(MatMul) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(MatMul) = {{"transpose_x1", ATTR_DESC(transpose_x1, AnyTraits<bool>())},
                    {"transpose_x2", ATTR_DESC(transpose_x2, AnyTraits<bool>())}};
OUTPUT_MAP(MatMul) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatMul, kNameMatMul, ADPT_DESC(MatMulV2))

// MatMulV2
INPUT_MAP(MatMulV2) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(bias)}};
ATTR_MAP(MatMulV2) = {{"transpose_a", ATTR_DESC(transpose_x1, AnyTraits<bool>())},
                      {"transpose_b", ATTR_DESC(transpose_x2, AnyTraits<bool>())}};
OUTPUT_MAP(MatMulV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatMulV2, prim::kPrimMatMul->name(), ADPT_DESC(MatMulV2))
REG_ADPT_DESC(MatMulV2Duplicate, prim::kPrimMatMulV2->name(), ADPT_DESC(MatMulV2))

// MatrixDiagD
INPUT_MAP(MatrixDiagD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(assist)}};
ATTR_MAP(MatrixDiagD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(MatrixDiagD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixDiag, kMatrixDiagOpName, ADPT_DESC(MatrixDiagD))
REG_ADPT_DESC(MatrixDiagD, kMatrixDiagDOpName, ADPT_DESC(MatrixDiagD))

// MatrixDiagPartD
INPUT_MAP(MatrixDiagPartD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(assist)}};
ATTR_MAP(MatrixDiagPartD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(MatrixDiagPartD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixDiagPartD, kMatrixDiagPartDOpName, ADPT_DESC(MatrixDiagPartD))
REG_ADPT_DESC(MatrixDiagPart, kMatrixDiagPartOpName, ADPT_DESC(MatrixDiagPartD))

// MatrixSetDiagD
INPUT_MAP(MatrixSetDiagD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(diagonal)}, {3, INPUT_DESC(assist)}};
ATTR_MAP(MatrixSetDiagD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(MatrixSetDiagD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixSetDiagD, kNameMatrixSetDiagD, ADPT_DESC(MatrixSetDiagD))
REG_ADPT_DESC(MatrixSetDiag, kMatrixSetDiagOpName, ADPT_DESC(MatrixSetDiagD))

// MatrixDiagPartV3
INPUT_MAP(MatrixDiagPartV3) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(k)}, {3, INPUT_DESC(padding_value)}};
ATTR_MAP(MatrixDiagPartV3) = {{"align", ATTR_DESC(align, AnyTraits<std::string>())}};
OUTPUT_MAP(MatrixDiagPartV3) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixDiagPartV3, kMatrixDiagPartV3OpName, ADPT_DESC(MatrixDiagPartV3))

// MatrixSetDiagV3
INPUT_MAP(MatrixSetDiagV3) = {{1, INPUT_DESC(input)}, {2, INPUT_DESC(diagonal)}, {3, INPUT_DESC(k)}};
ATTR_MAP(MatrixSetDiagV3) = {{"align", ATTR_DESC(align, AnyTraits<std::string>())}};
OUTPUT_MAP(MatrixSetDiagV3) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(MatrixSetDiagV3, kMatrixSetDiagV3OpName, ADPT_DESC(MatrixSetDiagV3))

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
ATTR_MAP(BatchMatMul) = {{"transpose_a", ATTR_DESC(adj_x1, AnyTraits<bool>())},
                         {"transpose_b", ATTR_DESC(adj_x2, AnyTraits<bool>())}};
OUTPUT_MAP(BatchMatMul) = {{0, OUTPUT_DESC(y)}};

// BatchMatMul->BatchMatMulV2
INPUT_MAP(BatchMatMulV2) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(bias)}};
ATTR_MAP(BatchMatMulV2) = {{"transpose_a", ATTR_DESC(adj_x1, AnyTraits<bool>())},
                           {"transpose_b", ATTR_DESC(adj_x2, AnyTraits<bool>())}};
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

// TensorScatterSub
INPUT_MAP(TensorScatterSub) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(TensorScatterSub) = EMPTY_ATTR_MAP;
OUTPUT_MAP(TensorScatterSub) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(TensorScatterSub, kNameTensorScatterSub, ADPT_DESC(TensorScatterSub))

// TensorScatterMin
INPUT_MAP(TensorScatterMin) = {{1, INPUT_DESC(input)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(TensorScatterMin) = EMPTY_ATTR_MAP;
OUTPUT_MAP(TensorScatterMin) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(TensorScatterMin, kNameTensorScatterMin, ADPT_DESC(TensorScatterMin))

// TensorScatterMax
INPUT_MAP(TensorScatterMax) = {{1, INPUT_DESC(input)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(TensorScatterMax) = EMPTY_ATTR_MAP;
OUTPUT_MAP(TensorScatterMax) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(TensorScatterMax, kNameTensorScatterMax, ADPT_DESC(TensorScatterMax))

// Triu
INPUT_MAP(Triu) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Triu) = {{"diagonal", ATTR_DESC(diagonal, AnyTraits<int64_t>())}};
OUTPUT_MAP(Triu) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Triu, kNameTriu, ADPT_DESC(Triu))

INPUT_MAP(MatrixDiagV3) = {{1, INPUT_DESC(x)},
                           {2, INPUT_DESC(k)},
                           {3, INPUT_DESC(num_rows)},
                           {4, INPUT_DESC(num_cols)},
                           {5, INPUT_DESC(padding_value)}};
ATTR_MAP(MatrixDiagV3) = {{"align", ATTR_DESC(align, AnyTraits<std::string>())}};
OUTPUT_MAP(MatrixDiagV3) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixDiagV3, kNameMatrixDiagV3, ADPT_DESC(MatrixDiagV3))

// Tril
INPUT_MAP(Tril) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Tril) = {{"diagonal", ATTR_DESC(diagonal, AnyTraits<int64_t>())}};
OUTPUT_MAP(Tril) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Tril, kNameTril, ADPT_DESC(Tril))

// Eye
INPUT_MAP(Eye) = EMPTY_INPUT_MAP;
ATTR_MAP(Eye) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(Eye) = {{1, ATTR_DESC(num_rows, AnyTraits<int64_t>())},
                       {2, ATTR_DESC(num_columns, AnyTraits<int64_t>())},
                       {3, ATTR_DESC(dtype, AnyTraits<GEType>())}};
OUTPUT_MAP(Eye) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Eye, kNameEye, ADPT_DESC(Eye));

// FillDiagonal
INPUT_MAP(FillDiagonal) = {{1, INPUT_DESC(x)}};
ATTR_MAP(FillDiagonal) = {{"fill_value", ATTR_DESC(fill_value, AnyTraits<float>())},
                          {"wrap", ATTR_DESC(wrap, AnyTraits<bool>())}};
OUTPUT_MAP(FillDiagonal) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(FillDiagonal, kNameFillDiagonal, ADPT_DESC(FillDiagonal));

// Trace
INPUT_MAP(Trace) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Trace) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Trace) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Trace, prim::kPrimTrace->name(), ADPT_DESC(Trace));

// TraceGrad
CUST_INPUT_MAP(TraceGrad) = {{1, INPUT_DESC(y_grad)}, {2, INPUT_DESC(x_shape)}};
CUST_ATTR_MAP(TraceGrad) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(TraceGrad) = {{0, OUTPUT_DESC(x_grad)}};
REG_ADPT_DESC(TraceGrad, prim::kPrimTraceGrad->name(), CUST_ADPT_DESC(TraceGrad));

// SwinAttentionFFN
INPUT_MAP(SwinAttentionFFN) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(bias)}, {4, INPUT_DESC(x3)}};
ATTR_MAP(SwinAttentionFFN) = {{"shifts", ATTR_DESC(shifts, AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(SwinAttentionFFN) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(SwinAttentionFFN, kNameSwinAttentionFFN, ADPT_DESC(SwinAttentionFFN));

// SwinTransformerLnQKV
INPUT_MAP(SwinTransformerLnQKV) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(gamma)}, {3, INPUT_DESC(beta)}, {4, INPUT_DESC(weight)}, {5, INPUT_DESC(bias)}};
ATTR_MAP(SwinTransformerLnQKV) = {{"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())},
                                  {"head_dim", ATTR_DESC(head_dim, AnyTraits<int64_t>())},
                                  {"head_num", ATTR_DESC(head_num, AnyTraits<int64_t>())},
                                  {"seq_length", ATTR_DESC(seq_length, AnyTraits<int64_t>())},
                                  {"shifts", ATTR_DESC(shifts, AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(SwinTransformerLnQKV) = {
  {0, OUTPUT_DESC(query_output)}, {1, OUTPUT_DESC(key_output)}, {2, OUTPUT_DESC(value_output)}};
REG_ADPT_DESC(SwinTransformerLnQKV, kNameSwinTransformerLnQKV, ADPT_DESC(SwinTransformerLnQKV));

// SwinAttentionScore
INPUT_MAP(SwinAttentionScore) = {{1, INPUT_DESC(query)},         {2, INPUT_DESC(key)},           {3, INPUT_DESC(value)},
                                 {4, INPUT_DESC(padding_mask1)}, {5, INPUT_DESC(padding_mask2)}, {6, INPUT_DESC(scale)},
                                 {7, INPUT_DESC(drop_mask)}};
ATTR_MAP(SwinAttentionScore) = {{"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())},
                                {"query_transpose", ATTR_DESC(query_transpose, AnyTraits<bool>())},
                                {"key_transpose", ATTR_DESC(key_transpose, AnyTraits<bool>())},
                                {"bmm_score_transpose_a", ATTR_DESC(bmm_score_transpose_a, AnyTraits<bool>())},
                                {"bmm_score_transpose_b", ATTR_DESC(bmm_score_transpose_b, AnyTraits<bool>())},
                                {"softmax_axes", ATTR_DESC(softmax_axes, AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(SwinAttentionScore) = {{0, OUTPUT_DESC(attention_score)}, {1, OUTPUT_DESC(softmax)}};
REG_ADPT_DESC(SwinAttentionScore, kNameSwinAttentionScore, ADPT_DESC(SwinAttentionScore));

// SparseTensorDenseMatMul
INPUT_MAP(SparseTensorDenseMatMul) = {
  {1, INPUT_DESC(x1_indices)}, {2, INPUT_DESC(x1_values)}, {3, INPUT_DESC(x1_shape)}, {4, INPUT_DESC(x2)}};
ATTR_MAP(SparseTensorDenseMatMul) = {{"adjoint_a", ATTR_DESC(adjoint_a, AnyTraits<bool>())},
                                     {"adjoint_b", ATTR_DESC(adjoint_b, AnyTraits<bool>())}};
OUTPUT_MAP(SparseTensorDenseMatMul) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(SparseTensorDenseMatMul, kNameSparseTensorDenseMatmul, ADPT_DESC(SparseTensorDenseMatMul));
}  // namespace mindspore::transform
