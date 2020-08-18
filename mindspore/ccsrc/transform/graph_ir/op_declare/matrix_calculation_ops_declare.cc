/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

// MatMulV2
INPUT_MAP(MatMulV2) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(bias)}};
ATTR_MAP(MatMulV2) = {{"transpose_a", ATTR_DESC(transpose_x1, AnyTraits<bool>())},
                      {"transpose_b", ATTR_DESC(transpose_x2, AnyTraits<bool>())}};
OUTPUT_MAP(MatMulV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatMulV2, prim::kPrimMatMul->name(), ADPT_DESC(MatMulV2))

// DiagPart
INPUT_MAP(DiagPart) = {{1, INPUT_DESC(x)}};
ATTR_MAP(DiagPart) = EMPTY_ATTR_MAP;
OUTPUT_MAP(DiagPart) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DiagPart, kNameDiagPart, ADPT_DESC(DiagPart))

// BatchMatMul
INPUT_MAP(BatchMatMul) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(BatchMatMul) = {{"transpose_x1", ATTR_DESC(adj_x1, AnyTraits<bool>())},
                         {"transpose_x2", ATTR_DESC(adj_x2, AnyTraits<bool>())}};
OUTPUT_MAP(BatchMatMul) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(BatchMatMul, kNameBatchMatMul, ADPT_DESC(BatchMatMul))

// L2Loss
INPUT_MAP(L2Loss) = {{1, INPUT_DESC(x)}};
ATTR_MAP(L2Loss) = EMPTY_ATTR_MAP;
OUTPUT_MAP(L2Loss) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(L2Loss, kNameL2Loss, ADPT_DESC(L2Loss))
}  // namespace mindspore::transform
