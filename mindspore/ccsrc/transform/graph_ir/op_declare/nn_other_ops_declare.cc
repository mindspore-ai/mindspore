/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/nn_other_ops_declare.h"
#include <string>
#include <vector>
#include "ops/nn_ops.h"

namespace mindspore::transform {
// RotaryMul
INPUT_MAP(RotaryMul) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(r1)}, {3, INPUT_DESC(r2)}};
ATTR_MAP(RotaryMul) = EMPTY_ATTR_MAP;
OUTPUT_MAP(RotaryMul) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(RotaryMul, prim::kPrimRotaryMul->name(), ADPT_DESC(RotaryMul))

// RotaryMulGrad
INPUT_MAP(RotaryMulGrad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(r1)}, {3, INPUT_DESC(r2)}, {4, INPUT_DESC(dy)}};
ATTR_MAP(RotaryMulGrad) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(RotaryMulGrad) = {{5, ATTR_DESC(need_backward, AnyTraits<bool>())}};
OUTPUT_MAP(RotaryMulGrad) = {{0, OUTPUT_DESC(dx)}, {1, OUTPUT_DESC(dr1)}, {2, OUTPUT_DESC(dr2)}};
REG_ADPT_DESC(RotaryMulGrad, prim::kPrimRotaryMulGrad->name(), ADPT_DESC(RotaryMulGrad))
}  // namespace mindspore::transform
