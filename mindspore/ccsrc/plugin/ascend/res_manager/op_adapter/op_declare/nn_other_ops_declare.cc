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

#include "plugin/ascend/res_manager/op_adapter/op_declare/nn_other_ops_declare.h"
#include <vector>
#include <string>
#include "plugin/ascend/res_manager/op_adapter/op_declare/op_declare_macro.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"

namespace mindspore::device::ascend {
// ApplyRotaryPosEmb
INPUT_MAP(ApplyRotaryPosEmb) = {
  {1, INPUT_DESC(query)}, {2, INPUT_DESC(key)}, {3, INPUT_DESC(cos)}, {4, INPUT_DESC(sin)}};
ATTR_MAP(ApplyRotaryPosEmb) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(ApplyRotaryPosEmb) = {{6, ATTR_DESC(layout, AnyTraits<int64_t>())}};
OUTPUT_MAP(ApplyRotaryPosEmb) = {{0, OUTPUT_DESC(query)}, {1, OUTPUT_DESC(key)}};
REG_ADPT_DESC(ApplyRotaryPosEmb, kNameApplyRotaryPosEmb, ADPT_DESC(ApplyRotaryPosEmb))

// RotaryPositionEmbedding
INPUT_MAP(RotaryPositionEmbedding) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(cos)}, {3, INPUT_DESC(sin)}};
ATTR_MAP(RotaryPositionEmbedding) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(RotaryPositionEmbedding) = {{kIndex4, ATTR_DESC(mode, AnyTraits<int64_t>())}};
OUTPUT_MAP(RotaryPositionEmbedding) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(RotaryPositionEmbedding, prim::kPrimRotaryPositionEmbedding->name(), ADPT_DESC(RotaryPositionEmbedding))

// RotaryPositionEmbeddingGrad
INPUT_MAP(RotaryPositionEmbeddingGrad) = {
  {1, INPUT_DESC(dy)}, {2, INPUT_DESC(cos)}, {3, INPUT_DESC(sin)}, {4, INPUT_DESC(x)}};
ATTR_MAP(RotaryPositionEmbeddingGrad) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(RotaryPositionEmbeddingGrad) = {{5, ATTR_DESC(mode, AnyTraits<int64_t>())}};
OUTPUT_MAP(RotaryPositionEmbeddingGrad) = {{0, OUTPUT_DESC(dx)}, {1, OUTPUT_DESC(dcos)}, {2, OUTPUT_DESC(dsin)}};
REG_ADPT_DESC(RotaryPositionEmbeddingGrad, prim::kPrimRotaryPositionEmbeddingGrad->name(),
              ADPT_DESC(RotaryPositionEmbeddingGrad))
}  // namespace mindspore::device::ascend
