/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "mindspore/ops/infer/symbol_ops_impl/common.h"
#include "abstract/symbolic_shape/utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API IndexSelect : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  IndexSelect(const SymbolPtr &data, const SymbolPtr &idx) : InferShapeOp({data, idx}) {}
  ~IndexSelect() override = default;
  MS_DECLARE_PARENT(IndexSelect, InferShapeOp)

 protected:
  SymbolPtr Eval() override {
    auto x_shape = input_as<ListSymbol>(kIndex0);
    auto axis = input_as<IntSymbol>(kIndex1);
    auto index_shape = input_as<ListSymbol>(kIndex2);
    if (!x_shape->HasData()) {
      return GenVList();
    }
    auto rank = x_shape->size();
    if (!axis->HasData()) {
      return GenVIntList(rank);
    }
    auto i = LongToSize(NormAxis(axis->value(), rank));
    if (i >= rank) {
      MS_LOG(INTERNAL_EXCEPTION) << "The axis " << axis->value() << " is out of range of input rank " << rank;
    }
    auto result = x_shape->symbols();
    if (!index_shape->HasData()) {
      result[i] = GenVInt();
    }
    DoNotEvalOnRun();
    result[i] = index_shape->item(0);
    return GenList(std::move(result));
  }
};

REG_SYMBOL_OP_BUILDER("IndexSelect")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kShape})
  .SetShapeFuncWith<IndexSelect>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
