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
#include "mindspore/core/ops/symbol_ops_impl/reduce.h"
#include <memory>

namespace mindspore {
namespace symshape {
namespace ops {
bool Reduce::GetAxisSet(const SymbolPtr &axis, int64_t rank, bool skip_mode, HashSet<int64_t> *axis_set) const {
  auto axis_list = axis->as<ListSymbol>();
  if (axis_list != nullptr) {
    if (axis_list->symbols().empty()) {
      if (skip_mode) {
        return false;
      } else {
        for (int64_t i = 0; i < rank; i++) {
          (void)axis_set->insert(i);
        }
      }
    } else {
      for (auto &x : axis_list->symbols()) {
        (void)axis_set->insert(AsInt(x));
      }
    }
  } else {
    (void)axis_set->insert(AsInt(axis));
  }
  return true;
}

SymbolPtr Reduce::Eval() {
  auto data = input_as<ListSymbol>(kIndex0);
  auto axis = input(kIndex1);
  bool keep_dims = input_as<BoolSymbol>(kIndex2)->value();
  bool skip_mode = input_as<BoolSymbol>(kIndex3)->value();
  if (!data->HasData() || !axis->HasData()) {
    // for empty axis
    auto axis_list = axis->as<ListSymbol>();
    if (axis_list != nullptr && axis_list->HasData() && axis_list->size() == 0) {
      if (skip_mode) {
        DoNotEvalOnRun();
        return input(kIndex0);
      }
      // if not keepdims, and the axis is a reduce-all case, then output shape is empty.
      if (!keep_dims) {
        return GenList({});
      }
    }
    if (keep_dims && data->HasData()) {
      // axis has no data
      return GenVIntList(data->size());
    }
    return GenVList();
  }
  DoNotEvalOnRun();
  HashSet<int64_t> axis_set;
  auto &inp = data->symbols();
  auto rank = SizeToLong(inp.size());
  if (!GetAxisSet(axis, rank, skip_mode, &axis_set)) {
    return input(kIndex0);
  }
  SymbolPtrList out_shape;
  out_shape.reserve(inp.size());
  for (int64_t i = 0; i < rank; i++) {
    if (axis_set.count(i) != 0 || axis_set.count(i - rank) != 0) {
      if (keep_dims) {
        (void)out_shape.emplace_back(GenInt(1));
      }
    } else {
      (void)out_shape.emplace_back(inp[i]);
    }
  }
  return GenList(out_shape);
}

SymbolPtr ReduceShapeBuilder(OperationBuilder *b) {
  auto input = b->GetInputShape(kIndex0);
  auto axis = b->GetInputValue(kIndex1);
  auto keep_dims = b->GetInputOrAttr(kIndex2, kAttrKeepDims);
  MS_EXCEPTION_IF_NULL(keep_dims);
  // the skip_mode only exists in ReduceSum
  auto skip_mode = b->GetInputOrAttr(kIndex3, kAttrSkipMode);
  if (skip_mode == nullptr) {
    skip_mode = BoolSymbol::Make(false);
  }
  return b->Emit(std::make_shared<Reduce>(input, axis, keep_dims, skip_mode));
}

SymbolPtr ArgMinMaxShapeBuilder(OperationBuilder *b) {
  auto input = b->GetInputShape(kIndex0);
  auto axis = b->GetInputValue(kIndex1);
  auto s_false = BoolSymbol::Make(false);
  return b->Emit(std::make_shared<Reduce>(input, axis, s_false, s_false));
}

SymbolPtr ArgMinMaxWithValueShapeBuilder(OperationBuilder *b) {
  auto input = b->GetInputShape(kIndex0);
  auto axis = b->GetInputOrAttr(kIndex1, kAttrAxis);
  MS_EXCEPTION_IF_NULL(axis);
  auto keep_dims = b->GetInputOrAttr(kIndex2, kAttrKeepDims);
  MS_EXCEPTION_IF_NULL(keep_dims);
  auto ret = b->Emit(std::make_shared<Reduce>(input, axis, keep_dims, BoolSymbol::Make(false)));
  if (ret == nullptr) {
    return nullptr;
  }
  // ArgMinWithValue has 2 outputs of same shape
  return ListSymbol::Make({ret, ret});
}

REG_SYMBOL_OP_BUILDER("ReduceSum")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue, DependOn::kValue})  // last input is skip_mode
  .SetShapeFunc(ReduceShapeBuilder);
REG_SYMBOL_OP_BUILDER("ReduceMax")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFunc(ReduceShapeBuilder);
REG_SYMBOL_OP_BUILDER("ReduceMin")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFunc(ReduceShapeBuilder);
REG_SYMBOL_OP_BUILDER("ReduceMean")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFunc(ReduceShapeBuilder);

REG_SYMBOL_OP_BUILDER("ArgMax")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFunc(ArgMinMaxShapeBuilder);
REG_SYMBOL_OP_BUILDER("ArgMin")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFunc(ArgMinMaxShapeBuilder);
REG_SYMBOL_OP_BUILDER("ArgMaxWithValue")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFunc(ArgMinMaxWithValueShapeBuilder);
REG_SYMBOL_OP_BUILDER("ArgMinWithValue")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFunc(ArgMinMaxWithValueShapeBuilder);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
