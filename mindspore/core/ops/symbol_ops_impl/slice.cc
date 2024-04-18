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
#include "mindspore/core/ops/symbol_ops_impl/common.h"
#include "mindspore/core/ops/symbol_ops_impl/scalar_add.h"
#include "mindspore/core/ops/symbol_ops_impl/scalar_sub.h"
#include "mindspore/core/ops/symbol_ops_impl/scalar_div.h"
#include "mindspore/core/ops/symbol_ops_impl/scalar_min.h"
#include "symbolic_shape/symbol.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API Slice : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~Slice() override = default;
  MS_DECLARE_PARENT(Slice, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Slice::Eval() {
  auto data_sym = input_as_sptr<ListSymbol>(kIndex0);
  MS_EXCEPTION_IF_NULL(data_sym);
  auto begin_sym = input_as_sptr<ListSymbol>(kIndex1);
  MS_EXCEPTION_IF_NULL(begin_sym);
  auto size_sym = input_as_sptr<ListSymbol>(kIndex2);
  MS_EXCEPTION_IF_NULL(size_sym);

  if (data_sym->HasData() && begin_sym->HasData() && size_sym->HasData()) {
    auto rank = data_sym->size();
    if (begin_sym->size() != rank || size_sym->size() != rank) {
      MS_LOG(ERROR) << "For Slice, the shape of input|begin|size must be equal, but got " << rank << "|"
                    << begin_sym->size() << "|" << size_sym->size() << ".";
    }

    SymbolPtrList new_syms;
    new_syms.reserve(rank);
    bool is_change = false;
    for (size_t i = 0; i < rank; ++i) {
      auto size_s = size_sym->item_as_sptr<IntSymbol>(i);
      MS_EXCEPTION_IF_NULL(size_s);
      if (size_s->is_positive()) {
        new_syms.push_back(size_s);
        continue;
      }

      auto data_s = data_sym->item_as_sptr<IntSymbol>(i);
      MS_EXCEPTION_IF_NULL(data_s);
      auto begin_s = begin_sym->item_as_sptr<IntSymbol>(i);
      MS_EXCEPTION_IF_NULL(begin_s);
      if (!data_s->HasData() || !begin_s->HasData() || !size_s->HasData()) {
        new_syms.push_back(GenVInt());
        continue;
      }

      auto data_v = data_s->value();
      auto begin_v = begin_s->value();
      auto size_v = size_s->value();
      if (begin_v + size_v > data_v) {
        MS_EXCEPTION(ValueError) << "For Slice, the sum of begin[" << i << "](" << begin_v << ") and size[" << i << "]("
                                 << size_v << ") must be no greater than input_shape[" << i << "](" << data_v << ").";
      }
      if (size_v == -1) {
        size_v = data_v - begin_v;
        is_change = true;
        new_syms.push_back(GenInt(size_v));
        continue;
      }
      new_syms.push_back(size_s);
    }

    if (is_change) {
      size_sym = ListSymbol::Make(std::move(new_syms));
    }
  }

  return size_sym;
}

REG_SYMBOL_OP_BUILDER("Slice")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFunc(DefaultBuilder<Slice>);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
