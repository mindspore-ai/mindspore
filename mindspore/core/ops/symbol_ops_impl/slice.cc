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
#include "mindspore/core/ops/symbol_ops_impl/scalar_sub.h"

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
  auto data = input_as_sptr<ListSymbol>(kIndex0);
  auto begin = input_as_sptr<ListSymbol>(kIndex1);
  auto size = input_as_sptr<ListSymbol>(kIndex2);
  if (!data->HasData() || !begin->HasData() || !size->HasData()) {
    if (data->HasData()) {
      return GenVIntList(data->size());
    }
    if (begin->HasData()) {
      return GenVIntList(begin->size());
    }
    if (size->HasData()) {
      return GenVIntList(size->size());
    }
  }

  auto rank = data->size();
  if (begin->size() != rank || size->size() != rank) {
    MS_LOG(ERROR) << "For Slice, the shape of input|begin|size must be equal, but got " << rank << "|" << begin->size()
                  << "|" << size->size() << ".";
  }

  SymbolPtrList new_syms(rank);
  bool has_unknown_size = false;
  for (size_t i = 0; i < rank; ++i) {
    auto data_s = data->item_as_sptr<IntSymbol>(i);
    auto begin_s = begin->item_as_sptr<IntSymbol>(i);
    auto size_s = size->item_as_sptr<IntSymbol>(i);
    if (size_s->is_greater_equal(0)) {
      new_syms[i] = size_s;
    } else if (size_s->is_negative()) {
      // when the size is "-1", result is data[begin:]
      new_syms[i] = Emit(std::make_shared<ScalarSub>(data_s, begin_s));
    } else {
      has_unknown_size = true;
      new_syms[i] = GenVInt();
    }
  }
  if (!has_unknown_size) {
    DoNotEvalOnRun();
  }
  return GenList(std::move(new_syms));
}

REG_SYMBOL_OP_BUILDER("Slice")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFuncWith<Slice>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
