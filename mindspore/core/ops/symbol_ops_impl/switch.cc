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
#include "mindspore/core/ops/symbol_ops_impl/switch.h"
#include <algorithm>
#include <utility>
#include <memory>

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API ControlFlowJoin : public InferShapeOp {
 public:
  ControlFlowJoin(const SymbolPtr &cond, const SymbolPtr &true_branch, const SymbolPtr &false_branch)
      : InferShapeOp({cond, true_branch, false_branch}) {}
  ~ControlFlowJoin() override = default;
  MS_DECLARE_PARENT(ControlFlowJoin, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
  SymbolPtr ShapeJoin(const SymbolPtr &tb, const SymbolPtr &fb);
  SymbolPtr ItemJoin(const SymbolPtr &tb, const SymbolPtr &fb);
};

SymbolPtr ControlFlowJoin::ItemJoin(const SymbolPtr &tb, const SymbolPtr &fb) {
  return tb->EqualsTo(fb) ? tb : GenVInt();
}

SymbolPtr ControlFlowJoin::ShapeJoin(const SymbolPtr &tb, const SymbolPtr &fb) {
  if (tb->EqualsTo(fb)) {
    return tb;
  }
  if (tb->tid() != fb->tid()) {
    return DynamicSymbol::Make(shared_from_this());
  }

  if (auto tb_list = tb->as<ListSymbol>(); tb_list != nullptr) {
    auto fb_list = fb->as<ListSymbol>();
    MS_EXCEPTION_IF_NULL(fb_list);
    if (tb_list->size() != fb_list->size()) {
      return GenVList();
    }
    SymbolPtrList result(tb_list->size());
    (void)std::transform(tb_list->symbols().begin(), tb_list->symbols().end(), fb_list->symbols().begin(),
                         result.begin(), [this](const SymbolPtr &t, const SymbolPtr &f) { return ShapeJoin(t, f); });
    return GenList(std::move(result));
  } else {
    return ItemJoin(tb, fb);
  }
}

SymbolPtr ControlFlowJoin::Eval() {
  auto cond = input_as<BoolSymbol>(kIndex0);
  auto tb = input(kIndex1);
  auto fb = input(kIndex2);
  if (cond->HasData()) {
    DoNotEvalOnRun();
    return cond->value() ? tb : fb;
  }
  return ShapeJoin(tb, fb);
}

REG_SYMBOL_OP_BUILDER(kControlFlowJoin)
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto cond = b->GetInputValue(kIndex0);
    auto true_branch = b->GetInputShape(kIndex1);
    auto false_branch = b->GetInputShape(kIndex2);
    return b->Emit(std::make_shared<ControlFlowJoin>(cond, true_branch, false_branch));
  })
  .SetValueFunc([](OperationBuilder *b) -> SymbolPtr {
    auto cond = b->GetInputValue(kIndex0)->as_sptr<BoolSymbol>();
    // buildvalue for control flow only support constant folding.
    if (cond != nullptr && cond->HasData()) {
      return cond->value() ? b->GetInputValue(kIndex1) : b->GetInputValue(kIndex2);
    }
    return nullptr;
  });

REG_SYMBOL_OP_BUILDER("Switch").SetValueFunc([](OperationBuilder *b) -> SymbolPtr { return IntSymbol::Make(); });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
