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

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API GridSampler : public InferShapeOp {
 public:
  GridSampler(const SymbolPtr &x, const SymbolPtr &grid, size_t out_rank)
      : InferShapeOp({x, grid}), out_rank_(out_rank) {}
  ~GridSampler() override = default;
  MS_DECLARE_PARENT(GridSampler, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
  size_t out_rank_;
};

SymbolPtr GridSampler::Eval() {
  auto x = input_as<ListSymbol>(kIndex0);
  auto grid = input_as<ListSymbol>(kIndex1);
  if (!x->HasData() && !grid->HasData()) {
    return GenVIntList(out_rank_);
  }
  SymbolPtrList result;
  result.reserve(out_rank_);
  if (!grid->HasData()) {
    (void)result.emplace_back(x->item(kIndex0));
    (void)result.emplace_back(x->item(kIndex1));
    while (result.size() < out_rank_) {
      (void)result.emplace_back(GenVInt());
    }
    return GenList(std::move(result));
  }
  MS_EXCEPTION_IF_CHECK_FAIL(grid->size() == out_rank_, "Invalid grid shape");
  if (!x->HasData()) {
    (void)result.emplace_back(grid->item(kIndex0));
    (void)result.emplace_back(GenVInt());
  } else {
    DoNotEvalOnRun();
    (void)result.emplace_back(x->item(kIndex0));
    (void)result.emplace_back(x->item(kIndex1));
    if (is_building()) {
      // e.g. For GridSampler2D, x'shape is (N, C, Hi, Wi), grid's shape is (N, Ho, Wo, 2).
      //      Set the two 'N' equal.
      x->item_as_sptr<IntSymbol>(kIndex0)->SetEqual(grid->item_as_sptr<IntSymbol>(kIndex0));
    }
  }
  (void)result.insert(result.end(), grid->symbols().begin() + 1, grid->symbols().end() - 1);
  return ResultIntList(std::move(result));
}

constexpr size_t grid_sampler_2d_rank = 4;
constexpr size_t grid_sampler_3d_rank = 5;
template <bool IS_2D>
SymbolPtr GridSamplerShapeBuilder(OperationBuilder *b) {
  size_t out_rank = (IS_2D ? grid_sampler_2d_rank : grid_sampler_3d_rank);
  return b->Emit(std::make_shared<GridSampler>(b->GetInputShape(kIndex0), b->GetInputShape(kIndex1), out_rank));
}

REG_SYMBOL_OP_BUILDER("GridSampler2D")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape})
  .SetShapeFunc(GridSamplerShapeBuilder<true>);
REG_SYMBOL_OP_BUILDER("GridSampler3D")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape})
  .SetShapeFunc(GridSamplerShapeBuilder<false>);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
