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
namespace {
constexpr size_t kResize2DOutRank = 4;
}

class MS_CORE_API Resize2D : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Resize2D(const SymbolPtr &image_shape, const SymbolPtr &size_array) : InferShapeOp({image_shape, size_array}) {}
  ~Resize2D() override = default;
  MS_DECLARE_PARENT(Resize2D, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Resize2D::Eval() {
  auto image_shape = input_as<ListSymbol>(kIndex0);
  auto size_array = input_as<ListSymbol>(kIndex1);
  SymbolPtrList result(kResize2DOutRank, nullptr);

  bool all_have_data = true;
  if (image_shape->HasData()) {
    result[kIndex0] = image_shape->item(kIndex0);
    result[kIndex1] = image_shape->item(kIndex1);
  } else {
    all_have_data = false;
    result[kIndex0] = GenVInt();
    result[kIndex1] = GenVInt();
  }
  if (size_array->HasData()) {
    result[kIndex2] = size_array->item(kIndex0);
    result[kIndex3] = size_array->item(kIndex1);
  } else {
    all_have_data = false;
    result[kIndex2] = GenVInt();
    result[kIndex3] = GenVInt();
  }
  if (all_have_data) {
    DoNotEvalOnRun();
  }
  return GenList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("ResizeBicubic")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFuncWith<Resize2D>();
REG_SYMBOL_OP_BUILDER("ResizeBilinearV2")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFuncWith<Resize2D>();
REG_SYMBOL_OP_BUILDER("ResizeNearestNeighborV2Grad")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFuncWith<Resize2D>();
REG_SYMBOL_OP_BUILDER("ResizeNearestNeighborV2")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFuncWith<Resize2D>();
REG_SYMBOL_OP_BUILDER("ResizeNearestNeighbor")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFuncWith<Resize2D>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
