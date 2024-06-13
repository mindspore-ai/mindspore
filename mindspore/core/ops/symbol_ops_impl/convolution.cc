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
#include "mindspore/core/ops/symbol_ops_impl/operator_scope.h"

namespace mindspore {
namespace symshape {
namespace ops {
namespace {
constexpr size_t kOutRank = 4;
constexpr size_t kInputIdx = 0;
constexpr size_t kWeightIdx = 1;
constexpr size_t kStrideIdx = 2;
constexpr size_t kPaddingIdx = 3;
constexpr size_t kDilationIdx = 4;
constexpr size_t kTransposedIdx = 5;
constexpr size_t kOutputPaddingIdx = 6;
constexpr size_t kGroupsIdx = 7;
}  // namespace
class MS_CORE_API Convolution : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~Convolution() override = default;
  MS_DECLARE_PARENT(Convolution, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
  SymbolPtr CalcForTransposed(const SymbolPtr &x, const SymbolPtr &weight, const SymbolPtr &stride,
                              const SymbolPtr &padding, const SymbolPtr &dilation, const SymbolPtr &out_padding);
  SymbolPtr CalcForNotTransposed(const SymbolPtr &x, const SymbolPtr &weight, const SymbolPtr &stride,
                                 const SymbolPtr &padding, const SymbolPtr &dilation);
};

SymbolPtr Convolution::CalcForTransposed(const SymbolPtr &x, const SymbolPtr &weight, const SymbolPtr &stride,
                                         const SymbolPtr &padding, const SymbolPtr &dilation,
                                         const SymbolPtr &out_padding) {
  // (x - 1) * stride - 2 * padding + dilation * (weight - 1) + output_padding + 1
  OperatorScope h(emitter());
  auto v1 = h(kSym1);
  auto v2 = h(kSym2);
  return (x - v1) * stride - v2 * padding + dilation * (weight - v1) + out_padding + v1;
}

SymbolPtr Convolution::CalcForNotTransposed(const SymbolPtr &x, const SymbolPtr &weight, const SymbolPtr &stride,
                                            const SymbolPtr &padding, const SymbolPtr &dilation) {
  // (x + 2 * padding - (weight - 1) * dilation - 1) / stride + 1
  OperatorScope h(emitter(), OperatorScope::DivType::FLOOR_DIV);
  auto v1 = h(kSym1);
  auto v2 = h(kSym2);
  return (x + v2 * padding - dilation * (weight - v1) - v1) / stride + v1;
}

SymbolPtr Convolution::Eval() {
  auto input_shape = input_as<ListSymbol>(kInputIdx);
  if (!input_shape->HasData()) {
    return GenVIntList(kOutRank);
  }
  auto out_n = input_shape->item(kIndex0);
  auto weight_shape = input_as<ListSymbol>(kWeightIdx);
  auto transposed = input_as<BoolSymbol>(kTransposedIdx);
  if (!weight_shape->HasData() || !transposed->HasData()) {
    return GenList({out_n, GenVInt(), GenVInt(), GenVInt()});
  }
  auto stride = input_as<ListSymbol>(kStrideIdx);
  auto padding = input_as<ListSymbol>(kPaddingIdx);
  auto dilation = input_as<ListSymbol>(kDilationIdx);
  SymbolPtr out_c;
  SymbolPtr out_h;
  SymbolPtr out_w;
  if (transposed->value()) {
    out_c = Emit(std::make_shared<ScalarMul>(input(kGroupsIdx), weight_shape->item(kIndex1)));
  } else {
    out_c = weight_shape->item(kIndex0);
  }
  if (!stride->HasData() || !padding->HasData() || !dilation->HasData()) {
    return GenList({out_n, out_c, GenVInt(), GenVInt()});
  }
  if (transposed->value()) {
    auto output_padding = input_as<ListSymbol>(kOutputPaddingIdx);
    if (!output_padding->HasData()) {
      return GenList({out_n, out_c, GenVInt(), GenVInt()});
    }
    out_h = CalcForTransposed(input_shape->item(kIndex2), weight_shape->item(kIndex2), stride->item(kIndex0),
                              padding->item(kIndex0), dilation->item(kIndex0), output_padding->item(kIndex0));
    out_w = CalcForTransposed(input_shape->item(kIndex3), weight_shape->item(kIndex3), stride->item(kIndex1),
                              padding->item(kIndex1), dilation->item(kIndex1), output_padding->item(kIndex1));
  } else {
    out_h = CalcForNotTransposed(input_shape->item(kIndex2), weight_shape->item(kIndex2), stride->item(kIndex0),
                                 padding->item(kIndex0), dilation->item(kIndex0));
    out_w = CalcForNotTransposed(input_shape->item(kIndex3), weight_shape->item(kIndex3), stride->item(kIndex1),
                                 padding->item(kIndex1), dilation->item(kIndex1));
  }
  DoNotEvalOnRun();
  return GenList({out_n, out_c, out_h, out_w});
}

REG_SYMBOL_OP_BUILDER("Convolution")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape, DependOn::kNone, DependOn::kValue, DependOn::kValue,
                   DependOn::kValue, DependOn::kValue, DependOn::kValue, DependOn::kValue})
  .SetShapeFuncWith<Convolution>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
