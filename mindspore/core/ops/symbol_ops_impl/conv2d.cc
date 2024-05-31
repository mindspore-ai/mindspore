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
#include "mindspore/core/ops/conv2d.h"
#include "mindspore/core/ops/symbol_ops_impl/operator_scope.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace symshape {
namespace ops {
namespace {
constexpr size_t kNum2 = 2;
constexpr size_t kNum4 = 4;
}  // namespace
class MS_CORE_API Conv2D : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Conv2D(const SymbolPtr &x, const SymbolPtr &out_channel, const SymbolPtr &kernel_size, const SymbolPtr &pad_mode,
         const SymbolPtr &padding, const SymbolPtr &stride, const SymbolPtr &dilation, const SymbolPtr &format)
      : InferShapeOp({x, out_channel, kernel_size, pad_mode, padding, stride, dilation, format}) {}

  ~Conv2D() override = default;
  MS_DECLARE_PARENT(Conv2D, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
  SymbolPtr GenOutput(const SymbolPtr &n, const SymbolPtr &h, const SymbolPtr &w) const {
    auto out_channel = input(kIndex1);
    auto format = input_as<StrSymbol>(kIndex7)->value();
    return format == "NCHW" ? ListSymbol::Make({n, out_channel, h, w}) : ListSymbol::Make({n, h, w, out_channel});
  }
  SymbolPtr CalcForPadValid(const SymbolPtr &x, const SymbolPtr &kernel, const SymbolPtr &stride,
                            const SymbolPtr &dilation);
  SymbolPtr CalcForPadSame(const SymbolPtr &x, const SymbolPtr &stride) {
    return Emit(std::make_shared<ScalarCeilDiv>(x, stride));
  }
  SymbolPtr CalcForPadding(const SymbolPtr &x_shape, const SymbolPtr &kernel, const SymbolPtr &padding,
                           const SymbolPtr &stride, const SymbolPtr &dilation);

  ListSymbolPtr ProcessAttr(const SymbolPtr &attr, size_t begin_idx, size_t num) {
    if (attr->is<ListSymbol>()) {
      auto list = attr->as_sptr<ListSymbol>();
      if (list->size() == num) {
        return list;
      }
      SymbolPtrList res(list->symbols().begin() + begin_idx, list->symbols().begin() + begin_idx + num);
      return ListSymbol::Make(std::move(res));
    }
    SymbolPtrList res(num, attr);
    return ListSymbol::Make(std::move(res));
  }
};

SymbolPtr Conv2D::CalcForPadValid(const SymbolPtr &x, const SymbolPtr &kernel, const SymbolPtr &stride,
                                  const SymbolPtr &dilation) {
  // `(x - (kernel - 1) * dilation)) / stride`, to ceil
  OperatorScope h(emitter(), OperatorScope::DivType::CEIL_DIV);
  auto v1 = h(kSym1);
  return (x - (kernel - v1) * dilation) / stride;
}

SymbolPtr Conv2D::CalcForPadding(const SymbolPtr &x_shape, const SymbolPtr &kernel, const SymbolPtr &padding,
                                 const SymbolPtr &stride, const SymbolPtr &dilation) {
  //    `[(x + padding - kernel - (kernel - 1) * (dilation - 1)) / stride] + 1`, [] is to floor.
  // => `[(x + padding - kernel * (dilation - 1) - 1) / stride] + 1`.
  OperatorScope h(emitter(), OperatorScope::DivType::FLOOR_DIV);
  auto v1 = h(kSym1);
  auto x = h(x_shape);
  return ((x + padding - kernel * (dilation - v1) - v1) / stride) + v1;
}

SymbolPtr Conv2D::Eval() {
  auto x = input_as<ListSymbol>(kIndex0);
  auto pad_mode_sym = input(kIndex3);
  int64_t pad_mode;
  if (pad_mode_sym->is<StrSymbol>()) {
    CheckAndConvertUtils::GetPadModEnumValue(MakeValue(pad_mode_sym->as<StrSymbol>()->value()), &pad_mode);
  } else if (pad_mode_sym->is<IntSymbol>()) {
    pad_mode = pad_mode_sym->as<IntSymbol>()->value();
  } else {
    MS_LOG(EXCEPTION) << "Unsupported pad_mode " << pad_mode_sym->ToString();
  }
  auto format = input_as<StrSymbol>(kIndex7)->value();
  if (!x->HasData()) {
    return GenOutput(GenVInt(), GenVInt(), GenVInt());
  }
  size_t h_axis = kIndex2;
  size_t w_axis = kIndex3;
  if (format == "NHWC") {
    h_axis = kIndex1;
    w_axis = kIndex2;
  }
  auto out_n = x->item(kIndex0);
  SymbolPtr out_h;
  SymbolPtr out_w;
  if (pad_mode == PadMode::VALID) {
    auto kernel = ProcessAttr(input(kIndex2), kIndex0, kNum2);
    auto stride = ProcessAttr(input(kIndex5), kIndex2, kNum2);
    auto dilation = ProcessAttr(input(kIndex6), kIndex2, kNum2);
    out_h = CalcForPadValid(x->item(h_axis), kernel->item(kIndex0), stride->item(kIndex0), dilation->item(kIndex0));
    out_w = CalcForPadValid(x->item(w_axis), kernel->item(kIndex1), stride->item(kIndex1), dilation->item(kIndex1));
  } else if (pad_mode == PadMode::SAME) {
    auto stride = ProcessAttr(input(kIndex5), kIndex2, kNum2);
    out_h = CalcForPadSame(x->item(h_axis), stride->item(kIndex0));
    out_w = CalcForPadSame(x->item(w_axis), stride->item(kIndex1));
  } else if (pad_mode == PadMode::PAD) {
    auto kernel = ProcessAttr(input(kIndex2), kIndex0, kNum2);
    auto padding = ProcessAttr(input(kIndex4), kIndex0, kNum4);
    auto stride = ProcessAttr(input(kIndex5), kIndex2, kNum2);
    auto dilation = ProcessAttr(input(kIndex6), kIndex2, kNum2);
    auto padding_h = Emit(std::make_shared<ScalarAdd>(padding->item(kIndex0), padding->item(kIndex1)));
    auto padding_w = Emit(std::make_shared<ScalarAdd>(padding->item(kIndex2), padding->item(kIndex3)));
    out_h =
      CalcForPadding(x->item(h_axis), kernel->item(kIndex0), padding_h, stride->item(kIndex0), dilation->item(kIndex0));
    out_w =
      CalcForPadding(x->item(w_axis), kernel->item(kIndex1), padding_w, stride->item(kIndex1), dilation->item(kIndex1));
  } else {
    MS_LOG(DEBUG) << "The pad_mode " << pad_mode << " is not supported now.";
    return nullptr;
  }
  DoNotEvalOnRun();
  return GenOutput(out_n, out_h, out_w);
}

REG_SYMBOL_OP_BUILDER("Conv2D")
  .SetShapeDepend({DependOn::kShape, DependOn::kNone})
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto x = b->GetInputShape(kIndex0);
    auto out_channel = b->GetInputOrAttr(kIndex3, "out_channel");
    auto kernel_size = b->GetInputOrAttr(kIndex4, "kernel_size");
    auto pad_mode = b->GetInputOrAttr(kIndex6, "pad_mode");
    auto padding = b->GetInputOrAttr(kIndex7, "pad");
    auto stride = b->GetInputOrAttr(kIndex8, "stride");
    auto dilation = b->GetInputOrAttr(kIndex9, "dilation");
    auto format = b->GetInputOrAttr(kIndex11, "format");
    return b->Emit(std::make_shared<Conv2D>(x, out_channel, kernel_size, pad_mode, padding, stride, dilation, format));
  });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
