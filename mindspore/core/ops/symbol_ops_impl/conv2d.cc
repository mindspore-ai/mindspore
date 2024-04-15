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
#include "mindspore/core/ops/symbol_ops_impl/scalar_div.h"

namespace mindspore {
namespace symshape {
namespace ops {
namespace {
constexpr size_t kNum2 = 2;
}
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
  SymbolPtr CalcForPadSame(const SymbolPtr &x, const SymbolPtr &stride) {
    return Emit(std::make_shared<ScalarCeilDiv>(x, stride));
  }

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

SymbolPtr Conv2D::Eval() {
  auto x = input_as<ListSymbol>(kIndex0);
  auto pad_mode = input_as<IntSymbol>(kIndex3)->value();
  auto stride = ProcessAttr(input(kIndex5), kIndex2, kNum2);
  auto format = input_as<StrSymbol>(kIndex7)->value();
  if (pad_mode != PadMode::SAME) {
    // only support SAME pad now.
    return nullptr;
  }
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
  auto out_h = CalcForPadSame(x->item(h_axis), stride->item(kIndex0));
  auto out_w = CalcForPadSame(x->item(w_axis), stride->item(kIndex1));
  return GenOutput(out_n, out_h, out_w);
}

REG_SYMBOL_OP_BUILDER("Conv2D").SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
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
