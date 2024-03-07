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
#include "backend/common/graph_kernel/expander/base/ir_builder.h"

namespace mindspore::graphkernel::expander {
REG_EXPANDER_FUNC("Sigmoid").SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->input(kIndex0);
  auto const_one = ib->Tensor(1.0, input_x->GetDtype());
  auto neg_x = ib->Neg(input_x);
  auto exp_neg_x = ib->Exp(neg_x);
  auto add_exp = ib->Add(exp_neg_x, const_one);
  auto result = ib->Div(const_one, add_exp);
  return {result};
});

REG_EXPANDER_FUNC("SoftmaxBackward").SetBody(BODYFUNC(ib) {
  auto dout = ib->input(kIndex0);
  auto out = ib->input(kIndex1);
  auto dim = ib->input(kIndex2);
  auto dim_value_ptr = dim->GetValue();
  if (dim_value_ptr == nullptr || dim_value_ptr->isa<ValueAny>() || dim_value_ptr->isa<None>()) {
    MS_LOG(INFO) << "dim is not const value";
    return {};
  }
  auto dim_value = GetValue<int64_t>(dim_value_ptr);
  auto shp = out->GetShape();
  bool is_last_axis = true;
  if (IsDynamicRank(shp)) {
    is_last_axis = (dim_value == -1);
  } else {
    auto nd = SizeToLong(shp.size());
    is_last_axis = dim_value < 0 ? (dim_value == -1) : (dim_value == nd - 1);
  }
  if (!is_last_axis) {
    MS_LOG(INFO) << "dim is not last axis";
    return {};
  }
  ShapeVector axis{-1};
  auto result = ib->Mul(out, ib->Sub(dout, ib->ReduceSum(ib->Mul(out, dout), ib->Value(axis), ib->Value(true))));
  return {result};
});
}  // namespace mindspore::graphkernel::expander
