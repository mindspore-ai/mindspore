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
#include "backend/common/graph_kernel/expander/base/ir_builder.h"
#include "backend/common/graph_kernel/expander/base/utils.h"

namespace mindspore::graphkernel::expander {
REG_EXPANDER_FUNC("AddN").SetBody(BODYFUNC(ib) {
  if (!CheckAllFormatsSame(ib)) {
    return {};
  }
  // Check Inputs
  constexpr size_t min_inputs = 2;
  if (ib->inputs().size() < min_inputs) {
    MS_LOG(INFO) << "For 'AddN', the inputs num should be greater than 1, but got " << ib->inputs().size();
    return {};
  }

  auto result = ib->input(0);
  for (size_t i = 1; i < ib->inputs().size(); ++i) {
    result = ib->Add(result, ib->input(i));
  }
  return {result};
});

REG_EXPANDER_FUNC("EqualCount").SetBody(BODYFUNC(ib) {
  // Check inputs
  auto it = std::find_if(std::begin(ib->inputs()), std::end(ib->inputs()), [](const NodePtr &input) {
    return input->GetDtype() != TypeIdToType(kNumberTypeFloat32) &&
           input->GetDtype() != TypeIdToType(kNumberTypeFloat16) && input->GetDtype() != TypeIdToType(kNumberTypeInt32);
  });
  if (it != std::end(ib->inputs())) {
    MS_LOG(INFO) << "In EqualCount, input's dtype must be float16 or float32 or int32, But input's type is "
                 << (*it)->GetDtype()->ToString();
    return {};
  }
  const auto &input_x = ib->input(0);
  const auto &input_y = ib->input(1);
  if (input_x->GetDtype() != input_y->GetDtype()) {
    MS_LOG(INFO) << "In EqualCount, the inputs data type should be same, But input_x's type is " << input_x->GetDtype()
                 << " input_y's type is " << input_y->GetDtype();
    return {};
  }
  if (input_x->GetShape() != input_y->GetShape()) {
    MS_LOG(INFO) << "In EqualCount, the inputs data shape should be same, But input_x's shape is "
                 << input_x->GetShape() << " input_y's shape is " << input_y->GetShape();
    return {};
  }
  // Expand
  auto dtype = input_x->GetDtype();
  auto eql_val = ib->Equal(input_x, input_y);
  auto cast_val = ib->Cast(eql_val, kNumberTypeFloat32);
  auto shape_size = input_x->GetShape().size();
  std::vector<int64_t> axis(shape_size);
  for (size_t i = 0; i < shape_size; ++i) {
    axis[i] = SizeToLong(i);
  }
  auto result = ib->ReduceSum(cast_val, axis, false);
  result = ib->Reshape(result, ib->Tensor({1}));
  if (result->GetDtype() != dtype) {
    result = ib->Cast(result, dtype->type_id());
  }
  return {result};
});

REG_EXPANDER_FUNC("Tanh").SetBody(BODYFUNC(ib) {
  auto result = ib->Tanh(ib->input(kIndex0));
  return {result};
});

REG_EXPANDER_FUNC("Sinh").SetBody(BODYFUNC(ib) {
  auto result = ib->Sinh(ib->input(kIndex0));
  return {result};
});

REG_EXPANDER_FUNC("Cosh").SetBody(BODYFUNC(ib) {
  auto result = ib->Cosh(ib->input(kIndex0));
  return {result};
});

REG_EXPANDER_FUNC("LogicalXor").SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->input(kIndex0);
  const auto &input_y = ib->input(kIndex1);

  auto result_b = ib->LogicalAnd(input_x, ib->LogicalNot(input_y));
  auto result_a = ib->LogicalAnd(input_y, ib->LogicalNot(input_x));
  return {ib->LogicalOr(result_a, result_b)};
});

NodePtrList FastGeluExpand(const DefaultIrBuilder *ib) {
  const auto &input_x = ib->input(kIndex0);
  const double val = 1.7020000219345093;
  auto const_0 = ib->Tensor(-val, input_x->GetDtype());
  auto const_1 = ib->Tensor(val / 2, input_x->GetDtype());
  auto const_2 = ib->Tensor(1, input_x->GetDtype());

  auto abs = ib->Abs(input_x);
  auto sub = ib->Sub(input_x, abs);
  auto exp_0 = ib->Exp(ib->Mul(const_1, sub));
  auto n = ib->Mul(input_x, exp_0);
  auto exp_1 = ib->Exp(ib->Mul(const_0, abs));
  auto d = ib->Add(exp_1, const_2);

  return {ib->Div(n, d)};
}

NodePtrList FastGeluGradExpand(const DefaultIrBuilder *ib) {
  const auto &input_x = ib->input(kIndex1);
  const auto &dout = ib->input(kIndex0);
  const double val = 1.7020000219345093;
  auto const_0 = ib->Tensor(val, input_x->GetDtype());
  auto const_1 = ib->Tensor(-val, input_x->GetDtype());
  auto const_2 = ib->Tensor(1, input_x->GetDtype());

  auto abs = ib->Abs(input_x);
  auto mul_1 = ib->Exp(ib->Mul(const_1, abs));
  auto mul_3 = ib->Mul(input_x, mul_1);
  mul_3 = ib->Mul(const_0, mul_3);
  mul_3 = ib->Add(mul_3, mul_1);

  auto sub = ib->Sub(input_x, abs);
  sub = ib->Exp(ib->Mul(sub, const_0));

  mul_3 = ib->Add(sub, mul_3);
  mul_1 = ib->Add(mul_1, const_2);
  mul_1 = ib->Mul(mul_1, mul_1);

  return {ib->Mul(ib->Div(mul_3, mul_1), dout)};
}

REG_EXPANDER_FUNC("FastGelu").SetBody(FastGeluExpand);
REG_EXPANDER_FUNC("FastGeLU").SetBody(FastGeluExpand);
REG_EXPANDER_FUNC("FastGeluGrad").SetBody(FastGeluGradExpand);
REG_EXPANDER_FUNC("FastGeLUGrad").SetBody(FastGeluGradExpand);
}  // namespace mindspore::graphkernel::expander
