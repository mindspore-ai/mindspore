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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/cummin_ext.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct CumminExtShapeParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr axis;
};

class TestCumminExt : public TestOps, public testing::WithParamInterface<CumminExtShapeParams> {};
class TestCumminExtSimpleInfer : public TestOps, public testing::WithParamInterface<CumminExtShapeParams> {};

TEST_P(TestCumminExt, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto axis = param.axis->ToAbstract();

  auto values_shape = std::make_shared<abstract::Shape>(param.x_shape);
  auto indices_shape = std::make_shared<abstract::Shape>(param.x_shape);
  auto expect_shape = std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{values_shape, indices_shape});
  auto expect_type = std::make_shared<Tuple>(std::vector<TypePtr>{std::make_shared<TensorType>(param.x_type), kInt64});

  CumminExtFuncImpl cummin_ext_func_impl;
  auto prim = std::make_shared<Primitive>("CumminExt");

  auto out_dtype = cummin_ext_func_impl.InferType(prim, {x, axis});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = cummin_ext_func_impl.InferShape(prim, {x, axis});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

TEST_P(TestCumminExtSimpleInfer, simple_infer) {
  const auto &param = GetParam();
  auto x = std::make_shared<tensor::BaseTensor>(param.x_type->type_id(), param.x_shape);
  auto expect_shape = ShapeArray{param.x_shape, param.x_shape};
  auto expect_type = TypePtrList{param.x_type, kInt64};

  CumminExtFuncImpl cummin_ext_func_impl;
  auto prim = std::make_shared<Primitive>("CumminExt");
  ValuePtrList input_values;
  input_values.push_back(std::move(x));
  input_values.push_back(std::move(param.axis));
  auto out_dtype = cummin_ext_func_impl.InferType(prim, input_values);
  TypeCompare(out_dtype, expect_type);
  auto out_shape = cummin_ext_func_impl.InferShape(prim, input_values);
  ShapeCompare(out_shape, expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestCumminExt, TestCumminExt,
  testing::Values(CumminExtShapeParams{{3, 4, 5}, kFloat32, CreateScalar<int64_t>(2)},
                  CumminExtShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(0)},
                  CumminExtShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(-3)},
                  CumminExtShapeParams{{2, 3, 4, 5}, kInt32, CreateScalar<int64_t>(-2)},
                  CumminExtShapeParams{{-1, -1, -1}, kUInt64, CreateScalar<int64_t>(2)},
                  CumminExtShapeParams{{-2}, kFloat32, CreateScalar<int64_t>(2)}));

INSTANTIATE_TEST_CASE_P(
  TestCumminExtSimpleInfer, TestCumminExtSimpleInfer,
  testing::Values(CumminExtShapeParams{{3, 4, 5}, kFloat32, CreateScalar<int64_t>(2)},
                  CumminExtShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(0)},
                  CumminExtShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(-3)},
                  CumminExtShapeParams{{2, 3, 4, 5}, kInt32, CreateScalar<int64_t>(-2)},
                  CumminExtShapeParams{{-1, -1, -1}, kUInt64, CreateScalar<int64_t>(2)},
                  CumminExtShapeParams{{-2}, kFloat32, CreateScalar<int64_t>(2)}));
}  // namespace ops
}  // namespace mindspore
