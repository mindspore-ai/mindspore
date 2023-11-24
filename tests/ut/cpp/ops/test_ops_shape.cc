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

#include "ops/test_ops.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/op_name.h"
#include "ops/ops_func_impl/shape.h"
#include "ops/ops_frontend_func_impl.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_value_utils.h"
#include "abstract/dshape.h"

namespace mindspore {
namespace ops {
struct ShapeShape {
  ShapeVector x_shape;
  ValuePtr output;
  bool dyn_rank;
  bool is_compile_only;
};

struct ShapeDType {
  TypePtr x_dtype;
  TypePtr out_dtype;
};

class TestShape : public TestOps, public testing::WithParamInterface<std::tuple<ShapeShape, ShapeDType>> {};

TEST_P(TestShape, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  auto prim = std::make_shared<Primitive>("Shape");
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_dtype, shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto expect_value = shape_param.output->ToAbstract();
  auto expect_shape = expect_value->GetShape();
  auto expect_dtype = std::make_shared<Tuple>(TypePtrList(shape_param.x_shape.size(), dtype_param.out_dtype));

  if (shape_param.is_compile_only) {
    std::vector<abstract::AbstractBasePtr> input_args{std::move(x)};
    auto infer_impl = GetOpFrontendFuncImplPtr("Shape");
    ASSERT_NE(infer_impl, nullptr);
    auto shape_func_impl = infer_impl->InferAbstract(prim, input_args);
    ASSERT_NE(shape_func_impl, nullptr);

    auto infer_shape = shape_func_impl->GetShape();
    ASSERT_NE(infer_shape, nullptr);
    if (shape_param.dyn_rank) {
      ASSERT_TRUE(infer_shape->isa<abstract::DynamicSequenceShape>());
    } else {
      ASSERT_TRUE(*infer_shape == *expect_shape);
    }
    auto infer_dtype = shape_func_impl->GetType();
    ASSERT_NE(infer_dtype, nullptr);
    ASSERT_TRUE(infer_dtype->number_type() == expect_dtype->number_type());
  } else {
    auto shape_func_impl = std::make_shared<ShapeFuncImpl>();

    auto infer_shape = shape_func_impl->InferShape(prim, {x});
    ASSERT_NE(infer_shape, nullptr);
    if (shape_param.dyn_rank) {
      ASSERT_TRUE(infer_shape->isa<abstract::DynamicSequenceShape>());
    } else {
      ASSERT_TRUE(*infer_shape == *expect_shape);
    }
    auto infer_dtype = shape_func_impl->InferType(prim, {x});
    ASSERT_NE(infer_dtype, nullptr);
    ASSERT_TRUE(infer_dtype->number_type() == expect_dtype->number_type());
  }
}

auto ShapeDynTestCase = testing::ValuesIn({
  ShapeShape{{2, 2}, CreateTuple({2, 2}), false, false},
  ShapeShape{{-1, -1}, CreateTuple({-1, -1}), false, true},
  ShapeShape{{-2}, CreateTuple({kValueAny}), true, true},
});

auto ShapeDTypeTestCase = testing::ValuesIn({
  ShapeDType{kFloat16, kInt64},
  ShapeDType{kFloat32, kInt64},
  ShapeDType{kFloat64, kInt64},
});

INSTANTIATE_TEST_CASE_P(TestShapeGroup, TestShape, testing::Combine(ShapeDynTestCase, ShapeDTypeTestCase));
}  // namespace ops
}  // namespace mindspore
