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
#include <vector>
#include <memory>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops/ops_func_impl/non_zero_ext.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_ops.h"

namespace mindspore {
namespace ops {
struct NonZeroExtOpParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeArray out_shapes;
  TypePtr out_type;
  bool is_compile_only;
};
class TestNonZeroExt : public TestOps, public testing::WithParamInterface<NonZeroExtOpParams> {};

TEST_P(TestNonZeroExt, non_zero_ext_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("Nonzeroext");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);

  if (param.is_compile_only) {
    AbstractBasePtrList output_abs;
    for (auto output_shape : param.out_shapes) {
      auto key_element = std::make_shared<abstract::AbstractTensor>(param.out_type, output_shape);
      output_abs.push_back(key_element);
    }
    auto expect_abs = std::make_shared<abstract::AbstractTuple>(output_abs);
    ASSERT_NE(expect_abs, nullptr);
    if (param.x_shape.size() == 1 && param.x_shape[0] == -2) {
      expect_abs->CheckAndConvertToDynamicLenSequence();
    }
    auto expect_shape = expect_abs->GetShape();
    auto expect_type = expect_abs->GetType();
    // infer 
    auto infer_impl = GetOpFrontendFuncImplPtr("NonZeroExt");
    ASSERT_NE(infer_impl, nullptr);
    std::vector<abstract::AbstractBasePtr> input_args{std::move(x)};
    auto infer_shape_type = infer_impl->InferAbstract(primitive, input_args);
    ASSERT_NE(infer_shape_type, nullptr);
    auto infer_shape = infer_shape_type->GetShape();
    ASSERT_NE(infer_shape, nullptr);
    auto infer_type = infer_shape_type->GetType();
    ASSERT_NE(infer_type, nullptr);
    ASSERT_TRUE(*infer_shape == *expect_shape);
    ASSERT_TRUE(*infer_type == *expect_type);
  } else {
    abstract::BaseShapePtrList output_shapes;
    TypePtrList output_types;
    output_shapes.reserve(param.out_shapes.size());
    for (auto output_shape : param.out_shapes) {
      auto out = std::make_shared<abstract::TensorShape>(output_shape);
      ASSERT_NE(out, nullptr);
      output_shapes.push_back(out);
      auto type = std::make_shared<TensorType>(param.out_type);
      output_types.push_back(type);
    }
    auto expect_shape = std::make_shared<abstract::TupleShape>(output_shapes);
    ASSERT_NE(expect_shape, nullptr);
    auto expect_type = std::make_shared<Tuple>(output_types);
    ASSERT_NE(expect_type, nullptr);
    DoFuncImplInferAndCompare<NonZeroExtFuncImpl>("NonZeroExt", {x}, expect_shape, expect_type);
  }
}

INSTANTIATE_TEST_CASE_P(TestNonZeroExtGroup, TestNonZeroExt,
                        testing::Values(
                          NonZeroExtOpParams{{2, 3}, kFloat, {{6},{6}}, kInt64, false},
                          NonZeroExtOpParams{{2, 2, 3}, kFloat, {{12},{12},{12}}, kInt64, false},
                          NonZeroExtOpParams{{3, 4}, kFloat, {{-1},{-1}}, kInt64, true},
                          NonZeroExtOpParams{{-1, -1, -1}, kFloat, {{-1},{-1},{-1}}, kInt64, true},
                          NonZeroExtOpParams{{-2}, kFloat, {{-2}}, kInt64, true}));
}  // namespace ops
}  // namespace mindspore
