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
#include "common/common_test.h"
#include "ops/ops_func_impl/median_ext.h"
#include "ops/ops_func_impl/median_dim.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct MedianDimShapeParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector values_shape;
  TypePtr values_type;
  ShapeVector indexes_shape;
  TypePtr indexes_type;
  AbstractBasePtr axis;
  AbstractBasePtr keep_dims;
};

static auto value_none = std::make_shared<abstract::AbstractScalar>(kValueAny, kTypeNone);
static auto keep_dims_true = std::make_shared<BoolImm>(true)->ToAbstract();
static auto keep_dims_false = std::make_shared<BoolImm>(false)->ToAbstract();

AbstractBasePtr CreateAxis(const int &value) {
  return CreatePyInt(value)->ToAbstract();
}

class TestMedianDim : public TestOps, public testing::WithParamInterface<MedianDimShapeParams> {};

TEST_P(TestMedianDim, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto axis = param.axis;
  auto keep_dims = param.keep_dims;

  auto values_shape = std::make_shared<abstract::Shape>(param.values_shape);
  auto slice_shape = std::make_shared<abstract::Shape>(param.indexes_shape);

  auto expect_shape = std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{values_shape, values_shape});
  auto expect_type = std::make_shared<Tuple>(std::vector<TypePtr>{std::make_shared<TensorType>(param.values_type), std::make_shared<TensorType>(param.indexes_type)});

  MedianDimFuncImpl median_dim_func_impl;
  auto prim = std::make_shared<Primitive>("MedianDim");
  auto input_args = std::vector<AbstractBasePtr>{x, axis, keep_dims};
  auto out_dtype = median_dim_func_impl.InferType(prim, input_args);
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = median_dim_func_impl.InferShape(prim, input_args);
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestMedianDim, TestMedianDim,
  testing::Values(MedianDimShapeParams{{3, 4, 5}, kFloat32, {1, 4, 5}, kFloat32, {1, 4, 5}, kInt64, CreateAxis(0), keep_dims_true},
                  MedianDimShapeParams{{3, 4, 5}, kInt64, {4, 5}, kInt64, {4, 5}, kInt64, CreateAxis(0), keep_dims_false},
                  MedianDimShapeParams{{2, 3, 4, 5}, kInt32, {2, 1, 4, 5}, kInt32, {2, 1, 4, 5}, kInt64, CreateAxis(1), keep_dims_true},
                  MedianDimShapeParams{{-1, -1, -1}, kUInt8, {1, -1, -1}, kUInt8, {1, -1, -1}, kInt64, CreateAxis(0), keep_dims_true},
                  MedianDimShapeParams{{-2}, kFloat32, {-2}, kFloat32, {-2}, kInt64, CreateAxis(0), keep_dims_true}
  ));

class TestMedianDimSimple : public TestOps, public testing::WithParamInterface<MedianDimShapeParams> {};

TEST_P(TestMedianDimSimple, simple_infer) {
  const auto &param = GetParam();
  auto x = std::make_shared<tensor::BaseTensor>(param.x_type->type_id(), param.x_shape);
  auto axis = param.axis;
  auto keep_dims = param.keep_dims;

  auto expect_shape = ShapeArray{param.values_shape, param.indexes_shape};
  auto expect_type =  TypePtrList{param.values_type, param.indexes_type};

  ValuePtrList input_values;
  input_values.push_back(std::move(x));
  input_values.push_back(std::move(param.axis->GetValue()));
  input_values.push_back(std::move(param.keep_dims->GetValue()));

  MedianDimFuncImpl median_dim_func_impl;
  auto prim = std::make_shared<Primitive>("MedianDim");
  auto out_dtype = median_dim_func_impl.InferType(prim, input_values);
  auto out_shape = median_dim_func_impl.InferShape(prim, input_values);
  ShapeCompare(out_shape, expect_shape);
  TypeCompare(out_dtype, expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestMedianDimSimple, TestMedianDimSimple,
  testing::Values(MedianDimShapeParams{{3, 4, 5}, kFloat32, {1, 4, 5}, kFloat32, {1, 4, 5}, kInt64, CreateAxis(0), keep_dims_true},
                  MedianDimShapeParams{{3, 4, 5}, kInt64, {4, 5}, kInt64, {4, 5}, kInt64, CreateAxis(0), keep_dims_false},
                  MedianDimShapeParams{{2, 3, 4, 5}, kInt32, {2, 1, 4, 5}, kInt32, {2, 1, 4, 5}, kInt64, CreateAxis(1), keep_dims_true}
  ));

struct MedianExtShapeParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector values_shape;
  TypePtr values_type;
};

class TestMedianExt : public TestOps, public testing::WithParamInterface<MedianExtShapeParams> {};

TEST_P(TestMedianExt, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);

  auto expect_shape = std::make_shared<abstract::Shape>(param.values_shape);
  auto expect_type = std::make_shared<TensorType>(param.values_type);

  MedianExtFuncImpl median_ext_func_impl;
  auto prim = std::make_shared<Primitive>("MedianExt");
  auto input_args = std::vector<AbstractBasePtr>{x};
  auto out_dtype = median_ext_func_impl.InferType(prim, input_args);
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = median_ext_func_impl.InferShape(prim, input_args);
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestMedianExt, TestMedianExt,
  testing::Values(MedianExtShapeParams{{3, 4, 5}, kFloat32, {}, kFloat32},
                  MedianExtShapeParams{{3, 4, 5}, kInt64, {}, kInt64},
                  MedianExtShapeParams{{2, 3, 4, 5}, kInt32, {}, kInt32},
                  MedianExtShapeParams{{-1, -1, -1}, kUInt8, {}, kUInt8},
                  MedianExtShapeParams{{-2}, kFloat32, {}, kFloat32}));

class TestMedianExtSimple : public TestOps, public testing::WithParamInterface<MedianExtShapeParams> {};

TEST_P(TestMedianExtSimple, simple_infer) {
  const auto &param = GetParam();
  auto x = std::make_shared<tensor::BaseTensor>(param.x_type->type_id(), param.x_shape);

  auto expect_shape = ShapeArray{param.values_shape};
  auto expect_type = TypePtrList{param.values_type};

  ValuePtrList input_values;
  input_values.push_back(std::move(x));

  MedianExtFuncImpl median_ext_func_impl;
  auto prim = std::make_shared<Primitive>("MedianExt");
  auto out_dtype = median_ext_func_impl.InferType(prim, input_values);
  auto out_shape = median_ext_func_impl.InferShape(prim, input_values);
  ShapeCompare(out_shape, expect_shape);
  TypeCompare(out_dtype, expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestMedianExtSimple, TestMedianExtSimple,
  testing::Values(MedianExtShapeParams{{3, 4, 5}, kFloat32, {}, kFloat32},
                  MedianExtShapeParams{{3, 4, 5}, kInt64, {}, kInt64},
                  MedianExtShapeParams{{2, 3, 4, 5}, kInt32, {}, kInt32}));
}  // namespace ops
}  // namespace mindspore
