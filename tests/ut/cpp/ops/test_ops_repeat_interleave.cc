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
#include "ops/ops_func_impl/repeat_interleave_int.h"
#include "ops/ops_func_impl/repeat_interleave_tensor.h"
#include "ops/op_name.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct RepeatInterleaveTensorParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector repeats_shape;
  TypeId repeats_type;
  std::vector<int64_t> repeats_data;
  ValuePtr dim;
  ShapeVector out_shape;
  TypePtr out_type;
};

struct RepeatInterleaveIntParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr repeats;
  ValuePtr dim;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestRepeatInterleaveTensor : public TestOps, public testing::WithParamInterface<RepeatInterleaveTensorParams> {};

TEST_P(TestRepeatInterleaveTensor, repeat_interleave_tensor_dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto repeats_tensor = std::make_shared<tensor::Tensor>(param.repeats_type, param.repeats_shape,
                                                         (void *)&param.repeats_data[0], param.repeats_type);
  auto repeats = repeats_tensor->ToAbstract();
  ASSERT_NE(x, nullptr);
  ASSERT_NE(repeats, nullptr);

  auto dim = param.dim->ToAbstract();
  ASSERT_NE(dim, nullptr);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);

  RepeatInterleaveTensorFuncImpl repeat_interleave_tensor_func_impl;
  auto prim = std::make_shared<Primitive>("RepeatInterleaveTensor");

  auto out_dtype = repeat_interleave_tensor_func_impl.InferType(prim, {x, repeats, dim});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = repeat_interleave_tensor_func_impl.InferShape(prim, {x, repeats, dim});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestRepeatInterleaveTensor, TestRepeatInterleaveTensor,
  testing::Values(
    RepeatInterleaveTensorParams{{2, 3, 4}, kFloat32, {1}, kNumberTypeInt64, {2}, CreatePyInt(0), {4, 3, 4}, kFloat32},
    RepeatInterleaveTensorParams{{2, 3, 4}, kFloat16, {1}, kNumberTypeInt64, {2}, CreatePyInt(-1), {2, 3, 8}, kFloat16},
    RepeatInterleaveTensorParams{{2, 3, 4}, kBool, {2}, kNumberTypeInt64, {2, 5}, CreatePyInt(0), {7, 3, 4}, kBool},
    RepeatInterleaveTensorParams{{-2}, kInt8, {1}, kNumberTypeInt64, {2}, CreatePyInt(0), {-2}, kInt8},
    RepeatInterleaveTensorParams{{2, 3, 4}, kInt16, {1}, kNumberTypeInt64, {2}, kValueAny, {-1, -1, -1}, kInt16},
    RepeatInterleaveTensorParams{{-2}, kInt32, {1}, kNumberTypeInt64, {2}, kValueAny, {-2}, kInt32},
    RepeatInterleaveTensorParams{
      {2, 3, -1}, kInt64, {3}, kNumberTypeInt64, {2, 3, 5}, CreatePyInt(1), {2, 10, -1}, kInt64}));

class TestRepeatInterleaveInt : public TestOps, public testing::WithParamInterface<RepeatInterleaveIntParams> {};

TEST_P(TestRepeatInterleaveInt, repeat_interleave_int_dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);

  auto repeats = param.repeats->ToAbstract();
  ASSERT_NE(repeats, nullptr);
  auto dim = param.dim->ToAbstract();
  ASSERT_NE(dim, nullptr);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);

  RepeatInterleaveIntFuncImpl repeat_interleave_int_func_impl;
  auto prim = std::make_shared<Primitive>("RepeatInterleaveInt");

  auto out_dtype = repeat_interleave_int_func_impl.InferType(prim, {x, repeats, dim});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = repeat_interleave_int_func_impl.InferShape(prim, {x, repeats, dim});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestRepeatInterleaveInt, TestRepeatInterleaveInt,
  testing::Values(RepeatInterleaveIntParams{{2, 3, 4}, kFloat32, CreatePyInt(2), CreatePyInt(0), {4, 3, 4}, kFloat32},
                  RepeatInterleaveIntParams{{2, 3, 4}, kFloat16, CreatePyInt(2), CreatePyInt(-1), {2, 3, 8}, kFloat16},
                  RepeatInterleaveIntParams{{-2}, kInt8, CreatePyInt(2), CreatePyInt(0), {-2}, kInt8},
                  RepeatInterleaveIntParams{{2, 3, 4}, kInt16, CreatePyInt(2), kValueAny, {-1, -1, -1}, kInt16},
                  RepeatInterleaveIntParams{{-2}, kInt32, CreatePyInt(2), kValueAny, {-2}, kInt32},
                  RepeatInterleaveIntParams{{2, 3, -1}, kInt64, CreatePyInt(2), CreatePyInt(1), {2, 6, -1}, kInt64}));
}  // namespace ops
}  // namespace mindspore
