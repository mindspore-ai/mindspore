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
#include "ir/tensor_new.h"
#include "common/common_test.h"
#include "infer/ops_func_impl/repeat_interleave_grad.h"
#include "primitive/op_name.h"
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
struct RepeatInterleaveGradParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector repeats_shape;
  TypeId repeats_type;
  std::vector<int64_t> repeats_data;
  ValuePtr dim;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestRepeatInterleaveGrad : public TestOps, public testing::WithParamInterface<RepeatInterleaveGradParams> {};

TEST_P(TestRepeatInterleaveGrad, repeat_interleave_grad_dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto repeats_tensor = tensor::from_buffer(param.repeats_type, param.repeats_shape,
                                                         (void *)&param.repeats_data[0], param.repeats_type);
  auto repeats = repeats_tensor->ToAbstract();
  ASSERT_NE(x, nullptr);
  ASSERT_NE(repeats, nullptr);

  auto dim = param.dim->ToAbstract();
  ASSERT_NE(dim, nullptr);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);

  RepeatInterleaveGradFuncImpl repeat_interleave_grad_func_impl;
  auto prim = std::make_shared<Primitive>("RepeatInterleaveGrad");

  auto out_dtype = repeat_interleave_grad_func_impl.InferType(prim, {x, repeats, dim});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = repeat_interleave_grad_func_impl.InferShape(prim, {x, repeats, dim});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestRepeatInterleaveGrad, TestRepeatInterleaveGrad,
  testing::Values(
    RepeatInterleaveGradParams{{4, 3, 4}, kFloat32, {1}, kNumberTypeInt64, {2}, CreatePyInt(0), {2, 3, 4}, kFloat32},
    RepeatInterleaveGradParams{{2, 3, 8}, kFloat16, {1}, kNumberTypeInt64, {2}, CreatePyInt(-1), {2, 3, 4}, kFloat16},
    RepeatInterleaveGradParams{{7, 3, 4}, kFloat32, {2}, kNumberTypeInt64, {2, 5}, CreatePyInt(0), {2, 3, 4}, kFloat32},
    RepeatInterleaveGradParams{{-2}, kFloat32, {1}, kNumberTypeInt64, {2}, CreatePyInt(0), {-2}, kFloat32}));


// Additional UTs to verify enhanced error messages.
TEST(TestRepeatInterleaveGradError, repeats_contains_negative_value_should_throw_with_index_and_value) {
  ShapeVector x_shape{4, 3, 4};
  TypePtr x_type = kFloat32;
  auto x = std::make_shared<abstract::AbstractTensor>(x_type, x_shape);

  // repeats = [2, -1]
  ShapeVector repeats_shape{2};
  std::vector<int64_t> repeats_data{2, -1};
  auto repeats_tensor = tensor::from_buffer(kNumberTypeInt64, repeats_shape, (void *)&repeats_data[0], kNumberTypeInt64);
  auto repeats = repeats_tensor->ToAbstract();

  auto dim = CreatePyInt(0)->ToAbstract();

  RepeatInterleaveGradFuncImpl impl;
  auto prim = std::make_shared<Primitive>("RepeatInterleaveGrad");
  try {
    (void)impl.InferShape(prim, {x, repeats, dim});
    FAIL() << "Expected exception for negative repeats value";
  } catch (const std::exception &e) {
    std::string msg = e.what();
    // Expect message contains index and offending value.
    EXPECT_NE(msg.find("can not be negative"), std::string::npos);
    EXPECT_NE(msg.find("repeats[1]"), std::string::npos);
    EXPECT_NE(msg.find("-1"), std::string::npos);
  }
}

TEST(TestRepeatInterleaveGradError, repeats_zero_scalar_should_throw_with_value) {
  ShapeVector x_shape{4, 3, 4};
  TypePtr x_type = kFloat32;
  auto x = std::make_shared<abstract::AbstractTensor>(x_type, x_shape);

  // repeats = [0]
  ShapeVector repeats_shape{1};
  std::vector<int64_t> repeats_data{0};
  auto repeats_tensor = tensor::from_buffer(kNumberTypeInt64, repeats_shape, (void *)&repeats_data[0], kNumberTypeInt64);
  auto repeats = repeats_tensor->ToAbstract();

  auto dim = CreatePyInt(0)->ToAbstract();

  RepeatInterleaveGradFuncImpl impl;
  auto prim = std::make_shared<Primitive>("RepeatInterleaveGrad");
  try {
    (void)impl.InferShape(prim, {x, repeats, dim});
    FAIL() << "Expected exception for zero repeats value";
  } catch (const std::exception &e) {
    std::string msg = e.what();
    // Expect message mentions repeats[0] and zero.
    EXPECT_NE(msg.find("must not be zero"), std::string::npos);
    EXPECT_NE(msg.find("repeats[0]"), std::string::npos);
    EXPECT_NE(msg.find("0"), std::string::npos);
  }
}

TEST(TestRepeatInterleaveGradError, repeats_tensor_rank_gt_1_should_throw_with_dimension_value) {
  ShapeVector x_shape{4, 3, 4};
  TypePtr x_type = kFloat32;
  auto x = std::make_shared<abstract::AbstractTensor>(x_type, x_shape);

  // repeats is a 2-D tensor of shape [2, 2]
  ShapeVector repeats_shape{2, 2};
  std::vector<int64_t> repeats_data{1, 1, 1, 1};
  auto repeats_tensor = tensor::from_buffer(kNumberTypeInt64, repeats_shape, (void *)&repeats_data[0], kNumberTypeInt64);
  auto repeats = repeats_tensor->ToAbstract();

  auto dim = CreatePyInt(0)->ToAbstract();

  RepeatInterleaveGradFuncImpl impl;
  auto prim = std::make_shared<Primitive>("RepeatInterleaveGrad");
  try {
    (void)impl.InferShape(prim, {x, repeats, dim});
    FAIL() << "Expected exception for repeats with rank > 1";
  } catch (const std::exception &e) {
    std::string msg = e.what();
    // Expect message contains dimension value 2.
    EXPECT_NE(msg.find("0-dim or 1-dim tensor"), std::string::npos);
    EXPECT_NE(msg.find("dimension = 2"), std::string::npos);
  }
}
}  // namespace ops
}  // namespace mindspore
