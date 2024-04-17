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
#include "test_value_utils.h"
#include "ir/value.h"
#include "ir/primitive.h"
#include "ir/dtype/number.h"
#include "ir/tensor.h"
#include "ir/anf.h"
#include "mindapi/base/type_id.h"
#include "mindapi/base/shape_vector.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/randperm_v2.h"

namespace mindspore {
namespace ops {
struct RandpermV2OpParams {
  int64_t n;
  int64_t seed;
  int64_t offset;
  int64_t dtype;
  ShapeVector expect_shape;
  TypePtr expect_dtype;
};

class TestRandpermV2 : public TestOps, public testing::WithParamInterface<RandpermV2OpParams> {};

TEST_P(TestRandpermV2, dyn_shape) {
  const auto &param = GetParam();

  // input
  RandpermV2FuncImpl randperm_v2_func_impl;
  auto primitive = std::make_shared<Primitive>("RandpermV2");
  ASSERT_NE(primitive, nullptr);

  auto n = CreateScalar(param.n)->ToAbstract();
  auto seed = CreateScalar(param.seed)->ToAbstract();
  auto offset = CreateScalar(param.offset)->ToAbstract();
  auto dtype = CreateScalar(param.dtype)->ToAbstract();
  std::vector<AbstractBasePtr> input_args = {n, seed, offset, dtype};

  // execute
  auto out_shape = randperm_v2_func_impl.InferShape(primitive, input_args);
  auto out_dtype = randperm_v2_func_impl.InferType(primitive, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *std::make_shared<abstract::Shape>(param.expect_shape));
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *param.expect_dtype);
}

INSTANTIATE_TEST_CASE_P(TestOpsFuncImpl, TestRandpermV2,
                        testing::Values(RandpermV2OpParams{256, 0, 0, kNumberTypeUInt8, {256}, kUInt8},
                                        RandpermV2OpParams{128, 0, 0, kNumberTypeInt8, {128}, kInt8},
                                        RandpermV2OpParams{32678, 0, 0, kNumberTypeInt16, {32678}, kInt16},
                                        RandpermV2OpParams{2147483648, -1, 0, kNumberTypeInt32, {2147483648}, kInt32},
                                        RandpermV2OpParams{9223372036854775807, -1, 0, kNumberTypeInt64, {9223372036854775807}, kInt64}));
}  // namespace ops
}  // namespace mindspore
