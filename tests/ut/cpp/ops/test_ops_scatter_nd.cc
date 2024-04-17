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
#include "ops/ops_func_impl/scatter_nd.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct TestScatterNdParams {
  ShapeVector indices_shape;
  TypePtr indices_type;
  ShapeVector updates_shape;
  TypePtr updates_type;
  ShapeVector shape_value;
  bool is_dyn_len;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestScatterNd : public TestOps, public testing::WithParamInterface<TestScatterNdParams> {};

TEST_P(TestScatterNd, scatter_nd_dyn_shape) {
  const auto &param = GetParam();
  auto indices = std::make_shared<abstract::AbstractTensor>(param.indices_type, param.indices_shape);
  auto updates = std::make_shared<abstract::AbstractTensor>(param.updates_type, param.updates_shape);
  ASSERT_NE(indices, nullptr);
  ASSERT_NE(updates, nullptr);
  AbstractBasePtrList values{};
  abstract::AbstractScalarPtr dim_abs = nullptr;
  for (auto dim : param.shape_value) {
    if (dim == kUnknown) {
      dim_abs = std::make_shared<abstract::AbstractScalar>(kValueAny);
    } else {
      dim_abs = std::make_shared<abstract::AbstractScalar>(dim);
    }
    (void)values.emplace_back(dim_abs);
  }
  auto shape = std::make_shared<abstract::AbstractTuple>(values);
  ASSERT_NE(shape, nullptr);
  if (param.is_dyn_len) {
    shape->CheckAndConvertToDynamicLenSequence();
  }
  auto infer_impl = std::make_shared<ScatterNdFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(indices), std::move(updates), std::move(shape)};
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  DoFuncImplInferAndCompare<ScatterNdFuncImpl>(kNameScatterNd, input_args, expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestScatterNd, TestScatterNd,
  testing::Values(
  TestScatterNdParams{{2, 1}, kInt32, {2, 3, 4}, kFloat32, {kUnknown, 3, 4}, false, {-1, 3, 4}, kFloat32},
  TestScatterNdParams{
      {2, 1}, kInt32, {2, 3, 4}, kFloat64, {kUnknown, kUnknown, kUnknown}, false, {-1, 3, 4}, kFloat64},
  TestScatterNdParams{
      {2, 3, 2}, kInt32, {2, 3, 4, 5}, kInt32, {kUnknown, kUnknown, kUnknown, kUnknown}, false, {-1, -1, 4, 5}, kInt32},
  TestScatterNdParams{
      {2, 1}, kInt64, {2, -1, -1}, kFloat32, {kUnknown, kUnknown, kUnknown}, false, {-1, -1, -1}, kFloat32},
  TestScatterNdParams{{2, 1}, kInt32, {-2}, kFloat32, {}, true, {-2}, kFloat32},
  TestScatterNdParams{{-1, -1}, kInt32, {2, 3, 4}, kFloat32, {5, kUnknown, kUnknown}, false, {5, -1, -1}, kFloat32},
  TestScatterNdParams{{2, 2}, kInt32, {2, 3, 4}, kFloat32, {}, true, {-1, -1, 3, 4}, kFloat32},
  TestScatterNdParams{{-1, 1}, kInt32, {2, 3, 4}, kFloat32, {5, kUnknown, kUnknown}, false, {5, 3, 4}, kFloat32},
  TestScatterNdParams{
      {-1, -1}, kInt32, {-1, -1, -1}, kFloat32, {kUnknown, kUnknown, kUnknown}, false, {-1, -1, -1}, kFloat32},
  TestScatterNdParams{{-2}, kInt32, {-2}, kFloat32, {}, true, {-2}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
