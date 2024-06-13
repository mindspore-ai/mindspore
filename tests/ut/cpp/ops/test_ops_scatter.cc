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

#include "ops/test_ops.h"
#include "ops/ops_func_impl/scatter.h"
#include "ops/test_value_utils.h"
#include "mindapi/base/types.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/auto_generate/gen_ops_name.h"

namespace mindspore {
namespace ops {
struct ScatterParams {
  ShapeVector x_shape;
  TypePtr x_dtype;
  ShapeVector index_shape;
  TypePtr index_dtype;
  ShapeVector src_shape;
  TypePtr src_dtype;
  ShapeVector out_shape;
  TypePtr out_dtype;
};

class TestScatter : public TestOps, public testing::WithParamInterface<ScatterParams> {};

TEST_P(TestScatter, scatter_dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_dtype, param.x_shape);
  auto dim = std::make_shared<abstract::AbstractScalar>(static_cast<int64_t>(0));
  auto index = std::make_shared<abstract::AbstractTensor>(param.index_dtype, param.index_shape);
  auto src = std::make_shared<abstract::AbstractTensor>(param.src_dtype, param.src_shape);
  auto reduce = std::make_shared<abstract::AbstractScalar>(static_cast<int64_t>(Reduce::REDUCE_NONE));
  auto expect = std::make_shared<abstract::AbstractTensor>(param.out_dtype, param.out_shape);

  std::vector<abstract::AbstractBasePtr> input_args{std::move(x), std::move(dim), std::move(index), std::move(src),
                                                    std::move(reduce)};
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_dtype);
  DoFuncImplInferAndCompare<ScatterFuncImpl>(kNameScatter, input_args, expect_shape, expect_type);
}

auto scatter_cases = testing::Values(
  /* static */
  ScatterParams{{4, 5, 6}, kFloat64, {2, 3, 4}, kInt64, {3, 4, 5}, kFloat64, {4, 5, 6}, kFloat64},
  /* -1 */
  ScatterParams{{-1, 3, -1}, kFloat32, {3, 3, 3}, kInt64, {3, 3, 3}, kFloat32, {-1, 3, -1}, kFloat32},
  /* -2 */
  ScatterParams{{-2}, kFloat64, {-2}, kInt64, {-2}, kFloat64, {-2}, kFloat64},
  ScatterParams{{-2}, kFloat64, {2, 3, 4}, kInt64, {-2}, kFloat64, {-1, -1, -1}, kFloat64},
  ScatterParams{{-2}, kFloat32, {-2}, kInt64, {-1, -1}, kFloat32, {-1, -1}, kFloat32});
INSTANTIATE_TEST_CASE_P(TestScatter, TestScatter, scatter_cases);
}  // namespace ops
}  // namespace mindspore
