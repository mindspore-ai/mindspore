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

#include <memory>
#include "common/common_test.h"
#include "ops/test_ops.h"
#include "abstract/dshape.h"
#include "ops/ops_func_impl/rank.h"

namespace mindspore {
namespace ops {
struct RankOpParams {
  ShapeVector input_shape;
  TypePtr input_type;
  std::shared_ptr<abstract::NoShape> out_shape;
  TypePtr out_type;
};

class TestRank : public TestOps, public testing::WithParamInterface<RankOpParams> {};

TEST_P(TestRank, Rank_DynamicShape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  ASSERT_NE(input, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(input)};

  auto primitive = std::make_shared<Primitive>("Rank");
  OpFuncImplPtr infer_impl = std::make_shared<RankFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto infer_shape = infer_impl->InferShape(primitive, input_args);
  ASSERT_NE(infer_shape, nullptr);
  auto infer_type = infer_impl->InferType(primitive, input_args);
  ASSERT_NE(infer_type, nullptr);

  ASSERT_TRUE(infer_shape == param.out_shape);
  ASSERT_TRUE(infer_type == param.out_type);
}

INSTANTIATE_TEST_CASE_P(TestOpsFuncImpl, TestRank,
                        testing::Values(RankOpParams{{2, 3}, kFloat32, abstract::kNoShape, kInt64},
                                        RankOpParams{{-1, -1}, kFloat32, abstract::kNoShape, kInt64},
                                        RankOpParams{{-2}, kFloat32, abstract::kNoShape, kInt64}));
}  // namespace ops
}  // namespace mindspore
