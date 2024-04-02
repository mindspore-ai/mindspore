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
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "ops/ops_func_impl/group_norm.h"

namespace mindspore {
namespace ops {
struct GroupNormOpParams {
  ShapeVector input_x_shape;
  TypePtr input_x_type;
  ValuePtr num_groups;
  ShapeVector gamma_shape;
  TypePtr gamma_type;
  ShapeVector beta_shape;
  TypePtr beta_type;
  ValuePtr epsilon;

  ShapeVector output_x_shape;
  TypePtr output_x_type;
  ShapeVector mean_shape;
  TypePtr mean_type;
  ShapeVector rstd_shape;
  TypePtr rstd_type;
};

class TestGroupNorm : public TestOps, public testing::WithParamInterface<GroupNormOpParams> {};

TEST_P(TestGroupNorm, group_norm_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("GroupNorm");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();
  auto input_x = std::make_shared<abstract::AbstractTensor>(param.input_x_type, param.input_x_shape);
  ASSERT_NE(input_x, nullptr);
  auto gamma = std::make_shared<abstract::AbstractTensor>(param.gamma_type, param.gamma_shape);
  ASSERT_NE(gamma, nullptr);
  auto beta = std::make_shared<abstract::AbstractTensor>(param.beta_type, param.beta_shape);
  ASSERT_NE(beta, nullptr);
  auto num_groups = param.num_groups->ToAbstract();
  ASSERT_NE(num_groups, nullptr);
  auto epsilon = param.epsilon->ToAbstract();
  ASSERT_NE(epsilon, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{
    std::move(input_x), std::move(num_groups), std::move(gamma), std::move(beta), std::move(epsilon)};
  auto infer_impl = std::make_shared<GroupNormFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto infer_shapes_ptr = infer_impl->InferShape(primitive, input_args);
  std::shared_ptr<abstract::TupleShape> infer_shapes =
    std::dynamic_pointer_cast<abstract::TupleShape>(infer_shapes_ptr);
  ASSERT_NE(infer_shapes, nullptr);
  auto infer_types_ptr = infer_impl->InferType(primitive, input_args);
  std::shared_ptr<Tuple> infer_types = std::dynamic_pointer_cast<Tuple>(infer_types_ptr);
  ASSERT_NE(infer_types, nullptr);
  auto expect_output_x_shape = std::make_shared<abstract::TensorShape>(param.output_x_shape);
  ASSERT_NE(expect_output_x_shape, nullptr);
  auto expect_output_x_type = std::make_shared<TensorType>(param.output_x_type);
  ASSERT_NE(expect_output_x_type, nullptr);
  auto expect_mean_shape = std::make_shared<abstract::TensorShape>(param.mean_shape);
  ASSERT_NE(expect_mean_shape, nullptr);
  auto expect_mean_type = std::make_shared<TensorType>(param.mean_type);
  ASSERT_NE(expect_mean_type, nullptr);
  auto expect_rstd_shape = std::make_shared<abstract::TensorShape>(param.rstd_shape);
  ASSERT_NE(expect_rstd_shape, nullptr);
  auto expect_rstd_type = std::make_shared<TensorType>(param.rstd_type);
  ASSERT_NE(expect_rstd_type, nullptr);
  ASSERT_TRUE(*((*infer_shapes)[0]) == *expect_output_x_shape);
  ASSERT_TRUE(*((*infer_shapes)[1]) == *expect_mean_shape);
  ASSERT_TRUE(*((*infer_shapes)[2]) == *expect_rstd_shape);
  ASSERT_TRUE(*((*infer_types)[0]) == *expect_output_x_type);
  ASSERT_TRUE(*((*infer_types)[1]) == *expect_mean_type);
  ASSERT_TRUE(*((*infer_types)[2]) == *expect_rstd_type);
}

INSTANTIATE_TEST_CASE_P(TestGroupNormGroup, TestGroupNorm,
                        testing::Values(GroupNormOpParams{{1, 2, 4, 4},
                                                          kFloat32,
                                                          CreateScalar<int64_t>(2),
                                                          {2},
                                                          kFloat32,
                                                          {2},
                                                          kFloat32,
                                                          CreateScalar<double>(1e-5f),
                                                          {1, 2, 4, 4},
                                                          kFloat32,
                                                          {1, 2},
                                                          kFloat32,
                                                          {1, 2},
                                                          kFloat32},
                                        GroupNormOpParams{{1, 2, 4, -1},
                                                          kFloat32,
                                                          CreateScalar<int64_t>(2),
                                                          {2},
                                                          kFloat32,
                                                          {2},
                                                          kFloat32,
                                                          CreateScalar<double>(1e-5f),
                                                          {1, 2, 4, -1},
                                                          kFloat32,
                                                          {1, 2},
                                                          kFloat32,
                                                          {1, 2},
                                                          kFloat32},
                                        GroupNormOpParams{{1, -1, 4, -1},
                                                          kFloat32,
                                                          CreateScalar<int64_t>(-1),
                                                          {-1},
                                                          kFloat32,
                                                          {-1},
                                                          kFloat32,
                                                          CreateScalar<double>(1e-5f),
                                                          {1, -1, 4, -1},
                                                          kFloat32,
                                                          {1, -1},
                                                          kFloat32,
                                                          {1, -1},
                                                          kFloat32},
                                        GroupNormOpParams{{-1, -1, -1, -1},
                                                          kFloat32,
                                                          CreateScalar<int64_t>(-1),
                                                          {-1},
                                                          kFloat32,
                                                          {-1},
                                                          kFloat32,
                                                          CreateScalar<double>(1e-5f),
                                                          {-1, -1, -1, -1},
                                                          kFloat32,
                                                          {-1, -1},
                                                          kFloat32,
                                                          {-1, -1},
                                                          kFloat32},
                                        GroupNormOpParams{{-2},
                                                          kFloat32,
                                                          CreateScalar<int64_t>(2),
                                                          {-2},
                                                          kFloat32,
                                                          {-2},
                                                          kFloat32,
                                                          CreateScalar<double>(1e-5f),
                                                          {-2},
                                                          kFloat32,
                                                          {-2},
                                                          kFloat32,
                                                          {-2},
                                                          kFloat32}));
}  // namespace ops
}  // namespace mindspore
