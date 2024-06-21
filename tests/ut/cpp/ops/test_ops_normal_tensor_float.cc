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
#include "ops/ops_func_impl/normal_tensor_float.h"
#include "ops/test_ops.h"
#include "test_value_utils.h"

namespace mindspore {
namespace ops {
struct NormalTensorFloatOpParams {
  ShapeVector mean_shape;
  TypePtr mean_type;
  ValuePtr std_float;
  ValuePtr seed_;
  ValuePtr offset_;
  ShapeVector output_shape;
  TypePtr output_type;
};
class TestNormalTensorFloat : public TestOps, public testing::WithParamInterface<NormalTensorFloatOpParams> {};

TEST_P(TestNormalTensorFloat, normal_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("NormalTensorFloat");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();
  auto mean = std::make_shared<abstract::AbstractTensor>(param.mean_type, param.mean_shape);
  ASSERT_NE(mean, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(mean)};

  auto std = std::make_shared<abstract::AbstractScalar>(param.std_float);
  ASSERT_NE(std, nullptr);
  input_args.push_back(std::move(std));

  auto seed_ = param.seed_->ToAbstract();
  ASSERT_NE(seed_, nullptr);

  input_args.push_back(std::move(seed_));
  auto offset_ = param.offset_->ToAbstract();
  ASSERT_NE(offset_, nullptr);
  input_args.push_back(std::move(offset_));

  auto infer_impl = std::make_shared<NormalTensorFloatFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto infer_shape = infer_impl->InferShape(primitive, input_args);
  ASSERT_NE(infer_shape, nullptr);
  auto infer_type = infer_impl->InferType(primitive, input_args);
  ASSERT_NE(infer_type, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_type = std::make_shared<TensorType>(param.output_type);
  ASSERT_NE(expect_type, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  ASSERT_TRUE(*infer_type == *expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestNormalTensorFloatGroup, TestNormalTensorFloat,
  testing::Values(
    NormalTensorFloatOpParams{
      {2, 2}, kFloat16, CreateScalar<float>(1.0), CreateScalar<float>(1.0), CreateScalar<float>(1.0), {2, 2}, kFloat16},
    NormalTensorFloatOpParams{{2, 2},
                              kFloat32,
                              CreateScalar<float>(1.0),
                              CreateScalar<float>(1.0),
                              CreateScalar<float>(1.0),
                              {2, 2},
                              kFloat32}));
}  // namespace ops
}  // namespace mindspore
