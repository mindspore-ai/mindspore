/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include <exception>
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "include/backend/common/pass_manager/helper.h"
#include "ops/test_ops.h"
#include "infer/ops_func_impl/rmsnorm_quant.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"
#include "primitive/auto_generate/gen_ops_name_r.h"

namespace mindspore {
namespace ops {

struct RmsNormQuantShape {
  std::vector<int64_t> x_shape;
  std::vector<int64_t> gamma_shape;
  std::vector<int64_t> beta_shape;
  std::vector<int64_t> scale_shape;
  std::vector<int64_t> offset_shape;
  ValuePtr eps;

  std::vector<int64_t> out_shape;
};

struct RmsNormQuantType {
  TypePtr x_type;
  TypePtr gamma_type;
  TypePtr beta_type;
  TypePtr scale_type;
  TypePtr offset_type;

  TypePtr out_type;
};

class TestRmsNormQuant : public TestOps,
                         public testing::WithParamInterface<std::tuple<RmsNormQuantShape, RmsNormQuantType>> {};

class TestRmsNormQuantException : public TestOps,
                                  public testing::WithParamInterface<std::tuple<RmsNormQuantShape, RmsNormQuantType>> {
};

TEST_P(TestRmsNormQuant, RmsNormQuant_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  RmsNormQuantFuncImpl RmsNormQuant_func_impl;
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  auto gamma = std::make_shared<abstract::AbstractTensor>(dtype_param.gamma_type, shape_param.gamma_shape);
  auto beta = std::make_shared<abstract::AbstractTensor>(dtype_param.beta_type, shape_param.beta_shape);
  auto scale = std::make_shared<abstract::AbstractTensor>(dtype_param.scale_type, shape_param.scale_shape);
  auto offset = std::make_shared<abstract::AbstractTensor>(dtype_param.offset_type, shape_param.offset_shape);

  auto expect_out_shape = std::make_shared<abstract::TensorShape>(shape_param.out_shape);

  auto expect_out_type = std::make_shared<TensorType>(dtype_param.out_type);

  auto prim = std::make_shared<Primitive>(kNameRmsNormQuant);
  auto out_dtype =
    RmsNormQuant_func_impl.InferType(prim, {x, gamma, beta, scale, offset, shape_param.eps->ToAbstract()});
  ASSERT_TRUE(*out_dtype == *expect_out_type);
  auto out_shape =
    RmsNormQuant_func_impl.InferShape(prim, {x, gamma, beta, scale, offset, shape_param.eps->ToAbstract()});
  ASSERT_TRUE(*out_shape == *expect_out_shape);
}

TEST_P(TestRmsNormQuantException, RmsNormQuant_exception) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  RmsNormQuantFuncImpl RmsNormQuant_func_impl;
  auto prim = std::make_shared<Primitive>(kNameRmsNormQuant);
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  auto gamma = std::make_shared<abstract::AbstractTensor>(dtype_param.gamma_type, shape_param.gamma_shape);
  auto beta = std::make_shared<abstract::AbstractTensor>(dtype_param.beta_type, shape_param.beta_shape);
  auto scale = std::make_shared<abstract::AbstractTensor>(dtype_param.scale_type, shape_param.scale_shape);
  auto offset = std::make_shared<abstract::AbstractTensor>(dtype_param.offset_type, shape_param.offset_shape);

  bool raise_exception = false;
  try {
    (void)RmsNormQuant_func_impl.InferShape(prim, {x, gamma, beta, scale, offset, shape_param.eps->ToAbstract()});
  } catch (std::exception &e) {
    raise_exception = true;
  }

  ASSERT_TRUE(raise_exception);
}

auto RmsNormQuantOpShapeTestCases = testing::ValuesIn(
  {RmsNormQuantShape{{1, 1, 7168}, {7168}, {7168}, {1}, {1}, CreateScalar<float>(0.000001), {1, 1, 7168}},
   RmsNormQuantShape{{-1, -1, 7168}, {7168}, {7168}, {1}, {1}, CreateScalar<float>(0.000001), {-1, -1, 7168}}});

auto RmsNormQuantOpTypeTestCases = testing::ValuesIn({
  RmsNormQuantType{kFloat16, kFloat16, kFloat16, kFloat16, kInt8, kInt8},
  RmsNormQuantType{kBFloat16, kBFloat16, kBFloat16, kBFloat16, kInt8, kInt8},
});

auto RmsNormQuantOpInvalidShapeTestCases = testing::ValuesIn(
  {RmsNormQuantShape{{1, 1, 7168}, {7167}, {7168}, {1}, {1}, CreateScalar<float>(0.000001), {1, 1, 7168}},
   RmsNormQuantShape{{1, 1, 7168}, {7168}, {7167}, {1}, {1}, CreateScalar<float>(0.000001), {1, 1, 7168}},
   RmsNormQuantShape{{1, 1, 7168}, {7168}, {7168}, {2}, {1}, CreateScalar<float>(0.000001), {1, 1, 7168}},
   RmsNormQuantShape{{1, 1, 7168}, {7168}, {7168}, {1}, {2}, CreateScalar<float>(0.000001), {1, 1, 7168}}});

INSTANTIATE_TEST_CASE_P(TestRmsNormQuant, TestRmsNormQuant,
                        testing::Combine(RmsNormQuantOpShapeTestCases, RmsNormQuantOpTypeTestCases));

INSTANTIATE_TEST_CASE_P(TestRmsNormQuantException, TestRmsNormQuantException,
                        testing::Combine(RmsNormQuantOpInvalidShapeTestCases, RmsNormQuantOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
