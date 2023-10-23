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

#include "ops/test_ops.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/op_name.h"
#include "ops/ops_func_impl/silu.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {
struct SiLUShape {
  ShapeVector x_shape;
  ShapeVector out_shape;
};

struct SiLUDType {
  TypePtr x_dtype;
  TypePtr out_dtype;
};

class TestSiLU : public TestOps, public testing::WithParamInterface<std::tuple<SiLUShape, SiLUDType>> {};

TEST_P(TestSiLU, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  auto silu_func_impl = std::make_shared<SiLUFuncImpl>();
  auto prim = std::make_shared<Primitive>("SiLU");

  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_dtype, shape_param.x_shape);
  ASSERT_NE(x, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_dtype);

  auto infer_shape = silu_func_impl->InferShape(prim, {x});
  ASSERT_NE(infer_shape, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  auto infer_dtype = silu_func_impl->InferType(prim, {x});
  ASSERT_NE(infer_dtype, nullptr);
  ASSERT_TRUE(*infer_dtype == *expect_dtype);
}

auto SiLUDynTestCase = testing::ValuesIn({
  SiLUShape{{1}, {1}},
  SiLUShape{{-1}, {-1}},
  SiLUShape{{-2}, {-2}},
});

auto SiLUDTypeTestCase = testing::ValuesIn({
  SiLUDType{kFloat16, kFloat16},
  SiLUDType{kFloat32, kFloat32},
  SiLUDType{kFloat64, kFloat64},
});

INSTANTIATE_TEST_CASE_P(TestSiLUGroup, TestSiLU, testing::Combine(SiLUDynTestCase, SiLUDTypeTestCase));
}  // namespace ops
}  // namespace mindspore
