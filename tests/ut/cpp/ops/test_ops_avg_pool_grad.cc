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
#include "ops/ops_func_impl/avg_pool_grad.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {
struct AvgPoolGradShape {
  ShapeVector x_origin_shape;
  ShapeVector out_shape;
};

struct AvgPoolGradDType {
  TypePtr x_origin_dtype;
  TypePtr out_dtype;
};

class TestAvgPoolGrad : public TestOps,
                        public testing::WithParamInterface<std::tuple<AvgPoolGradShape, AvgPoolGradDType>> {};

TEST_P(TestAvgPoolGrad, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  auto avg_pool_grad_func_impl = std::make_shared<AvgPoolGradFuncImpl>();
  auto prim = std::make_shared<Primitive>("AvgPoolGrad");

  auto x_origin = std::make_shared<abstract::AbstractTensor>(dtype_param.x_origin_dtype, shape_param.x_origin_shape);
  ASSERT_NE(x_origin, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_dtype);

  auto infer_shape = avg_pool_grad_func_impl->InferShape(prim, {x_origin});
  ASSERT_NE(infer_shape, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  auto infer_dtype = avg_pool_grad_func_impl->InferType(prim, {x_origin});
  ASSERT_NE(infer_dtype, nullptr);
  ASSERT_TRUE(*infer_dtype == *expect_dtype);
}

auto AvgPoolGradDynTestCase = testing::ValuesIn({
  AvgPoolGradShape{{1, 3, 5, 5}, {1, 3, 5, 5}},
  AvgPoolGradShape{{1, 3, -1, -1}, {1, 3, -1, -1}},
  AvgPoolGradShape{{-1, -1, -1, -1}, {-1, -1, -1, -1}},
  AvgPoolGradShape{{-2}, {-2}},
});

auto AvgPoolGradDTypeTestCase = testing::ValuesIn({
  AvgPoolGradDType{kFloat16, kFloat16},
  AvgPoolGradDType{kFloat32, kFloat32},
  AvgPoolGradDType{kFloat64, kFloat64},
});

INSTANTIATE_TEST_CASE_P(TestAvgPoolGradGroup, TestAvgPoolGrad,
                        testing::Combine(AvgPoolGradDynTestCase, AvgPoolGradDTypeTestCase));
}  // namespace ops
}  // namespace mindspore
