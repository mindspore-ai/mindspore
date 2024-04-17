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
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/geqrf.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct GeqrfShape {
  ShapeVector x_shape;
  ShapeVector y_shape;
  ShapeVector tau_shape;
};

struct GeqrfDtype {
  TypePtr x_type;
  TypePtr y_type;
  TypePtr tau_type;
};

class TestGeqrf : public TestOps, public testing::WithParamInterface<std::tuple<GeqrfShape, GeqrfDtype>> {};

TEST_P(TestGeqrf, geqrf_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  GeqrfFuncImpl geqrf_func_impl;
  auto prim = std::make_shared<Primitive>("Geqrf");
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);

  std::vector<BaseShapePtr> shapes_list = {std::make_shared<abstract::Shape>(shape_param.y_shape),
                                           std::make_shared<abstract::Shape>(shape_param.tau_shape)};
  auto expect_shape = std::make_shared<abstract::TupleShape>(std::vector<BaseShapePtr>{shapes_list});
  auto y_dtype_out = std::make_shared<TensorType>(dtype_param.y_type);
  auto tau_dtype_out = std::make_shared<TensorType>(dtype_param.tau_type);
  auto expect_dtype = std::make_shared<Tuple>(std::vector<TypePtr>{y_dtype_out, tau_dtype_out});

  auto out_shape = geqrf_func_impl.InferShape(prim, {x});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = geqrf_func_impl.InferType(prim, {x});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto GeqrfOpShapeTestCases = testing::ValuesIn({
  /* static */
  GeqrfShape{{2, 3, 4}, {2, 3, 4}, {2, 3}},
  /* dynamic shape */
  GeqrfShape{{-1, 2, 4}, {-1, 2, 4}, {-1, 2}},
  GeqrfShape{{5, 3, -1, 2, 1}, {5, 3, -1, 2, 1}, {5, 3, -1, 1}},
  GeqrfShape{{5, 3, -1, -1, 1, 4, 7, 4}, {5, 3, -1, -1, 1, 4, 7, 4}, {5, 3, -1, -1, 1, 4, 4}},
  GeqrfShape{{2, -1, -1, 5}, {2, -1, -1, 5}, {2, -1, -1}},
  GeqrfShape{{2, -1}, {2, -1}, {-1}},
  /* dynamic rank */
  GeqrfShape{{-2}, {-2}, {-2}},
});

auto GeqrfOpTypeTestCases = testing::ValuesIn({
  GeqrfDtype{kFloat32, kFloat32, kFloat32},
  GeqrfDtype{kFloat64, kFloat64, kFloat64},
  GeqrfDtype{kComplex64, kComplex64, kComplex64},
  GeqrfDtype{kComplex128, kComplex128, kComplex128},
});

INSTANTIATE_TEST_CASE_P(TestGeqrf, TestGeqrf, testing::Combine(GeqrfOpShapeTestCases, GeqrfOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
