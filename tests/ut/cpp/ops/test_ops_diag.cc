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
#include "include/backend/optimizer/helper.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/diag.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct DiagShape {
  ShapeVector input_x_shape;
  ShapeVector output_shape;
};

struct DiagDtype {
  TypePtr input_x_dtype;
  TypePtr output_dtype;
};

class TestDiag : public TestOps, public testing::WithParamInterface<std::tuple<DiagShape, DiagDtype>> {};

TEST_P(TestDiag, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  DiagFuncImpl diag_func_impl;
  auto prim = std::make_shared<Primitive>("Diag");

  auto input_x = std::make_shared<abstract::AbstractTensor>(dtype_param.input_x_dtype, shape_param.input_x_shape);
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.output_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.output_dtype);

  auto out_shape = diag_func_impl.InferShape(prim, {input_x});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = diag_func_impl.InferType(prim, {input_x});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto DiagDynTestCase = testing::ValuesIn({
  /* static */
  DiagShape{{20, 30, 30}, {20, 30, 30, 20, 30, 30}},
  DiagShape{{2, 4}, {2, 4, 2, 4}},
  /* dynamic shape */
  DiagShape{{-1, -1, -1}, {-1, -1, -1, -1, -1, -1}},
  DiagShape{{-1, 2, -1}, {-1, 2, -1, -1, 2, -1}},
  DiagShape{{-1, 9}, {-1, 9, -1, 9}},
  /* dynamic rank */
  DiagShape{{-2}, {-2}},
});

auto DiagOpTypeCases = testing::ValuesIn({
  DiagDtype{kInt32, kInt32},
  DiagDtype{kInt64, kInt64},
  DiagDtype{kFloat32, kFloat32},
  DiagDtype{kFloat32, kFloat32},
  DiagDtype{kComplex64, kComplex64},
  DiagDtype{kComplex128, kComplex128},
});

INSTANTIATE_TEST_CASE_P(TestDiag, TestDiag, testing::Combine(DiagDynTestCase, DiagOpTypeCases));
}  // namespace ops
}  // namespace mindspore
