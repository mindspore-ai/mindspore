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
#include "common/common_test.h"
#include "ops/ops_func_impl/avg_pool2d.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore::ops {
struct AvgPool2DParams {
  ShapeVector input_shape;
  ValuePtr kernel_size;  // tuple[int]
  ValuePtr stride;       // tuple[int]
  ValuePtr padding;      // tuple[int]
  ValuePtr ceil_mode;    // bool
  ShapeVector out_shape;
};

class TestAvgPool2D : public TestOps, public testing::WithParamInterface<AvgPool2DParams> {};

TEST_P(TestAvgPool2D, dyn_shape) {
  const auto &param = GetParam();

  auto input = std::make_shared<abstract::AbstractTensor>(kFloat32, param.input_shape);
  ASSERT_NE(input, nullptr);
  auto kernel_size = param.kernel_size->ToAbstract();
  ASSERT_NE(kernel_size, nullptr);
  auto stride = param.stride->ToAbstract();
  ASSERT_NE(stride, nullptr);
  auto padding = param.padding->ToAbstract();
  ASSERT_NE(padding, nullptr);
  auto ceil_mode = param.ceil_mode->ToAbstract();
  ASSERT_NE(ceil_mode, nullptr);
  auto count_include_pad = CreateScalar<bool>(true)->ToAbstract();
  auto divisor_override = CreateScalar<int64_t>(int64_t(1))->ToAbstract();
  std::vector<AbstractBasePtr> input_args{input,     kernel_size,       stride,          padding,
                                          ceil_mode, count_include_pad, divisor_override};

  auto prim = std::make_shared<Primitive>(kNameAvgPool2D);
  auto infer_impl = std::make_shared<AvgPool2DFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto inferred_shape = infer_impl->InferShape(prim, input_args);

  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);

  ShapeCompare(inferred_shape, expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestAvgPool2D, TestAvgPool2D,
  testing::Values(AvgPool2DParams{ShapeVector{-2}, CreatePyIntTuple({2}), CreatePyIntTuple({1}),
                                  CreatePyIntTuple({0, 0}), CreateScalar<bool>(true), ShapeVector{-2}},

                  AvgPool2DParams{ShapeVector{-1, -1, 10}, CreatePyIntTuple({4}), CreatePyIntTuple({4, 4}),
                                  CreatePyIntTuple({0}), CreateScalar<bool>(false), ShapeVector{-1, -1, 2}},

                  AvgPool2DParams{ShapeVector{3, 10, 10}, CreatePyIntTuple({4, 4}), CreatePyIntTuple({4, 4}),
                                  CreatePyIntTuple({0, 0}), CreateScalar(kValueAny), ShapeVector{3, -1, -1}},
                  AvgPool2DParams{ShapeVector{4, 3, 10, 10}, kValueAny, CreatePyIntTuple({4, 4}),
                                  CreatePyIntTuple({0, 0}), CreateScalar<bool>(false), ShapeVector{4, 3, -1, -1}},
                  AvgPool2DParams{ShapeVector{2, 10, 10}, CreatePyIntTuple({4, 4}), kValueAny, CreatePyIntTuple({0, 0}),
                                  CreateScalar<bool>(false), ShapeVector{2, -1, -1}},
                  AvgPool2DParams{ShapeVector{5, 3, 10, 10}, CreatePyIntTuple({4, 4}), CreatePyIntTuple({4, 4}),
                                  kValueAny, CreateScalar<bool>(false), ShapeVector{5, 3, -1, -1}},

                  AvgPool2DParams{ShapeVector{4, 3, 10, 10}, CreatePyIntTuple({4, 4}), CreatePyIntTuple({kValueAny}),
                                  CreatePyIntTuple({0, 0}), CreateScalar<bool>(true), ShapeVector{4, 3, -1, -1}},
                  AvgPool2DParams{ShapeVector{4, 3, 10, 10}, CreatePyIntTuple({4, kValueAny}), CreatePyIntTuple({4}),
                                  CreatePyIntTuple({0}), CreateScalar<bool>(false), ShapeVector{4, 3, 2, -1}}));
}  // namespace mindspore::ops