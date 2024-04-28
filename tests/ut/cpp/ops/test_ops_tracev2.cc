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
#include <memory>
#include <vector>
#include <tuple>
#include "common/common_test.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/primal_attr.h"
#include "mindapi/base/shape_vector.h"
#include "test_value_utils.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/trace_v2.h"
#include "include/c_api/ms/base/types.h"

namespace mindspore {
namespace ops {

struct TraceV2Shape {
  ShapeVector input_shape;
  ValuePtr offset;
  ValuePtr axis1;
  ValuePtr axis2;
  ShapeVector output_shape;
};

struct TraceV2Type {
  TypePtr input_type;
  ValuePtr dtype;
  TypePtr output_type;
};

class TestTraceV2 : public TestOps, public testing::WithParamInterface<std::tuple<TraceV2Shape, TraceV2Type>> {};

TEST_P(TestTraceV2, dyn_shape) {
  // prepare
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());

  // input
  TraceV2FuncImpl tracev2_func_impl;
  auto primitive = std::make_shared<Primitive>("TraceV2");
  ASSERT_NE(primitive, nullptr);
  auto input_tensor = std::make_shared<abstract::AbstractTensor>(type_param.input_type, shape_param.input_shape);
  ASSERT_NE(input_tensor, nullptr);
  auto offset = shape_param.offset->ToAbstract();
  auto axis1 = shape_param.axis1->ToAbstract();
  auto axis2 = shape_param.axis2->ToAbstract();
  auto dtype = type_param.dtype->ToAbstract();
  std::vector<AbstractBasePtr> input_args = {input_tensor, offset, axis1, axis2, dtype};

  // expect output
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.output_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_dtype = std::make_shared<TensorType>(type_param.output_type);
  ASSERT_NE(expect_dtype, nullptr);

  // execute
  auto out_shape = tracev2_func_impl.InferShape(primitive, input_args);
  auto out_dtype = tracev2_func_impl.InferType(primitive, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto tracev2_shape_cases = testing::Values(
  TraceV2Shape{{5, 5}, CreateScalar<int64_t>(0), CreateScalar<int64_t>(1), CreateScalar<int64_t>(0), {}},
  TraceV2Shape{{5, 5, 5}, CreateScalar<int64_t>(0), CreateScalar<int64_t>(2), CreateScalar<int64_t>(0), {5}},
  TraceV2Shape{{-1, -1, -1}, CreateScalar<int64_t>(0), CreateScalar<int64_t>(2), CreateScalar<int64_t>(0), {-1}},
  TraceV2Shape{{-2}, CreateScalar<int64_t>(0), CreateScalar<int64_t>(1), CreateScalar<int64_t>(2), {-2}});

auto tracev2_type_cases = testing::ValuesIn({
  TraceV2Type{kFloat16, CreateScalar<int64_t>(kNumberTypeBool), kBool},
  TraceV2Type{kFloat32, CreateScalar<int64_t>(kNumberTypeInt8), kInt8},
  TraceV2Type{kFloat64, CreateScalar<int64_t>(kNumberTypeInt16), kInt16},
  TraceV2Type{kFloat16, CreateScalar<int64_t>(kNumberTypeInt32), kInt32},
  TraceV2Type{kFloat16, CreateScalar<int64_t>(kNumberTypeInt64), kInt64},
  TraceV2Type{kFloat32, CreateScalar<int64_t>(kNumberTypeUInt8), kUInt8},
  TraceV2Type{kFloat64, CreateScalar<int64_t>(kNumberTypeUInt16), kUInt16},
  TraceV2Type{kFloat16, CreateScalar<int64_t>(kNumberTypeUInt32), kUInt32},
  TraceV2Type{kFloat16, CreateScalar<int64_t>(kNumberTypeUInt64), kUInt64},
  TraceV2Type{kFloat16, CreateScalar<int64_t>(kNumberTypeFloat16), kFloat16},
  TraceV2Type{kFloat16, CreateScalar<int64_t>(kNumberTypeFloat32), kFloat32},
  TraceV2Type{kFloat16, CreateScalar<int64_t>(kNumberTypeFloat64), kFloat64},
  TraceV2Type{kFloat16, CreateScalar<int64_t>(kNumberTypeComplex64), kComplex64},
  TraceV2Type{kFloat16, CreateScalar<int64_t>(kNumberTypeComplex128), kComplex128},
});

INSTANTIATE_TEST_CASE_P(TestOpsFuncImpl, TestTraceV2, testing::Combine(tracev2_shape_cases, tracev2_type_cases));

}  // namespace ops
}  // namespace mindspore
