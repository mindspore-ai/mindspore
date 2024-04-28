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
#include "ops/ops_func_impl/tracev2_grad.h"
#include "include/c_api/ms/base/types.h"

namespace mindspore {
namespace ops {

struct TraceV2GradShape {
  ShapeVector dout_shape;
  ValuePtr in_shape;  // shape is tuple[int]
  ValuePtr offset;
  ValuePtr axis1;
  ValuePtr axis2;
  ShapeVector din_shape;
};

struct TraceV2GradType {
  TypePtr dout_type;
  TypePtr din_type;
};

class TestTraceV2Grad : public TestOps,
                        public testing::WithParamInterface<std::tuple<TraceV2GradShape, TraceV2GradType>> {};

TEST_P(TestTraceV2Grad, dyn_shape) {
  // prepare
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());

  // input
  TraceV2GradFuncImpl tracev2_grad_func_impl;
  auto primitive = std::make_shared<Primitive>("TraceV2");
  ASSERT_NE(primitive, nullptr);
  auto dout_tensor = std::make_shared<abstract::AbstractTensor>(type_param.dout_type, shape_param.dout_shape);
  ASSERT_NE(dout_tensor, nullptr);
  auto in_shape = shape_param.in_shape->ToAbstract();
  ASSERT_NE(in_shape, nullptr);
  auto offset = shape_param.offset->ToAbstract();
  auto axis1 = shape_param.axis1->ToAbstract();
  auto axis2 = shape_param.axis2->ToAbstract();
  std::vector<AbstractBasePtr> input_args = {dout_tensor, in_shape, offset, axis1, axis2};

  // expect output
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.din_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_dtype = std::make_shared<TensorType>(type_param.din_type);
  ASSERT_NE(expect_dtype, nullptr);

  // execute
  auto out_shape = tracev2_grad_func_impl.InferShape(primitive, input_args);
  auto out_dtype = tracev2_grad_func_impl.InferType(primitive, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto tracev2_grad_shape_cases = testing::Values(TraceV2GradShape{{5, 5},
                                                                 CreatePyIntTuple({5, 5, 5, 5}),
                                                                 CreateScalar<int64_t>(0),
                                                                 CreateScalar<int64_t>(1),
                                                                 CreateScalar<int64_t>(0),
                                                                 {5, 5, 5, 5}},
                                                TraceV2GradShape{{-1, -1, -1},
                                                                 CreatePyIntTuple({5, 5, 5, 5, 5}),
                                                                 CreateScalar<int64_t>(0),
                                                                 CreateScalar<int64_t>(2),
                                                                 CreateScalar<int64_t>(0),
                                                                 {5, 5, 5, 5, 5}},
                                                TraceV2GradShape{{-2},
                                                                 CreatePyIntTuple({5, 5, 5, 5, 5}),
                                                                 CreateScalar<int64_t>(0),
                                                                 CreateScalar<int64_t>(1),
                                                                 CreateScalar<int64_t>(2),
                                                                 {5, 5, 5, 5, 5}});

auto tracev2_grad_type_cases = testing::ValuesIn({
  TraceV2GradType{kBool, kBool},
  TraceV2GradType{kInt8, kInt8},
  TraceV2GradType{kInt16, kInt16},
  TraceV2GradType{kInt32, kInt32},
  TraceV2GradType{kInt64, kInt64},
  TraceV2GradType{kUInt8, kUInt8},
  TraceV2GradType{kUInt16, kUInt16},
  TraceV2GradType{kUInt32, kUInt32},
  TraceV2GradType{kUInt64, kUInt64},
  TraceV2GradType{kFloat16, kFloat16},
  TraceV2GradType{kFloat32, kFloat32},
  TraceV2GradType{kFloat64, kFloat64},
  TraceV2GradType{kComplex64, kComplex64},
  TraceV2GradType{kComplex128, kComplex128},
});

INSTANTIATE_TEST_CASE_P(TestOpsFuncImpl, TestTraceV2Grad,
                        testing::Combine(tracev2_grad_shape_cases, tracev2_grad_type_cases));

}  // namespace ops
}  // namespace mindspore
