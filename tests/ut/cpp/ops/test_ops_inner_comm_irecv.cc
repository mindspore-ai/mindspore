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
#include "common/common_test.h"
#include "infer/ops_func_impl/communication/inner_comm_irecv.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct IrecvShapeParams {
  ValuePtr tag;
  ValuePtr src;
  ValuePtr shape;
  ValuePtr group;
  ValuePtr dtype;
  ShapeVector output_shape;
  TypePtr output_type;
};

class TestIrecv : public TestOps, public testing::WithParamInterface<IrecvShapeParams> {};

TEST_P(TestIrecv, dyn_shape) {
  const auto &param = GetParam();
  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_dtype = std::make_shared<TensorType>(param.output_type);

  InnerCommIrecvFuncImpl irecv_func_impl;
  auto prim = std::make_shared<Primitive>("InnerCommIrecv");

  auto tag_abs = param.tag->ToAbstract();
  auto src_abs = param.src->ToAbstract();
  auto shape_abs = param.shape->ToAbstract();
  auto group_abs = param.group->ToAbstract();
  auto dtype_abs = param.dtype->ToAbstract();

  auto out_dtype = irecv_func_impl.InferType(prim, {tag_abs, src_abs, shape_abs, group_abs, dtype_abs});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
  auto out_shape = irecv_func_impl.InferShape(prim, {tag_abs, src_abs, shape_abs, group_abs, dtype_abs});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(TestIrecv, TestIrecv,
                        testing::Values(IrecvShapeParams{CreateScalar<int64_t>(0),
                                                         CreateScalar<int64_t>(0),
                                                         CreatePyIntTuple({2, 2}),
                                                         CreateScalar<int64_t>(0),
                                                         CreateScalar<int64_t>(kNumberTypeFloat32),
                                                         {2, 2},
                                                         kFloat32},
                                        IrecvShapeParams{CreateScalar<int64_t>(0),
                                                         CreateScalar<int64_t>(0),
                                                         CreatePyIntTuple({-1, -1}),
                                                         CreateScalar<int64_t>(0),
                                                         CreateScalar<int64_t>(kNumberTypeFloat32),
                                                         {-1, -1},
                                                         kFloat32},
                                        IrecvShapeParams{CreateScalar<int64_t>(0),
                                                         CreateScalar<int64_t>(0),
                                                         CreatePyIntTuple({-2}),
                                                         CreateScalar<int64_t>(0),
                                                         CreateScalar<int64_t>(kNumberTypeInt64),
                                                         {-2},
                                                         kInt64}));
}  // namespace ops
}  // namespace mindspore
