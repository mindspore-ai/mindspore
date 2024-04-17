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
#include "ops/ops_func_impl/strided_slice.h"
#include "ops/op_name.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct StridedSliceParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr begin;  // begin is tuple[int]
  ValuePtr end;
  ValuePtr strides;
  ValuePtr begin_mask;  // begin_mask is int
  ValuePtr end_mask;
  ValuePtr ellipsis_mask;
  ValuePtr new_axis_mask;
  ValuePtr shrink_axis_mask;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestStridedSlice : public TestOps, public testing::WithParamInterface<StridedSliceParams> {};

TEST_P(TestStridedSlice, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);
  auto begin = param.begin->ToAbstract();
  ASSERT_NE(begin, nullptr);
  auto end = param.end->ToAbstract();
  ASSERT_NE(end, nullptr);
  auto strides = param.strides->ToAbstract();
  ASSERT_NE(strides, nullptr);
  auto begin_mask = param.begin_mask->ToAbstract();
  ASSERT_NE(begin_mask, nullptr);
  auto end_mask = param.end_mask->ToAbstract();
  ASSERT_NE(end_mask, nullptr);
  auto ellipsis_mask = param.ellipsis_mask->ToAbstract();
  ASSERT_NE(ellipsis_mask, nullptr);
  auto new_axis_mask = param.new_axis_mask->ToAbstract();
  ASSERT_NE(new_axis_mask, nullptr);
  auto shrink_axis_mask = param.shrink_axis_mask->ToAbstract();
  ASSERT_NE(shrink_axis_mask, nullptr);


  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  DoFuncImplInferAndCompare<StridedSliceFuncImpl>(kNameStridedSlice, {x, begin, end, strides, begin_mask, end_mask,
    ellipsis_mask, new_axis_mask, shrink_axis_mask}, expect_shape, expect_type);
}

auto strided_slice_cases = testing::Values(
  // static shape
  StridedSliceParams{{6, 7, 8, 9, 10},
                      kFloat32,
                      CreatePyIntTuple({1, 2, 3, 2, 1}),
                      CreatePyIntTuple({5, 6, 7, 8, 9}),
                      CreatePyIntTuple({1, 2, 3, 2, 1}),
                      CreatePyInt(3),
                      CreatePyInt(5),
                      CreatePyInt(0),
                      CreatePyInt(8),
                      CreatePyInt(2),
                      {6, 2, 1, 8, 10},
                      kFloat32},
  StridedSliceParams{{6, 7, 8, 9, 10},
                      kFloat32,
                      CreatePyIntTuple({1, 2, 3, 2, 1}),
                      CreatePyIntTuple({5, 6, 7, 8, 9}),
                      CreatePyIntTuple({1, 2, 3, 2, 1}),
                      CreatePyInt(5),
                      CreatePyInt(2),
                      CreatePyInt(0),
                      CreatePyInt(10),
                      CreatePyInt(6),
                      {5, 1, 1, 7, 9, 10},
                      kFloat32},
  StridedSliceParams{{6, 7, 8, 9, 10},
                      kFloat32,
                      CreatePyIntTuple({1, 2, 3, 2, 1}),
                      CreatePyIntTuple({5, 6, 7, 8, 9}),
                      CreatePyIntTuple({1, 2, 3, 2, 1}),
                      CreatePyInt(3),
                      CreatePyInt(3),
                      CreatePyInt(0),
                      CreatePyInt(13),
                      CreatePyInt(5),
                      {1, 3, 1, 1, 6, 8, 9, 10},
                      kFloat32},
  StridedSliceParams{{6, 7, 8, 9, 10},
                      kFloat32,
                      CreatePyIntTuple({1, 2, 3, 2, 1}),
                      CreatePyIntTuple({5, 6, 7, 8, 9}),
                      CreatePyIntTuple({1, 2, 3, 2, 1}),
                      CreatePyInt(0),
                      CreatePyInt(0),
                      CreatePyInt(0),
                      CreatePyInt(15),
                      CreatePyInt(12),
                      {1, 1, 1, 1, 5, 7, 8, 9, 10},
                      kFloat32},
  // dynamic shape
  StridedSliceParams{{-1, 7, 8, -1, -1},
                      kFloat32,
                      CreatePyIntTuple({1, kValueAny, kValueAny, 2, 1}),
                      CreatePyIntTuple({5, 6, 7, 8, kValueAny}),
                      CreatePyIntTuple({1, 2, 3, kValueAny, 1}),
                      CreatePyInt(3),
                      CreatePyInt(5),
                      CreatePyInt(0),
                      CreatePyInt(8),
                      CreatePyInt(2),
                      {-1, -1, 1, -1, -1},
                      kFloat32},
  StridedSliceParams{{-1, 7, 8, -1, -1},
                      kFloat32,
                      CreatePyIntTuple({1, 2, 3, 2, 1}),
                      CreatePyIntTuple({5, 6, 7, 8, 9}),
                      CreatePyIntTuple({1, 2, 3, 2, 1}),
                      CreatePyInt(3),
                      CreatePyInt(5),
                      CreatePyInt(0),
                      CreatePyInt(8),
                      CreatePyInt(2),
                      {-1, 2, 1, -1, -1},
                      kFloat32},
  StridedSliceParams{{-1, 7, 8, -1, -1},
                      kFloat32,
                      CreatePyIntTuple({1, 2, 3, 2, 1}),
                      CreatePyIntTuple({5, 6, 7, 8, 9}),
                      CreatePyIntTuple({1, 2, 3, 2, 1}),
                      CreatePyInt(5),
                      CreatePyInt(2),
                      CreatePyInt(0),
                      CreatePyInt(10),
                      CreatePyInt(6),
                      {-1, 1, 1, 7, -1, -1},
                      kFloat32},
  StridedSliceParams{{-1, 7, 8, -1, -1},
                      kFloat32,
                      CreatePyIntTuple({1, 2, 3, 2, 1}),
                      CreatePyIntTuple({5, 6, 7, 8, 9}),
                      CreatePyIntTuple({1, 2, 3, 2, 1}),
                      CreatePyInt(3),
                      CreatePyInt(3),
                      CreatePyInt(0),
                      CreatePyInt(13),
                      CreatePyInt(5),
                      {1, -1, 1, 1, 6, 8, -1, -1},
                      kFloat32},
  StridedSliceParams{{-1, 7, 8, -1, -1},
                      kFloat32,
                      CreatePyIntTuple({1, 2, 3, 2, 1}),
                      CreatePyIntTuple({5, 6, 7, 8, 9}),
                      CreatePyIntTuple({1, 2, 3, 2, 1}),
                      CreatePyInt(0),
                      CreatePyInt(0),
                      CreatePyInt(0),
                      CreatePyInt(15),
                      CreatePyInt(12),
                      {1, 1, 1, 1, -1, 7, 8, -1, -1},
                      kFloat32});

INSTANTIATE_TEST_CASE_P(TestStridedSliceGroup, TestStridedSlice, strided_slice_cases);
}  // namespace ops
}  // namespace mindspore
