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
#include "ops/test_value_utils.h"
#include "ops/ops_func_impl/sinh.h"

namespace mindspore {
namespace ops {

struct SinhShape {
  ShapeVector input_shape;
  ShapeVector output_shape;
};
struct SinhType {
  TypePtr input_type;
  TypePtr output_type;
};

class TestSinh : public TestOps, public testing::WithParamInterface<std::tuple<SinhShape, SinhType>> {};

TEST_P(TestSinh, sinh_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  SinhFuncImpl sinh_func_impl;
  auto prim = std::make_shared<Primitive>("Sinh");
  auto input = std::make_shared<abstract::AbstractTensor>(dtype_param.input_type, shape_param.input_shape);
  auto expect_shape = std::make_shared<abstract::TensorShape>(shape_param.output_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.output_type);

  auto output_shape = sinh_func_impl.InferShape(prim, {input});
  auto output_dtype = sinh_func_impl.InferType(prim, {input});
  ASSERT_TRUE(*output_shape == *expect_shape);
  ASSERT_TRUE(*output_dtype == *expect_dtype);
}

auto SinhOpShapeTestCases = testing::ValuesIn({
  /* static */
  SinhShape{{2}, {2}},
  SinhShape{{2, 3, 4}, {2, 3, 4}},
  /* dynamic shape */
  SinhShape{{-1}, {-1}},
  SinhShape{{-1, 2, 4}, {-1, 2, 4}},
  SinhShape{{5, 3, -1, 2, 1}, {5, 3, -1, 2, 1}},
  /* dynamic rank */
  SinhShape{{-2}, {-2}},
});

auto SinhOpTypeTestCases = testing::ValuesIn(
  {SinhType{kBool, kFloat32}, SinhType{kUInt8, kFloat32}, SinhType{kInt8, kFloat32}, SinhType{kInt16, kFloat32},
   SinhType{kInt32, kFloat32}, SinhType{kInt64, kFloat32}, SinhType{kFloat16, kFloat16}, SinhType{kFloat32, kFloat32},
   SinhType{kFloat64, kFloat64}, SinhType{kComplex64, kComplex64}, SinhType{kComplex128, kComplex128},
   SinhType{kBFloat16, kBFloat16}});

INSTANTIATE_TEST_CASE_P(TestSinh, TestSinh, testing::Combine(SinhOpShapeTestCases, SinhOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
