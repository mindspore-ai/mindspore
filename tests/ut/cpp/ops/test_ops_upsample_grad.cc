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
#include "ops/ops_func_impl/upsample_bilinear2d_grad.h"
#include "ops/ops_func_impl/upsample_linear1d_grad.h"
#include "ops/ops_func_impl/upsample_nearest1d_grad.h"
#include "ops/ops_func_impl/upsample_nearest2d_grad.h"
#include "ops/ops_func_impl/upsample_nearest3d_grad.h"
#include "ops/ops_func_impl/upsample_trilinear3d_grad.h"
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
struct UpsampleBackwardParams {
  ShapeVector dout_shape;
  ValuePtr input_size;  // tuple[int]
  ValuePtr output_size; // tuple[int]
  ValuePtr scales;      // tuple[float]
  ShapeVector out_shape;
};

static std::map<std::string, OpFuncImplPtr> upsample_backward_func_impl = {
  {kNameUpsampleBilinear2DGrad, std::make_shared<UpsampleBilinear2DGradFuncImpl>()},
  {kNameUpsampleLinear1DGrad, std::make_shared<UpsampleLinear1DGradFuncImpl>()},
  {kNameUpsampleNearest1DGrad, std::make_shared<UpsampleNearest1DGradFuncImpl>()},
  {kNameUpsampleNearest2DGrad, std::make_shared<UpsampleNearest2DGradFuncImpl>()},
  {kNameUpsampleNearest3DGrad, std::make_shared<UpsampleNearest3DGradFuncImpl>()},
  {kNameUpsampleTrilinear3DGrad, std::make_shared<UpsampleTrilinear3DGradFuncImpl>()},
};

class TestUpsampleBackward : public TestOps,
                             public testing::WithParamInterface<std::tuple<const char *, UpsampleBackwardParams>> {};

TEST_P(TestUpsampleBackward, dyn_shape) {
  const auto &upsample_mode = std::get<0>(GetParam());
  const auto &param = std::get<1>(GetParam());

  auto dout = std::make_shared<abstract::AbstractTensor>(kFloat32, param.dout_shape);
  ASSERT_NE(dout, nullptr);
  auto input_size = param.input_size->ToAbstract();
  ASSERT_NE(input_size, nullptr);
  auto output_size = param.output_size->ToAbstract();
  ASSERT_NE(output_size, nullptr);
  auto scales = param.scales->ToAbstract();
  ASSERT_NE(scales, nullptr);
  std::vector<AbstractBasePtr> input_args{dout, input_size, output_size, scales};

  static std::set<std::string> white_list{kNameUpsampleLinear1DGrad, kNameUpsampleBilinear2DGrad,
                                          kNameUpsampleTrilinear3DGrad};
  if (white_list.find(upsample_mode) != white_list.end()) {
    auto align_corners = CreateScalar<bool>(true)->ToAbstract();
    input_args.push_back(align_corners);
  }

  auto op_itr = upsample_backward_func_impl.find(upsample_mode);
  ASSERT_TRUE(op_itr != upsample_backward_func_impl.end());
  auto op_impl = op_itr->second;
  ASSERT_NE(op_impl, nullptr);

  auto prim = std::make_shared<Primitive>(upsample_mode);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto inferred_shape = op_impl->InferShape(prim, input_args);
  ShapeCompare(inferred_shape, expect_shape);
}

namespace {
float scale = 0.5;
auto Upsample3DDynTestCase = testing::ValuesIn(
  {UpsampleBackwardParams{
     {1, 3, 8, 8, 8}, CreatePyIntTuple({1, 3, 4, 4, 4}), CreatePyIntTuple({8, 8, 8}), kNone, {1, 3, 4, 4, 4}},
   UpsampleBackwardParams{{1, 3, 8, 8, 8},
                          CreatePyIntTuple({1, 3, kValueAny, kValueAny, kValueAny}),
                          CreatePyIntTuple({8, 8, 8}),
                          kNone,
                          {1, 3, -1, -1, -1}},
   UpsampleBackwardParams{
     {1, 3, 8, 8, 8}, kValueAny, kNone, CreateTuple({scale, scale, kValueAny}), {-1, -1, -1, -1, -1}}});

auto Upsample2DDynTestCase = testing::ValuesIn(
  {UpsampleBackwardParams{{1, 3, 8, 8}, CreatePyIntTuple({1, 3, 4, 4}), CreatePyIntTuple({8, 8}), kNone, {1, 3, 4, 4}},
   UpsampleBackwardParams{
     {1, 3, 8, 8}, CreatePyIntTuple({1, 3, kValueAny, kValueAny}), CreatePyIntTuple({8, 8}), kNone, {1, 3, -1, -1}},
   UpsampleBackwardParams{{1, 3, 8, 8}, kValueAny, kNone, CreateTuple({scale, kValueAny}), {-1, -1, -1, -1}}});

auto Upsample1DDynTestCase = testing::ValuesIn(
  {UpsampleBackwardParams{{1, 3, 8}, CreatePyIntTuple({1, 3, 4}), CreatePyIntTuple({8}), kNone, {1, 3, 4}},
   UpsampleBackwardParams{{1, 3, 8}, CreatePyIntTuple({1, 3, kValueAny}), CreatePyIntTuple({8}), kNone, {1, 3, -1}},
   UpsampleBackwardParams{{1, 3, 8}, kValueAny, kNone, CreateTuple({kValueAny}), {-1, -1, -1}}});
}  // namespace

INSTANTIATE_TEST_CASE_P(TestUpsampleNearest1DGradGroup, TestUpsampleBackward,
                        testing::Combine(testing::ValuesIn({kNameUpsampleNearest1DGrad}), Upsample1DDynTestCase));
INSTANTIATE_TEST_CASE_P(TestUpsampleLinear1DGradGroup, TestUpsampleBackward,
                        testing::Combine(testing::ValuesIn({kNameUpsampleLinear1DGrad}), Upsample1DDynTestCase));

INSTANTIATE_TEST_CASE_P(TestUpsampleBilinear2DGradGroup, TestUpsampleBackward,
                        testing::Combine(testing::ValuesIn({kNameUpsampleBilinear2DGrad}), Upsample2DDynTestCase));
INSTANTIATE_TEST_CASE_P(TestUpsampleNearest2DGradGroup, TestUpsampleBackward,
                        testing::Combine(testing::ValuesIn({kNameUpsampleNearest2DGrad}), Upsample2DDynTestCase));

INSTANTIATE_TEST_CASE_P(TestUpsampleNearest3DGradGroup, TestUpsampleBackward,
                        testing::Combine(testing::ValuesIn({kNameUpsampleNearest3DGrad}), Upsample3DDynTestCase));
INSTANTIATE_TEST_CASE_P(TestUpsampleTrilinear3DGradGroup, TestUpsampleBackward,
                        testing::Combine(testing::ValuesIn({kNameUpsampleTrilinear3DGrad}), Upsample3DDynTestCase));
}  // namespace ops
}  // namespace mindspore
