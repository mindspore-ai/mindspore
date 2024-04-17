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

// testcase for ResizeBicubicGrad, ResizeBilinearGrad

#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ops/ops_func_impl/resize_bicubic_grad.h"
#include "ops/ops_func_impl/resize_bilinear_grad.h"
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
struct Resize2DGradParams {
  ShapeVector grads_shape;
  TypePtr grads_type;
  ShapeVector image_shape;
  TypePtr image_type;
  ValuePtr align_corners;       // bool
  ValuePtr half_pixel_centers;  // bool
  ShapeVector out_shape;
  TypePtr out_type;
};

static std::map<std::string, OpFuncImplPtr> resize_grad_func_impl = {
  {kNameResizeBicubicGrad, std::make_shared<ResizeBicubicGradFuncImpl>()},
  {kNameResizeBilinearGrad, std::make_shared<ResizeBilinearGradFuncImpl>()},
};

class TestResize2DGrad : public TestOps,
                         public testing::WithParamInterface<std::tuple<const char *, Resize2DGradParams>> {};

TEST_P(TestResize2DGrad, dyn_shape) {
  const auto &resize_grad_mode = std::get<0>(GetParam());
  const auto &param = std::get<1>(GetParam());
  auto grads = std::make_shared<abstract::AbstractTensor>(param.grads_type, param.grads_shape);
  ASSERT_NE(grads, nullptr);
  auto image = std::make_shared<abstract::AbstractTensor>(param.image_type, param.image_shape);
  ASSERT_NE(image, nullptr);
  auto align_corners = param.align_corners->ToAbstract();
  ASSERT_NE(align_corners, nullptr);
  auto half_pixel_centers = param.half_pixel_centers->ToAbstract();
  ASSERT_NE(half_pixel_centers, nullptr);

  auto resize_grad_op_itr = resize_grad_func_impl.find(resize_grad_mode);
  ASSERT_TRUE(resize_grad_op_itr != resize_grad_func_impl.end());
  auto op_impl = resize_grad_op_itr->second;
  ASSERT_NE(op_impl, nullptr);

  auto prim = std::make_shared<Primitive>(resize_grad_mode);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  auto inferred_shape = op_impl->InferShape(prim, {grads, image, align_corners, half_pixel_centers});
  auto inferred_type = op_impl->InferType(prim, {grads, image, align_corners, half_pixel_centers});
  ShapeCompare(inferred_shape, expect_shape);
  TypeCompare(inferred_type, expect_type);
}

namespace {
auto ResizeGradDynTestCase = testing::ValuesIn(
  {Resize2DGradParams{
     {1, 3, 4, 4}, kFloat32, {1, 3, 8, 8}, kFloat32, CreateScalar(true), CreateScalar(true), {1, 3, 8, 8}, kFloat32},
   Resize2DGradParams{
     {1, 3, -1, -1}, kFloat32, {1, 3, 8, 8}, kFloat32, CreateScalar(true), CreateScalar(true), {1, 3, 8, 8}, kFloat32},
   Resize2DGradParams{{-1, -1, -1, -1},
                      kFloat32,
                      {-1, -1, -1, -1},
                      kFloat32,
                      CreateScalar(true),
                      CreateScalar(true),
                      {-1, -1, -1, -1},
                      kFloat32},
   Resize2DGradParams{{-2}, kFloat32, {-2}, kFloat32, CreateScalar(true), CreateScalar(true), {-2}, kFloat32}});
}

INSTANTIATE_TEST_CASE_P(TestResizeBicubicGradGroup, TestResize2DGrad,
                        testing::Combine(testing::ValuesIn({kNameResizeBicubicGrad}), ResizeGradDynTestCase));
INSTANTIATE_TEST_CASE_P(TestResizeBilinearGradGroup, TestResize2DGrad,
                        testing::Combine(testing::ValuesIn({kNameResizeBilinearGrad}), ResizeGradDynTestCase));
}  // namespace ops
}  // namespace mindspore
