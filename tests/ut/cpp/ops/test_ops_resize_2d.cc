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

// testcase for ResizeBicubic, ResizeBilinearV2, ResizeNearestNeighborV2

#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ops/ops_func_impl/resize_bicubic.h"
#include "ops/ops_func_impl/resize_bilinear_v2.h"
#include "ops/ops_func_impl/resize_nearest_neighbor_v2.h"
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
struct Resize2DParams {
  ShapeVector image_shape;
  TypePtr image_type;
  ValuePtr size;                // tuple[int*2]
  ValuePtr align_corners;       // bool
  ValuePtr half_pixel_centers;  // bool
  ShapeVector out_shape;
  TypePtr out_type;
};

static std::map<std::string, OpFuncImplPtr> resize_func_impl = {
  {kNameResizeBicubic, std::make_shared<ResizeBicubicFuncImpl>()},
  {kNameResizeBilinearV2, std::make_shared<ResizeBilinearV2FuncImpl>()},
  {kNameResizeNearestNeighborV2, std::make_shared<ResizeNearestNeighborV2FuncImpl>()},
};

class TestResize2D : public TestOps, public testing::WithParamInterface<std::tuple<const char *, Resize2DParams>> {};

TEST_P(TestResize2D, dyn_shape) {
  const auto &resize_mode = std::get<0>(GetParam());
  const auto &param = std::get<1>(GetParam());
  auto image = std::make_shared<abstract::AbstractTensor>(param.image_type, param.image_shape);
  ASSERT_NE(image, nullptr);
  auto size = param.size->ToAbstract();
  ASSERT_NE(size, nullptr);
  auto align_corners = param.align_corners->ToAbstract();
  ASSERT_NE(align_corners, nullptr);
  auto half_pixel_centers = param.half_pixel_centers->ToAbstract();
  ASSERT_NE(half_pixel_centers, nullptr);

  auto resize_op_itr = resize_func_impl.find(resize_mode);
  ASSERT_TRUE(resize_op_itr != resize_func_impl.end());
  auto op_impl = resize_op_itr->second;
  ASSERT_NE(op_impl, nullptr);

  auto prim = std::make_shared<Primitive>(resize_mode);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  auto inferred_shape = op_impl->InferShape(prim, {image, size, align_corners, half_pixel_centers});
  auto inferred_type = op_impl->InferType(prim, {image, size, align_corners, half_pixel_centers});
  ShapeCompare(inferred_shape, expect_shape);
  TypeCompare(inferred_type, expect_type);
}

namespace {
auto ResizeDynTestCase = testing::ValuesIn(
  {Resize2DParams{
     {1, 3, 8, 8}, kFloat32, CreatePyIntTuple({4, 4}), CreateScalar(true), CreateScalar(true), {1, 3, 4, 4}, kFloat32},
   Resize2DParams{{1, 3, 8, 8},
                  kFloat32,
                  CreatePyIntTuple({4, kValueAny}),
                  CreateScalar(true),
                  CreateScalar(true),
                  {1, 3, 4, -1},
                  kFloat32},
   Resize2DParams{{1, 3, 8, 8},
                  kFloat32,
                  CreatePyIntTuple({kValueAny, kValueAny}),
                  CreateScalar(true),
                  CreateScalar(true),
                  {1, 3, -1, -1},
                  kFloat32},
   Resize2DParams{{1, 3, 8, 8}, kFloat32, kValueAny, CreateScalar(true), CreateScalar(true), {1, 3, -1, -1}, kFloat32},
   Resize2DParams{{1, 3, -1, -1},
                  kFloat32,
                  CreatePyIntTuple({4, 4}),
                  CreateScalar(true),
                  CreateScalar(true),
                  {1, 3, 4, 4},
                  kFloat32},
   Resize2DParams{{1, 3, -1, -1},
                  kFloat32,
                  CreatePyIntTuple({4, kValueAny}),
                  CreateScalar(true),
                  CreateScalar(true),
                  {1, 3, 4, -1},
                  kFloat32},
   Resize2DParams{{1, 3, -1, -1},
                  kFloat32,
                  CreatePyIntTuple({kValueAny, kValueAny}),
                  CreateScalar(true),
                  CreateScalar(true),
                  {1, 3, -1, -1},
                  kFloat32},
   Resize2DParams{
     {1, 3, -1, -1}, kFloat32, kValueAny, CreateScalar(true), CreateScalar(true), {1, 3, -1, -1}, kFloat32},
   Resize2DParams{
     {-2}, kFloat32, CreatePyIntTuple({4, 4}), CreateScalar(true), CreateScalar(true), {-1, -1, 4, 4}, kFloat32},
   Resize2DParams{{-2},
                  kFloat32,
                  CreatePyIntTuple({kValueAny, kValueAny}),
                  CreateScalar(true),
                  CreateScalar(true),
                  {-1, -1, -1, -1},
                  kFloat32},
   Resize2DParams{{-2}, kFloat32, kValueAny, CreateScalar(true), CreateScalar(true), {-1, -1, -1, -1}, kFloat32}});
}

INSTANTIATE_TEST_CASE_P(TestResizeBicubicGroup, TestResize2D,
                        testing::Combine(testing::ValuesIn({kNameResizeBicubic}), ResizeDynTestCase));
INSTANTIATE_TEST_CASE_P(TestResizeBilinearV2Group, TestResize2D,
                        testing::Combine(testing::ValuesIn({kNameResizeBilinearV2}), ResizeDynTestCase));
INSTANTIATE_TEST_CASE_P(TestResizeNeighborV2Group, TestResize2D,
                        testing::Combine(testing::ValuesIn({kNameResizeNearestNeighborV2}), ResizeDynTestCase));
}  // namespace ops
}  // namespace mindspore
