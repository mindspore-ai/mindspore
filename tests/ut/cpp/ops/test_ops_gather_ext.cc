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
#include "ops/test_ops.h"
#include "ops/ops_func_impl/gather_ext.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
//  输入输出参数，包括Tensor输入的shape、type，属性转输入的ValuePtr，期望输出Tensor的shape、type
struct GatherExtShapeParams {
  ShapeVector input_shape;
  TypePtr input_type;
  ValuePtr dim;
  ShapeVector index_shape;
  TypePtr index_type;
};
class TestGatherExt : public TestOps, public testing::WithParamInterface<GatherExtShapeParams> {};
class TestGatherExtException : public TestOps, public testing::WithParamInterface<GatherExtShapeParams> {};
//  测试代码主体
TEST_P(TestGatherExt, dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto dim = param.dim->ToAbstract();
  auto index = std::make_shared<abstract::AbstractTensor>(param.index_type, param.index_shape);

  auto expect_shape = index->GetShape();
  auto expect_type = input->GetType();

  //  生成算子infer实例
  GatherExtFuncImpl gather_ext_func_impl;
  auto prim = std::make_shared<Primitive>("GatherExt");
  //  调用算子InferSahpe & InferType并与期望进行对比
  auto out_dtype = gather_ext_func_impl.InferType(prim, {input, dim, index});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = gather_ext_func_impl.InferShape(prim, {input, dim, index});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

TEST_P(TestGatherExtException, exception) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto dim = param.dim->ToAbstract();
  auto index = std::make_shared<abstract::AbstractTensor>(param.index_type, param.index_shape);

  auto expect_shape = index->GetShape();
  auto expect_type = input->GetType();

  //  生成算子infer实例
  GatherExtFuncImpl gather_ext_func_impl;
  auto prim = std::make_shared<Primitive>("GatherExt");

  try {
    (void)gather_ext_func_impl.CheckValidation(prim, {input, dim, index});
  } catch (std::exception &e) {
    ASSERT_TRUE(true);
    return;
  }

  ASSERT_TRUE(false);
}
//  测试用例批量测试参数
INSTANTIATE_TEST_CASE_P(
  TestGatherExt, TestGatherExt,
  testing::Values(GatherExtShapeParams{{3, 4, 5}, kFloat32, CreateScalar<int64_t>(2), {3, 4, 5}, kInt64},
                  GatherExtShapeParams{{}, kFloat32, CreateScalar<int64_t>(0), {}, kInt64},
                  GatherExtShapeParams{{1}, kFloat32, CreateScalar<int64_t>(0), {}, kInt64},
                  GatherExtShapeParams{{}, kFloat32, CreateScalar<int64_t>(0), {1}, kInt64},
                  GatherExtShapeParams{{-1, -1, -1}, kInt64, CreateScalar<int64_t>(0), {3, 4, 5}, kInt64},
                  GatherExtShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(0), {-1, -1, -1}, kInt64},
                  GatherExtShapeParams{{3, 4, 5, 6}, kInt64, CreateScalar<int64_t>(0), {-2}, kInt64},
                  GatherExtShapeParams{{-2}, kInt64, CreateScalar<int64_t>(0), {-1, -1, -1, -1}, kInt64}));
INSTANTIATE_TEST_CASE_P(
  TestGatherExtException, TestGatherExtException,
  testing::Values(
    GatherExtShapeParams{{3, 4, 5}, kFloat32, CreateScalar<int64_t>(3), {3, 4, 5}, kInt64},  // dim invalid
    GatherExtShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(-4), {3, 4, 5}, kInt64},   // dim invalid
    GatherExtShapeParams{{}, kFloat32, CreateScalar<int64_t>(1), {1}, kInt64},               // dim invalid
    GatherExtShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(0), {2, 3}, kInt64},       // rank not equal
    GatherExtShapeParams{{3, 4}, kInt64, CreateScalar<int64_t>(1), {4, 3}, kInt64}));
}  // namespace ops
}  // namespace mindspore
