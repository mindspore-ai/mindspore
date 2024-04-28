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
#include "ops/test_ops_cmp_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/chunk.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct ChunkParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr chunks;  // can not be kValueAny.
  ValuePtr dims;  // can not be kValueAny.
  std::vector<ShapeVector> out_shape;
  std::vector<TypePtr> out_type;
};
class TestChunk : public TestOps, public testing::WithParamInterface<ChunkParams> {};

TEST_P(TestChunk, chunk_dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(x), std::move(param.chunks->ToAbstract()),
                                                    std::move(param.dims->ToAbstract())};
  auto expect = MakeOutputTupleShapeAndType(param.out_shape, param.out_type);
  auto expect_shape = expect.first;
  ASSERT_NE(expect_shape, nullptr);
  auto expect_type = expect.second;
  ASSERT_NE(expect_type, nullptr);
  DoFuncImplInferAndCompare<ChunkFuncImpl>(kNameChunk, input_args, expect_shape, expect_type);
};

INSTANTIATE_TEST_CASE_P(
  TestChunkGroup, TestChunk,
  testing::Values(
    ChunkParams{{2, 2}, kFloat32, CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), {{2, 2}}, {kFloat32}},
    ChunkParams{
      {2, 3}, kFloat32, CreateScalar<int64_t>(3), CreateScalar<int64_t>(0), {{1, 3}, {1, 3}}, {kFloat32, kFloat32}},
    ChunkParams{
      {4, 3}, kFloat32, CreateScalar<int64_t>(3), CreateScalar<int64_t>(0), {{2, 3}, {2, 3}}, {kFloat32, kFloat32}},
    ChunkParams{
      {3, 3}, kFloat32, CreateScalar<int64_t>(2), CreateScalar<int64_t>(0), {{2, 3}, {1, 3}}, {kFloat32, kFloat32}},
    ChunkParams{
      {-1, 2}, kFloat32, CreateScalar<int64_t>(2), CreateScalar<int64_t>(1), {{-1, 1}, {-1, 1}}, {kFloat32, kFloat32}},
    ChunkParams{
      {2, -1}, kFloat32, CreateScalar<int64_t>(1), CreateScalar<int64_t>(0), {{2, -1}}, {kFloat32}}));
}  // namespace ops
}  // namespace mindspore
