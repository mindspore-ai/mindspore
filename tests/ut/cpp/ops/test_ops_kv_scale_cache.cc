/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "include/backend/common/pass_manager/helper.h"
#include "ops/test_ops.h"
#include "infer/ops_func_impl/kv_scale_cache.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct KvScaleCacheShapeParams {
  ShapeVector key_scale_shape;
  TypePtr key_scale_type;
  ShapeVector value_scale_shape;
  TypePtr value_scale_type;
  ShapeVector key_value_scale_cache_shape;
  TypePtr key_value_scale_cache_type;
  ShapeVector batch_valid_length_shape;
  TypePtr batch_valid_length_type;
  ValuePtr cache_mode;
};

class TestKvScaleCache : public TestOps, public testing::WithParamInterface<KvScaleCacheShapeParams> {};

TEST_P(TestKvScaleCache, DynShape) {
  const auto &param = GetParam();
  auto key_scale = std::make_shared<abstract::AbstractTensor>(param.key_scale_type, param.key_scale_shape);
  auto value_scale = std::make_shared<abstract::AbstractTensor>(param.value_scale_type, param.value_scale_shape);
  auto batch_valid_length =
    std::make_shared<abstract::AbstractTensor>(param.batch_valid_length_type, param.batch_valid_length_shape);
  auto key_value_scale_cache =
    std::make_shared<abstract::AbstractTensor>(param.key_value_scale_cache_type, param.key_value_scale_cache_shape);
  auto cache_mode = param.cache_mode->ToAbstract();
  auto key_value_scale_cache_shape = std::make_shared<abstract::Shape>(param.key_value_scale_cache_shape);
  auto expect_shape = key_value_scale_cache_shape;
  auto expect_type = param.key_value_scale_cache_type;

  KvScaleCacheFuncImpl func_impl;
  auto prim = std::make_shared<Primitive>("KvScaleCache");
  auto out_dtype =
    func_impl.InferType(prim, {key_scale, value_scale, key_value_scale_cache, batch_valid_length, cache_mode});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape =
    func_impl.InferShape(prim, {key_scale, value_scale, key_value_scale_cache, batch_valid_length, cache_mode});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestKvScaleCache, TestKvScaleCache,
  testing::Values(
    KvScaleCacheShapeParams{
      {1, 4}, kFloat32, {1, 4}, kFloat32, {2, 14, 1024}, kFloat32, {12}, kInt32, CreateScalar<int64_t>(1)},
    KvScaleCacheShapeParams{
      {12, 1}, kFloat32, {12, 1}, kFloat32, {2, 14, 1024}, kFloat32, {12}, kInt32, CreateScalar<int64_t>(0)}));
}  // namespace ops
}  // namespace mindspore
