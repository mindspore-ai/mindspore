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
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/reshape_and_cache.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct ReshapeAndCacheShapeParams {
 ShapeVector key_shape;
 TypePtr key_type;
 ShapeVector value_shape;
 TypePtr value_type;
 ShapeVector key_cache_shape;
 TypePtr key_cache_type;
 ShapeVector value_cache_shape;
 TypePtr value_cache_type;
 ShapeVector slot_mapping_shape;
 TypePtr slot_mapping_type;
};

class TestReshapeAndCache : public TestOps, public testing::WithParamInterface<ReshapeAndCacheShapeParams> {};

TEST_P(TestReshapeAndCache, DynShape) {
 const auto &param = GetParam();
 auto key = std::make_shared<abstract::AbstractTensor>(param.key_type, param.key_shape);
 auto value = std::make_shared<abstract::AbstractTensor>(param.value_type, param.value_shape);
 auto key_cache = std::make_shared<abstract::AbstractTensor>(param.key_cache_type, param.key_cache_shape);
 auto value_cache = std::make_shared<abstract::AbstractTensor>(param.value_cache_type, param.value_cache_shape);
 auto slot_mapping = std::make_shared<abstract::AbstractTensor>(param.slot_mapping_type, param.slot_mapping_shape);

 auto key_shape = std::make_shared<abstract::Shape>(param.key_shape);
 auto expect_shape = key_shape;
 auto expect_type = param.key_type;

 ReshapeAndCacheFuncImpl func_impl;
 auto prim = std::make_shared<Primitive>("ReshapeAndCache");

 auto out_dtype = func_impl.InferType(prim, {key, value, key_cache, value_cache, slot_mapping});
 ASSERT_TRUE(*out_dtype == *expect_type);
 auto out_shape = func_impl.InferShape(prim, {key, value, key_cache, value_cache, slot_mapping});
 ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestReshapeAndCache, TestReshapeAndCache,
  testing::Values(
    ReshapeAndCacheShapeParams{{3, 4, 5}, kFloat16, {3, 4, 5}, kFloat16, {2, 3, 4, 5}, kFloat16, {2, 3, 4, 5}, kFloat16, {12}, kInt32},
    ReshapeAndCacheShapeParams{{-1, 4, 5}, kFloat16, {-1, 4, 5}, kFloat16, {2, 3, 4, 5}, kFloat16, {2, 3, 4, 5}, kFloat16, {-1}, kInt32}
  ));
}  // namespace ops
}  // namespace mindspore
