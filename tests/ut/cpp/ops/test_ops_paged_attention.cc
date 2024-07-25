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
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "utils/tensor_construct_utils.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/paged_attention.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct PagedAttentionShapeParams {
  ShapeVector query_shape;
  TypePtr query_type;
  ShapeVector key_cache_shape;
  TypePtr key_cache_type;
  ShapeVector value_cache_shape;
  TypePtr value_cache_type;
  ShapeVector block_tables_shape;
  TypePtr block_tables_type;
  ShapeVector context_lens_shape;
  TypePtr context_lens_type;
  ValuePtr num_head;
  ValuePtr scale_value;
  ValuePtr kv_head;
};

class TestPagedAttention : public TestOps, public testing::WithParamInterface<PagedAttentionShapeParams> {};

TEST_P(TestPagedAttention, DynShape) {
  const auto &param = GetParam();
  auto query = std::make_shared<abstract::AbstractTensor>(param.query_type, param.query_shape);
  auto key_cache = std::make_shared<abstract::AbstractTensor>(param.key_cache_type, param.key_cache_shape);
  auto value_cache = std::make_shared<abstract::AbstractTensor>(param.value_cache_type, param.value_cache_shape);
  auto block_tables = std::make_shared<abstract::AbstractTensor>(param.block_tables_type, param.block_tables_shape);
  auto context_lens = std::make_shared<abstract::AbstractTensor>(param.context_lens_type, param.context_lens_shape);
  auto antiquant_scale = std::make_shared<abstract::AbstractNone>();
  auto antiquant_offset = std::make_shared<abstract::AbstractNone>();
  auto query_shape = std::make_shared<abstract::Shape>(param.query_shape);
  auto expect_shape = query_shape;
  auto expect_type = param.query_type;
  auto num_head = param.num_head->ToAbstract();
  auto scale_value = param.scale_value->ToAbstract();
  auto kv_head = param.kv_head->ToAbstract();

  PagedAttentionFuncImpl func_impl;
  auto prim = std::make_shared<Primitive>("PagedAttention");

  auto out_dtype = func_impl.InferType(prim, {query, key_cache, value_cache, block_tables, context_lens,
                                              antiquant_scale, antiquant_offset, num_head, scale_value, kv_head});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = func_impl.InferShape(prim, {query, key_cache, value_cache, block_tables, context_lens,
                                               antiquant_scale, antiquant_offset, num_head, scale_value, kv_head});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(TestPagedAttention, TestPagedAttention,
                        testing::Values(PagedAttentionShapeParams{{2, 40, 128},
                                                                  kFloat16,
                                                                  {256, 16, 40, 128},
                                                                  kFloat16,
                                                                  {256, 16, 40, 128},
                                                                  kFloat16,
                                                                  {2, 32},
                                                                  kInt32,
                                                                  {2},
                                                                  kInt32,
                                                                  CreateScalar<int>(40),
                                                                  CreateScalar<float>(1.0),
                                                                  CreateScalar<int>(40)},
                                        PagedAttentionShapeParams{{-1, 40, 128},
                                                                  kFloat16,
                                                                  {256, 16, 40, 128},
                                                                  kFloat16,
                                                                  {256, 16, 40, 128},
                                                                  kFloat16,
                                                                  {-1, 32},
                                                                  kInt32,
                                                                  {-1},
                                                                  kInt32,
                                                                  CreateScalar<int>(40),
                                                                  CreateScalar<float>(1.0),
                                                                  CreateScalar<int>(40)}));
}  // namespace ops
}  // namespace mindspore
