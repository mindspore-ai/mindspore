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
#include "ops/ops_func_impl/kv_cache_scatter_update.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct TestKVCacheScatterUpdateParams {
  ShapeVector var_shape;
  TypePtr var_type;
  ShapeVector indices_shape;
  TypePtr indices_type;
  ShapeVector updates_shape;
  TypePtr updates_type;
  ValuePtr axis;
  ValuePtr reduce;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestKVCacheScatterUpdate : public TestOps, public testing::WithParamInterface<TestKVCacheScatterUpdateParams> {};

TEST_P(TestKVCacheScatterUpdate, scatter_dyn_shape) {
  const auto &param = GetParam();
  auto var = std::make_shared<abstract::AbstractTensor>(param.var_type, param.var_shape);
  auto indices = std::make_shared<abstract::AbstractTensor>(param.indices_type, param.indices_shape);
  auto updates = std::make_shared<abstract::AbstractTensor>(param.updates_type, param.updates_shape);
  ASSERT_NE(var, nullptr);
  ASSERT_NE(indices, nullptr);
  ASSERT_NE(updates, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(var), std::move(indices), std::move(updates),
                                                    param.axis->ToAbstract(), param.reduce->ToAbstract()};
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  DoFuncImplInferAndCompare<KVCacheScatterUpdateFuncImpl>(kNameKVCacheScatterUpdate, input_args, expect_shape,
                                                          expect_type);
}

INSTANTIATE_TEST_CASE_P(TestKVCacheScatterUpdateGroup, TestKVCacheScatterUpdate,
                        testing::Values(TestKVCacheScatterUpdateParams{{1, 5, 32, 4096},
                                                                       kFloat32,
                                                                       {1},
                                                                       kInt32,
                                                                       {1, 5, 32, 1},
                                                                       kFloat32,
                                                                       CreateScalar<int64_t>(-1),
                                                                       CreateScalar<int64_t>(3),
                                                                       {1, 5, 32, 4096},
                                                                       kFloat32},
                                        TestKVCacheScatterUpdateParams{{1, 5, 32, 4096},
                                                                       kFloat32,
                                                                       {1},
                                                                       kInt64,
                                                                       {1, 5, 32, 1},
                                                                       kFloat32,
                                                                       CreateScalar<int64_t>(-1),
                                                                       CreateScalar<int64_t>(3),
                                                                       {1, 5, 32, 4096},
                                                                       kFloat32}));
}  // namespace ops
}  // namespace mindspore
