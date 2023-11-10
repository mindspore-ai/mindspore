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
#include "ops/test_ops_cmp_utils.h"
#include "ir/dtype/number.h"
#include "ops/ops_func_impl/argmax.h"
#include "ops/test_value_utils.h"
#include "abstract/dshape.h"

namespace mindspore {
namespace ops {
static std::vector<MultiInputOpParams> GetCases() {
  auto dyn_rank = abstract::TensorShape::kShapeRankAny;
  auto dyn_dim = abstract::TensorShape::kShapeDimAny;
  std::vector<MultiInputOpParams> cases = {
    MultiInputOpParams{{{4, 2, 3}}, {kFloat16}, {{4, 3}}, {kInt64}, {CreatePyInt(1), CreatePyInt(kNumberTypeInt64)}},
    MultiInputOpParams{
      {{dyn_rank}}, {kFloat16}, {{dyn_rank}}, {kInt64}, {CreatePyInt(1), CreatePyInt(kNumberTypeInt64)}},
    MultiInputOpParams{{{4, 2, 3}},
                       {kFloat16},
                       {{dyn_dim, dyn_dim}},
                       {kInt64},
                       {CreateScalar(kValueAny), CreatePyInt(kNumberTypeInt64)}},
    MultiInputOpParams{
      {{4, dyn_dim, 3}}, {kFloat16}, {{4, 3}}, {kInt64}, {CreatePyInt(1), CreatePyInt(kNumberTypeInt64)}},
  };
  return cases;
}

#define ARG_OP_FUNC_IMPL_TEST_WITH_DEFAULT_CASES(op_name) \
  OP_FUNC_IMPL_TEST_DECLARE(op_name, MultiInputOpParams)  \
  INSTANTIATE_TEST_CASE_P(Test##op_name, Test##op_name, testing::ValuesIn(GetCases()));

ARG_OP_FUNC_IMPL_TEST_WITH_DEFAULT_CASES(Argmax)
}  // namespace ops
}  // namespace mindspore
