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
#include "ops/ops_func_impl/addcdiv.h"
#include "ops/ops_func_impl/addcmul.h"

#ifndef TESTS_UT_CPP_OPS_TEST_OPS_ADDC_OP_H_
#define TESTS_UT_CPP_OPS_TEST_OPS_ADDC_OP_H_

namespace mindspore {
namespace ops {

struct ThreeInputBroadcastShape {
  std::vector<int64_t> x_shape;
  std::vector<int64_t> y_shape;
  std::vector<int64_t> z_shape;
  std::vector<int64_t> output_shape;
};

static std::vector<MultiInputOpParams> GetCases() {
  std::vector<MultiInputOpParams> cases;
  std::vector<TypePtr> input_dtypes = {kFloat16, kFloat16, kFloat16, kFloat16};
  std::vector<TypePtr> output_dtypes = {kFloat16};
  std::vector<ThreeInputBroadcastShape> shape_cases = {
    ThreeInputBroadcastShape{{4, 2, 3}, {4, 2, 3}, {4, 2, 3}, {4, 2, 3}},
    ThreeInputBroadcastShape{{3}, {2, 3}, {4, 2, 3}, {4, 2, 3}},
    ThreeInputBroadcastShape{{1}, {2, -1}, {4, -1, -1}, {4, 2, -1}},
    ThreeInputBroadcastShape{{3}, {2, -1}, {4, -1, -1}, {4, 2, 3}},
    ThreeInputBroadcastShape{{3}, {-2}, {4, -1, -1}, {-2}},
    ThreeInputBroadcastShape{{-2}, {-2}, {-2}, {-2}},
    ThreeInputBroadcastShape{{-1, -1, -1}, {-1, -1, -1}, {4, -1, -1}, {4, -1, -1}}};
  for (const auto &shape_case : shape_cases) {
    cases.push_back(MultiInputOpParams{{shape_case.x_shape, shape_case.y_shape, shape_case.z_shape, {}},
                                       input_dtypes,
                                       {shape_case.output_shape},
                                       output_dtypes,
                                       {}});
  }
  return cases;
}

#define ADDC_OP_FUNC_IMPL_TEST_WITH_DEFAULT_CASES(op_name) \
  OP_FUNC_IMPL_TEST_DECLARE(op_name, MultiInputOpParams)   \
  INSTANTIATE_TEST_CASE_P(Test##op_name, Test##op_name, testing::ValuesIn(GetCases()));

ADDC_OP_FUNC_IMPL_TEST_WITH_DEFAULT_CASES(Addcdiv);
ADDC_OP_FUNC_IMPL_TEST_WITH_DEFAULT_CASES(Addcmul);
}  // namespace ops
}  // namespace mindspore
#endif  // TESTS_UT_CPP_OPS_TEST_OPS_ADDC_OP_H_
