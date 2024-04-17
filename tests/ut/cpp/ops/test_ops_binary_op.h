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

#ifndef TESTS_UT_CPP_OPS_TEST_OPS_BINARY_OP_H_
#define TESTS_UT_CPP_OPS_TEST_OPS_BINARY_OP_H_

namespace mindspore {
namespace ops {

static std::vector<MultiInputOpParams> GetBinaryOpDefaultCases() {
  std::vector<MultiInputOpParams> cases;
  std::vector<TypePtr> input_dtypes = {kFloat16, kFloat16};
  std::vector<TypePtr> output_dtypes = {kFloat16};
  // these case is from the test_ops_dyn_cases.h:
  std::vector<BroadcastOpShapeParams> BroadcastOpShapeTensorTensorCases = {
    /* y is number */
    BroadcastOpShapeParams{{10}, {}, {10}}, BroadcastOpShapeParams{{10, 1, 2}, {}, {10, 1, 2}},
    BroadcastOpShapeParams{{10, 4, 2}, {}, {10, 4, 2}}, BroadcastOpShapeParams{{10, 1, -1}, {}, {10, 1, -1}},
    BroadcastOpShapeParams{{-2}, {}, {-2}},
    /* x is number */
    BroadcastOpShapeParams{{}, {10}, {10}}, BroadcastOpShapeParams{{}, {10, 1, 2}, {10, 1, 2}},
    BroadcastOpShapeParams{{}, {10, 4, 2}, {10, 4, 2}}, BroadcastOpShapeParams{{}, {10, 1, -1}, {10, 1, -1}},
    BroadcastOpShapeParams{{}, {-2}, {-2}},
    /* x and y both tensor */
    BroadcastOpShapeParams{{4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}},
    BroadcastOpShapeParams{{2, 1, 4, 5, 6, 9}, {9}, {2, 1, 4, 5, 6, 9}},
    BroadcastOpShapeParams{{2, 3, 4, -1}, {2, 3, 4, 5}, {2, 3, 4, 5}},
    BroadcastOpShapeParams{{2, 3, 4, -1}, {-1, -1, 4, 5}, {2, 3, 4, 5}},
    BroadcastOpShapeParams{{2, 1, 4, -1}, {-1, -1, 4, 5}, {2, -1, 4, 5}},
    BroadcastOpShapeParams{{2, 1, 4, 5, 6, 9}, {-2}, {-2}}, BroadcastOpShapeParams{{2, 1, 4, 5, -1, 9}, {-2}, {-2}},
    BroadcastOpShapeParams{{-2}, {2, 1, 4, 5, 6, 9}, {-2}}, BroadcastOpShapeParams{{-2}, {2, 1, 4, 5, -1, 9}, {-2}},
    BroadcastOpShapeParams{{-2}, {-2}, {-2}}};
  for (const auto &broad_case : BroadcastOpShapeTensorTensorCases) {
    cases.push_back(MultiInputOpParams{
      {broad_case.x_shape, broad_case.y_shape}, input_dtypes, {broad_case.out_shape}, output_dtypes, {}});
  }
  return cases;
}

#define BINARY_OP_FUNC_IMPL_TEST_WITH_DEFAULT_CASES(op_name) \
  OP_FUNC_IMPL_TEST_DECLARE(op_name, MultiInputOpParams)     \
  INSTANTIATE_TEST_CASE_P(Test##op_name, Test##op_name, testing::ValuesIn(GetBinaryOpDefaultCases()));
}  // namespace ops
}  // namespace mindspore

#endif  // TESTS_UT_CPP_OPS_TEST_OPS_BINARY_OP_H_
