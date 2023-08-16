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
#include <string>
#include "ops/test_ops.h"
#include "common/common_test.h"
#include "ops/ops_func_impl/op_func_impl.h"

#ifndef MINDSPORE_TESTS_UT_CPP_OPS_TEST_OPS_CMP_UTILS_H_
#define MINDSPORE_TESTS_UT_CPP_OPS_TEST_OPS_CMP_UTILS_H_

namespace mindspore {
namespace ops {
void TestOpFuncImplWithEltwiseOpParams(const OpFuncImplPtr &infer_impl, const std::string &prim_name,
                                       const EltwiseOpParams &param);
void TestOpFuncImplWithMutiInputOpParams(const OpFuncImplPtr &infer_impl, const std::string &prim_name,
                                         const MutiInputOpParams &param);
#define OP_FUNC_IMPL_TEST_DECLARE(op_name, param_name)                                           \
  class Test##op_name : public TestOps, public testing::WithParamInterface<param_name> {};       \
  TEST_P(Test##op_name, op_name##_DynamicShape) {                                                \
    TestOpFuncImplWith##param_name(std::make_shared<op_name##FuncImpl>(), #op_name, GetParam()); \
  }

#define OP_FUNC_IMPL_TEST_CASES(op_name, cases) \
  INSTANTIATE_TEST_CASE_P(TestOpsFuncImpl, Test##op_name, cases);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_TESTS_UT_CPP_OPS_TEST_OPS_CMP_UTILS_H_
