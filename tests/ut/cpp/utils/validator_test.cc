/**
 * Copyright 2020-2026 Huawei Technologies Co., Ltd
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
#include <iostream>
#include <string>
#include "common/common_test.h"

#include "primitive/arithmetic_op_name.h"
#include "utils/log_adapter.h"
#include "frontend/jit/ps/validator.h"
#include "frontend/jit/ps/parse/parse.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "ir/value.h"
#include "frontend/jit/ps/static_analysis/prim.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace validator {
class TestValidator : public UT::Common {
 public:
  TestValidator() {}
  virtual ~TestValidator() {}

  virtual void TearDown() {}
};

/// Feature: Validate operation for primitives in graph
/// Description: Validate a ScalarAdd primitive node in normal graph
/// Expectation: No exception because ScalarAdd is a legal primitive
TEST_F(TestValidator, ValidateOperation01) {
  auto node = NewValueNode(std::make_shared<Primitive>("ScalarAdd"));
  ValidateOperation(node);
  // normally, the above statement should not exit, so expected the following statement execute
  EXPECT_TRUE(true);
}

/// Feature: Validate abstract for nodes in graph
/// Description: Validate abstract of a ValueNode with int64 value in a FuncGraph
/// Expectation: No exception because int64 abstract is legal
TEST_F(TestValidator, ValidateAbstract01) {
  AnfNodePtr node = NewValueNode(static_cast<int64_t>(1));
  abstract::AbstractBasePtr abstract_v1 = abstract::FromValue(static_cast<int64_t>(1), false);
  node->set_abstract(abstract_v1);
  auto func = std::make_shared<FuncGraph>();
  ValidateAbstract(node, func);
  // normally, the above statement should not exit, so expected the following statement execute
  EXPECT_TRUE(true);
}

/// Feature: Validate operation security for MindIR loading
/// Description: Validate a PyExecute primitive with is_load attribute from MindIR
/// Expectation: Throw exception because PyExecute is not allowed from MindIR load
TEST_F(TestValidator, ValidateOperationRejectPyExecuteFromMindIR) {
  auto prim = std::make_shared<Primitive>("PyExecute");
  prim->set_attr("is_load", MakeValue(true));
  auto node = NewValueNode(prim);
  EXPECT_THROW(ValidateOperation(node), std::exception);
}

/// Feature: Validate operation security for MindIR loading
/// Description: Validate a PyInterpret primitive with is_load attribute from MindIR
/// Expectation: Throw exception because PyInterpret is not allowed from MindIR load
TEST_F(TestValidator, ValidateOperationRejectPyInterpretFromMindIR) {
  auto prim = std::make_shared<Primitive>("PyInterpret");
  prim->set_attr("is_load", MakeValue(true));
  auto node = NewValueNode(prim);
  EXPECT_THROW(ValidateOperation(node), std::exception);
}

/// Feature: Validate operation security for MindIR loading
/// Description: Validate a PyFunc primitive with is_load attribute from MindIR
/// Expectation: Throw exception because PyFunc is not allowed from MindIR load
TEST_F(TestValidator, ValidateOperationRejectPyFuncFromMindIR) {
  auto prim = std::make_shared<Primitive>("PyFunc");
  prim->set_attr("is_load", MakeValue(true));
  auto node = NewValueNode(prim);
  EXPECT_THROW(ValidateOperation(node), std::exception);
}

/// Feature: Validate operation security for MindIR loading
/// Description: Validate a PyExecute primitive without is_load attribute in normal graph
/// Expectation: No exception because PyExecute without is_load is allowed in normal execution
TEST_F(TestValidator, ValidateOperationAllowPyExecute) {
  auto node = NewValueNode(std::make_shared<Primitive>("PyExecute"));
  ValidateOperation(node);
  EXPECT_TRUE(true);
}
}  // namespace validator
}  // namespace mindspore
