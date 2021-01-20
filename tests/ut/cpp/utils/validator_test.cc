/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "utils/log_adapter.h"
#include "pipeline/jit/validator.h"
#include "pipeline/jit/parse/parse.h"
#include "ir/manager.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "frontend/operator/ops.h"
#include "base/core_ops.h"

namespace mindspore {
namespace validator {
class TestValidator : public UT::Common {
 public:
  TestValidator() {}
  virtual ~TestValidator() {}

  virtual void TearDown() {}
};

TEST_F(TestValidator, ValidateOperation01) {
  auto node = NewValueNode(std::make_shared<Primitive>(prim::kScalarAdd));
  ValidateOperation(node);
  // normally, the above statement should not exit, so expected the following statement execute
  EXPECT_TRUE(true);
}

TEST_F(TestValidator, ValidateAbstract01) {
  AnfNodePtr node = NewValueNode(static_cast<int64_t>(1));
  abstract::AbstractBasePtr abstract_v1 = abstract::FromValue(static_cast<int64_t>(1), false);
  node->set_abstract(abstract_v1);
  ValidateAbstract(node);
  // normally, the above statement should not exit, so expected the following statement execute
  EXPECT_TRUE(true);
}
}  // namespace validator
}  // namespace mindspore
