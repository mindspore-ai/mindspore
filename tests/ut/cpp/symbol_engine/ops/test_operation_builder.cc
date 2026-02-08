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

#include "abstract/symbolic_shape/operation_builder.h"
#include "common/common_test.h"

namespace mindspore::symshape::test {
class TestOperationBuilder : public UT::Common {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

/// Feature: Symbolic operation builder
/// Description: When the ops register a shape (or value) builder, it should register shape (or value) depend status
///               together.
/// Expectation: success.
TEST_F(TestOperationBuilder, reg_buildfunc_with_depend) {
  std::set<std::string> no_input_ops = {"GetNext"};
  const auto &builders = OperationBuilderInfoRegistry::Instance().builders();
  for (auto &[op, build_info] : builders) {
    if (no_input_ops.count(op) > 0) {
      continue;
    }
    if (build_info.build_shape_func != nullptr) {
      EXPECT_TRUE(build_info.shape_depend_func != nullptr || !build_info.shape_depend_list.empty())
        << "  Op " << op << " has build symbolic Shape function but has not shape depend info.";
    }
    if (build_info.build_value_func != nullptr) {
      EXPECT_TRUE(build_info.value_depend_func != nullptr || !build_info.value_depend_list.empty())
        << "  Op " << op << " has build symbolic Value function but has not value depend info.";
    }
  }
}
}  // namespace mindspore::symshape::test
