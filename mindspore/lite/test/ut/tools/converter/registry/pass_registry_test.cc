/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "common/common_test.h"
#include "backend/optimizer/common/pass.h"
#include "include/registry/pass_registry.h"

namespace mindspore {
class PassRegistryTest : public mindspore::CommonTest {
 public:
  PassRegistryTest() = default;
};

namespace opt {
class TestFusion : public Pass {
 public:
  TestFusion() : Pass("test_fusion") {}
  bool Run(const FuncGraphPtr &func_graph) override { return true; }
};
REG_PASS(POSITION_BEGIN, TestFusion)
REG_PASS(POSITION_END, TestFusion)
}  // namespace opt

TEST_F(PassRegistryTest, TestRegistry) {
  auto passes = opt::PassRegistry::GetInstance()->GetPasses();
  ASSERT_EQ(passes.size(), 2);
  auto begin_pass = passes[opt::POSITION_BEGIN];
  ASSERT_NE(begin_pass, nullptr);
  auto begin_pass_test = std::dynamic_pointer_cast<opt::TestFusion>(begin_pass);
  ASSERT_NE(begin_pass_test, nullptr);
  auto res = begin_pass_test->Run(nullptr);
  ASSERT_EQ(res, true);
  auto end_pass = passes[opt::POSITION_END];
  ASSERT_NE(end_pass, nullptr);
  auto end_pass_test = std::dynamic_pointer_cast<opt::TestFusion>(end_pass);
  ASSERT_NE(end_pass_test, nullptr);
  res = end_pass_test->Run(nullptr);
  ASSERT_EQ(res, true);
}
}  // namespace mindspore
