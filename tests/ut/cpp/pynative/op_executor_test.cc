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

#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"

#include "backend/graph_compiler/backend.h"
#include "backend/common/session/session_basic.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_compiler.h"

class OpExecuteTest : public testing::Test {
 protected:
  virtual void SetUp() {}

  virtual void TearDown() { GlobalMockObject::verify(); }

  static void SetUpTestCase() {}

  static void TearDownTestCase() {}
};

namespace mindspore {
namespace compile {
TEST_F(OpExecuteTest, TestNeedSync) {
  const auto &context = MsContext::GetInstance();
  ASSERT_NE(context, nullptr);
  const auto &executor = runtime::OpExecutor::GetInstance();

  context->set_param<int>(mindspore::MS_CTX_EXECUTION_MODE, kPynativeMode);
  context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE, false);
  ASSERT_EQ(executor.NeedSync(), false);
  context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE, true);
  ASSERT_EQ(executor.NeedSync(), true);

  context->set_param<int>(mindspore::MS_CTX_EXECUTION_MODE, kGraphMode);
  context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE, false);
  ASSERT_EQ(executor.NeedSync(), true);
  context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE, true);
  ASSERT_EQ(executor.NeedSync(), true);
}

TEST_F(OpExecuteTest, TestRegisterForwardCallback) {
  auto &executor = runtime::OpExecutor::GetInstance();
  int x = 0;
  executor.RegisterForwardCallback([&x]() { x += 1; });
  executor.WaitAll();
  ASSERT_EQ(x, 1);

  executor.WaitAll();
  ASSERT_EQ(x, 2);

  executor.RegisterForwardCallback([]() {});
  ASSERT_EQ(x, 2);
}

TEST_F(OpExecuteTest, TestPushOpRunTask) {
  auto &executor = runtime::OpExecutor::GetInstance();

  auto op_compiler_info =
    std::make_shared<pynative::OpCompilerInfo>("", 0, nullptr, nullptr, false, false, std::vector<KernelWithIndex>(),
                                               std::vector<size_t>(), std::vector<std::string>(), nullptr);

  int x = 0;

  auto op_context = std::make_shared<runtime::OpTaskContext>(0, nullptr, nullptr, op_compiler_info, true);
  auto task1 = std::make_shared<runtime::DeviceOpRunTask>(
    op_context, [&x](const std::shared_ptr<runtime::OpTaskContext> &context) { x += 1; });
  executor.PushOpRunTask(task1);
  executor.Wait();
  ASSERT_EQ(x, 1);

  auto task2 = std::make_shared<runtime::DeviceOpRunTask>(
    op_context, [&x](const std::shared_ptr<runtime::OpTaskContext> &context) { x += 1; });
  executor.PushOpRunTask(task2);
  executor.Wait();
  ASSERT_EQ(x, 2);

  auto task3 = std::make_shared<runtime::PyBoostDeviceTask>([&x](){ x += 1; });
  executor.PushOpRunTask(task3);
  executor.Wait();
  ASSERT_EQ(x, 3);

  auto task4 = std::make_shared<runtime::PassthroughDeviceTask>([&x](){ x += 1; });
  executor.PushSimpleOpRunTask(task4);
  executor.Wait();
  ASSERT_EQ(x, 4);

  ASSERT_EQ(executor.RunQueueEmpty(), true);

  mindspore::runtime::OpExecutor::DispatchLaunchTask([&x](){ x += 1; });
  executor.WaitAll();
  ASSERT_EQ(x, 5);
}
}  // namespace compile
}  // namespace mindspore