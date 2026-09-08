/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include <exception>
#include <memory>
#include <stdexcept>

#include "common/common_test.h"
#include "include/runtime/pipeline/async_rqueue.h"
#include "runtime/pipeline/task/task.h"

namespace mindspore {
namespace runtime {
namespace {
class ThrowingTask : public AsyncTask {
 public:
  ThrowingTask() : AsyncTask(kFrontendTask) {}

  void Run() override { throw std::runtime_error("task failed"); }

  void SetException(const std::exception_ptr &exception) override { exception_ = exception; }

  const std::exception_ptr &exception() const { return exception_; }

 private:
  std::exception_ptr exception_{nullptr};
};
}  // namespace

class AsyncRQueueTest : public UT::Common {};

/// Feature: Synchronous AsyncRQueue execution.
/// Description: A synchronous task throws while running.
/// Expectation: The task receives the exception before it is rethrown to the caller.
TEST_F(AsyncRQueueTest, SynchronousPushSetsTaskExceptionBeforeRethrowing) {
  AsyncRQueue queue("sync_exception", kLevelFrontend);
  queue.SetMultiThreadDisabled(true);
  auto task = std::make_shared<ThrowingTask>();

  EXPECT_THROW(queue.Push(task), std::runtime_error);

  ASSERT_NE(task->exception(), nullptr);
  EXPECT_THROW(std::rethrow_exception(task->exception()), std::runtime_error);
}
}  // namespace runtime
}  // namespace mindspore
