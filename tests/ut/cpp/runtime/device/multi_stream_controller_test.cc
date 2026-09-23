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
#include <vector>
#include <string>
#include <memory>
#include "common/common_test.h"
#include "common/test_device_address.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "runtime/hardware_abstract/event/device_event.h"

namespace mindspore {
namespace device {
class MultiStreamControllerTest : public UT::Common {
 public:
  MultiStreamControllerTest() = default;
  void SetUp() override {
    res_manager_ = std::make_shared<runtime::test::TestResManager>();
    controller_ = std::make_shared<MultiStreamController>(res_manager_.get());
  }
  void TearDown() override {}

 protected:
  std::shared_ptr<runtime::test::TestResManager> res_manager_;
  MultiStreamControllerPtr controller_;
};

/// Feature: MultiStreamController CrossStreamTaskId
/// Description: Test GetCrossStreamTaskId returns 0 for unknown stream id
/// Expectation: Returns 0
TEST_F(MultiStreamControllerTest, GetCrossStreamTaskIdReturnsZeroForUnknownStream) {
  EXPECT_EQ(controller_->GetCrossStreamTaskId(0), 0);
  EXPECT_EQ(controller_->GetCrossStreamTaskId(1), 0);
  EXPECT_EQ(controller_->GetCrossStreamTaskId(999), 0);
}

/// Feature: MultiStreamController CrossStreamTaskId
/// Description: Test SetCrossStreamTaskId and GetCrossStreamTaskId basic set/get
/// Expectation: Get returns the value that was set
TEST_F(MultiStreamControllerTest, SetAndGetCrossStreamTaskId) {
  controller_->SetCrossStreamTaskId(1, 100);
  EXPECT_EQ(controller_->GetCrossStreamTaskId(1), 100);

  controller_->SetCrossStreamTaskId(2, 200);
  EXPECT_EQ(controller_->GetCrossStreamTaskId(2), 200);
  EXPECT_EQ(controller_->GetCrossStreamTaskId(1), 100);
}

/// Feature: MultiStreamController CrossStreamTaskId
/// Description: Test SetCrossStreamTaskId overwrites previous value
/// Expectation: Get returns the latest set value
TEST_F(MultiStreamControllerTest, SetCrossStreamTaskIdOverwritesPrevious) {
  controller_->SetCrossStreamTaskId(1, 100);
  EXPECT_EQ(controller_->GetCrossStreamTaskId(1), 100);

  controller_->SetCrossStreamTaskId(1, 300);
  EXPECT_EQ(controller_->GetCrossStreamTaskId(1), 300);
}

/// Feature: MultiStreamController CrossStreamTaskId
/// Description: Test multiple stream ids can be cached independently
/// Expectation: Each stream id maintains its own task id
TEST_F(MultiStreamControllerTest, MultipleStreamIdsIndependent) {
  constexpr uint32_t kNumStreams = 10;
  for (uint32_t i = 0; i < kNumStreams; ++i) {
    controller_->SetCrossStreamTaskId(i, static_cast<int64_t>(i * 10));
  }
  for (uint32_t i = 0; i < kNumStreams; ++i) {
    EXPECT_EQ(controller_->GetCrossStreamTaskId(i), static_cast<int64_t>(i * 10));
  }
}

/// Feature: MultiStreamController CrossStreamTaskId
/// Description: Test task id selection logic (max of cached vs current)
/// Expectation: std::max selects the larger task id correctly
TEST_F(MultiStreamControllerTest, MaxTaskIdSelection) {
  controller_->SetCrossStreamTaskId(1, 50);
  int64_t current_task_id = 30;
  auto cached = controller_->GetCrossStreamTaskId(1);
  auto max_task_id = std::max(cached, current_task_id);
  EXPECT_EQ(max_task_id, 50);

  current_task_id = 80;
  max_task_id = std::max(cached, current_task_id);
  EXPECT_EQ(max_task_id, 80);
}
namespace {
class MemoryReuseEvent : public DeviceEvent {
 public:
  explicit MemoryReuseEvent(std::vector<std::pair<char, uint32_t>> *calls) : calls_(calls) {}
  bool IsReady() const override { return true; }
  void RecordEvent(uint32_t stream_id) override { calls_->emplace_back('R', stream_id); }
  bool WaitEvent(uint32_t stream_id) override {
    calls_->emplace_back('W', stream_id);
    return true;
  }
  void RecordEvent() override {}
  void WaitEvent() override {}
  void WaitEventWithoutReset() override {}
  bool NeedWait() override { return false; }
  void SyncEvent() override {}
  bool QueryEvent() override { return true; }
  void ElapsedTime(float *, const DeviceEvent *) override {}
  bool DestroyEvent() override { return true; }
  void set_wait_stream(void *) override {}
  void set_record_stream(void *) override {}

 private:
  std::vector<std::pair<char, uint32_t>> *calls_;
};

class MemoryReuseResManager : public runtime::test::TestResManager {
 public:
  size_t QueryStreamSize() const override { return 7; }
  DeviceEventPtr CreateRuntimeEvent(bool, bool) override { return std::make_shared<MemoryReuseEvent>(&calls); }
  std::vector<std::pair<char, uint32_t>> calls;
};
}  // namespace

class MemoryReuseSynchronizationTest : public UT::Common {
 protected:
  void SetUp() override {
    res_manager_ = std::make_shared<MemoryReuseResManager>();
    controller_ = std::make_shared<MultiStreamController>(res_manager_.get());
    controller_->Refresh();
  }
  std::shared_ptr<MemoryReuseResManager> res_manager_;
  MultiStreamControllerPtr controller_;
};

/// Feature: Communication buffer reuse synchronization.
/// Description: The static event already covers all allocation-stream tasks.
/// Expectation: No additional device event is dispatched.
TEST_F(MemoryReuseSynchronizationTest, CurrentStaticEventNeedsNoAdditionalEvent) {
  auto send_task = controller_->LaunchTaskIdOnStream(0);
  (void)controller_->WaitEvent(send_task, 0, 6);
  EXPECT_FALSE(controller_->DispatchRecordWaitEventForMemoryReuse(6, 0));
  EXPECT_TRUE(res_manager_->calls.empty());
}

/// Feature: Communication buffer reuse synchronization.
/// Description: Another graph submits two default-stream tasks after the static send.
/// Expectation: One record/wait pair is dispatched, in the correct direction, without duplicate waits.
TEST_F(MemoryReuseSynchronizationTest, InterleavedTasksRefreshDependencyOnce) {
  auto send_task = controller_->LaunchTaskIdOnStream(0);
  (void)controller_->WaitEvent(send_task, 0, 6);
  (void)controller_->LaunchTaskIdOnStream(0);  // Add-op63.
  auto last_reader = controller_->LaunchTaskIdOnStream(0);  // Add-op64.
  EXPECT_TRUE(controller_->DispatchRecordWaitEventForMemoryReuse(6, 0));
  const std::vector<std::pair<char, uint32_t>> expected = {{'R', 0}, {'W', 6}};
  EXPECT_EQ(res_manager_->calls, expected);
  EXPECT_EQ(controller_->QueryTaskIdOnStream(0, 6), last_reader);
  EXPECT_FALSE(controller_->DispatchRecordWaitEventForMemoryReuse(6, 0));
  EXPECT_EQ(res_manager_->calls, expected);
}

/// Feature: Communication buffer reuse synchronization.
/// Description: A dependency in the reverse direction does not cover default-stream accesses.
/// Expectation: A default-to-communication event is still dispatched.
TEST_F(MemoryReuseSynchronizationTest, ReverseDependencyDoesNotSuppressSynchronization) {
  (void)controller_->LaunchTaskIdOnStream(0);
  (void)controller_->UpdateTaskIdOnStream(100, 6, 0);
  EXPECT_TRUE(controller_->DispatchRecordWaitEventForMemoryReuse(6, 0));
  const std::vector<std::pair<char, uint32_t>> expected = {{'R', 0}, {'W', 6}};
  EXPECT_EQ(res_manager_->calls, expected);
}

/// Feature: Communication buffer reuse synchronization.
/// Description: No device synchronization is necessary within the same stream.
/// Expectation: No record/wait pair is dispatched.
TEST_F(MemoryReuseSynchronizationTest, SameStreamNeedsNoEvent) {
  (void)controller_->LaunchTaskIdOnStream(0);
  EXPECT_FALSE(controller_->DispatchRecordWaitEventForMemoryReuse(0, 0));
  EXPECT_TRUE(res_manager_->calls.empty());
}
}  // namespace device
}  // namespace mindspore
