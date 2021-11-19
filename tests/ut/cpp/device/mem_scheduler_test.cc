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

#include <vector>
#include <map>
#include "common/common_test.h"
#include "runtime/device/memory_scheduler.h"
namespace mindspore::device {
constexpr size_t kDeviceMemSize = 1 * 1024 * 1024 * 1024;
constexpr size_t kMaxVirtualCount = 1 * 1024 * 1024;
class MemHandlerImpl : public MemHandler {
 public:
  MemHandlerImpl() {
    device_mem_.resize(kMaxVirtualCount, 0);
    host_mem_.resize(kMaxVirtualCount, 1);
  }
  size_t GetAvailableMemSize() override { return kDeviceMemSize; }
  void *MallocDevice(size_t mem_size) override {
    auto ret = device_mem_.data() + device_virtual_count_;
    ++device_virtual_count_;
    device_mem_size_.emplace(ret, mem_size);
    return ret;
  }
  void FreeDevice(void *ptr) override {
    auto iter = device_mem_size_.find(ptr);
    if (iter != device_mem_size_.end()) {
      device_mem_size_.erase(iter);
    }
  }
  void *MallocHost(size_t mem_size) override {
    auto ret = host_mem_.data() + host_virtual_count_;
    ++host_virtual_count_;
    host_mem_size_.emplace(ret, mem_size);
    return ret;
  }
  void FreeHost(void *ptr) override {
    auto iter = host_mem_size_.find(ptr);
    if (iter != host_mem_size_.end()) {
      host_mem_size_.erase(iter);
    }
  }
  void SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) override {}
  void SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) override {}

 private:
  std::vector<uint8_t> device_mem_;
  std::vector<uint8_t> host_mem_;
  size_t device_virtual_count_;
  size_t host_virtual_count_;
  std::map<void *, size_t> device_mem_size_;
  std::map<void *, size_t> host_mem_size_;
};

class TestMemScheduler : public UT::Common {
 public:
  TestMemScheduler() {}
};

/// Feature: MemSchedulerManager
/// Description: Test MemSchedulerManager GetOrCreateMemScheduler interface
/// Expectation: Create MemScheduler
TEST_F(TestMemScheduler, test_mem_scheduler_manager) {
  MemSchedulerManager mem_scheduler_manager;
  auto ret = mem_scheduler_manager.GetMemScheduler(0);
  ASSERT_EQ(ret, nullptr);
  ret = mem_scheduler_manager.GetOrCreateMemScheduler(0);
  ASSERT_NE(ret, nullptr);
  ret = mem_scheduler_manager.GetMemScheduler(0);
  ASSERT_NE(ret, nullptr);
}

/// Feature: MemScheduler
/// Description: Test MemScheduler interface
/// Expectation: MemScheduler GetOrMalloc return valid ptr
TEST_F(TestMemScheduler, test_mem_scheduler) {
  MemSchedulerManager mem_scheduler_manager;
  auto scheduler = mem_scheduler_manager.GetOrCreateMemScheduler(0);
  ASSERT_NE(scheduler, nullptr);
  auto need_record = scheduler->need_record_event();
  ASSERT_EQ(need_record, true);
  auto optimized = scheduler->optimized();
  ASSERT_EQ(optimized, false);
  std::shared_ptr<MemHandler> mem_handler = std::make_shared<MemHandlerImpl>();
  ASSERT_NE(mem_handler, nullptr);
  scheduler->SetMemHandler(mem_handler);

  constexpr size_t kUsedTensors = 10;
  constexpr size_t kTimeSlice = 7;
  std::vector<uint8_t> tensor_keys(kUsedTensors, 0);
  std::vector<uint8_t> tensor_datas(kUsedTensors, 0);
  std::vector<size_t> init_tensors = {0, 2, 4};
  std::vector<std::vector<size_t>> step_tensors = {{0, 1}, {1, 2, 3}, {3, 4, 5}, {5, 6}, {6, 7}, {2, 7, 8}, {4, 8, 9}};
  void *stream = nullptr;
  scheduler->SetTotalStep(kTimeSlice);
  // record
  for (auto index : init_tensors) {
    scheduler->Init(tensor_keys.data() + index, tensor_datas.data() + index, 1, kMemPriorityHigh);
  }
  for (size_t i = 0; i < kTimeSlice; ++i) {
    auto &tensors = step_tensors[i];
    for (auto j : tensors) {
      scheduler->GetOrMalloc(tensor_keys.data() + j, 1);
    }
    scheduler->PostCompute(stream);
  }
  scheduler->set_need_record_event(false);

  // optimize
  scheduler->Optimize();

  // run
  scheduler->ResetCurrentStep();
  for (auto index : init_tensors) {
    scheduler->Init(tensor_keys.data() + index, tensor_datas.data() + index, 1, kMemPriorityHigh);
  }
  for (size_t i = 0; i < kTimeSlice; ++i) {
    scheduler->PreCompute(stream);
    auto &tensors = step_tensors[i];
    for (auto j : tensors) {
      auto addr = scheduler->GetOrMalloc(tensor_keys.data() + j, 1);
      ASSERT_NE(addr, nullptr);
    }
    scheduler->PostCompute(stream);
  }
}
}  // namespace mindspore::device