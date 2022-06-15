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
constexpr size_t kDeviceMemSize = 5;
constexpr size_t kMaxVirtualCount = 1024;
class MemoryManagerStub : public MemoryManager {
 public:
  MemoryManagerStub() {
    device_mem_.resize(kMaxVirtualCount, 0);
  }
  void Initialize() override {}
  void Finalize() override {}

  size_t GetAvailableMemSize() override { return kDeviceMemSize; }

  void *MallocMemFromMemPool(size_t mem_size, bool useless = false) override {
    if (device_virtual_count_ >= kDeviceMemSize) {
      return nullptr;
    }
    auto ret = device_mem_.data() + device_virtual_count_;
    ++device_virtual_count_;
    device_mem_size_.emplace(ret, mem_size);
    return ret;
  }

  void FreeMemFromMemPool(void *ptr) override {
    --device_virtual_count_;
    auto iter = device_mem_size_.find(ptr);
    if (iter != device_mem_size_.end()) {
      device_mem_size_.erase(iter);
    }
  }

  std::vector<void *> MallocContinuousMemFromMemPool(const std::vector<size_t> &size_list) override {
    const size_t total_size = std::accumulate(size_list.begin(), size_list.end(), 0);
    std::vector<void *> ret;
    if (device_virtual_count_ + total_size > kDeviceMemSize) {
      return ret;
    }
    for (const auto &size : size_list) {
      auto ptr = device_mem_.data() + device_virtual_count_;
      device_mem_size_.emplace(ptr, size);
      ret.emplace_back(ptr);
      ++device_virtual_count_;
    }
    return ret;
  }

  void SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) override {}

  void SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) override {}

 protected:
  uint8_t *MallocStaticMem(size_t size, bool communication_mem, uint32_t graph_id) { return nullptr; }

 private:
  std::vector<uint8_t> device_mem_;
  size_t device_virtual_count_{0};
  std::map<void *, size_t> device_mem_size_;
};

class TestMemScheduler : public UT::Common {
 public:
  TestMemScheduler() {}

 protected:
  size_t used_tensor_num_{1};
  size_t total_step_{1};
  std::vector<uint8_t> tensor_keys_;
  std::vector<uint8_t> tensor_datas_;
  std::vector<size_t> init_tensors_;
  std::vector<std::vector<size_t>> step_used_tensors_;

  void Record(const std::shared_ptr<MemScheduler> &scheduler) {
    void *stream = nullptr;
    for (auto index : init_tensors_) {
      scheduler->Init(tensor_keys_.data() + index, tensor_datas_.data() + index, 1, kMemPriorityHigh);
    }
    for (size_t i = 0; i < total_step_; ++i) {
      auto &tensors = step_used_tensors_[i];
      for (auto j : tensors) {
        scheduler->GetOrMalloc(tensor_keys_.data() + j, 1);
      }
      scheduler->PostCompute(stream);
    }
    scheduler->set_need_record_event(false);
  }

  void Run(const std::shared_ptr<MemScheduler> &scheduler) {
    void *stream = nullptr;
    scheduler->Reset();
    scheduler->Update();
    for (auto index : init_tensors_) {
      scheduler->Init(tensor_keys_.data() + index, tensor_datas_.data() + index, 1, kMemPriorityHigh);
    }
    for (size_t i = 0; i < total_step_; ++i) {
      scheduler->PreCompute(stream);
      auto &tensors = step_used_tensors_[i];
      for (auto j : tensors) {
        auto addr = scheduler->GetOrMalloc(tensor_keys_.data() + j, 1);
        ASSERT_NE(addr, nullptr);
      }
      scheduler->PostCompute(stream);
    }
  }
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
/// Expectation: MemScheduler GetOrMalloc return valid ptr for continuous mem
TEST_F(TestMemScheduler, test_mem_scheduler) {
  MemSchedulerManager mem_scheduler_manager;
  auto scheduler = mem_scheduler_manager.GetOrCreateMemScheduler(0);
  ASSERT_NE(scheduler, nullptr);
  auto need_record = scheduler->need_record_event();
  ASSERT_EQ(need_record, true);
  std::shared_ptr<MemHandler> mem_handler = std::make_shared<MemHandler>(std::make_shared<MemoryManagerStub>());
  ASSERT_NE(mem_handler, nullptr);
  scheduler->SetMemHandler(mem_handler);

  // input data
  used_tensor_num_ = 10;
  total_step_ = 8;
  std::vector<uint8_t> tensor_keys(used_tensor_num_, 0);
  std::vector<uint8_t> tensor_datas(used_tensor_num_, 0);
  std::vector<size_t> init_tensors = {0, 2, 4};
  // 8 step tensor usage
  //
  // 0
  // 1  1-----------------1
  //    2--------------2
  //    3  3--------3
  //       4-----4
  //       5  5
  //          6  6
  //             7  7
  //                8  8
  //                   9  9
  std::vector<std::vector<size_t>> step_used_tensors = {{0, 1},    {1, 2, 3}, {3, 4, 5}, {5, 6},
                                                        {4, 6, 7}, {3, 7, 8}, {2, 8, 9}, {1, 9}};
  tensor_keys_.swap(tensor_keys);
  tensor_datas_.swap(tensor_datas);
  init_tensors_.swap(init_tensors);
  step_used_tensors_.swap(step_used_tensors);
  scheduler->SetTotalStep(total_step_);

  // record
  Record(scheduler);
  // optimize
  scheduler->Optimize();
  // run
  Run(scheduler);
}

/// Feature: MemScheduler
/// Description: Test MemScheduler interface
/// Expectation: MemScheduler GetOrMalloc return valid ptr
TEST_F(TestMemScheduler, test_manual_mem_scheduler) {
  MemSchedulerManager mem_scheduler_manager;
  auto scheduler = mem_scheduler_manager.GetOrCreateMemScheduler(0);
  ASSERT_NE(scheduler, nullptr);
  auto need_record = scheduler->need_record_event();
  ASSERT_EQ(need_record, true);
  std::shared_ptr<MemHandler> mem_handler = std::make_shared<MemHandler>(std::make_shared<MemoryManagerStub>());
  ASSERT_NE(mem_handler, nullptr);
  scheduler->SetMemHandler(mem_handler);

  // input data
  used_tensor_num_ = 10;
  total_step_ = 8;
  std::vector<uint8_t> tensor_keys(used_tensor_num_, 0);
  std::vector<uint8_t> tensor_datas(used_tensor_num_, 0);
  std::vector<size_t> init_tensors = {0, 2, 4};
  std::vector<size_t> offload_tensor = {1, 2, 3};
  // 8 step tensor usage
  //
  // 0
  // 1  1-----------------1
  //    2--------------2
  //    3  3--------3
  //       4-----4
  //       5  5
  //          6  6
  //             7  7
  //                8  8
  //                   9  9
  std::vector<std::vector<size_t>> step_used_tensors = {{0, 1},    {1, 2, 3}, {3, 4, 5}, {5, 6},
                                                        {4, 6, 7}, {3, 7, 8}, {2, 8, 9}, {1, 9}};
  tensor_keys_.swap(tensor_keys);
  tensor_datas_.swap(tensor_datas);
  init_tensors_.swap(init_tensors);
  step_used_tensors_.swap(step_used_tensors);
  scheduler->SetTotalStep(total_step_);

  // set offload key
  for (auto index : offload_tensor) {
    scheduler->SetOffload(tensor_keys_.data() + index);
  }
  // record
  Record(scheduler);
  // optimize
  scheduler->Optimize();
  // run
  Run(scheduler);
}

/// Feature: MemScheduler
/// Description: Test MemScheduler interface
/// Expectation: MemScheduler GetOrMalloc return valid ptr
TEST_F(TestMemScheduler, test_mem_scheduler_with_continuous_mem) {
MemSchedulerManager mem_scheduler_manager;
auto scheduler = mem_scheduler_manager.GetOrCreateMemScheduler(0);
ASSERT_NE(scheduler, nullptr);
auto need_record = scheduler->need_record_event();
ASSERT_EQ(need_record, true);
std::shared_ptr<MemHandler> mem_handler = std::make_shared<MemHandler>(std::make_shared<MemoryManagerStub>());
ASSERT_NE(mem_handler, nullptr);
scheduler->SetMemHandler(mem_handler);

// input data
used_tensor_num_ = 8;
total_step_ = 8;
std::vector<uint8_t> tensor_keys(used_tensor_num_, 0);
std::vector<uint8_t> tensor_datas(used_tensor_num_, 0);
std::vector<size_t> init_tensors = {0, 2, 6, 7};
// 8 step tensor usage
//
// 0-----0-----0
//    1--1--1
// 2-----2--------2
//       3-----------3
//       4--------------4
// 6--------------6
//    7--------7
//
std::vector<std::vector<size_t>> step_used_tensors = {{0, 2, 6}, {1, 7}, {0, 1, 2, 3, 4}, {1},
                                                      {0, 7}, {2, 6}, {3}, {4}};
tensor_keys_.swap(tensor_keys);
tensor_datas_.swap(tensor_datas);
init_tensors_.swap(init_tensors);
step_used_tensors_.swap(step_used_tensors);
scheduler->SetTotalStep(total_step_);

// record
Record(scheduler);
// Add continuous memory info
scheduler->AddContinuousMemInfo(true, 2, 3, {1, 1, 1},
                                {tensor_keys_.data(), tensor_keys_.data() + 1, tensor_keys_.data() + 2});
scheduler->AddContinuousMemInfo(false, 2, 2, {1, 1}, {tensor_keys_.data() + 3, tensor_keys_.data() + 4});
// optimize
scheduler->Optimize();
// run
Run(scheduler);
}
}  // namespace mindspore::device