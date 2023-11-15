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
#include <random>
#include <unordered_set>

#include "common/common_test.h"
#include "include/backend/mem_reuse/mem_dynamic_allocator.h"

namespace mindspore {
namespace device {
constexpr size_t kSize1G = 1 << 30;
constexpr size_t kTotalMem = kSize1G * 10;
constexpr size_t expected_size_zero = 0;
constexpr size_t expected_size_one = 1;
constexpr size_t expected_size_two = 2;
constexpr size_t expected_size_three = 3;

class DummyPool : public DynamicMemPoolBestFit {
 public:
  DummyPool() { SetMemAllocUintSize(kSize1G, kSize1G); }

  ~DummyPool() {
    for (auto addr : allocated_mems_) {
      free(addr);
    }
    allocated_mems_.clear();
  }

  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) override {
    *addr = malloc(size);
    allocated_size_ += kSize1G;
    (void)allocated_mems_.emplace(*addr);
    return size;
  }

  bool FreeDeviceMem(const DeviceMemPtr &addr) override {
    if (allocated_mems_.count(addr) == 0) {
      return false;
    }
    (void)allocated_mems_.erase(addr);
    free(addr);
    allocated_size_ -= kSize1G;
    return true;
  }

  size_t free_mem_size() override { return kTotalMem - allocated_size_; }

 private:
  size_t allocated_size_{0};
  std::unordered_set<DeviceMemPtr> allocated_mems_;
};

class TestMemDynamicAllocator : public UT::Common {
 public:
  TestMemDynamicAllocator() = default;
  virtual ~TestMemDynamicAllocator() = default;

  void SetUp() override {}
  void TearDown() override {}

  DummyPool mem_pool_;
};

/// Feature: test basic memory allocation from mem dynamic allocator.
/// Description: test mem dynamic allocatordata structure and interface.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMemDynamicAllocator, test_basic_allocation) {
  auto persitent_mem_pool = mem_pool_.persistent_mem();
  auto common_mem_pool = mem_pool_.common_mem();
  // Malloc from persistent mem pool.
  auto addr1 = mem_pool_.AllocTensorMem(kSize1G >> 1, true, true);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_.size(), expected_size_one);
  EXPECT_EQ(persitent_mem_pool->idle_mem_buf_map_.size(), expected_size_one);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_[0]->device_addr(), addr1);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_[0]->block_all_mem_buf_map_.count(addr1), expected_size_one);
  // Malloc 512M from common mem pool.
  auto addr2 = mem_pool_.AllocTensorMem(kSize1G >> 1, false, true);
  EXPECT_EQ(common_mem_pool->mem_block_list_.size(), expected_size_one);
  EXPECT_EQ(common_mem_pool->idle_mem_buf_map_.size(), expected_size_one);
  EXPECT_EQ(common_mem_pool->mem_block_list_[0]->device_addr(), addr2);
  EXPECT_EQ(common_mem_pool->mem_block_list_[0]->block_all_mem_buf_map_.count(addr2), expected_size_one);
  // Malloc more 1g from persistent pool, set need_recycle to false, so triggle extension of common pool.
  auto addr3 = mem_pool_.AllocTensorMem(kSize1G, true, false);
  EXPECT_EQ(common_mem_pool->mem_block_list_.size(), expected_size_two);
  EXPECT_EQ(common_mem_pool->idle_mem_buf_map_.size(), expected_size_one);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_.size(), expected_size_one);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_[0]->block_all_mem_buf_map_.count(addr3), expected_size_zero);
  // Malloc more 1g from persistent pool, triggle extension of persistent pool.
  auto addr4 = mem_pool_.AllocTensorMem(kSize1G, true, true);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_.size(), expected_size_two);
  // Direction of mmap is from top to down, address of new block is less than prev addresses.
  EXPECT_EQ(persitent_mem_pool->mem_block_list_[0]->device_addr(), addr4);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_[0]->block_all_mem_buf_map_.count(addr4), expected_size_one);
  EXPECT_EQ(persitent_mem_pool->idle_mem_buf_map_.size(), expected_size_one);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_[1]->device_addr(), addr1);
  // Malloc another 512M from persistent pool.
  auto addr5 = mem_pool_.AllocTensorMem(kSize1G >> 1, true, true);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_.size(), expected_size_two);
  EXPECT_EQ(persitent_mem_pool->idle_mem_buf_map_.size(), expected_size_zero);
  mem_pool_.FreeTensorMem(addr1);
  EXPECT_EQ(persitent_mem_pool->idle_mem_buf_map_.size(), expected_size_one);
  mem_pool_.FreeTensorMem(addr5);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_.size(), expected_size_two);
  EXPECT_EQ(persitent_mem_pool->idle_mem_buf_map_.size(), expected_size_one);
  // Malloc another 512M from common pool.
  mem_pool_.AllocTensorMem(kSize1G >> 1, false, true);
  EXPECT_EQ(common_mem_pool->idle_mem_buf_map_.size(), expected_size_zero);
  mem_pool_.FreeTensorMem(addr4);
  EXPECT_EQ(persitent_mem_pool->idle_mem_buf_map_.size(), expected_size_two);
  // Malloc another 1g from common mem pool.
  mem_pool_.AllocTensorMem(kSize1G, false, true);
  EXPECT_EQ(common_mem_pool->mem_block_list_.size(), expected_size_three);
  // Free 512M from common pool.
  mem_pool_.FreeTensorMem(addr2);
  EXPECT_EQ(common_mem_pool->idle_mem_buf_map_.size(), expected_size_one);
}

/// Feature: test persitent memory malloc from common pool of mem dynamic allocator.
/// Description: test mem dynamic allocatordata pool allocation strategy.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMemDynamicAllocator, test_malloc_persistent_from_common) {
  auto persitent_mem_pool = mem_pool_.persistent_mem();
  auto common_mem_pool = mem_pool_.common_mem();
  // Malloc from persistent mem pool.
  auto addr1 = mem_pool_.AllocTensorMem(kSize1G >> 1, true, true);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_.size(), expected_size_one);
  EXPECT_EQ(persitent_mem_pool->idle_mem_buf_map_.size(), expected_size_one);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_[0]->device_addr(), addr1);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_[0]->block_all_mem_buf_map_.count(addr1), expected_size_one);
  // Malloc from common mem pool.
  const int block_size = 9;
  for (auto i = 0; i < block_size; i++) {
    mem_pool_.AllocTensorMem(kSize1G, false, true);
  }
  // Try to malloc common memory from persistent pool.
  auto addr = mem_pool_.AllocTensorMem(kSize1G >> 1, false, true);
  auto addr_oom = mem_pool_.AllocTensorMem(kSize1G >> 1, false, true);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_.size(), expected_size_one);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_[0]->block_all_mem_buf_map_.at(addr)->size_, kSize1G >> 1);
  EXPECT_EQ(addr_oom, nullptr);
}

/// Feature: test common memory malloc from persistent pool of mem dynamic allocator.
/// Description: test mem dynamic allocatordata pool allocation strategy.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMemDynamicAllocator, test_malloc_common_from_persistent) {
  auto persitent_mem_pool = mem_pool_.persistent_mem();
  auto common_mem_pool = mem_pool_.common_mem();
  // Malloc from persistent mem pool.
  auto addr = mem_pool_.AllocTensorMem(kSize1G >> 1, true, true);
  mem_pool_.FreeTensorMem(addr);
  const int block_size = 9;
  // Malloc from common mem pool.
  for (auto i = 0; i < block_size; i++) {
    mem_pool_.AllocTensorMem(kSize1G, false, true);
  }
  EXPECT_EQ(persitent_mem_pool->mem_block_list_.size(), expected_size_one);
  EXPECT_EQ(common_mem_pool->mem_block_list_.size(), block_size);
  // Try to malloc common memory from persistent pool.
  EXPECT_EQ(persitent_mem_pool->idle_mem_buf_map_.size(), expected_size_one);
  mem_pool_.AllocTensorMem(kSize1G, false, true);
  // Make sure allocation from persistent pool.
  EXPECT_EQ(persitent_mem_pool->idle_mem_buf_map_.size(), expected_size_zero);
}
}  // namespace device
}  // namespace mindspore
