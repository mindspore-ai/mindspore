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

#include "common/common_test.h"
#include "include/backend/mem_reuse/mem_dynamic_allocator.h"

namespace mindspore {
namespace device {

const size_t kSize1G = 1 << 30;

class NormalPool : public DynamicMemPoolBestFit {
  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) override {
    *addr = malloc(size);
    return size;
  }
  bool FreeDeviceMem(const DeviceMemPtr &addr) override {
    free(addr);
    return true;
  }
  size_t free_mem_size() override {
    return kSize1G * 10;
  }
};

class TestMemDynamicAllocator : public UT::Common {
 public:
  TestMemDynamicAllocator() = default;
  virtual ~TestMemDynamicAllocator() = default;

  void SetUp() override {}
  void TearDown() override {}

  NormalPool normal_pool_;
};

/// Feature: test mem dynamic allocator.
/// Description: test mem dynamic allocatordata structure and interface.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMemDynamicAllocator, test_base_malloc_free) {
  auto persitent_mem_pool = normal_pool_.persistent_mem();
  auto common_mem_pool = normal_pool_.common_mem();
  // Malloc from persistent mem pool.
  auto addr1 = normal_pool_.AllocTensorMem(kSize1G >> 1, true, true);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_.size(), 1);
  EXPECT_EQ(persitent_mem_pool->idle_mem_buf_map_.size(), 1);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_[0]->device_addr(), addr1);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_[0]->block_all_mem_buf_map_.count(addr1), 1);
  // Malloc from common mem pool.
  auto addr2 = normal_pool_.AllocTensorMem(kSize1G >> 1, false, true);
  EXPECT_EQ(common_mem_pool->mem_block_list_.size(), 1);
  EXPECT_EQ(common_mem_pool->idle_mem_buf_map_.size(), 1);
  EXPECT_EQ(common_mem_pool->mem_block_list_[0]->device_addr(), addr2);
  EXPECT_EQ(common_mem_pool->mem_block_list_[0]->block_all_mem_buf_map_.count(addr2), 1);
  // Malloc more.
  auto addr3 = normal_pool_.AllocTensorMem(kSize1G, true, true);
  auto addr4 = normal_pool_.AllocTensorMem(kSize1G, false, true);

  EXPECT_EQ(persitent_mem_pool->mem_block_list_.size(), 1);
  EXPECT_EQ(persitent_mem_pool->idle_mem_buf_map_.size(), 1);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_[0]->device_addr(), addr1);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_[0]->block_all_mem_buf_map_.count(addr3), 0);
  EXPECT_EQ(common_mem_pool->mem_block_list_.size(), 3);
  EXPECT_EQ(common_mem_pool->idle_mem_buf_map_.size(), 1);
  EXPECT_EQ(common_mem_pool->mem_block_list_[0]->device_addr(), addr4);
  EXPECT_EQ(common_mem_pool->mem_block_list_[0]->block_all_mem_buf_map_.count(addr4), 1);

  auto addr5 = normal_pool_.AllocTensorMem(kSize1G >> 1, true, true);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_.size(), 1);
  EXPECT_EQ(persitent_mem_pool->idle_mem_buf_map_.size(), 0);
  normal_pool_.FreeTensorMem(addr1);
  EXPECT_EQ(persitent_mem_pool->idle_mem_buf_map_.size(), 1);
  normal_pool_.FreeTensorMem(addr5);
  EXPECT_EQ(persitent_mem_pool->mem_block_list_.size(), 1);
  EXPECT_EQ(persitent_mem_pool->idle_mem_buf_map_.size(), 1);
  normal_pool_.AllocTensorMem(kSize1G >> 1, false, true);
  normal_pool_.FreeTensorMem(addr4);
  EXPECT_EQ(common_mem_pool->idle_mem_buf_map_.size(), 1);
}
}  // namespace device
}  // namespace mindspore
