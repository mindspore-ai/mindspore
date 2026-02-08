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
#include <map>
#include <vector>

#include "common/common_test.h"
#include "include/runtime/memory/mem_pool/abstract_dynamic_mem_pool.h"

namespace mindspore {
namespace device {
size_t MallocLinearMem(size_t size, DeviceMemPtr *addr) {
  // actual peak size need base addr is not zero
  static size_t base_addr = 1 << 30;
  *addr = (void *)base_addr;
  base_addr += size;
  return size;
}

MemBufAllocatorPtr GenerateMemBufAllocatorPtr(size_t block_size = (1 << 30)) {
  bool is_persistent = true;
  uint32_t stream_id = kDefaultStreamIndex;
  std::function<MemBlock *(size_t)> mem_block_expander = [&, is_persistent = is_persistent, stream_id = stream_id,
                                                          block_size = block_size](size_t size) -> MemBlock * {
    size_t alloc_size = std::max(block_size, size);
    void *addr = nullptr;
    MallocLinearMem(alloc_size, &addr);
    auto mem_block = new MemBlock(alloc_size, addr, stream_id);
    return mem_block;
  };
  std::function<bool(MemBlock *)> mem_block_cleaner = [&](MemBlock *mem_block) { return true; };
  std::function<size_t(size_t size, void *addr)> mem_mapper = [&](size_t size, void *addr) { return size; };
  std::function<size_t(void *addr, const size_t size)> mem_eager_freer = [&](void *addr, const size_t size) {
    return size;
  };
  auto test_ptr = std::make_shared<MemStat>();
  return std::make_shared<MemBufAllocator>(mem_block_expander, mem_block_cleaner, mem_mapper, mem_eager_freer, true,
                                           is_persistent, stream_id, false, test_ptr);
}

class TestMemBufAllocator : public UT::Common {
 public:
  TestMemBufAllocator() = default;
  virtual ~TestMemBufAllocator() = default;
};

/// Feature: test brief info for MemBufAllocator.
/// Description: test brief ino.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMemBufAllocator, test_brief_info) {
  auto allocator = GenerateMemBufAllocatorPtr();
  const auto &brief_info = allocator->BriefInfo();
  EXPECT_EQ("Mem buf allocator, enable vmm : 1, is persistent : 1, stream id : 0, is small : 0, is customized : 0.",
            brief_info);
}

/// Feature: test actual peak size for MemBufAllocator.
/// Description: test actual peak size.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMemBufAllocator, test_actual_peak_size) {
  auto allocator = GenerateMemBufAllocatorPtr();
  auto mem_buf1 = allocator->Malloc(1 << 10);
  auto mem_buf2 = allocator->Malloc(1 << 9);
  auto mem_buf3 = allocator->Malloc(1 << 10);
  allocator->Free(mem_buf2);

  EXPECT_EQ((1 << 11) + (1 << 9), allocator->ActualPeakSize());
  allocator->Free(mem_buf1);
  allocator->Free(mem_buf3);
}

/// Feature: test malloc for MemBufAllocator.
/// Description: test malloc.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMemBufAllocator, test_malloc) {
  auto allocator = GenerateMemBufAllocatorPtr();
  auto mem_buf = allocator->Malloc(1 << 10);
  EXPECT_EQ(mem_buf->mem_block_->ToJson(), allocator->mem_blocks_.front()->ToJson());
  allocator->Free(mem_buf);
}

/// Feature: test allocate and free for MemBufAllocator.
/// Description: test allocate free.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMemBufAllocator, test_allocate_free) {
  auto allocator = GenerateMemBufAllocatorPtr(6 * kDynamicMemAlignSize);
  // test forward merge
  std::vector<size_t> front_merge_sizes{1 * kDynamicMemAlignSize, 2 * kDynamicMemAlignSize, 3 * kDynamicMemAlignSize};
  std::vector<MemBuf *> front_merge_mem_bufs;
  for (auto size : front_merge_sizes) {
    auto mem_buf = allocator->Malloc(size);
    (void)front_merge_mem_bufs.emplace_back(mem_buf);
  }
  EXPECT_EQ(front_merge_mem_bufs[0]->next_, front_merge_mem_bufs[1]);
  EXPECT_EQ(front_merge_mem_bufs[1]->next_, front_merge_mem_bufs[2]);
  EXPECT_EQ(front_merge_mem_bufs[2]->prev_, front_merge_mem_bufs[1]);
  EXPECT_EQ(front_merge_mem_bufs[1]->prev_, front_merge_mem_bufs[0]);

  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  allocator->Free(front_merge_mem_bufs[0]);
  allocator->Free(front_merge_mem_bufs[1]);
  auto first_buf = front_merge_mem_bufs[2]->prev_;
  EXPECT_EQ(first_buf->size_, 3 * kDynamicMemAlignSize);
  EXPECT_EQ(first_buf->prev_, nullptr);
  allocator->Free(front_merge_mem_bufs[2]);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);
  EXPECT_EQ((*allocator->free_mem_bufs_.begin())->size_, 6 * kDynamicMemAlignSize);
  front_merge_mem_bufs.clear();

  // test backward merge
  std::vector<size_t> back_merge_sizes{1 * kDynamicMemAlignSize, 2 * kDynamicMemAlignSize, 3 * kDynamicMemAlignSize};
  std::vector<MemBuf *> back_merge_mem_bufs;
  for (auto size : back_merge_sizes) {
    (void)back_merge_mem_bufs.emplace_back(allocator->Malloc(size));
  }
  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  allocator->Free(back_merge_mem_bufs[2]);
  allocator->Free(back_merge_mem_bufs[1]);
  auto last_buf = back_merge_mem_bufs[0]->next_;
  EXPECT_EQ(last_buf->size_, 5 * kDynamicMemAlignSize);
  EXPECT_EQ(last_buf->next_, nullptr);
  allocator->Free(back_merge_mem_bufs[0]);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);
  EXPECT_EQ((*allocator->free_mem_bufs_.begin())->size_, 6 * kDynamicMemAlignSize);
  back_merge_mem_bufs.clear();

  // test forward and backward merge
  std::vector<size_t> merge_sizes{3 * kDynamicMemAlignSize, 2 * kDynamicMemAlignSize, 1 * kDynamicMemAlignSize};
  std::vector<MemBuf *> merge_mem_bufs;
  for (auto size : merge_sizes) {
    (void)merge_mem_bufs.emplace_back(allocator->Malloc(size));
  }
  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  allocator->Free(merge_mem_bufs[2]);
  allocator->Free(merge_mem_bufs[0]);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 2);
  allocator->Free(merge_mem_bufs[1]);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);
  EXPECT_EQ((*allocator->free_mem_bufs_.begin())->size_, 6 * kDynamicMemAlignSize);
  merge_mem_bufs.clear();
}


/// Feature: test search logic for MemBufAllocator.
/// Description: test search available mem buf.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMemBufAllocator, test_search_available_mem_buf) {
  auto allocator = GenerateMemBufAllocatorPtr();
  auto ptr_1 = allocator->Malloc(1 << 10);
  auto ptr_2 = allocator->Malloc(1 << 10);
  auto ptr_3 = allocator->Malloc(1 << 10);
  bool ret = allocator->Free(ptr_2);
  // used, idle, used
  EXPECT_TRUE(ptr_1->status_ == MemBufStatus::kMemBufUsed);
  EXPECT_TRUE(ptr_2->status_ == MemBufStatus::kMemBufIdle);
  EXPECT_TRUE(ptr_3->status_ == MemBufStatus::kMemBufUsed);
  auto free_size_pair = allocator->FreeIdleMemsByEagerFree();
  EXPECT_EQ(free_size_pair.first, 1 << 10);
  EXPECT_TRUE(ptr_1->status_ == MemBufStatus::kMemBufUsed);
  EXPECT_TRUE(ptr_2->status_ == MemBufStatus::kMemBufEagerFree);
  EXPECT_TRUE(ptr_3->status_ == MemBufStatus::kMemBufUsed);
  // used, eager_free, used
  allocator->Free(ptr_1);
  // idle, eager_free, used
  auto mem_buf = allocator->Malloc(2 << 10);
  // used, used
  EXPECT_EQ(ptr_3->prev_, mem_buf);
  EXPECT_TRUE(ptr_3->next_ != nullptr && ptr_3->next_->status_ == MemBufStatus::kMemBufEagerFree);
  ptr_1 = allocator->Malloc(1 << 10);
  ptr_2 = allocator->Malloc(1 << 10);
  // used, used, + used(ptr_1), + used(ptr_2)
  auto ptr_4 = allocator->Malloc(2 << 10);
  // used, used, + used(ptr_1), + used(ptr_2), + used(ptr_4)
  allocator->Free(ptr_2);
  allocator->Free(ptr_4);
  allocator->FreeIdleMemsByEagerFree();
  allocator->Free(ptr_1);
  mem_buf = allocator->Malloc(2 << 10);
  EXPECT_EQ(ptr_3->next_, mem_buf);
}

class LinearDynamicMemPool : public AbstractDynamicMemPool {
 public:
  LinearDynamicMemPool() { SetEnableVmm(true); }
  ~LinearDynamicMemPool() override = default;

  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) override {
    static size_t base_addr = 0;
    *addr = (void *)base_addr;
    alloc_infos_[*addr] = size;
    base_addr += size;
    return size;
  }

  bool FreeDeviceMem(const DeviceMemPtr &addr) override {
    auto &&iter = alloc_infos_.find(addr);
    if (iter == alloc_infos_.end()) {
      return false;
    }
    alloc_infos_.erase(iter);
    return true;
  }

  size_t MmapDeviceMem(const size_t size, const DeviceMemPtr addr) override {
    vmm_mmap_size_ += size;
    return size;
  }

  size_t GetMaxUsedMemSize() const override { return SIZE_MAX; }

  size_t GetVmmUsedMemSize() const override { return vmm_mmap_size_; }

  size_t free_mem_size() override { return SIZE_MAX; }

  uint64_t total_mem_size() const override {
    // when enable eager free, this size effect block size, use small size.
    return 1 << 30;
  }

  size_t ReservedMemorySize() {
    size_t reserved_memory_size = 0;
    for (auto &alloc_info : alloc_infos_) {
      reserved_memory_size += alloc_info.second;
    }
    return reserved_memory_size;
  }

 protected:
  // The related interface of device memory eager free.
  const bool IsEnableEagerFree() const override { return false; }

  const bool SyncAllStreams() override { return true; }
  size_t AllocDeviceMemByEagerFree(size_t size, DeviceMemPtr *addr) override { return AllocDeviceMem(size, addr); }

  size_t FreeDeviceMemByEagerFree(const DeviceMemPtr addr, const size_t size) override { return size; }

 private:
  size_t vmm_mmap_size_{0};
  std::map<void *, size_t> alloc_infos_;
};

class TestAbstractDynamicMemPool : public UT::Common {
 public:
  TestAbstractDynamicMemPool() = default;
  virtual ~TestAbstractDynamicMemPool() = default;
};

/// Feature: test basic memory allocation for abstract dynamic mem pool.
/// Description: test basic allocation.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_basic_allocation) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  // Malloc 512M
  EXPECT_EQ(mem_pool->TotalMemStatistics(), 0);

  auto addr1 = mem_pool->AllocTensorMem(kGBToByte / 2);
  EXPECT_EQ((size_t)addr1, 0);

  // Malloc 512M
  auto addr2 = mem_pool->AllocTensorMem(kGBToByte / 2);
  EXPECT_EQ((size_t)addr2, kGBToByte / 2);

  // Malloc more 1g from persistent pool
  auto addr3 = mem_pool->AllocTensorMem(kGBToByte);
  EXPECT_EQ((size_t)addr3, kGBToByte);

  // Malloc more 1g
  auto addr4 = mem_pool->AllocTensorMem(kGBToByte);
  EXPECT_EQ((size_t)addr4, kGBToByte * 2);

  // Malloc another 512M
  auto addr5 = mem_pool->AllocTensorMem(kGBToByte / 2);
  EXPECT_EQ((size_t)addr5, kGBToByte * 3);

  mem_pool->FreeTensorMem(addr1);
  EXPECT_EQ(mem_pool->stream_id_allocators_.size(), 1);
  auto allocator = mem_pool->stream_id_allocators_.begin()->second;
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);

  // Malloc another 512M
  auto another_addr = mem_pool->AllocTensorMem(kGBToByte / 2);
  EXPECT_EQ((size_t)addr1, (size_t)another_addr);

  mem_pool->FreeTensorMem(addr4);
  // Malloc another 512M
  another_addr = mem_pool->AllocTensorMem(kGBToByte / 2, false, true);
  EXPECT_EQ((size_t)addr4, (size_t)another_addr);

  // Free 512M from common pool
  mem_pool->FreeTensorMem(addr2);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 2);
}

/// Feature: test alloc aligned continuous tensor mem for abstract dynamic mem pool.
/// Description: test alloc continuous tensor mem with aligned size.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_alloc_continuous_tensor_mem_with_aligned_size) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  std::vector<size_t> aligned_sizes{1 * kDynamicMemAlignSize, 2 * kDynamicMemAlignSize, 3 * kDynamicMemAlignSize,
                                   4 * kDynamicMemAlignSize};
  const auto &aligned_addresses = mem_pool->AllocContinuousTensorMem(aligned_sizes, kDefaultStreamIndex);
  EXPECT_EQ(aligned_addresses.size(), aligned_sizes.size());
  EXPECT_EQ(reinterpret_cast<size_t>(aligned_addresses[0]) + kDynamicMemAlignSize,
            reinterpret_cast<size_t>(aligned_addresses[1]));
  EXPECT_EQ(reinterpret_cast<size_t>(aligned_addresses[1]) + 2 * kDynamicMemAlignSize,
            reinterpret_cast<size_t>(aligned_addresses[2]));
  EXPECT_EQ(reinterpret_cast<size_t>(aligned_addresses[2]) + 3 * kDynamicMemAlignSize,
            reinterpret_cast<size_t>(aligned_addresses[3]));
  // assert mem buf counts is 4
  EXPECT_EQ(mem_pool->stream_id_allocators_.size(), (size_t)1);
  EXPECT_EQ(mem_pool->addr_mem_buf_allocators_.size(), aligned_sizes.size());
  // malloc more mem buf
  void *addr = mem_pool->AllocTensorMem(1);
  EXPECT_EQ(reinterpret_cast<size_t>(aligned_addresses[3]) + 4 * kDynamicMemAlignSize, reinterpret_cast<size_t>(addr));
  // free first two address
  mem_pool->FreeTensorMem(aligned_addresses[0]);
  mem_pool->FreeTensorMem(aligned_addresses[1]);
  // malloc again to assert result first two address
  void *first_addr = mem_pool->AllocTensorMem(aligned_sizes[0]);
  void *second_addr = mem_pool->AllocTensorMem(aligned_sizes[1]);
  EXPECT_EQ(aligned_addresses[0], first_addr);
  EXPECT_EQ(aligned_addresses[1], second_addr);
  // free used addresses
  mem_pool->FreeTensorMem(addr);
  mem_pool->FreeTensorMem(first_addr);
  mem_pool->FreeTensorMem(second_addr);
  mem_pool->FreeTensorMem(aligned_addresses[2]);
  mem_pool->FreeTensorMem(aligned_addresses[3]);
  // assert free mem buf size is 1
  EXPECT_EQ(mem_pool->stream_id_allocators_.size(), 1);
  EXPECT_EQ(mem_pool->addr_mem_buf_allocators_.size(), 0);
  auto allocator = mem_pool->stream_id_allocators_.begin()->second;
  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);
  EXPECT_EQ(allocator->eager_free_mem_bufs_.size(), 1);
}

/// Feature: test alloc unaligned continuous tensor mem for abstract dynamic mem pool.
/// Description: test alloc continuous tensor mem with unaligned size.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_alloc_continuous_tensor_mem_with_unaligned_size) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  std::vector<size_t> sizes{1, 2, 3, 4};
  const auto &addresses = mem_pool->AllocContinuousTensorMem(sizes, kDefaultStreamIndex);
  EXPECT_EQ(addresses.size(), sizes.size());
  EXPECT_EQ(reinterpret_cast<size_t>(addresses[0]) + 1, reinterpret_cast<size_t>(addresses[1]));
  EXPECT_EQ(reinterpret_cast<size_t>(addresses[1]) + 2, reinterpret_cast<size_t>(addresses[2]));
  EXPECT_EQ(reinterpret_cast<size_t>(addresses[2]) + 3, reinterpret_cast<size_t>(addresses[3]));
  // assert mem buf counts is 4.
  EXPECT_EQ(mem_pool->stream_id_allocators_.size(), 1);
  EXPECT_EQ(mem_pool->addr_mem_buf_allocators_.size(), sizes.size());
  auto allocator = mem_pool->stream_id_allocators_.begin()->second;
  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 0);
  EXPECT_EQ(allocator->eager_free_mem_bufs_.size(), 1);
  auto mem_buf = *(allocator->eager_free_mem_bufs_.begin());
  auto four_mem_buf = mem_buf->prev_;
  EXPECT_EQ(four_mem_buf->size_, 4);
  auto three_mem_buf = four_mem_buf->prev_;
  EXPECT_EQ(three_mem_buf->size_, 3);
  auto two_mem_buf = three_mem_buf->prev_;
  EXPECT_EQ(two_mem_buf->size_, 2);
  auto one_mem_buf = two_mem_buf->prev_;
  EXPECT_EQ(one_mem_buf->size_, 1);
}

/// Feature: test free part tensor mem keep first part for abstract dynamic mem pool.
/// Description: test free part tensor mems keep first part.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_free_part_tensor_mems_keep_first_part) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  void *first_addr = mem_pool->AllocTensorMem(3 * kDynamicMemAlignSize, true, true);
  std::vector<void *> first_free_addrs;
  (void)first_free_addrs.emplace_back(first_addr);
  std::vector<void *> first_keep_addrs;
  (void)first_keep_addrs.emplace_back(first_addr);
  std::vector<size_t> first_keep_addr_sizes;
  (void)first_keep_addr_sizes.emplace_back(kDynamicMemAlignSize);
  mem_pool->FreePartTensorMems(first_free_addrs, first_keep_addrs, first_keep_addr_sizes);
  EXPECT_EQ(mem_pool->stream_id_allocators_.size(), 1);
  auto allocator = mem_pool->stream_id_allocators_.begin()->second;
  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);
  EXPECT_EQ(allocator->eager_free_mem_bufs_.size(), 1);
}

/// Feature: test free part tensor mem keep middle part for abstract dynamic mem pool.
/// Description: test free part tensor mems keep middle part.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_free_part_tensor_mems_keep_middle_part) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  void *middle_addr = mem_pool->AllocTensorMem(3 * kDynamicMemAlignSize, true, true);
  std::vector<void *> middle_free_addrs;
  (void)middle_free_addrs.emplace_back(middle_addr);
  std::vector<void *> middle_keep_addrs;
  (void)middle_keep_addrs.emplace_back(static_cast<int8_t *>(middle_addr) + kDynamicMemAlignSize);
  std::vector<size_t> middle_keep_addr_sizes;
  (void)middle_keep_addr_sizes.emplace_back(kDynamicMemAlignSize);
  mem_pool->FreePartTensorMems(middle_free_addrs, middle_keep_addrs, middle_keep_addr_sizes);
  EXPECT_EQ(mem_pool->stream_id_allocators_.size(), 1);
  auto allocator = mem_pool->stream_id_allocators_.begin()->second;
  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 2);
  EXPECT_EQ(allocator->eager_free_mem_bufs_.size(), 1);
  EXPECT_EQ(mem_pool->addr_mem_buf_allocators_.size(), 1);
}

/// Feature: test free part tensor mem keep last part for abstract dynamic mem pool.
/// Description: test free part tensor mems keep last part.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_free_part_tensor_mems_keep_last_part) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  void *last_addr = mem_pool->AllocTensorMem(3 * kDynamicMemAlignSize, true, true);
  std::vector<void *> last_free_addrs;
  (void)last_free_addrs.emplace_back(last_addr);
  std::vector<void *> last_keep_addrs;
  (void)last_keep_addrs.emplace_back(static_cast<int8_t *>(last_addr) + 2 * kDynamicMemAlignSize);
  std::vector<size_t> last_keep_addr_sizes;
  (void)last_keep_addr_sizes.emplace_back(kDynamicMemAlignSize);
  mem_pool->FreePartTensorMems(last_free_addrs, last_keep_addrs, last_keep_addr_sizes);
  EXPECT_EQ(mem_pool->stream_id_allocators_.size(), 1);
  auto allocator = mem_pool->stream_id_allocators_.begin()->second;
  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);
  EXPECT_EQ(allocator->eager_free_mem_bufs_.size(), 1);
  EXPECT_EQ(mem_pool->addr_mem_buf_allocators_.size(), 1);
}

/// Feature: test free part tensor mem keep multi part for abstract dynamic mem pool.
/// Description: test free part tensor mems keep multi part.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_free_part_tensor_mems_keep_multi_part) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  void *addr = mem_pool->AllocTensorMem(4 * kDynamicMemAlignSize, true, true);
  std::vector<void *> free_addrs;
  (void)free_addrs.emplace_back(addr);
  std::vector<void *> keep_addrs;
  (void)keep_addrs.emplace_back(addr);
  (void)keep_addrs.emplace_back(static_cast<int8_t *>(addr) + 2 * kDynamicMemAlignSize);
  (void)keep_addrs.emplace_back(static_cast<int8_t *>(addr) + 3 * kDynamicMemAlignSize);
  std::vector<size_t> keep_addr_sizes;
  (void)keep_addr_sizes.emplace_back(kDynamicMemAlignSize);
  (void)keep_addr_sizes.emplace_back(kDynamicMemAlignSize);
  (void)keep_addr_sizes.emplace_back(kDynamicMemAlignSize);
  mem_pool->FreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
  EXPECT_EQ(mem_pool->stream_id_allocators_.size(), 1);
  auto allocator = mem_pool->stream_id_allocators_.begin()->second;
  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);
  EXPECT_EQ(allocator->eager_free_mem_bufs_.size(), 1);
  EXPECT_EQ(mem_pool->addr_mem_buf_allocators_.size(), 3);
}

/// Feature: test memory stats for abstract dynamic mem pool.
/// Description: test memory stats.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_memory_stats) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  auto &&block_counts = mem_pool->BlockCountsStatistics();
  EXPECT_EQ(block_counts.size(), 2);
  EXPECT_EQ(block_counts.count(kPersistentMemPoolType), 1);
  EXPECT_EQ(block_counts[kPersistentMemPoolType], 0);
  EXPECT_EQ(block_counts.count(kCommonMemPoolType), 1);
  EXPECT_EQ(block_counts[kCommonMemPoolType], 0);

  auto &&block_unit_size = mem_pool->BlockUnitSizeStatistics();
  EXPECT_EQ(block_unit_size.size(), 2);
  EXPECT_EQ(block_unit_size.count(kPersistentMemPoolType), 1);
  EXPECT_EQ(block_unit_size[kPersistentMemPoolType], kGBToByte);
  EXPECT_EQ(block_unit_size.count(kCommonMemPoolType), 1);
  EXPECT_EQ(block_unit_size[kPersistentMemPoolType], kGBToByte);

  auto &&common_mem_blocks_info = mem_pool->CommonMemBlocksInfoStatistics();
  EXPECT_EQ(common_mem_blocks_info.size(), 0);

  auto &&persistent_mem_blocks_info = mem_pool->PersistentMemBlocksInfoStatistics();
  EXPECT_EQ(persistent_mem_blocks_info.size(), 0);

  std::vector<size_t> sizes{1 * kDynamicMemAlignSize, 2 * kDynamicMemAlignSize, 3 * kDynamicMemAlignSize,
                            4 * kDynamicMemAlignSize, 5 * kDynamicMemAlignSize, 6 * kDynamicMemAlignSize};
  std::vector<void *> addresses;
  for (auto size : sizes) {
    (void)addresses.emplace_back(mem_pool->AllocTensorMem(size));
  }
  auto &&cur_block_counts = mem_pool->BlockCountsStatistics();
  EXPECT_EQ(cur_block_counts.size(), 2);
  EXPECT_EQ(cur_block_counts.count(kPersistentMemPoolType), 1);
  EXPECT_EQ(cur_block_counts[kPersistentMemPoolType], 0);
  EXPECT_EQ(cur_block_counts.count(kCommonMemPoolType), 1);
  EXPECT_EQ(cur_block_counts[kCommonMemPoolType], 1);
  // assert block count.
  EXPECT_EQ(mem_pool->CommonMemBlocksInfoStatistics().size(), 1);
  EXPECT_EQ(mem_pool->PersistentMemBlocksInfoStatistics().size(), 0);
}

/// Feature: test memory reserved size for abstract dynamic mem pool.
/// Description: test memory reserved size.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_memory_reserved_dize) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  std::vector<void *> addrs;
  for (size_t i = 0; i < 100; i++) {
    (void)addrs.emplace_back(mem_pool->AllocTensorMem(i));
  }
  EXPECT_EQ(mem_pool->ReservedMemorySize(), kGBToByte);
  mem_pool->ReleaseDeviceRes();
  EXPECT_EQ(mem_pool->ReservedMemorySize(), 0);
}

class LinearDynamicMemPoolWithoutVmm : public LinearDynamicMemPool {
 public:
  LinearDynamicMemPoolWithoutVmm() { SetEnableVmm(false); }

  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) override {
    auto ret = LinearDynamicMemPool::AllocDeviceMem(size, addr);
    free_mem_size_ -= ret;
    return ret;
  }

  uint64_t total_mem_size() const override { return total_mem_size_; }

  size_t free_mem_size() override { return free_mem_size_; }

 private:
  size_t total_mem_size_{100 * kGBToByte};
  size_t free_mem_size_{total_mem_size_};
};

/// Feature: test persistent mem block limit for abstract dynamic mem pool.
/// Description: test persistent mem block limit.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_persistent_block_limit) {
  auto mem_pool = std::make_shared<LinearDynamicMemPoolWithoutVmm>();
  std::vector<void *> addrs;
  for (size_t i = 0; i < 10; i++) {
    (void)addrs.emplace_back(mem_pool->AllocTensorMem(kGBToByte, true));
  }
  const auto &allocators_map = mem_pool->stream_id_allocators_;
  EXPECT_EQ(allocators_map.size(), 2);
  auto persistent_allocators_it = allocators_map.find(AllocatorInfo{0, true, false});
  EXPECT_TRUE(persistent_allocators_it != allocators_map.end());
  EXPECT_TRUE(persistent_allocators_it->second->mem_blocks_.size() == 1);
  auto common_allocators_it = allocators_map.find(AllocatorInfo{0, false, false});
  EXPECT_TRUE(common_allocators_it != allocators_map.end());
  EXPECT_TRUE(common_allocators_it->second->mem_blocks_.size() == 9);
}

/// Feature: test malloc common memory from persistent pool.
/// Description: test memory pool malloc mem block limit.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_common_malloc_from_persistent) {
  // Total memory is 100Gb.
  //    First, malloc 50GB persistent memory and 50GB common memory.
  //    Second, free 50GB persistent memory.
  //    Third, malloc 1GB common memory.
  auto mem_pool = std::make_shared<LinearDynamicMemPoolWithoutVmm>();
  auto persistent_memory = mem_pool->AllocTensorMem(50 * kGBToByte, true);
  auto common_memory = mem_pool->AllocTensorMem(50 * kGBToByte, false);
  auto more_memory = mem_pool->AllocTensorMem(1, false);
  EXPECT_TRUE(more_memory == nullptr);
  mem_pool->FreeTensorMem(persistent_memory);
  more_memory = mem_pool->AllocTensorMem(1, false);
  EXPECT_TRUE(more_memory != nullptr);
  mem_pool->FreeTensorMem(common_memory);
  mem_pool->FreeTensorMem(more_memory);
}

/// Feature: test release free blocks for abstract dynamic mem pool.
/// Description: test release free blocks.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_release_free_blocks) {
  auto mem_pool = std::make_shared<LinearDynamicMemPoolWithoutVmm>();
  std::vector<void *> addrs;
  for (size_t i = 0; i < 10; i++) {
    (void)addrs.emplace_back(mem_pool->AllocTensorMem(kGBToByte));
  }
  EXPECT_EQ(mem_pool->TotalMemStatistics(), kGBToByte * 10);
  mem_pool->FreeTensorMem(addrs[0]);
  mem_pool->FreeTensorMem(addrs[9]);
  EXPECT_EQ(mem_pool->TotalMemStatistics(), kGBToByte * 10);
  EXPECT_EQ(mem_pool->TotalUsedMemStatistics(), kGBToByte * 8);
  size_t release_size = mem_pool->ReleaseFreeBlocks();
  EXPECT_EQ(release_size, kGBToByte * 2);
  EXPECT_EQ(mem_pool->TotalMemStatistics(), kGBToByte * 8);
  EXPECT_EQ(mem_pool->TotalUsedMemStatistics(), kGBToByte * 8);
}

/// Feature: test empty cache for abstract dynamic mem pool.
/// Description: test empty cache.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_empty_cache) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  std::vector<void *> addrs;
  for (size_t i = 0; i < 10; i++) {
    (void)addrs.emplace_back(mem_pool->AllocTensorMem(kGBToByte));
  }
  for (size_t i = 0; i < 10; i++) {
    mem_pool->FreeTensorMem(addrs[i]);
  }
  EXPECT_EQ(mem_pool->TotalMemStatistics(), kGBToByte * 10);
  EXPECT_EQ(mem_pool->TotalUsedMemStatistics(), 0);
  // Empty cache is not directly implements in memory pool.
  size_t expected_size = -1L;
  EXPECT_EQ(mem_pool->EmptyCache(), expected_size);
}

/// Feature: test insert and erase eager free buf for MemBufAllocator.
/// Description: test insert and erase eager free buf functionality.
/// Expectation: insert and erase operation works correctly and updates statistics.
TEST_F(TestMemBufAllocator, test_insert_erase_eager_free_buf) {
  const auto allocator = GenerateMemBufAllocatorPtr();
  const auto base_addr = reinterpret_cast<void *>(0x100000);
  const auto mem_block = new MemBlock(1024, base_addr, 0);
  auto mem_buf = new MemBuf(512, base_addr, 0, mem_block, MemBufStatus::kMemBufEagerFree);
  const size_t initial_eager_free_size = allocator->mem_stat_ptr_->eager_free_size_;

  allocator->InsertEagerFreeBuf(mem_buf);
  EXPECT_EQ(allocator->eager_free_mem_bufs_.size(), 1);
  EXPECT_EQ(allocator->mem_stat_ptr_->eager_free_size_, initial_eager_free_size + 512);
  auto it = allocator->eager_free_mem_bufs_.find(mem_buf);
  EXPECT_EQ(*it, mem_buf);

  allocator->EraseEagerFreeBuf(mem_buf);
  it = allocator->eager_free_mem_bufs_.find(mem_buf);
  EXPECT_EQ(allocator->eager_free_mem_bufs_.size(), 0);
  EXPECT_EQ(allocator->mem_stat_ptr_->eager_free_size_, initial_eager_free_size);
}

/// Feature: test merge mem buf for MemBufAllocator.
/// Description: test merge mem buf functionality through friend class.
/// Expectation: merge operation works correctly in all scenarios.
TEST_F(TestMemBufAllocator, test_merge_mem_buf_with_friend) {
  const auto allocator = GenerateMemBufAllocatorPtr();
  std::uintptr_t base_addr = 0x100000;
  const auto mem_block = new MemBlock(10240, reinterpret_cast<void *>(base_addr), 0);
  auto mem_buf1 = new MemBuf(1024, reinterpret_cast<void *>(base_addr), 0, mem_block, MemBufStatus::kMemBufIdle);
  auto mem_buf2 = new MemBuf(1024, reinterpret_cast<void *>(base_addr + 1024), 0, mem_block, MemBufStatus::kMemBufIdle);

  mem_buf1->next_ = mem_buf2;
  mem_buf2->prev_ = mem_buf1;
  allocator->free_mem_bufs_.insert(mem_buf1);
  allocator->free_mem_bufs_.insert(mem_buf2);
  // merge mem_buf1 to mem_buf2, MergeMemBuff keeps mem_buf2 and delete mem_buf1
  allocator->MergeMemBuf(mem_buf1, mem_buf2);
  EXPECT_EQ(mem_buf2->size_, 2048);
  EXPECT_EQ(mem_buf2->addr_, reinterpret_cast<void *>(base_addr));
  EXPECT_EQ(mem_buf2->prev_, nullptr);
  EXPECT_EQ(mem_buf2->next_, nullptr);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);
  auto mem_buf2_it = allocator->free_mem_bufs_.find(mem_buf2);
  EXPECT_EQ(*mem_buf2_it, mem_buf2);

  auto mem_buf3 = new MemBuf(1024, reinterpret_cast<void *>(base_addr + 2048), 0, mem_block, MemBufStatus::kMemBufIdle);
  mem_buf2->next_ = mem_buf3;
  mem_buf3->prev_ = mem_buf2;
  allocator->free_mem_bufs_.insert(mem_buf3);
  // merge mem_buf3 to mem_buf2, MergeMemBuff keeps mem_buf2 and delete mem_buf3
  allocator->MergeMemBuf(mem_buf3, mem_buf2);
  EXPECT_EQ(mem_buf2->size_, 3072);
  EXPECT_EQ(mem_buf2->addr_, reinterpret_cast<void *>(base_addr));
  EXPECT_EQ(mem_buf2->prev_, nullptr);
  EXPECT_EQ(mem_buf2->next_, nullptr);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);
  mem_buf2_it = allocator->free_mem_bufs_.find(mem_buf2);
  EXPECT_EQ(*mem_buf2_it, mem_buf2);
}
}  // namespace device
}  // namespace mindspore
