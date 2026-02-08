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
#include <atomic>

#include "common/common_test.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#define private public
#define protected public
#include "plugin/ascend/res_manager/mem_manager/ascend_memory_pool.h"
#undef private
#undef protected

namespace mindspore {
namespace device {
namespace ascend {
class TestAscendMemoryPool : public UT::Common {
 public:
  TestAscendMemoryPool() = default;
  virtual ~TestAscendMemoryPool() = default;
};

struct ConstCounter {
  ConstCounter(size_t val = 0) : val_(new size_t(val)) {}
  ~ConstCounter() { delete val_; }

  void operator++(int) const { *val_ = *val_ + 1; }

  size_t Get() { return *val_; }

  size_t *const val_;
};

class DefaultAscendMemoryPoolImpl : public DefaultAscendMemoryPool {
 public:
  DefaultAscendMemoryPoolImpl() = default;
  ~DefaultAscendMemoryPoolImpl() override = default;

  size_t release_device_res_{0};
  void ReleaseDeviceRes() override { release_device_res_++; }

  size_t alloc_tensor_mem_{0};
  DeviceMemPtr AllocTensorMem(size_t size, bool from_persistent_mem = false, bool need_recycle = false,
                              uint32_t stream_id = kDefaultStreamIndex) override {
    alloc_tensor_mem_++;
    return reinterpret_cast<void *>(1);
  }

  size_t alloc_continuous_tensor_mem_{0};
  std::vector<DeviceMemPtr> AllocContinuousTensorMem(const std::vector<size_t> &size_list,
                                                     uint32_t stream_id = kDefaultStreamIndex) override {
    alloc_continuous_tensor_mem_++;
    return {nullptr};
  }

  size_t free_tensor_mem_{0};
  void FreeTensorMem(const DeviceMemPtr &device_addr) override { free_tensor_mem_++; }

  size_t do_free_tensor_mem_{0};
  bool DoFreeTensorMem(const DeviceMemPtr &device_addr) override {
    do_free_tensor_mem_++;
    return false;
  }

  size_t free_part_tensor_mems_{0};
  void FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs, const std::vector<DeviceMemPtr> &keep_addrs,
                          const std::vector<size_t> &keep_addr_sizes) override {
    free_part_tensor_mems_++;
  }

  size_t do_free_part_tensor_mems_{0};
  std::vector<MemBuf *> DoFreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                             const std::vector<DeviceMemPtr> &keep_addrs,
                                             const std::vector<size_t> &keep_addr_sizes) override {
    do_free_part_tensor_mems_++;
    return {};
  }

  size_t record_event_{0};
  bool RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                   const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                   const DeviceEventPtr &event) override {
    record_event_++;
    return false;
  }

  size_t wait_event_{0};
  bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) override {
    wait_event_++;
    return false;
  }

  bool WaitEvent(int64_t task_id_on_stream, uint32_t memory_stream_id) override {
    wait_event_++;
    return false;
  }

  size_t sync_all_events_{0};
  bool SyncAllEvents() override {
    sync_all_events_++;
    return false;
  }

  ConstCounter align_memory_size_;
  size_t AlignMemorySize(size_t size) const override {
    align_memory_size_++;
    return size;
  }

  size_t cal_mem_block_alloc_size_{0};
  size_t CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool need_recycle = false) override {
    cal_mem_block_alloc_size_++;
    return kDynamicMemAllocUnitSize;
  }

  size_t set_mem_pool_block_size_{0};
  void SetMemPoolBlockSize(size_t available_device_mem_size) override { set_mem_pool_block_size_++; }

  ConstCounter mem_alloc_unit_size_{0};
  size_t MemAllocUnitSize(bool from_persistent_mem) const override {
    mem_alloc_unit_size_++;
    return kDynamicMemAllocUnitSize;
  }

  size_t set_mem_alloc_unit_size_{0};
  void SetMemAllocUintSize(size_t common_size, size_t persist_size = kDynamicMemAllocUnitSize) override {
    set_mem_alloc_unit_size_++;
  }

  ConstCounter get_min_using_memory_addr_{0};
  void *GetMinUsingMemoryAddr() const override {
    get_min_using_memory_addr_++;
    return nullptr;
  }

  size_t alloc_device_mem_{0};
  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) override {
    alloc_device_mem_++;
    static size_t base = 0;
    *addr = reinterpret_cast<void *>(base);
    base += size;
    return size;
  }

  size_t free_device_mem_{0};
  bool FreeDeviceMem(const DeviceMemPtr &addr) override {
    free_device_mem_++;
    return true;
  }

  size_t free_mem_size_{0};
  size_t free_mem_size() override {
    free_mem_size_++;
    return SIZE_MAX;
  }

  ConstCounter total_mem_size_{0};
  uint64_t total_mem_size() const override {
    total_mem_size_++;
    return SIZE_MAX;
  }

  ConstCounter get_max_used_mem_size_{0};
  size_t GetMaxUsedMemSize() const override {
    get_max_used_mem_size_++;
    return SIZE_MAX;
  }

  ConstCounter get_vmm_used_mem_size_{0};
  size_t GetVmmUsedMemSize() const override {
    get_vmm_used_mem_size_++;
    return 0;
  }

  size_t defrag_memory_{0};
  void DefragMemory() override { defrag_memory_++; }

  size_t dump_dynamic_mem_pool_state_info_{0};
  void DumpDynamicMemPoolStateInfo() override { dump_dynamic_mem_pool_state_info_++; }

  size_t dump_dynamic_mem_pool_debug_info_{0};
  void DumpDynamicMemPoolDebugInfo() { dump_dynamic_mem_pool_debug_info_++; }

  ConstCounter total_mem_statistics_{0};
  size_t TotalMemStatistics() const override {
    total_mem_statistics_++;
    return 0;
  };

  ConstCounter total_used_mem_statistics_{0};
  size_t TotalUsedMemStatistics() const override {
    total_used_mem_statistics_++;
    return 0;
  };

  ConstCounter total_used_by_event_mem_statistics_{0};
  size_t TotalUsedByEventMemStatistics() const override {
    total_used_by_event_mem_statistics_++;
    return 0;
  }

  ConstCounter total_idle_mem_statistics_{0};
  size_t TotalIdleMemStatistics() const override {
    total_idle_mem_statistics_++;
    return 0;
  }

  ConstCounter total_eager_free_mem_statistics_{0};
  size_t TotalEagerFreeMemStatistics() const override {
    total_eager_free_mem_statistics_++;
    return 0;
  }

  ConstCounter used_mem_peak_statistics_{0};
  size_t UsedMemPeakStatistics() const override {
    used_mem_peak_statistics_++;
    return 0;
  }

  ConstCounter max_mem_allocated_statistics_{0};
  size_t MaxMemAllocatedStatistics() const override {
    max_mem_allocated_statistics_++;
    return 0;
  }

  ConstCounter max_mem_reserved_statistics_{0};
  size_t MaxMemReservedStatistics() const override {
    max_mem_reserved_statistics_++;
    return 0;
  }

  ConstCounter actual_peak_statistics_{0};
  size_t ActualPeakStatistics() const override {
    actual_peak_statistics_++;
    return 0;
  }

  ConstCounter block_counts_statistics_{0};
  std::unordered_map<std::string, std::size_t> BlockCountsStatistics() const override {
    block_counts_statistics_++;
    return {};
  }

  ConstCounter block_unit_size_statistics_{0};
  std::unordered_map<std::string, std::size_t> BlockUnitSizeStatistics() const override {
    block_unit_size_statistics_++;
    return {};
  }

  ConstCounter common_mem_blocks_info_statistics_{0};
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> CommonMemBlocksInfoStatistics()
    const override {
    common_mem_blocks_info_statistics_++;
    return {};
  }

  ConstCounter persistent_mem_blocks_info_statistics_{0};
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> PersistentMemBlocksInfoStatistics()
    const override {
    persistent_mem_blocks_info_statistics_++;
    return {};
  }

  size_t reset_max_mem_reserved_{0};
  void ResetMaxMemReserved() override { reset_max_mem_reserved_++; }

  size_t reset_max_mem_allocated_{0};
  void ResetMaxMemAllocated() override { reset_max_mem_allocated_++; }

  ConstCounter is_enable_eager_free_{0};
  const bool IsEnableEagerFree() const override {
    is_enable_eager_free_++;
    return false;
  }

  ConstCounter is_enable_vmm_{0};
  const bool IsEnableVmm() const override {
    is_enable_vmm_++;
    return false;
  }

  size_t set_enable_vmm_{0};
  void SetEnableVmm(bool enable_vmm) override { set_enable_vmm_++; }

  ConstCounter sync_all_streams_{0};
  const bool SyncAllStreams() override {
    sync_all_streams_++;
    return false;
  }

  size_t alloc_device_mem_by_eager_free_{0};
  size_t AllocDeviceMemByEagerFree(size_t size, DeviceMemPtr *addr) override {
    alloc_device_mem_by_eager_free_++;
    return 0;
  }

  size_t free_device_mem_by_eager_free_{0};
  size_t FreeDeviceMemByEagerFree(const DeviceMemPtr addr, const size_t size) override {
    free_device_mem_by_eager_free_++;
    return 0;
  }

  size_t mmap_device_mem_{0};
  size_t MmapDeviceMem(size_t size, DeviceMemPtr addr) {
    mmap_device_mem_++;
    return 0;
  }

  ConstCounter free_idle_mems_by_eager_free_{0};
  const std::pair<size_t, size_t> FreeIdleMemsByEagerFree() {
    free_idle_mems_by_eager_free_++;
    return {0, 0};
  }

  size_t release_free_blocks_{0};
  size_t ReleaseFreeBlocks() override {
    release_free_blocks_++;
    return 0;
  }

  size_t is_enable_time_event_{0};
  bool IsEnableTimeEvent() {
    is_enable_time_event_++;
    return false;
  }

  size_t set_enable_time_event_{0};
  void SetEnableTimeEvent(bool enable_time_event) { set_enable_time_event_++; }

  size_t report_memory_pool_info_{0};
  void ReportMemoryPoolInfo() override { report_memory_pool_info_++; }

  size_t report_free_memory_pool_info_to_mstx_{0};
  void ReportMemoryPoolFreeInfoToMstx(void *ptr) override { report_free_memory_pool_info_to_mstx_++; }

  size_t report_malloc_memory_pool_info_to_mstx_{0};
  void ReportMemoryPoolMallocInfoToMstx(void *ptr, size_t size) override { report_malloc_memory_pool_info_to_mstx_++; };

  size_t gen_allocate_memory_time_event_{0};
  MemoryTimeEventPtr GenAllocateMemoryTimeEvent(const void *addr, size_t size, uint32_t stream_id, bool from_persistent,
                                                bool is_persistent) override {
    gen_allocate_memory_time_event_++;
    return nullptr;
  }

  size_t gen_free_memory_time_event_{0};
  MemoryTimeEventPtr GenFreeMemoryTimeEvent(const void *addr) override {
    gen_free_memory_time_event_++;
    return nullptr;
  }

  ConstCounter reset_idle_mem_buf_{0};
  void ResetIdleMemBuf() const override { reset_idle_mem_buf_++; }

  std::string GetMemoryPoolType() const override { return "DefaultAscendMemoryPoolImpl"; }

  size_t gen_set_rank_id_getter_{0};
  void SetRankIdGetter(const std::function<size_t()> &rank_id_getter) override { gen_set_rank_id_getter_++; }
};
using DefaultAscendMemoryPoolImplPtr = std::shared_ptr<DefaultAscendMemoryPoolImpl>;

/// Feature: test ascend vmm adapter.
/// Description: test basic allocation.
/// Expectation: can alloc memory and can not throw exception.
TEST_F(TestAscendMemoryPool, test_default_enhanced_ascend_memory_pool_proxy) {
  auto pool = std::make_shared<DefaultAscendMemoryPoolImpl>();
  auto enhanced_pool = std::make_shared<DefaultEnhancedAscendMemoryPool>(pool);

  EXPECT_EQ(pool->set_enable_vmm_, 1);
  EXPECT_EQ(pool->actual_peak_statistics_.Get(), 0);

  EXPECT_EQ(pool->release_device_res_, 0);
  enhanced_pool->ReleaseDeviceRes();
  EXPECT_EQ(pool->release_device_res_, 1);
  EXPECT_EQ(pool->is_enable_time_event_, 0);

  enhanced_pool->AllocTensorMem(0);
  EXPECT_EQ(pool->alloc_tensor_mem_, 0);
  EXPECT_EQ(pool->dump_dynamic_mem_pool_state_info_, 0);
  EXPECT_EQ(pool->dump_dynamic_mem_pool_debug_info_, 0);
  EXPECT_EQ(pool->is_enable_time_event_, 1);
  EXPECT_EQ(pool->actual_peak_statistics_.Get(), 0);
  EXPECT_EQ(pool->align_memory_size_.Get(), 1);
  EXPECT_EQ(pool->report_memory_pool_info_, 1);
  EXPECT_EQ(pool->report_malloc_memory_pool_info_to_mstx_, 1);

  enhanced_pool->AllocContinuousTensorMem({});
  EXPECT_EQ(pool->alloc_continuous_tensor_mem_, 1);

  enhanced_pool->FreeTensorMem(nullptr);
  EXPECT_EQ(pool->free_tensor_mem_, 0);
  EXPECT_EQ(pool->do_free_tensor_mem_, 1);

  enhanced_pool->DoFreeTensorMem(nullptr);
  EXPECT_EQ(pool->do_free_tensor_mem_, 2);
  EXPECT_EQ(pool->actual_peak_statistics_.Get(), 0);

  enhanced_pool->FreePartTensorMems({}, {}, {});
  EXPECT_EQ(pool->free_part_tensor_mems_, 0);
  EXPECT_EQ(pool->is_enable_time_event_, 3);
  EXPECT_EQ(pool->do_free_part_tensor_mems_, 1);
  EXPECT_EQ(pool->actual_peak_statistics_.Get(), 0);

  enhanced_pool->DoFreePartTensorMems({}, {}, {});
  EXPECT_EQ(pool->do_free_part_tensor_mems_, 2);

  enhanced_pool->RecordEvent(0, 0, {}, nullptr);
  EXPECT_EQ(pool->record_event_, 0);

  enhanced_pool->WaitEvent(0, 0, 0);
  EXPECT_EQ(pool->wait_event_, 0);

  enhanced_pool->WaitEvent(0, 0);
  EXPECT_EQ(pool->wait_event_, 0);

  enhanced_pool->SyncAllEvents();
  EXPECT_EQ(pool->sync_all_events_, 1);

  enhanced_pool->AlignMemorySize(1);
  EXPECT_EQ(pool->align_memory_size_.Get(), 2);

  enhanced_pool->CalMemBlockAllocSize(0, true);
  EXPECT_EQ(pool->cal_mem_block_alloc_size_, 2);

  enhanced_pool->SetMemPoolBlockSize(0);
  EXPECT_EQ(pool->set_mem_pool_block_size_, 1);

  enhanced_pool->MemAllocUnitSize(true);
  EXPECT_EQ(pool->mem_alloc_unit_size_.Get(), 1);

  enhanced_pool->SetMemAllocUintSize(0);
  EXPECT_EQ(pool->set_mem_alloc_unit_size_, 1);

  enhanced_pool->GetMinUsingMemoryAddr();
  EXPECT_EQ(pool->get_min_using_memory_addr_.Get(), 1);

  void *addr;
  EXPECT_EQ(pool->alloc_device_mem_, 1);
  enhanced_pool->AllocDeviceMem(0, &addr);
  EXPECT_EQ(pool->alloc_device_mem_, 2);

  enhanced_pool->FreeDeviceMem(nullptr);
  EXPECT_EQ(pool->free_device_mem_, 1);

  enhanced_pool->free_mem_size();
  EXPECT_EQ(pool->free_mem_size_, 1);

  enhanced_pool->total_mem_size();
  EXPECT_EQ(pool->total_mem_size_.Get(), 1);

  enhanced_pool->GetMaxUsedMemSize();
  EXPECT_EQ(pool->get_max_used_mem_size_.Get(), 1);

  enhanced_pool->GetVmmUsedMemSize();
  EXPECT_EQ(pool->get_vmm_used_mem_size_.Get(), 2);

  enhanced_pool->DefragMemory();
  EXPECT_EQ(pool->defrag_memory_, 1);

  enhanced_pool->DumpDynamicMemPoolStateInfo();
  EXPECT_EQ(pool->dump_dynamic_mem_pool_state_info_, 1);

  enhanced_pool->DumpDynamicMemPoolDebugInfo();
  EXPECT_EQ(pool->dump_dynamic_mem_pool_debug_info_, 1);

  enhanced_pool->TotalMemStatistics();
  EXPECT_EQ(pool->total_mem_statistics_.Get(), 1);

  enhanced_pool->TotalUsedMemStatistics();
  EXPECT_EQ(pool->total_used_mem_statistics_.Get(), 1);

  enhanced_pool->TotalUsedByEventMemStatistics();
  EXPECT_EQ(pool->total_used_by_event_mem_statistics_.Get(), 1);

  enhanced_pool->TotalIdleMemStatistics();
  EXPECT_EQ(pool->total_idle_mem_statistics_.Get(), 1);

  enhanced_pool->TotalEagerFreeMemStatistics();
  EXPECT_EQ(pool->total_eager_free_mem_statistics_.Get(), 1);

  enhanced_pool->UsedMemPeakStatistics();
  EXPECT_EQ(pool->used_mem_peak_statistics_.Get(), 1);

  enhanced_pool->MaxMemAllocatedStatistics();
  EXPECT_EQ(pool->max_mem_allocated_statistics_.Get(), 1);

  enhanced_pool->MaxMemReservedStatistics();
  EXPECT_EQ(pool->max_mem_reserved_statistics_.Get(), 1);

  EXPECT_EQ(pool->actual_peak_statistics_.Get(), 1);
  enhanced_pool->ActualPeakStatistics();
  EXPECT_EQ(pool->actual_peak_statistics_.Get(), 2);

  enhanced_pool->BlockCountsStatistics();
  EXPECT_EQ(pool->block_counts_statistics_.Get(), 1);

  enhanced_pool->BlockUnitSizeStatistics();
  EXPECT_EQ(pool->block_unit_size_statistics_.Get(), 1);

  enhanced_pool->CommonMemBlocksInfoStatistics();
  EXPECT_EQ(pool->common_mem_blocks_info_statistics_.Get(), 1);

  enhanced_pool->PersistentMemBlocksInfoStatistics();
  EXPECT_EQ(pool->persistent_mem_blocks_info_statistics_.Get(), 1);

  enhanced_pool->ResetMaxMemReserved();
  EXPECT_EQ(pool->reset_max_mem_reserved_, 1);

  enhanced_pool->ResetMaxMemAllocated();
  EXPECT_EQ(pool->reset_max_mem_allocated_, 1);

  enhanced_pool->IsEnableEagerFree();
  EXPECT_EQ(pool->is_enable_eager_free_.Get(), 3);

  enhanced_pool->IsEnableVmm();
  EXPECT_EQ(pool->is_enable_vmm_.Get(), 3);

  enhanced_pool->SetEnableVmm(true);
  EXPECT_EQ(pool->set_enable_vmm_, 2);

  enhanced_pool->SyncAllStreams();
  EXPECT_EQ(pool->sync_all_streams_.Get(), 1);

  enhanced_pool->AllocDeviceMemByEagerFree(0, &addr);
  EXPECT_EQ(pool->alloc_device_mem_by_eager_free_, 1);

  enhanced_pool->FreeDeviceMemByEagerFree(addr, 0);
  EXPECT_EQ(pool->free_device_mem_by_eager_free_, 1);

  enhanced_pool->MmapDeviceMem(0, addr);
  EXPECT_EQ(pool->mmap_device_mem_, 1);

  enhanced_pool->FreeIdleMemsByEagerFree();
  EXPECT_EQ(pool->free_idle_mems_by_eager_free_.Get(), 1);

  enhanced_pool->ReleaseFreeBlocks();
  EXPECT_EQ(pool->release_free_blocks_, 1);

  EXPECT_EQ(pool->is_enable_time_event_, 3);
  enhanced_pool->IsEnableTimeEvent();
  EXPECT_EQ(pool->is_enable_time_event_, 4);

  enhanced_pool->SetEnableTimeEvent(true);
  EXPECT_EQ(pool->set_enable_time_event_, 1);

  enhanced_pool->ReportMemoryPoolInfo();
  EXPECT_EQ(pool->report_memory_pool_info_, 2);

  void *ptr;
  enhanced_pool->ReportMemoryPoolFreeInfoToMstx(ptr);
  EXPECT_EQ(pool->report_free_memory_pool_info_to_mstx_, 1);

  size_t size{0};
  enhanced_pool->ReportMemoryPoolMallocInfoToMstx(ptr, size);
  EXPECT_EQ(pool->report_malloc_memory_pool_info_to_mstx_, 2);

  enhanced_pool->GenAllocateMemoryTimeEvent(nullptr, 0, 0, true, true);
  EXPECT_EQ(pool->gen_allocate_memory_time_event_, 1);

  enhanced_pool->GenFreeMemoryTimeEvent(nullptr);
  EXPECT_EQ(pool->gen_free_memory_time_event_, 1);

  enhanced_pool->ResetIdleMemBuf();
  EXPECT_EQ(pool->reset_idle_mem_buf_.Get(), 1);

  EXPECT_EQ(enhanced_pool->GetMemoryPoolType(), "DefaultEnhancedAscendMemoryPool");

  EXPECT_EQ(pool->gen_set_rank_id_getter_, 0);
  enhanced_pool->SetRankIdGetter([]() { return 0L; });
  EXPECT_EQ(pool->gen_set_rank_id_getter_, 1);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore