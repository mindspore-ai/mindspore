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
#include "common/common_test.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#define private public
#define protected public
#include "plugin/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#undef private
#undef protected

namespace mindspore {
namespace device {
namespace ascend {
class TestAscendVmmAdapter : public UT::Common {
 public:
  TestAscendVmmAdapter() = default;
  virtual ~TestAscendVmmAdapter() = default;

  void SetUp() override {
    mindspore::device::ascend::aclrtUnmapMem_ = aclrtUnmapMem;
    mindspore::device::ascend::aclrtReserveMemAddress_ = aclrtReserveMemAddress;
    mindspore::device::ascend::aclrtMallocPhysical_ = aclrtMallocPhysical;
    mindspore::device::ascend::aclrtMapMem_ = aclrtMapMem;
    mindspore::device::ascend::aclrtFreePhysical_ = aclrtFreePhysical;
    mindspore::device::ascend::aclrtReleaseMemAddress_ = aclrtReleaseMemAddress;
    common::ResetConfig("MS_ALLOC_CONF");
  }
  void TearDown() override {}

  AscendVmmAdapter ascend_vmm_adapter_;
};

/// Feature: test ascend vmm adapter.
/// Description: test basic allocation.
/// Expectation: can alloc memory and can not throw exception.
TEST_F(TestAscendVmmAdapter, test_basic_allocation) {
  size_t block = ascend_vmm_adapter_.vmm_align_size_;
  DeviceMemPtr addr = reinterpret_cast<DeviceMemPtr>(block);
  size_t size = 1024;
  size_t ret = ascend_vmm_adapter_.AllocDeviceMem(size, &addr);
  EXPECT_EQ(ret, ascend_vmm_adapter_.vmm_align_size_);
}

/// Feature: test ascend vmm adapter.
/// Description: test set align size.
/// Expectation: can set align size.
TEST_F(TestAscendVmmAdapter, test_set_align_size) {
  common::SetEnv("MS_ALLOC_CONF", "vmm_align_size:20MB");
  size_t mb = 1024 * 1024;
  AscendVmmAdapter test_vmm_adapter;
  EXPECT_EQ(test_vmm_adapter.vmm_align_size_, 20 * mb);
  common::SetEnv("MS_ALLOC_CONF", "");
  common::ResetConfig("MS_ALLOC_CONF");
}

/// Feature: test ascend vmm adapter.
/// Description: test set align size throw.
/// Expectation: can throw exception.
TEST_F(TestAscendVmmAdapter, test_set_align_size_throw) {
  common::SetEnv("MS_ALLOC_CONF", "vmm_align_size:balabala");
  EXPECT_ANY_THROW(AscendVmmAdapter test_vmm_adapter);
  common::SetEnv("MS_ALLOC_CONF", "");
  common::ResetConfig("MS_ALLOC_CONF");
}

/// Feature: test ascend vmm adapter.
/// Description: test round up.
/// Expectation: can round up.
TEST_F(TestAscendVmmAdapter, test_round_up) {
  size_t block = ascend_vmm_adapter_.vmm_align_size_;
  size_t size = block + 33;
  size_t ret = ascend_vmm_adapter_.GetRoundUpAlignSize(size);
  EXPECT_EQ(ret, block * 2);
}

/// Feature: test ascend vmm adapter.
/// Description: test round up.
/// Expectation: can round up.
TEST_F(TestAscendVmmAdapter, test_round_up_1) {
  common::SetEnv("MS_ALLOC_CONF", "vmm_align_size:20MB");
  AscendVmmAdapter test_vmm_adapter;
  size_t block = test_vmm_adapter.vmm_align_size_;
  size_t size = block + 33;
  size_t ret = test_vmm_adapter.GetRoundUpAlignSize(size);
  EXPECT_EQ(ret, block * 2);
  common::SetEnv("MS_ALLOC_CONF", "");
  common::ResetConfig("MS_ALLOC_CONF");
}

/// Feature: test ascend vmm adapter.
/// Description: test round down.
/// Expectation: can round down.
TEST_F(TestAscendVmmAdapter, test_round_down) {
  size_t block = ascend_vmm_adapter_.vmm_align_size_;
  size_t size = block + 33;
  size_t ret = ascend_vmm_adapter_.GetRoundDownAlignSize(size);
  EXPECT_EQ(ret, block);
}

/// Feature: test ascend vmm adapter.
/// Description: test round down.
/// Expectation: can round down.
TEST_F(TestAscendVmmAdapter, test_round_down_1) {
  common::SetEnv("MS_ALLOC_CONF", "vmm_align_size:20MB");
  AscendVmmAdapter test_vmm_adapter;
  size_t block = test_vmm_adapter.vmm_align_size_;
  size_t size = block + 33;
  size_t ret = test_vmm_adapter.GetRoundDownAlignSize(size);
  EXPECT_EQ(ret, block);
  common::SetEnv("MS_ALLOC_CONF", "");
  common::ResetConfig("MS_ALLOC_CONF");
}

/// Feature: test ascend vmm adapter.
/// Description: test handle size.
/// Expectation: can get handle size.
TEST_F(TestAscendVmmAdapter, test_handle_size) {
  size_t block = ascend_vmm_adapter_.vmm_align_size_;
  size_t size = block * 99;
  size_t ret = ascend_vmm_adapter_.GetHandleSize(size);
  EXPECT_EQ(ret, 99);
}

/// Feature: test ascend vmm adapter.
/// Description: test handle size throw.
/// Expectation: can throw exception.
TEST_F(TestAscendVmmAdapter, test_handle_size_throw) {
  size_t block = ascend_vmm_adapter_.vmm_align_size_;
  size_t size = block * 99 + 233;
  EXPECT_ANY_THROW(ascend_vmm_adapter_.GetHandleSize(size));
}

/// Feature: test ascend vmm adapter.
/// Description: test find vmm segment.
/// Expectation: can find vmm segment.
TEST_F(TestAscendVmmAdapter, test_find_vmm_segment) {
  size_t size = ascend_vmm_adapter_.vmm_align_size_;
  DeviceMemPtr addr = reinterpret_cast<DeviceMemPtr>(size);
  auto ret = ascend_vmm_adapter_.AllocDeviceMem(size * 7, &addr);
  EXPECT_EQ(ret, size * 7);
  ret = ascend_vmm_adapter_.MmapDeviceMem(size, addr, SIZE_MAX);
  EXPECT_EQ(ret, size);
  DeviceMemPtr addr2 = reinterpret_cast<DeviceMemPtr>(size * 3);
  ret = ascend_vmm_adapter_.MmapDeviceMem(size, addr2, SIZE_MAX);
  EXPECT_EQ(ret, size);
  DeviceMemPtr addr3 = reinterpret_cast<DeviceMemPtr>(size + 233);
  DeviceMemPtr ret_addr = ascend_vmm_adapter_.FindVmmSegment(addr3);
  EXPECT_EQ(ret_addr, addr);
}

/// Feature: test ascend vmm adapter.
/// Description: test mmap device memory.
/// Expectation: can mmap device memory.
TEST_F(TestAscendVmmAdapter, test_mmap_device_mem) {
  size_t size = ascend_vmm_adapter_.vmm_align_size_;
  DeviceMemPtr addr = reinterpret_cast<DeviceMemPtr>(size);
  auto ret = ascend_vmm_adapter_.AllocDeviceMem(size * 20, &addr);
  EXPECT_EQ(ret, size * 20);
  ret = ascend_vmm_adapter_.MmapDeviceMem(size, addr, SIZE_MAX);
  EXPECT_EQ(ret, size);
  DeviceMemPtr addr2 = reinterpret_cast<DeviceMemPtr>(size * 7);
  ret = ascend_vmm_adapter_.MmapDeviceMem(size, addr2, SIZE_MAX);
  EXPECT_EQ(ret, size);
  auto iter0 = ascend_vmm_adapter_.vmm_map_.find(addr);
  EXPECT_NE(iter0, ascend_vmm_adapter_.vmm_map_.end());
  auto iter1 = ascend_vmm_adapter_.vmm_map_.find(addr2);
  EXPECT_NE(iter1, ascend_vmm_adapter_.vmm_map_.end());
}

/// Feature: test ascend vmm adapter.
/// Description: test mmap device memory.
/// Expectation: can't mmap device memory.
TEST_F(TestAscendVmmAdapter, test_mmap_device_mem_err) {
  size_t size = ascend_vmm_adapter_.vmm_align_size_;
  DeviceMemPtr addr = reinterpret_cast<DeviceMemPtr>(size);
  auto ret = ascend_vmm_adapter_.AllocDeviceMem(size * 20, &addr);
  EXPECT_EQ(ret, size * 20);
  DeviceMemPtr addr2 = reinterpret_cast<DeviceMemPtr>(size * 24 + 233);
  ret = ascend_vmm_adapter_.MmapDeviceMem(size, addr2, SIZE_MAX);
  EXPECT_EQ(ret, 0);
}

/// Feature: test ascend vmm adapter.
/// Description: test eager free memory.
/// Expectation: can eager free memory.
TEST_F(TestAscendVmmAdapter, test_eager_free_0) {
  size_t size = ascend_vmm_adapter_.vmm_align_size_;
  DeviceMemPtr addr = reinterpret_cast<DeviceMemPtr>(size);
  ascend_vmm_adapter_.vmm_map_[addr] = reinterpret_cast<void *>(1);
  size_t ret_size = ascend_vmm_adapter_.EagerFreeDeviceMem(addr, size);
  EXPECT_EQ(ret_size, size);
  EXPECT_EQ(ascend_vmm_adapter_.vmm_map_[addr], nullptr);
}

/// Feature: test ascend vmm adapter.
/// Description: test eager free memory.
/// Expectation: can eager free memory.
TEST_F(TestAscendVmmAdapter, test_eager_free_1) {
  size_t size = ascend_vmm_adapter_.vmm_align_size_;
  DeviceMemPtr not_null = reinterpret_cast<void *>(1);
  DeviceMemPtr addr = reinterpret_cast<DeviceMemPtr>(size);
  ascend_vmm_adapter_.vmm_map_[addr] = not_null;
  DeviceMemPtr addr2 = reinterpret_cast<DeviceMemPtr>(size * 2);
  ascend_vmm_adapter_.vmm_map_[addr2] = not_null;
  DeviceMemPtr addr3 = reinterpret_cast<DeviceMemPtr>(size * 3);
  ascend_vmm_adapter_.vmm_map_[addr3] = not_null;
  DeviceMemPtr addr4 = reinterpret_cast<DeviceMemPtr>(size * 4);
  ascend_vmm_adapter_.vmm_map_[addr4] = not_null;

  DeviceMemPtr free_addr = reinterpret_cast<DeviceMemPtr>(size + 233);
  size_t ret_size = ascend_vmm_adapter_.EagerFreeDeviceMem(free_addr, size * 3);
  EXPECT_EQ(ret_size, size * 2);
  EXPECT_NE(ascend_vmm_adapter_.vmm_map_[addr], nullptr);
  EXPECT_EQ(ascend_vmm_adapter_.vmm_map_[addr2], nullptr);
  EXPECT_EQ(ascend_vmm_adapter_.vmm_map_[addr3], nullptr);
  EXPECT_NE(ascend_vmm_adapter_.vmm_map_[addr4], nullptr);
}

/// Feature: test ascend vmm adapter.
/// Description: test set server comm for vmm by group name.
/// Expectation: interface call is successful and hccl_comm keeps valid default in ut.
TEST_F(TestAscendVmmAdapter, test_set_server_comm_for_vmm) {
  ascend_vmm_adapter_.hccl_comm_ = reinterpret_cast<HcclComm>(0x1);
  EXPECT_NO_THROW(ascend_vmm_adapter_.SetServerCommForVmm("test_group"));
  EXPECT_EQ(ascend_vmm_adapter_.hccl_comm_, nullptr);
}

/// Feature: test ascend vmm adapter.
/// Description: test zero-copy set memory range call path.
/// Expectation: set-memory-range call path executes without exception.
TEST_F(TestAscendVmmAdapter, test_zero_copy_set_memory_range_call) {
  ascend_vmm_adapter_.hccl_comm_ = reinterpret_cast<HcclComm>(0x1);
  size_t size = ascend_vmm_adapter_.vmm_align_size_;
  DeviceMemPtr addr = reinterpret_cast<DeviceMemPtr>(size);
  size_t ret = 0;
  EXPECT_NO_THROW(ret = ascend_vmm_adapter_.AllocDeviceMem(size, &addr));
  EXPECT_EQ(ret, size);
}

/// Feature: test ascend vmm adapter.
/// Description: test zero-copy activate comm memory call path.
/// Expectation: activate-comm-memory call path executes without exception.
TEST_F(TestAscendVmmAdapter, test_zero_copy_activate_comm_memory_call) {
  ascend_vmm_adapter_.hccl_comm_ = reinterpret_cast<HcclComm>(0x1);
  size_t size = ascend_vmm_adapter_.vmm_align_size_;
  DeviceMemPtr addr = reinterpret_cast<DeviceMemPtr>(size);
  size_t ret = 0;
  EXPECT_NO_THROW(ret = ascend_vmm_adapter_.AllocDeviceMem(size, &addr));
  EXPECT_EQ(ret, size);
  EXPECT_NO_THROW(ret = ascend_vmm_adapter_.MmapDeviceMem(size, addr, SIZE_MAX));
  EXPECT_EQ(ret, size);
}

/// Feature: test ascend vmm adapter.
/// Description: test zero-copy deactivate comm memory call path.
/// Expectation: deactivate-comm-memory call path executes without exception.
TEST_F(TestAscendVmmAdapter, test_zero_copy_deactivate_comm_memory_call) {
  ascend_vmm_adapter_.hccl_comm_ = reinterpret_cast<HcclComm>(0x1);
  size_t size = ascend_vmm_adapter_.vmm_align_size_;
  DeviceMemPtr addr = reinterpret_cast<DeviceMemPtr>(size);
  ascend_vmm_adapter_.vmm_map_[addr] = reinterpret_cast<void *>(1);
  size_t ret_size = 0;
  EXPECT_NO_THROW(ret_size = ascend_vmm_adapter_.EagerFreeDeviceMem(addr, size));
  EXPECT_EQ(ret_size, size);
}

/// Feature: test ascend vmm adapter.
/// Description: test zero-copy unset memory range call path.
/// Expectation: unset-memory-range call path executes without exception.
TEST_F(TestAscendVmmAdapter, test_zero_copy_unset_memory_range_call) {
  ascend_vmm_adapter_.hccl_comm_ = reinterpret_cast<HcclComm>(0x1);
  DeviceMemPtr addr = reinterpret_cast<DeviceMemPtr>(ascend_vmm_adapter_.vmm_align_size_);
  ascend_vmm_adapter_.all_reserve_mems_.push_back(addr);
  EXPECT_NO_THROW(ascend_vmm_adapter_.ClearAllMemory());
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
