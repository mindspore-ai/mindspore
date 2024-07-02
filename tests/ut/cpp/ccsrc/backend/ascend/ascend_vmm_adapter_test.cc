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
#include "transform/symbol/acl_rt_symbol.h"
#define private public
#define protected public
#include "plugin/device/ascend/hal/device/ascend_vmm_adapter.h"
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
    mindspore::transform::aclrtUnmapMem_ = aclrtUnmapMem;
    mindspore::transform::aclrtReserveMemAddress_ = aclrtReserveMemAddress;
    mindspore::transform::aclrtMallocPhysical_ = aclrtMallocPhysical;
    mindspore::transform::aclrtMapMem_ = aclrtMapMem;
    mindspore::transform::aclrtFreePhysical_ = aclrtFreePhysical;
    mindspore::transform::aclrtReleaseMemAddress_ = aclrtReleaseMemAddress;
    common::ResetConfig("MS_ALLOC_CONF");
  }
  void TearDown() override {}

  AscendVmmAdapter ascend_vmm_adapter_;
};

/// Feature: test ascend vmm adapter.
/// Description: test basic allocation.
/// Expectation: can alloc memory and can not throw exception.
TEST_F(TestAscendVmmAdapter, test_basic_allocation) {
  size_t block = ascend_vmm_adapter_.kVmmAlignSize;
  DeviceMemPtr addr = reinterpret_cast<DeviceMemPtr>(block);;
  size_t size = 1024;
  size_t ret = ascend_vmm_adapter_.AllocDeviceMem(size, &addr);
  EXPECT_EQ(ret, ascend_vmm_adapter_.kVmmAlignSize);
}

/// Feature: test ascend vmm adapter.
/// Description: test set align size.
/// Expectation: can set align size.
TEST_F(TestAscendVmmAdapter, test_set_align_size) {
  common::SetEnv("MS_ALLOC_CONF", "vmm_align_size:20MB");
  size_t mb = 1024 * 1024;
  AscendVmmAdapter test_vmm_adapter;
  EXPECT_EQ(test_vmm_adapter.kVmmAlignSize, 20 * mb);
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
  size_t block = ascend_vmm_adapter_.kVmmAlignSize;
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
  size_t block = test_vmm_adapter.kVmmAlignSize;
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
  size_t block = ascend_vmm_adapter_.kVmmAlignSize;
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
  size_t block = test_vmm_adapter.kVmmAlignSize;
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
  size_t block = ascend_vmm_adapter_.kVmmAlignSize;
  size_t size = block * 99;
  size_t ret = ascend_vmm_adapter_.GetHandleSize(size);
  EXPECT_EQ(ret, 99);
}

/// Feature: test ascend vmm adapter.
/// Description: test handle size throw.
/// Expectation: can throw exception.
TEST_F(TestAscendVmmAdapter, test_handle_size_throw) {
  size_t block = ascend_vmm_adapter_.kVmmAlignSize;
  size_t size = block * 99 + 233;
  EXPECT_ANY_THROW(ascend_vmm_adapter_.GetHandleSize(size));
}

/// Feature: test ascend vmm adapter.
/// Description: test find vmm segment.
/// Expectation: can find vmm segment.
TEST_F(TestAscendVmmAdapter, test_find_vmm_segment) {
  size_t size = ascend_vmm_adapter_.kVmmAlignSize;
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
  size_t size = ascend_vmm_adapter_.kVmmAlignSize;
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
  size_t size = ascend_vmm_adapter_.kVmmAlignSize;
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
  size_t size = ascend_vmm_adapter_.kVmmAlignSize;
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
  size_t size = ascend_vmm_adapter_.kVmmAlignSize;
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
}  // namespace ascend
}  // namespace device
}  // namespace mindspore