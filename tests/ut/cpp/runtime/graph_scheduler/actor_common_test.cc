/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "tests/ut/cpp/common/device_common_test.h"

#include <memory>

#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "backend/ms_backend/runtime/actors/base/actor_common.h"

namespace mindspore {
namespace runtime {
using namespace test;
constexpr char kAscendDeviceName[] = "Ascend";
constexpr char kCpuDeviceName[] = "CPU";
constexpr size_t kTensorLen = 1024;
class ActorCommonTest : public UT::Common {
 public:
  ActorCommonTest() = default;

  void SetUp() override {
    MS_REGISTER_DEVICE(kAscendDeviceName, TestDeviceContext);
    auto ms_context = MsContext::GetInstance();
    ms_context->set_param<uint32_t>(MS_CTX_DEVICE_ID, 0);
    DeviceContextKey device_context_key{device::DeviceType::kAscend, device_id_};
    device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(device_context_key);
  }

  TestResManager *GetRawDeviceResManager() {
    return static_cast<TestResManager *>(device_context_->device_res_manager_.get());
  }

  std::pair<DeviceTensorPtr, DeviceTensorPtr> GenerateDeviceAddress(const std::string &dst_device_name,
                                                                    const std::string &src_device_name,
                                                                    uint32_t dst_stream_id, uint32_t src_stream_id) {
    auto dst_tensor =
      std::make_shared<device::DeviceAddress>(dst_arr_, kTensorLen, device::GetDeviceTypeByName(dst_device_name), 0);
    dst_tensor->set_stream_id(dst_stream_id);
    auto src_tensor =
      std::make_shared<device::DeviceAddress>(src_arr_, kTensorLen, device::GetDeviceTypeByName(src_device_name), 0);
    src_tensor->set_stream_id(src_stream_id);
    return std::make_pair(dst_tensor, src_tensor);
  }

 private:
  uint32_t device_id_ = 0;
  device::DeviceContext *device_context_;

  std::vector<int64_t> shape_{32, 32};
  char dst_arr_[kTensorLen]{0};
  char src_arr_[kTensorLen]{1};
};

/// Feature: sync stream on demand.
/// Description: Test switch.
/// Expectation: As expected.
TEST_F(ActorCommonTest, SwitchTest) {
  TestResManager *test_device_res_manager = GetRawDeviceResManager();
  uint32_t stream_id = 0;
  // dst is ascend so that we can reuse test_device_res_manager.
  auto [dst_tensor, src_tensor] = GenerateDeviceAddress(kAscendDeviceName, kCpuDeviceName, stream_id, stream_id);
  size_t sync_all_stream_before_count = test_device_res_manager->sync_all_stream_count_;
  size_t sync_stream_before_count = test_device_res_manager->sync_stream_counts_[stream_id];
  auto ret = SyncAllStreamForDeviceAddress(dst_tensor, src_tensor, stream_id, false);
  size_t sync_all_stream_after_count = test_device_res_manager->sync_all_stream_count_;
  size_t sync_stream_after_count = test_device_res_manager->sync_stream_counts_[stream_id];
  ASSERT_EQ(sync_all_stream_before_count + 1, sync_all_stream_after_count);
  ASSERT_EQ(sync_stream_before_count, sync_stream_after_count);
  ASSERT_TRUE(ret);
}

/// Feature: sync stream on demand.
/// Description: Test sync stream.
/// Expectation: As expected.
TEST_F(ActorCommonTest, SyncStreamTest) {
  TestResManager *test_device_res_manager = GetRawDeviceResManager();
  uint32_t stream_id = 0;
  // test for cpu vs cpu
  {
    auto [cpu_dst_addr, cpu_src_addr] = GenerateDeviceAddress(kCpuDeviceName, kCpuDeviceName, stream_id, stream_id);
    size_t sync_all_stream_before_count = test_device_res_manager->sync_all_stream_count_;
    size_t sync_stream_before_count = test_device_res_manager->sync_stream_counts_[stream_id];
    auto ret = SyncStreamOnDemandForDeviceAddress(cpu_dst_addr, cpu_src_addr, stream_id);
    size_t sync_all_stream_after_count = test_device_res_manager->sync_all_stream_count_;
    size_t sync_stream_after_count = test_device_res_manager->sync_stream_counts_[stream_id];
    ASSERT_EQ(sync_all_stream_before_count, sync_all_stream_after_count);
    ASSERT_EQ(sync_stream_before_count, sync_stream_after_count);
    ASSERT_TRUE(ret);
  }

  // test for src is ascend, src is ascend so that we can reuse test_device_res_manager
  {
    uint32_t src_stream_id = 1;
    auto [dst_tensor, src_tensor] = GenerateDeviceAddress(kCpuDeviceName, kAscendDeviceName, stream_id, src_stream_id);
    size_t sync_all_stream_before_count = test_device_res_manager->sync_all_stream_count_;
    size_t sync_stream_before_count = test_device_res_manager->sync_stream_counts_[src_stream_id];
    auto ret = SyncStreamOnDemandForDeviceAddress(dst_tensor, src_tensor, stream_id);
    size_t sync_all_stream_after_count = test_device_res_manager->sync_all_stream_count_;
    size_t sync_stream_after_count = test_device_res_manager->sync_stream_counts_[src_stream_id];
    ASSERT_EQ(sync_all_stream_before_count, sync_all_stream_after_count);
    ASSERT_EQ(sync_stream_before_count + 1, sync_stream_after_count);
    ASSERT_TRUE(ret);
  }

  // test for src is cpu and dst is ascend
  {
    uint32_t dst_stream_id = 1;
    auto [dst_tensor, src_tensor] = GenerateDeviceAddress(kAscendDeviceName, kCpuDeviceName, dst_stream_id, stream_id);
    size_t sync_all_stream_before_count = test_device_res_manager->sync_all_stream_count_;
    size_t sync_stream_before_count = test_device_res_manager->sync_stream_counts_[dst_stream_id];
    auto ret = SyncStreamOnDemandForDeviceAddress(dst_tensor, src_tensor, stream_id);
    size_t sync_all_stream_after_count = test_device_res_manager->sync_all_stream_count_;
    size_t sync_stream_after_count = test_device_res_manager->sync_stream_counts_[dst_stream_id];
    ASSERT_EQ(sync_all_stream_before_count, sync_all_stream_after_count);
    ASSERT_EQ(sync_stream_before_count + 1, sync_stream_after_count);
    ASSERT_TRUE(ret);
  }
}
}  // namespace runtime
}  // namespace mindspore
