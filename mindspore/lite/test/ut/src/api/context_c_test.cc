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
#include "include/c_api/context_c.h"
#include "common/common_test.h"

namespace mindspore {
class ContextCTest : public mindspore::CommonTest {
 public:
  ContextCTest() {}
};

TEST_F(ContextCTest, common_test) {
  MSDeviceInfoHandle npu_device_info = MSDeviceInfoCreate(kMSDeviceTypeKirinNPU);
  ASSERT_TRUE(npu_device_info != nullptr);
  ASSERT_EQ(MSDeviceInfoGetDeviceType(npu_device_info), kMSDeviceTypeKirinNPU);

  MSDeviceInfoSetProvider(npu_device_info, "vendor name");
  ASSERT_STREQ(MSDeviceInfoGetProvider(npu_device_info), "vendor name");

  MSDeviceInfoSetProviderDevice(npu_device_info, "npu_a");
  ASSERT_STREQ(MSDeviceInfoGetProviderDevice(npu_device_info), "npu_a");

  MSDeviceInfoSetFrequency(npu_device_info, 3);
  ASSERT_EQ(MSDeviceInfoGetFrequency(npu_device_info), 3);

  MSContextHandle context = MSContextCreate();
  ASSERT_TRUE(context != nullptr);

  MSContextSetThreadNum(context, 4);
  ASSERT_EQ(MSContextGetThreadNum(context), 4);

  MSContextSetThreadAffinityMode(context, 2);
  ASSERT_EQ(MSContextGetThreadAffinityMode(context), 2);

  constexpr size_t core_num = 4;
  int32_t core_list[core_num] = {1, 3, 2, 0};
  MSContextSetThreadAffinityCoreList(context, core_list, core_num);
  size_t ret_core_num;
  const int32_t *ret_core_list = nullptr;
  ret_core_list = MSContextGetThreadAffinityCoreList(context, &ret_core_num);
  ASSERT_EQ(ret_core_num, core_num);
  for (size_t i = 0; i < ret_core_num; i++) {
    ASSERT_EQ(ret_core_list[i], core_list[i]);
  }

  MSContextSetEnableParallel(context, true);
  ASSERT_EQ(MSContextGetEnableParallel(context), true);

  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  MSDeviceInfoDestroy(&cpu_device_info);
  cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);

  MSDeviceInfoSetEnableFP16(cpu_device_info, true);
  ASSERT_EQ(MSDeviceInfoGetEnableFP16(cpu_device_info), true);

  MSContextAddDeviceInfo(context, cpu_device_info);
  MSContextAddDeviceInfo(context, npu_device_info);
  MSContextDestroy(&context);
}
}  // namespace mindspore
