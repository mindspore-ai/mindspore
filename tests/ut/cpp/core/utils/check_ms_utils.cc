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
#include "utils/ms_utils.h"

namespace mindspore {
class TestCheckMsUtils : public UT::Common {
 public:
  TestCheckMsUtils() = default;
  void SetUp() {}
};

// Feature: TestCheckMsUtils.
// Description: Check function of IsEnableRuntimeConfig and IsDisableRuntimeConfig in ms_utils.cc
// Expectation: Get right runtime config.
TEST_F(TestCheckMsUtils, test_read_runtime_config) {
  int ret = common::SetEnv("MS_DEV_RUNTIME_CONF", "inline:true,compile_statistic:True,memoty_statistic:false");
  ASSERT_EQ(ret, 0);

  ASSERT_TRUE(common::IsEnableRuntimeConfig("inline"));
  ASSERT_TRUE(common::IsEnableRuntimeConfig("compile_statistic"));
  ASSERT_FALSE(common::IsEnableRuntimeConfig("memoty_statistic"));
  ASSERT_TRUE(common::IsDisableRuntimeConfig("memoty_statistic"));
  ASSERT_FALSE(common::IsEnableRuntimeConfig("switch_inline"));
  ASSERT_FALSE(common::IsDisableRuntimeConfig("switch_inline"));
  
  (void)common::SetEnv("MS_DEV_RUNTIME_CONF", "");
}
}  // namespace mindspore
