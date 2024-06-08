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
#define private public
#define protected public
#include "utils/ms_context.h"
#undef private
#undef protected

namespace mindspore {
class TestCheckMsContext : public UT::Common {
 public:
  TestCheckMsContext() = default;
  void SetUp() {}
};

// Feature: TestCheckMsContext.
// Description: Check function of IsKByKExecutorMode in ms_context.cc
// Expectation: Get right exec mode.
TEST_F(TestCheckMsContext, test_kbk_mode) {
  MsContext ms_context_("", "");
  ms_context_.set_param<std::string>(MS_CTX_DEVICE_TARGET, "Ascend");
  ms_context_.set_ascend_soc_version("ascend910");
  ms_context_.set_param<int>(MS_CTX_EXECUTION_MODE, 0);
  ASSERT_FALSE(ms_context_.IsKByKExecutorMode());

  ms_context_.set_param<std::string>(MS_CTX_DEVICE_TARGET, "Ascend");
  ms_context_.set_param<int>(MS_CTX_EXECUTION_MODE, 1);
  ASSERT_TRUE(ms_context_.IsKByKExecutorMode());

  ms_context_.set_param<std::string>(MS_CTX_DEVICE_TARGET, "GPU");
  ASSERT_TRUE(ms_context_.IsKByKExecutorMode());

  ms_context_.set_param<std::string>(MS_CTX_DEVICE_TARGET, "Ascend");
  ms_context_.set_param<std::string>(MS_CTX_JIT_LEVEL, "O0");
  ASSERT_TRUE(ms_context_.IsKByKExecutorMode());
}
}  // namespace mindspore
