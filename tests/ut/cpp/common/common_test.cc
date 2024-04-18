/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "utils/log_adapter.h"
#include "resource.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

namespace UT {

void Common::SetUpTestCase() {}

void Common::TearDownTestCase() {}

void Common::SetUp() {}

void Common::TearDown() {
  const char *suite_name = testing::UnitTest::GetInstance()->current_test_suite()->name();
  const char *test_name = testing::UnitTest::GetInstance()->current_test_info()->name();
  UT::UTResourceManager::GetInstance()->DropFuncGraph(UTKeyInfo{suite_name, test_name});
}

}  // namespace UT

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
