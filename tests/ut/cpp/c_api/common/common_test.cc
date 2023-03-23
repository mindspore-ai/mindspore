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
#include "common_test.h"
#include "c_api/include/context.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

namespace UT {
void CApiCommon::SetUpTestCase() {}

void CApiCommon::TearDownTestCase() {}

void CApiCommon::SetUp() {
  char str_buf[10];
  STATUS ret = MSGetBackendPolicy(str_buf, 10);
  if (ret == RET_OK) {
    org_policy_ = str_buf;
  }
}

void CApiCommon::TearDown() { (void)MSSetBackendPolicy(org_policy_.c_str()); }
}  // namespace UT

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
