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
#include "common/common_test.h"
#include "c_api/include/context.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

namespace ST {
static std::string GetEnv(const std::string &envvar) {
  const char *value = std::getenv(envvar.c_str());
  if (value == nullptr) {
    return "";
  }

  return std::string(value);
}

void Common::SetUpTestCase() {}

void Common::TearDownTestCase() {}

void Common::SetUp() {
  char str_buf[10];
  STATUS ret = MSGetBackendPolicy(str_buf, 10);
  if (ret == RET_OK) {
    org_policy_ = str_buf;
  }
}

void Common::TearDown() { (void)MSSetBackendPolicy(org_policy_.c_str()); }

void Common::ContextAutoSet() {
  auto device_target_str = GetEnv("DEVICE_TARGET");
  if (device_target_str.empty()) {
    device_target_str = "CPU";  // default is CPU
  }

  auto device_id_str = GetEnv("DEVICE_ID");
  if (device_id_str.empty()) {
    device_id_str = "0";  // default is 0, only valid when device target is Ascend
  }
  uint32_t device_id = std::strtoul(device_id_str.c_str(), nullptr, 10);

  (void)MSSetDeviceTarget(device_target_str.c_str());
  (void)MSSetDeviceId(device_id);
}
}  // namespace ST

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
