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
#include "include/utils/env_vars.h"
#include <string>
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace common {
namespace {
// convert string to int value, return true when success, otherwise false
bool StringToInt(const std::string &str, int *value_ptr) {
  bool success = true;
  int value = 0;
  try {
    value = std::stoi(str);
  } catch (...) {
    success = false;
  }
  if (success && value_ptr != nullptr) {
    *value_ptr = value;
  }
  return success;
}

// get an integer value according to environment variable name
int GetEnvIntValue(const std::string &env_var, const int default_value, const int min_value, const int max_value) {
  auto env_val = common::GetEnv(env_var);
  if (env_val.empty()) {
    return default_value;
  }

  int value = default_value;
  bool success = StringToInt(env_val, &value);
  if (!success || value < min_value || value > max_value) {
    MS_LOG(WARNING) << "Ignore environment variable " << env_var << "'s invalid value " << env_val
                    << ", valid value should be in range [" << min_value << ", " << max_value << "].";
    return default_value;
  }
  return value;
}
}  // namespace

int GetDumpSliceSize() {
  // get print, tensordump slice size in MB
  constexpr char kVarDumpSliceSize[] = "MS_DUMP_SLICE_SIZE";
  constexpr int kMaxDumpSliceSize = 2048;
  static int print_slice_size = GetEnvIntValue(kVarDumpSliceSize, 0, 0, kMaxDumpSliceSize);
  return print_slice_size;
}

int GetDumpWaitTime() {
  // get print, tensordump wait time in seconds
  constexpr char kVarDumpWaitTime[] = "MS_DUMP_WAIT_TIME";
  constexpr int kMaxDumpWaitTime = 600;
  static int print_wait_time = GetEnvIntValue(kVarDumpWaitTime, 0, 0, kMaxDumpWaitTime);
  return print_wait_time;
}
}  // namespace common
}  // namespace mindspore
