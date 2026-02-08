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

#include "backend/ge_backend/dump/deprecated_env.h"
#include <chrono>
#include <string>
#include <thread>
#include <mutex>
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dump {
namespace {
std::once_flag alarm_flag;
}

void CheckDeprecatedDumpEnv() {
  if (!AnfAlgo::IsBackendGe()) {
    return;
  }
  const bool is_legacy_dump_set = !common::GetEnv("MINDSPORE_DUMP_CONFIG").empty();
  const bool is_ge_dump_enabled = common::GetEnv("ENABLE_MS_GE_DUMP") == "1";
  const bool using_deprecated_config = is_legacy_dump_set || is_ge_dump_enabled;

  if (!using_deprecated_config) {
    return;
  }
  std::call_once(alarm_flag, PrintDeprecatedWarning, is_legacy_dump_set, is_ge_dump_enabled);
}

void PrintDeprecatedWarning(bool is_legacy_dump_set, bool is_ge_dump_enabled) {
  std::string warning_msg;
  if (is_legacy_dump_set) {
    warning_msg +=
      "For 'Dump', in the scenario where 'backend' is 'GE', the 'MINDSPORE_DUMP_CONFIG' env has been deprecated "
      "since MindSpore 2.6. ";
  }
  if (is_ge_dump_enabled) {
    warning_msg += "For 'Dump', the 'ENABLE_MS_GE_DUMP' env has been deprecated since MindSpore 2.6. ";
  }

  constexpr int kMaxWarnings = 3;
  constexpr auto kWarningInterval = std::chrono::milliseconds(1000);

  for (int i = 1; i <= kMaxWarnings; ++i) {
    MS_LOG(WARNING)
      << "[Dump Deprecated Alert " << i << "/" << kMaxWarnings << "]: " << warning_msg
      << "Please use msprobe tool instead. "
      << "For more details, please refer to https://gitee.com/ascend/mstt/tree/master/debug/accuracy_tools/msprobe.";
    std::this_thread::sleep_for(kWarningInterval);
  }
}
}  // namespace dump
}  // namespace mindspore
