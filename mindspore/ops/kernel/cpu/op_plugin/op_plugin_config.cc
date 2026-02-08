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

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include <mutex>
#include "ops_utils/dl_utils.h"
#include "ops_utils/op_utils.h"
#include "ops_utils/callback.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"
#include "mindspore/core/include/utils/ms_utils.h"

namespace mindspore {
namespace ops {
namespace {
#define OP_PLUGIN_CONFIG_VERSION 1U
constexpr char kOpPluginConfSetupFunc[] = "OpPluginConfSetup";
constexpr char kConfOpThreadsNum[] = "op_threads_num";

typedef struct OpPluginConfigEntry {
  const char *key;
  const char *value;
} OpPluginConfigEntry;

typedef struct OpPluginConfig {
  // Set to sizeof(OpPluginConfig) for forward-compatible size checks.
  uint32_t size;
  uint32_t version;
  uint64_t entry_count;
  const OpPluginConfigEntry *entries;
  uint64_t reserved;
} OpPluginConfig;

using OpPluginConfigFn = int (*)(const OpPluginConfig *);

struct OpPluginConfigPayload {
  OpPluginConfig config{};
  std::vector<std::string> keys;
  std::vector<std::string> values;
  std::vector<OpPluginConfigEntry> entries;
};

std::shared_ptr<OpPluginConfigPayload> BuildPluginConfigPayload() {
  auto payload = std::make_shared<OpPluginConfigPayload>();
  payload->config.size = sizeof(OpPluginConfig);
  payload->config.version = OP_PLUGIN_CONFIG_VERSION;
  payload->config.entry_count = 0;
  payload->config.entries = nullptr;
  payload->config.reserved = 0;

  // config for op threads num
  auto runtime_conf = runtime::RuntimeConf::GetInstance();
  if (runtime_conf != nullptr) {
    if (runtime_conf->IsOpThreadsNumConfigured()) {
      payload->keys.emplace_back(kConfOpThreadsNum);
      payload->values.emplace_back(std::to_string(runtime_conf->op_threads_num()));
      payload->entries.push_back({payload->keys.back().c_str(), payload->values.back().c_str()});
    }
  }

  if (!payload->entries.empty()) {
    payload->config.entry_count = payload->entries.size();
    payload->config.entries = payload->entries.data();
  }

  return payload;
}

void SetupOpPluginConfig(void *handle) {
  if (handle == nullptr) {
    return;
  }

  (void)DL_ERROR();
#ifdef _WIN32
  SetLastError(0);
#endif
  auto config_func = reinterpret_cast<OpPluginConfigFn>(DL_SYM(handle, kOpPluginConfSetupFunc));
  if (config_func == nullptr) {
    MS_LOG(INFO) << "Op plugin does not expose " << kOpPluginConfSetupFunc << "; skip config forwarding.";
    return;
  }

  auto config = BuildPluginConfigPayload();
  if (config == nullptr) {
    MS_LOG(INFO) << "Op plugin config not set; calling " << kOpPluginConfSetupFunc << " is skipped.";
    return;
  }

  int ret = config_func(&config->config);
  if (ret != 0) {
    MS_LOG(WARNING) << kOpPluginConfSetupFunc << " returned non-zero: " << ret;
  } else {
    MS_LOG(INFO) << "Op plugin config applied successfully.";
  }
}
REGISTER_OPS_CALLBACK(SetupOpPluginConfig);
}  // namespace
}  // namespace ops
}  // namespace mindspore
