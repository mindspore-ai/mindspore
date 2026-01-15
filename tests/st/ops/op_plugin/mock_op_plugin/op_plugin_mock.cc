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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

const std::vector<const char *> kRegisteredOpNames = {
  "AddScalar", "CumsumExt", "InplaceNormal", "InplaceReLU", "LogicalAnd", "Randn", "StackExt", "SumExt", "Unique"};

const std::vector<const char *> kComputeDependOpNames = {"Unique"};

namespace {
inline bool StrEqual(const char *a, const char *b) { return std::strcmp(a, b) == 0; }
}  // namespace

// Configuration state tracking for tests
namespace {
// Match the OpPluginConfig structure from op_plugin_utils.cc
#define OP_PLUGIN_CONFIG_VERSION 1U

typedef struct OpPluginConfigEntry {
  const char *key;
  const char *value;
} OpPluginConfigEntry;

typedef struct OpPluginConfig {
  uint32_t size;
  uint32_t version;
  uint64_t entry_count;
  const OpPluginConfigEntry *entries;
  uint64_t reserved;
} OpPluginConfig;

// Static variables to track received configuration
static uint32_t g_received_op_threads_num = 0;
static bool g_op_threads_num_received = false;
static int g_config_call_count = 0;
static bool g_config_invalid_version = false;
static bool g_config_invalid_size = false;
}  // namespace

extern "C" {

bool IsKernelRegistered(const char *name) {
  if (name == nullptr) {
    return false;
  }
  return std::find_if(kRegisteredOpNames.begin(), kRegisteredOpNames.end(),
                      [name](const char *s) { return StrEqual(s, name); }) != kRegisteredOpNames.end();
}

int GetRegisteredOpCount() { return static_cast<int>(kRegisteredOpNames.size()); }

const char **GetAllRegisteredOps() { return const_cast<const char **>(kRegisteredOpNames.data()); }

int GetComputeDependOpCount() { return static_cast<int>(kComputeDependOpNames.size()); }

const char **GetAllOpPluginComputeDependOps() { return const_cast<const char **>(kComputeDependOpNames.data()); }

// Configuration setup function called by MindSpore
int OpPluginConfSetup(const OpPluginConfig *config) {
  g_config_call_count++;
  g_config_invalid_version = false;
  g_config_invalid_size = false;

  if (config == nullptr) {
    return 0;  // Null config is acceptable (no config to apply)
  }

  // Validate config structure
  if (config->size < sizeof(OpPluginConfig)) {
    g_config_invalid_size = true;
    return -1;  // Invalid size
  }

  if (config->version != OP_PLUGIN_CONFIG_VERSION) {
    g_config_invalid_version = true;
    return -1;  // Invalid version
  }

  // Iterate through entries to find op_threads_num
  for (uint64_t i = 0; i < config->entry_count; i++) {
    if (config->entries[i].key != nullptr && config->entries[i].value != nullptr) {
      if (std::strcmp(config->entries[i].key, "op_threads_num") == 0) {
        g_received_op_threads_num = static_cast<uint32_t>(std::atoi(config->entries[i].value));
        g_op_threads_num_received = true;
        break;
      }
    }
  }

  return 0;  // Success
}

// Query functions for test verification
uint32_t GetReceivedOpThreadsNum() { return g_received_op_threads_num; }

bool IsOpThreadsNumReceived() { return g_op_threads_num_received; }

int GetConfigCallCount() { return g_config_call_count; }

bool GetConfigInvalidVersion() { return g_config_invalid_version; }

bool GetConfigInvalidSize() { return g_config_invalid_size; }

void ResetConfigState() {
  g_received_op_threads_num = 0;
  g_op_threads_num_received = false;
  g_config_call_count = 0;
  g_config_invalid_version = false;
  g_config_invalid_size = false;
}

}  // extern "C"
