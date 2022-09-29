/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/nnapi/nnapi_implementation.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <sys/system_properties.h>
#include "src/common/log_adapter.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
namespace {
int32_t GetAndroidSdkVersion() {
  const char *sdk_prop = "ro.build.version.sdk";
  char prop_value[PROP_VALUE_MAX];
  (void)__system_property_get(sdk_prop, prop_value);
  std::function<bool(const std::string)> is_valid_num = [](const std::string &str) {
    return !str.empty() && std::all_of(str.begin(), str.end(), ::isdigit);
  };
  int sdk_version = 0;
  if (is_valid_num(std::string(prop_value))) {
    sdk_version = std::stoi(std::string(prop_value));
  }
  return sdk_version;
}

void *LoadNNAPIFunction(void *handle, const char *name, bool optional) {
  if (handle == nullptr) {
    return nullptr;
  }
  void *fn = dlsym(handle, name);
  if (fn == nullptr && !optional) {
    MS_LOG(ERROR) << "Load NNAPI function failed: " << name;
  }
  return fn;
}

#define LOAD_NNAPI_FUNCTION(handle, name) \
  nnapi.name = reinterpret_cast<name##_fn>(LoadNNAPIFunction(handle, #name, false));

#define LOAD_NNAPI_FUNCTION_OPTIONAL(handle, name) \
  nnapi.name = reinterpret_cast<name##_fn>(LoadNNAPIFunction(handle, #name, true));

const NNAPI LoadNNAPI() {
  NNAPI nnapi = {};
  nnapi.android_sdk_version = 0;

  void *libneuralnetworks = nullptr;
  // instances of nn api RT
  auto get_dl_error = []() -> std::string {
    auto error = dlerror();
    return error == nullptr ? "" : error;
  };
  static const char nnapi_library_name[] = "libneuralnetworks.so";
  libneuralnetworks = dlopen(nnapi_library_name, RTLD_LAZY | RTLD_LOCAL);
  if (libneuralnetworks == nullptr) {
    auto error = get_dl_error();
    MS_LOG(ERROR) << "dlopen " << nnapi_library_name << " failed, error: " << error;
    nnapi.nnapi_exists = false;
    return nnapi;
  }
  nnapi.nnapi_exists = true;
  nnapi.android_sdk_version = GetAndroidSdkVersion();
  if (nnapi.android_sdk_version < ANEURALNETWORKS_FEATURE_LEVEL_1) {
    MS_LOG(ERROR) << "Android NNAPI requires android sdk version to be at least 27";
    nnapi.nnapi_exists = false;
    return nnapi;
  }

  // NNAPI level 1 (API 27) methods.
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksMemory_createFromFd);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksMemory_free);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksModel_create);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksModel_free);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksModel_finish);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksModel_addOperand);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksModel_setOperandValue);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksModel_setOperandValueFromMemory);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksModel_addOperation);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksModel_identifyInputsAndOutputs);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksCompilation_create);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksCompilation_free);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksCompilation_setPreference);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksCompilation_finish);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_create);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_free);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_setInput);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_setInputFromMemory);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_setOutput);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_setOutputFromMemory);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_startCompute);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksEvent_wait);
  LOAD_NNAPI_FUNCTION(libneuralnetworks, ANeuralNetworksEvent_free);

  // NNAPI level 2 (API 28) methods.
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksModel_relaxComputationFloat32toFloat16);

  // NNAPI level 3 (API 29) methods.
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworks_getDeviceCount);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworks_getDevice);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksDevice_getName);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksDevice_getType);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksDevice_getVersion);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksDevice_getFeatureLevel);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksModel_getSupportedOperationsForDevices);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksCompilation_createForDevices);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksCompilation_setCaching);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksExecution_compute);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksExecution_getOutputOperandRank);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksExecution_getOutputOperandDimensions);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksBurst_create);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksBurst_free);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksExecution_burstCompute);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemory_createFromAHardwareBuffer);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksExecution_setMeasureTiming);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksExecution_getDuration);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksModel_setOperandSymmPerChannelQuantParams);

  // NNAPI level 4 (API 30) methods.
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemoryDesc_create);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemoryDesc_free);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemoryDesc_addInputRole);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemoryDesc_addOutputRole);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemoryDesc_setDimensions);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemoryDesc_finish);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemory_createFromDesc);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemory_copy);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksDevice_wait);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksModel_setOperandValueFromModel);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksCompilation_setPriority);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksCompilation_setTimeout);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksExecution_setTimeout);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksExecution_setLoopTimeout);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworks_getDefaultLoopTimeout);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworks_getMaximumLoopTimeout);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksEvent_createFromSyncFenceFd);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksEvent_getSyncFenceFd);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksExecution_startComputeWithDependencies);

  // NNAPI level 5 (API 31) methods.
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworks_getRuntimeFeatureLevel);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksExecution_enableInputAndOutputPadding);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksCompilation_getPreferredMemoryAlignmentForInput);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksCompilation_getPreferredMemoryPaddingForInput);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksCompilation_getPreferredMemoryPaddingForOutput);
  LOAD_NNAPI_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksExecution_setReusable);

  return nnapi;
}
}  // namespace

const NNAPI *NNAPIImplementation() {
  static const NNAPI nnapi = LoadNNAPI();
  return &nnapi;
}
}  // namespace lite
}  // namespace mindspore
