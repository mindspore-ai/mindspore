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

#include "runtime/device/ascend/profiling/profiling_callback_register.h"
#include "runtime/base.h"

namespace Analysis {
namespace Dvvp {
namespace ProfilerCommon {
extern int32_t MsprofilerInit();
}  // namespace ProfilerCommon
}  // namespace Dvvp
}  // namespace Analysis

namespace {
constexpr Status PROF_SUCCESS = 0;
constexpr Status PROF_FAILED = 0xFFFFFFFF;
}  // namespace

Status RegProfCtrlCallback(MsprofCtrlCallback func) {
  if (VMCallbackRegister::GetInstance().registered()) {
    return VMCallbackRegister::GetInstance().DoRegProfCtrlCallback(func);
  } else {
    return PROF_SUCCESS;
  }
}

Status RegProfSetDeviceCallback(MsprofSetDeviceCallback func) {
  if (VMCallbackRegister::GetInstance().registered()) {
    return VMCallbackRegister::GetInstance().DoRegProfSetDeviceCallback(func);
  } else {
    return PROF_SUCCESS;
  }
}

Status RegProfReporterCallback(MsprofReporterCallback func) {
  if (VMCallbackRegister::GetInstance().registered()) {
    return VMCallbackRegister::GetInstance().DoRegProfReporterCallback(func);
  } else {
    return PROF_SUCCESS;
  }
}

Status ProfCommandHandle(ProfCommandHandleType type, void *data, uint32_t len) {
  if (VMCallbackRegister::GetInstance().registered()) {
    return VMCallbackRegister::GetInstance().DoProfCommandHandle(type, data, len);
  } else {
    return PROF_SUCCESS;
  }
}

bool IsInitialize() { return true; }

VMCallbackRegister &VMCallbackRegister::GetInstance() {
  static VMCallbackRegister instance;
  return instance;
}

bool VMCallbackRegister::Register(Status (*pRegProfCtrlCallback)(MsprofCtrlCallback),
                                  Status (*pRegProfSetDeviceCallback)(MsprofSetDeviceCallback),
                                  Status (*pRegProfReporterCallback)(MsprofReporterCallback),
                                  Status (*pProfCommandHandle)(ProfCommandHandleType, void *, uint32_t)) {
  if (!registered_) {
    pRegProfCtrlCallback_ = pRegProfCtrlCallback;
    pRegProfSetDeviceCallback_ = pRegProfSetDeviceCallback;
    pRegProfReporterCallback_ = pRegProfReporterCallback;
    pProfCommandHandle_ = pProfCommandHandle;
    registered_ = true;
    ForceMsprofilerInit();
    return true;
  }
  return false;
}

void VMCallbackRegister::ForceMsprofilerInit() {
  if (!ms_profile_inited_) {
    Analysis::Dvvp::ProfilerCommon::MsprofilerInit();
    ms_profile_inited_ = true;
  }
}
