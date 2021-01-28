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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_CALLBACK_REGISTER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_CALLBACK_REGISTER_H_

#include "toolchain/prof_callback.h"

#define MAX_DEV_NUM (64)

using Status = uint32_t;
enum ProfCommandHandleType {
  kProfCommandhandleInit = 0,
  kProfCommandhandleStart,
  kProfCommandhandleStop,
  kProfCommandhandleFinalize,
  kProfCommandhandleModelSubscribe,
  kProfCommandhandleModelUnsubscribe
};

struct ProfCommandHandleData {
  uint64_t profSwitch;
  uint32_t devNums;  // length of device id list
  uint32_t devIdList[MAX_DEV_NUM];
  uint32_t modelId;
};

Status RegProfCtrlCallback(MsprofCtrlCallback func);
Status RegProfSetDeviceCallback(MsprofSetDeviceCallback func);
Status RegProfReporterCallback(MsprofReporterCallback func);
Status ProfCommandHandle(ProfCommandHandleType type, void *data, uint32_t len);
bool IsInitialize();

class __attribute__((visibility("default"))) VMCallbackRegister {
 public:
  static VMCallbackRegister &GetInstance();
  VMCallbackRegister(const VMCallbackRegister &) = delete;
  VMCallbackRegister &operator=(const VMCallbackRegister &) = delete;
  bool Register(Status (*pRegProfCtrlCallback)(MsprofCtrlCallback),
                Status (*pRegProfSetDeviceCallback)(MsprofSetDeviceCallback),
                Status (*pRegProfReporterCallback)(MsprofReporterCallback),
                Status (*pProfCommandHandle)(ProfCommandHandleType, void *, uint32_t));
  void ForceMsprofilerInit();
  bool registered() { return registered_; }
  Status DoRegProfCtrlCallback(MsprofCtrlCallback func) { return pRegProfCtrlCallback_(func); }
  Status DoRegProfSetDeviceCallback(MsprofSetDeviceCallback func) { return pRegProfSetDeviceCallback_(func); }
  Status DoRegProfReporterCallback(MsprofReporterCallback func) { return pRegProfReporterCallback_(func); }
  Status DoProfCommandHandle(ProfCommandHandleType type, void *data, uint32_t len) {
    return pProfCommandHandle_(type, data, len);
  }

 private:
  VMCallbackRegister()
      : registered_(false),
        ms_profile_inited_(false),
        pRegProfCtrlCallback_(nullptr),
        pRegProfSetDeviceCallback_(nullptr),
        pRegProfReporterCallback_(nullptr),
        pProfCommandHandle_(nullptr) {}
  ~VMCallbackRegister() = default;

  bool registered_;
  bool ms_profile_inited_;
  Status (*pRegProfCtrlCallback_)(MsprofCtrlCallback);
  Status (*pRegProfSetDeviceCallback_)(MsprofSetDeviceCallback);
  Status (*pRegProfReporterCallback_)(MsprofReporterCallback);
  Status (*pProfCommandHandle_)(ProfCommandHandleType, void *, uint32_t);
};
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_CALLBACK_REGISTER_H_
