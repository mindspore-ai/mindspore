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

#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "utils/ms_context.h"
#include "include/runtime/pipeline/pipeline.h"
#include "utils/device_manager_conf.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
OpStatus::OpStatus() { device_target = DeviceManagerConf::GetInstance()->device_type(); }

OpRunStatus &OpRunStatus::Get() {
  static OpRunStatus instance;
  return instance;
}

OpRunStatus::OpRunStatus() { cur_device_ = DeviceManagerConf::GetInstance()->device_type(); }

void OpRunStatus::HeterBarrier(device::DeviceType device) {
  if (cur_device_ != device) {
    MS_LOG(DEBUG) << "Current device " << device::GetDeviceNameByType(cur_device_) << " incoming device "
                  << device::GetDeviceNameByType(device);
    cur_device_ = device;
    runtime::Pipeline::Get().WaitAll();
  }
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
