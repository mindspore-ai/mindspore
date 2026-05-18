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

#include "include/pynative/utils/runtime/auto_grad_guard.h"
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

OpRunStatus::ThreadState &OpRunStatus::GetThreadState() {
  static thread_local ThreadState state;
  return state;
}

const OpStatus &OpRunStatus::op_status() { return GetThreadState().status; }

void OpRunStatus::set_run_info(OpStatus &&run_info) { GetThreadState().status = std::move(run_info); }

bool OpRunStatus::RequireGrad() const { return GetThreadState().require_grad; }

void OpRunStatus::SetRequireGrad(bool require_grad) { GetThreadState().require_grad = require_grad; }

device::DeviceType OpRunStatus::device_target() const { return GetThreadState().status.device_target; }

void OpRunStatus::ResetRequireGrad(bool require_grad) { GetThreadState().require_grad = require_grad; }

bool OpRunStatus::IsSafeView() const { return GetThreadState().is_safe_view; }

void OpRunStatus::SetIsSafeView(bool is_safe) { GetThreadState().is_safe_view = is_safe; }

void OpRunStatus::ResetIsSaveView(bool is_safe) { GetThreadState().is_safe_view = is_safe; }

void OpRunStatus::HeterBarrier(device::DeviceType device) {
  bool need_wait = false;
  {
    std::lock_guard<std::mutex> lock(heter_mutex_);
    if (cur_device_ != device) {
      MS_LOG(DEBUG) << "Current device " << device::GetDeviceNameByType(cur_device_) << " incoming device "
                    << device::GetDeviceNameByType(device);
      cur_device_ = device;
      need_wait = true;
    }
  }
  if (need_wait) {
    runtime::Pipeline::Get().WaitAll();
  }
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
