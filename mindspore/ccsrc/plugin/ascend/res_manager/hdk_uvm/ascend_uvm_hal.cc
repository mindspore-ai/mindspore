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

#include "plugin/ascend/res_manager/hdk_uvm/ascend_uvm_hal.h"
#include <memory>
#include <string>
#include "acl/acl_rt.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr int kRemoteNumaId = 0;
}  // namespace

AscendUvmHal &AscendUvmHal::GetInstance() {
  static AscendUvmHal instance;
  return instance;
}

void AscendUvmHal::SyncStream(void *stream_ptr) const {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  const auto sync_status = CALL_ASCEND_API(aclrtSynchronizeStreamWithTimeout, stream_ptr, -1);
  if (sync_status != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Failed to synchronize stream, ret: " << sync_status;
  }
}

void AscendUvmHal::WaitUpdateFinished(int32_t dst_device, void *addr, void *stream_ptr) const {
  MS_LOG(WARNING) << "WaitUpdateFinished is not implemented yet.";
}

void AscendUvmHal::WaitMemCpyFinished(int32_t dst_device, void *addr, void *stream_ptr) const {
  MS_LOG(WARNING) << "WaitMemCpyFinished is not implemented yet.";
}

void AscendUvmHal::SetUvmReadMostly(void *addr, size_t size) const {
  MS_LOG(WARNING) << "SetUvmReadMostly is not implemented yet.";
}

bool AscendUvmHal::UpdateDeviceToRemote(int32_t device_id, void *addr, size_t size, void *stream_ptr, size_t offset,
                                        bool sync) const {
  MS_LOG(WARNING) << "UpdateDeviceToRemote is not implemented yet.";
  return true;
}

bool AscendUvmHal::UpdateRemoteToDevice(int32_t device_id, void *addr, size_t size, void *stream_ptr, size_t offset,
                                        bool sync) const {
  MS_LOG(WARNING) << "UpdateRemoteToDevice is not implemented yet.";
  return true;
}

bool AscendUvmHal::HasDeviceMem(int32_t device_id, void *addr) const {
  MS_LOG(WARNING) << "HasDeviceMem is not implemented yet.";
  return true;
}

bool AscendUvmHal::IsUpdating(int32_t device_id, void *addr, size_t size) const {
  MS_LOG(WARNING) << "IsUpdating is not implemented yet.";
  return true;
}

bool AscendUvmHal::DetachDevice(int32_t device_id, void *addr, size_t size, bool sync, void *stream_ptr) const {
  MS_LOG(WARNING) << "DetachDevice is not implemented yet.";
  return true;
}

bool AscendUvmHal::CopyHostToRemote(void *src_addr, void *dst_addr, size_t size, void *stream_ptr, size_t offset,
                                    bool sync) const {
  MS_LOG(WARNING) << "CopyHostToRemote is not implemented yet.";
  return true;
}

bool AscendUvmHal::CopyRemoteToHost(void *src_addr, void *dst_addr, size_t size, void *stream_ptr, size_t offset,
                                    bool sync) const {
  MS_LOG(WARNING) << "CopyRemoteToHost is not implemented yet.";
  return true;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
