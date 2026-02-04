/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "include/pynative/utils/pyboost/comm_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

void CommUtils::SyncOpStream(const device::DeviceContext *device_ctx, size_t op_stream_id, size_t comm_stream_id) {
  // Sync op stream id by event
  MS_EXCEPTION_IF_NULL(device_ctx);

  auto comm_stream_ptr = device_ctx->device_res_manager_->GetStream(comm_stream_id);
  auto begin_event = CreateOrGetCommBeginEvent(device_ctx, op_stream_id, comm_stream_id);
  begin_event->RecordEvent(op_stream_id);
  begin_event->set_wait_stream(comm_stream_ptr);
  begin_event->WaitEventWithoutReset();
}

DeviceEventPtr CommUtils::CreateOrGetCommBeginEvent(const device::DeviceContext *device_ctx, size_t op_stream_id,
                                                    size_t comm_stream_id) {
  auto iter = comm_begin_events_.find({op_stream_id, comm_stream_id});
  if (iter != comm_begin_events_.end()) {
    return iter->second;
  }

  auto event = device_ctx->device_res_manager_->CreateEventWithFlag(false, false);
  comm_begin_events_[{op_stream_id, comm_stream_id}] = event;
  return event;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
