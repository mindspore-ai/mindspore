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

#include "utils/current_state.h"
#include "utils/log_adapter.h"

namespace mindspore {
static uint32_t g_current_stream_id = 0;
uint32_t CurrentStream::id() { return g_current_stream_id; }

void CurrentStream::set_id(uint32_t stream_id) {
  MS_LOG(DEBUG) << "Set Current Stream Id from " << g_current_stream_id << " to " << stream_id;
  g_current_stream_id = stream_id;
}

static const device::DeviceContext *g_current_device_context = nullptr;
const device::DeviceContext *CurrentDeviceContext::get() { return g_current_device_context; }

void CurrentDeviceContext::set(const device::DeviceContext *device_context) {
  MS_LOG(DEBUG) << "Set Current DeviceContext to " << device_context;
  g_current_device_context = device_context;
}
}  // namespace mindspore
