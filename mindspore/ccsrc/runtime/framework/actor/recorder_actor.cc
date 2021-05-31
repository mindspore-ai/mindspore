/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/framework/actor/recorder_actor.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void RecorderActor::RecordMemAddressInfo(const AnfNode *node, const KernelLaunchInfo *launch_info_,
                                         const DeviceContext *device_context, OpContext<DeviceTensor> *op_context) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(launch_info_);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(op_context);
  // todo record
}

void RecorderActor::ClearMemAddressInfo(OpContext<DeviceTensor> *op_context) {
  MS_EXCEPTION_IF_NULL(op_context);
  // todo clear
}

}  // namespace runtime
}  // namespace mindspore
