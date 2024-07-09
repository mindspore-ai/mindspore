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

#include "plugin/device/ascend/kernel/pyboost/customize/stress_detect.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
int StressDetectKernel(const device::DeviceContext *device_context, size_t stream_id) {
  auto cube_math_type = 1;
  auto ret = LAUNCH_ACLNN_SYNC_WITH_RETURN(aclnnStressDetect, device_context, stream_id, cube_math_type);
  return ret;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
