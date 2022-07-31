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

#include "src/litert/delegate/tensorrt/distribution/distribution_base.h"
#include <unistd.h>
#include <thread>
#include <string>
#include "plugin/device/gpu/hal/device/distribution/collective_wrapper.h"
#include "src/litert/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
int GetGPUGroupSize() { return GetGroupSize(NCCL_WORLD_GROUP); }

int GetRankID() { return GetRankIDByGroup(NCCL_WORLD_GROUP); }
}  // namespace mindspore::lite
