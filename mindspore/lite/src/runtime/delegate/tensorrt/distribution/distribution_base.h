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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_DISTRIBUTION_DISTRIBUTION_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_DISTRIBUTION_DISTRIBUTION_BASE_H_

#include <string>
#include "src/common/log_adapter.h"
#include "include/errorcode.h"

namespace mindspore::lite {
constexpr char NCCL_WORLD_GROUP[] = "nccl_world_group";

int GetGPUGroupSize();

int GetRankID();
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_DISTRIBUTION_DISTRIBUTION_BASE_H_
