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

#include "debug/dump_utils.h"

#include <string>
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/comm_manager.h"
#include "frontend/parallel/context.h"

namespace mindspore {
uint32_t DumpUtils::GetRankId() {
  uint32_t rank_id = 0;
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  if (parallel_mode == parallel::STAND_ALONE) {
    MS_LOG(INFO) << "parallel_mode is stand_alone, use 0 as default rank id.";
    return rank_id;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string world_group;
  std::string backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (backend == kAscendDevice) {
    world_group = kHcclWorldGroup;
  } else if (backend == kGPUDevice) {
    world_group = kNcclWorldGroup;
  } else {
    return rank_id;
  }

  if (!CommManager::GetInstance().GetRankID(world_group, &rank_id)) {
    MS_LOG(WARNING) << "Failed to get rank id.";
  }

  return rank_id;
}
}  // namespace mindspore
