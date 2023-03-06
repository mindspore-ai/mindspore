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

#include "frontend/parallel/pass/handle_group_info.h"
#include "frontend/parallel/device_manager.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"

namespace mindspore {
namespace parallel {
void HandleGroupInfo(const FuncGraphPtr &root) {
  if (g_device_manager == nullptr) {
    return;
  }
  auto group_info = g_device_manager->group_info();
  auto group_info_save_path = common::GetEnv("GROUP_INFO_FILE");
  if (!group_info_save_path.empty()) {
    ParallelContext::GetInstance()->set_group_ckpt_save_file(group_info_save_path);
  }

  if (StrategyCheckpoint::GetInstance().group_info_save_on()) {
    auto &strategy_ckt = StrategyCheckpoint::GetInstance();
    RankList comm_group = strategy_ckt.common_mirror_group();
    if (strategy_ckt.SaveGroupInfo(group_info, comm_group) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Save group info failed";
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
