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
#include "cxx_api/utils.h"
#include <string>
#include "mindspore/ccsrc/include/common/utils/comm_manager.h"
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"
namespace mindspore {
bool CreateGroupsByCkptFile(const std::string &file) {
  parallel::GroupInfoMap group_info_map;
  if (parallel::StrategyCheckpoint::GetInstance().LoadGroupInfo(file, &group_info_map) != parallel::SUCCESS) {
    return false;
  }

  for (const auto &[group_name, rank_ids] : group_info_map) {
    if (!CommManager::GetInstance().CreateGroupSync(group_name, rank_ids)) {
      MS_LOG(ERROR) << "Create group " << group_name << " rank ids " << rank_ids << " failed.";
      return false;
    }
  }

  MS_LOG(INFO) << "Create groups by checkpoint file success";
  return true;
}
}  // namespace mindspore
