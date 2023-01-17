/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <fstream>
#include <memory>
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
StrategyCheckpoint& StrategyCheckpoint::GetInstance() {
  static StrategyCheckpoint instance = StrategyCheckpoint();
  return instance;
}

bool StrategyCheckpoint::CheckPointExit(const std::string path) const { return false; }

Status StrategyCheckpoint::Load(StrategyMap* strategy_map) { return SUCCESS; }

Status StrategyCheckpoint::Save(const StrategyMap &strategy_map, const TensorInfoMap &tensor_info_map,
                                const ManualShapeMap &manual_shape_map) { return SUCCESS; }

Status StrategyCheckpoint::LoadGroupInfo(const std::string &file,
                                         GroupInfoMap *group_info_map) const { return SUCCESS; }

Status StrategyCheckpoint::SaveGroupInfo(const GroupInfoMap &group_info_map,
                                         const RankList &restore_rank_list) { return SUCCESS; }
}  // namespace parallel
}  // namespace mindspore
