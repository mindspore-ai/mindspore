/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_CHEKCPOINT_PARALLEL_STRATEGY_CHECKPOINT_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_CHEKCPOINT_PARALLEL_STRATEGY_CHECKPOINT_H_

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "utils/hash_map.h"
#include "frontend/parallel/strategy.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/tensor_layout/tensor_layout.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"
#include "frontend/parallel/strategy_checkpoint/strategy_checkpoint_info.h"

namespace mindspore {
namespace parallel {
class StrategyCheckpoint {
 public:
  StrategyCheckpoint() {
    load_file_ = "";
    save_file_ = "";
    group_info_save_file_ = "";
  }
  ~StrategyCheckpoint() = default;

  Status Load(StrategyMap *strategy_map);
  Status LoadGroupInfo(const std::string &file, GroupInfoMap *group_info_map) const;
  Status Save(const StrategyMap &strategy_map, const TensorInfoMap &tensor_info_map,
              const ManualShapeMap &manual_shape_map);
  Status SaveGroupInfo(const GroupInfoMap &group_info_map, const RankList &restore_rank_list);
  bool group_info_save_on() const { return group_info_save_on_; }

  static StrategyCheckpoint &GetInstance();
  bool LoadCheckPointOn() const { return load_checkpoint_on_; }
  bool SaveCheckPointOn() const { return save_checkpoint_on_; }

 private:
  std::string load_file_;
  std::string save_file_;
  bool load_checkpoint_on_ = false;
  bool save_checkpoint_on_ = false;
  bool CheckPointExit(const std::string path) const;
  bool CheckPath(const std::string path) const;
  int64_t current_stage_ = 0;
  std::string group_info_save_file_;
  bool group_info_save_on_ = false;
  bool load_format_json_ = true;
  bool save_format_json_ = true;
  StrategyCheckpointInfo strategy_checkpoint_info_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_CHEKCPOINT_PARALLEL_STRATEGY_CHECKPOINT_H_
