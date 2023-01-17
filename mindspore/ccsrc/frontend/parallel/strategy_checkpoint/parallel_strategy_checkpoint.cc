/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"

#include <fstream>
#include <vector>
#include <utility>
#include "include/common/utils/utils.h"
#include "utils/ms_utils.h"
#include "include/common/utils/convert_utils.h"
#include "utils/log_adapter.h"
#include "include/common/debug/common.h"
#include "mindspore/core/utils/file_utils.h"

namespace mindspore {
namespace parallel {
const uint32_t JSON_SUFFIX_LENGTH = 5;
StrategyCheckpoint &StrategyCheckpoint::GetInstance() {
  static StrategyCheckpoint instance = StrategyCheckpoint();
  if (ParallelContext::GetInstance() != nullptr) {
    instance.load_file_ = ParallelContext::GetInstance()->strategy_ckpt_load_file();
    instance.load_checkpoint_on_ = !ParallelContext::GetInstance()->strategy_ckpt_load_file().empty();
    instance.save_file_ = ParallelContext::GetInstance()->strategy_ckpt_save_file();
    instance.save_checkpoint_on_ = !ParallelContext::GetInstance()->strategy_ckpt_save_file().empty();
    instance.group_info_save_file_ = ParallelContext::GetInstance()->group_ckpt_save_file();
    instance.group_info_save_on_ = !ParallelContext::GetInstance()->group_ckpt_save_file().empty();
    instance.load_format_json_ = instance.load_file_.size() >= JSON_SUFFIX_LENGTH &&
                                 instance.load_file_.substr(instance.load_file_.size() - JSON_SUFFIX_LENGTH) == ".json";
    instance.save_format_json_ = instance.save_file_.size() >= JSON_SUFFIX_LENGTH &&
                                 instance.save_file_.substr(instance.save_file_.size() - JSON_SUFFIX_LENGTH) == ".json";
  }
  return instance;
}

bool StrategyCheckpoint::CheckPath(const std::string path) const {
  if (path.size() > PATH_MAX) {
    MS_LOG(ERROR) << "The checkpoit path " << path << " is too long";
    return false;
  }
  auto realpath = Common::CreatePrefixPath(path, true);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path=" << path;
    return false;
  }
  return true;
}

bool StrategyCheckpoint::CheckPointExit(const std::string path) const {
  std::ifstream fin(path);
  if (fin) {
    return true;
  }
  return false;
}

Status StrategyCheckpoint::LoadGroupInfo(const std::string &file, GroupInfoMap *group_info_map) const {
  MS_EXCEPTION_IF_NULL(group_info_map);
  if (!CheckPath(file)) {
    MS_LOG(EXCEPTION) << "CheckPoint file in invalid";
  }
  if (!CheckPointExit(file)) {
    MS_LOG(EXCEPTION) << "CheckPoint file is not found";
  }
  straspb::ParallelGroupMap parallel_group_map;
  std::fstream input(file, std::ios::in | std::ios::binary);
  if (!parallel_group_map.ParseFromIstream(&input)) {
    MS_LOG(ERROR) << "Load strategy file failed";
    return FAILED;
  }
  input.close();

  size_t group_num = LongToSize(parallel_group_map.parallel_group_item_size());
  for (size_t i = 0; i < group_num; ++i) {
    straspb::ParallelGroupItem parallel_group_item = parallel_group_map.parallel_group_item(SizeToInt(i));
    std::string group_name = parallel_group_item.group_name();

    straspb::ParallelGroupRanks parallel_group_ranks = parallel_group_item.parallel_group_ranks();
    size_t rank_num = LongToSize(parallel_group_ranks.dim_size());
    std::vector<uint32_t> ranks;
    for (size_t j = 0; j < rank_num; ++j) {
      uint32_t rank = parallel_group_ranks.dim(SizeToInt(j));
      ranks.push_back(rank);
    }

    std::pair<std::string, std::vector<uint32_t>> group = std::make_pair(group_name, ranks);
    group_info_map->push_back(group);
  }

  return SUCCESS;
}

Status StrategyCheckpoint::Load(StrategyMap *strategy_map) {
  if (strategy_map == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:strategy_map is nullptr";
  }
  if (!CheckPath(load_file_)) {
    MS_LOG(EXCEPTION) << "CheckPoint file in invalid";
  }
  if (!CheckPointExit(load_file_)) {
    MS_LOG(EXCEPTION) << "CheckPoint file is not found";
  }
  if (load_format_json_) {
    std::fstream input(load_file_, std::ios::in);
    nlohmann::json stra_ckpt_info_j;
    input >> stra_ckpt_info_j;
    strategy_checkpoint_info_.from_json(stra_ckpt_info_j);
  } else {
    straspb::ParallelStrategyMap parallel_strategy_map;
    std::fstream input(load_file_, std::ios::in | std::ios::binary);
    if (!parallel_strategy_map.ParseFromIstream(&input)) {
      MS_LOG(ERROR) << "Load strategy file failed";
      return FAILED;
    }
    input.close();
    strategy_checkpoint_info_.from_protobuf(parallel_strategy_map);
  }
  *strategy_map = strategy_checkpoint_info_.strategy_map();
  current_stage_ = SizeToLong(strategy_checkpoint_info_.current_stage());
  return SUCCESS;
}

Status StrategyCheckpoint::Save(const StrategyMap &strategy_map, const TensorInfoMap &tensor_info_map,
                                const ManualShapeMap &manual_shape_map) {
  if (!CheckPath(save_file_)) {
    MS_LOG(EXCEPTION) << "CheckPoint file in invalid";
  }
  strategy_checkpoint_info_.Init(strategy_map, tensor_info_map, manual_shape_map, ++current_stage_);
  if (save_format_json_) {
    auto stra_ckpt_info_j = strategy_checkpoint_info_.to_json();
    std::fstream output(save_file_, std::ios::out);
    stra_ckpt_info_j >> output;
    output.close();
  } else {
    auto parallel_strategy_map = strategy_checkpoint_info_.to_protobuf();
    std::fstream output(save_file_, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!parallel_strategy_map.SerializeToOstream(&output)) {
      MS_LOG(ERROR) << "Save strategy file failed";
      return FAILED;
    }
    output.close();
  }

  ChangeFileMode(save_file_, S_IRUSR | S_IWUSR);
  return SUCCESS;
}

Status StrategyCheckpoint::SaveGroupInfo(const GroupInfoMap &group_info_map, const RankList &restore_rank_list) {
  straspb::ParallelGroupMap parallel_group_map;
  for (auto &group : group_info_map) {
    straspb::ParallelGroupItem *parallel_group_item = parallel_group_map.add_parallel_group_item();
    MS_EXCEPTION_IF_NULL(parallel_group_item);
    parallel_group_item->set_group_name(group.first);
    straspb::ParallelGroupRanks *parallel_group_ranks = parallel_group_item->mutable_parallel_group_ranks();
    MS_EXCEPTION_IF_NULL(parallel_group_ranks);
    for (auto &rank : group.second) {
      parallel_group_ranks->add_dim(rank);
    }
  }
  straspb::ParallelGroupRanks *ckpt_restore_rank_list = parallel_group_map.mutable_ckpt_restore_rank_list();
  for (auto &restore_rank : restore_rank_list) {
    ckpt_restore_rank_list->add_dim(static_cast<uint32_t>(LongToSize(restore_rank)));
  }

  if (!CheckPath(group_info_save_file_)) {
    MS_LOG(EXCEPTION) << "CheckPoint file in invalid";
  }
  std::fstream output(group_info_save_file_, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!parallel_group_map.SerializeToOstream(&output)) {
    MS_LOG(ERROR) << "Save strategy file failed";
    return FAILED;
  }
  output.close();
  ChangeFileMode(group_info_save_file_, S_IRUSR | S_IWUSR);
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
