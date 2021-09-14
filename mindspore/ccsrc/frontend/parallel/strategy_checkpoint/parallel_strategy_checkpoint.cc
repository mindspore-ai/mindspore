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

#include "utils/ms_utils.h"
#include "utils/convert_utils.h"
#include "utils/log_adapter.h"
#include "debug/common.h"
#include "proto/node_strategy.pb.h"

namespace mindspore {
namespace parallel {
StrategyCheckpoint &StrategyCheckpoint::GetInstance() {
  static StrategyCheckpoint instance = StrategyCheckpoint();
  if (ParallelContext::GetInstance() != nullptr) {
    instance.load_file_ = ParallelContext::GetInstance()->strategy_ckpt_load_file();
    instance.load_checkpoint_on_ = !ParallelContext::GetInstance()->strategy_ckpt_load_file().empty();
    instance.save_file_ = ParallelContext::GetInstance()->strategy_ckpt_save_file();
    instance.save_checkpoint_on_ = !ParallelContext::GetInstance()->strategy_ckpt_save_file().empty();
    instance.group_info_save_file_ = ParallelContext::GetInstance()->group_ckpt_save_file();
    instance.group_info_save_on_ = !ParallelContext::GetInstance()->group_ckpt_save_file().empty();
  }
  return instance;
}

bool StrategyCheckpoint::CheckPath(const std::string path) const {
  if (path.size() > PATH_MAX) {
    MS_LOG(ERROR) << "The checkpoit path " << path << " is too long";
    return false;
  }
  auto realpath = Common::CreatePrefixPath(path);
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

Status StrategyCheckpoint::LoadGroupInfo(const std::string &file, GroupInfoMap *group_info_map) {
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
  straspb::ParallelStrategyMap parallel_strategy_map;
  std::fstream input(load_file_, std::ios::in | std::ios::binary);
  if (!parallel_strategy_map.ParseFromIstream(&input)) {
    MS_LOG(ERROR) << "Load strategy file failed";
    return FAILED;
  }
  input.close();
  size_t node_num = LongToSize(parallel_strategy_map.parallel_strategy_item_size());
  for (size_t i = 0; i < node_num; i++) {
    straspb::ParallelStrategyItem parallel_strategy_item = parallel_strategy_map.parallel_strategy_item(SizeToInt(i));
    std::string node_name = parallel_strategy_item.node_name();
    straspb::ParallelStrategys parallel_strategys = parallel_strategy_item.parallel_strategys();
    auto stage = (int64_t)parallel_strategys.stage();
    size_t strategys_num = LongToSize(parallel_strategys.parallel_strategy_size());
    Strategys strategy_inputs;
    for (size_t j = 0; j < strategys_num; j++) {
      straspb::ParallelStrategy parallel_strategy = parallel_strategys.parallel_strategy(SizeToInt(j));
      Dimensions dimension;
      size_t dim_num = LongToSize(parallel_strategy.dim_size());
      for (size_t k = 0; k < dim_num; k++) {
        dimension.push_back(parallel_strategy.dim(SizeToInt(k)));
      }
      strategy_inputs.push_back(dimension);
    }

    StrategyPtr strategy = NewStrategy(stage, strategy_inputs);
    (*strategy_map)[node_name] = strategy;
    current_stage_ = (int64_t)parallel_strategy_map.current_stage();
  }
  return SUCCESS;
}

Status StrategyCheckpoint::Save(const StrategyMap &strategy_map, const TensorInfoMap &tensor_info_map,
                                ManualShapeMap *manual_shape_map) {
  straspb::ParallelStrategyMap parallel_strategy_map;
  parallel_strategy_map.set_current_stage(UlongToUint(LongToUlong(++current_stage_)));
  for (auto &node_stra : strategy_map) {
    straspb::ParallelStrategyItem *parallel_strategy_item = parallel_strategy_map.add_parallel_strategy_item();
    MS_EXCEPTION_IF_NULL(parallel_strategy_item);
    parallel_strategy_item->set_node_name(node_stra.first);
    straspb::ParallelStrategys *parallel_strategys = parallel_strategy_item->mutable_parallel_strategys();
    MS_EXCEPTION_IF_NULL(parallel_strategys);
    MS_EXCEPTION_IF_NULL(node_stra.second);
    parallel_strategys->set_stage(UlongToUint(LongToUlong(node_stra.second->GetInputStage())));
    for (auto &dims : node_stra.second->GetInputDim()) {
      straspb::ParallelStrategy *parallel_strategy = parallel_strategys->add_parallel_strategy();
      MS_EXCEPTION_IF_NULL(parallel_strategy);
      for (auto stra_dim : dims) {
        parallel_strategy->add_dim(UlongToUint(LongToUlong(stra_dim)));
      }
    }
  }
  for (auto &node_tensor_info : tensor_info_map) {
    TensorLayoutPtr tensor_layout = node_tensor_info.second;
    MS_EXCEPTION_IF_NULL(tensor_layout);
    straspb::ParallelLayoutItem *parallel_layout_item = parallel_strategy_map.add_parallel_layout_item();
    MS_EXCEPTION_IF_NULL(parallel_layout_item);
    parallel_layout_item->set_param_name(node_tensor_info.first);
    straspb::ParallelLayouts *parallel_layouts = parallel_layout_item->mutable_parallel_layouts();
    straspb::DevMatrix *dev_matrix = parallel_layouts->add_dev_matrix();
    MS_EXCEPTION_IF_NULL(dev_matrix);
    for (auto dev_dim : tensor_layout->device_arrangement().array()) {
      dev_matrix->add_dim(UlongToUint(LongToUlong(dev_dim)));
    }
    straspb::TensorMap *tensor_map = parallel_layouts->add_tensor_map();
    MS_EXCEPTION_IF_NULL(tensor_map);
    for (auto map_dim : tensor_layout->tensor_map().array()) {
      tensor_map->add_dim(LongToInt(map_dim));
    }
    straspb::ParamSplitShape *param_split_shape = parallel_layouts->add_param_split_shape();
    straspb::IndicesOffset *indices_offset = parallel_layouts->add_indices_offset();
    MS_EXCEPTION_IF_NULL(manual_shape_map);
    auto manual_shape = (*manual_shape_map)[node_tensor_info.first];
    for (auto dim_pair : manual_shape) {
      param_split_shape->add_dim(dim_pair.first);
      indices_offset->add_dim(dim_pair.second);
    }
    parallel_layouts->set_field(LongToInt(tensor_layout->get_field_size()));
    parallel_layouts->set_opt_weight_shard_step(tensor_layout->opt_weight_shard_step());
    parallel_layouts->set_opt_weight_shard_size(tensor_layout->opt_weight_shard_size());
  }
  if (!CheckPath(save_file_)) {
    MS_LOG(EXCEPTION) << "CheckPoint file in invalid";
  }
  std::fstream output(save_file_, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!parallel_strategy_map.SerializeToOstream(&output)) {
    MS_LOG(ERROR) << "Save strategy file failed";
    return FAILED;
  }
  output.close();
  ChangeFileMode(save_file_, S_IRUSR | S_IWUSR);
  return SUCCESS;
}

Status StrategyCheckpoint::SaveGroupInfo(const GroupInfoMap &group_info_map) {
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
