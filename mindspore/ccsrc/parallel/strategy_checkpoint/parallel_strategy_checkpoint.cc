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

#include "parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"

#include <fstream>
#include <memory>
#include <vector>

#include "common/utils.h"
#include "utils/convert_utils.h"
#include "utils/log_adapter.h"
#include "proto/node_strategy.pb.h"

namespace mindspore {
namespace parallel {
StrategyCheckpoint &StrategyCheckpoint::GetInstance() {
  static StrategyCheckpoint instance = StrategyCheckpoint();
  return instance;
}

bool StrategyCheckpoint::CheckPointExit() const {
  std::ifstream fin(path_);
  if (fin) {
    return true;
  }
  return false;
}

Status StrategyCheckpoint::RemoveCheckPoint() const {
  if (std::remove(common::SafeCStr(path_)) == 0) {
    return SUCCESS;
  }
  return FAILED;
}

Status StrategyCheckpoint::Load(StrategyMap *strategy_map) {
  if (strategy_map == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:strategy_map is nullptr";
  }
  straspb::ParallelStrategyMap parallel_strategy_map;
  std::fstream input(path_, std::ios::in | std::ios::binary);
  if (!parallel_strategy_map.ParseFromIstream(&input)) {
    MS_LOG(ERROR) << "Load strategy file failed";
    return FAILED;
  }
  size_t node_num = IntToSize(parallel_strategy_map.parallel_strategy_item_size());
  for (size_t i = 0; i < node_num; i++) {
    straspb::ParallelStrategyItem parallel_strategy_item = parallel_strategy_map.parallel_strategy_item(SizeToInt(i));
    std::string node_name = parallel_strategy_item.node_name();
    straspb::ParallelStrategys parallel_strategys = parallel_strategy_item.parallel_strategys();
    auto stage = (int32_t)parallel_strategys.stage();
    size_t strategys_num = IntToSize(parallel_strategys.parallel_strategy_size());
    std::vector<std::vector<int32_t>> strategy_inputs;
    for (size_t j = 0; j < strategys_num; j++) {
      straspb::ParallelStrategy parallel_strategy = parallel_strategys.parallel_strategy(SizeToInt(j));
      std::vector<int32_t> dimension;
      size_t dim_num = IntToSize(parallel_strategy.dim_size());
      for (size_t k = 0; k < dim_num; k++) {
        dimension.push_back(parallel_strategy.dim(SizeToInt(k)));
      }
      strategy_inputs.push_back(dimension);
    }

    StrategyPtr strategy = NewStrategy(stage, strategy_inputs);
    (*strategy_map)[node_name] = strategy;
    current_train_time_ = (int32_t)parallel_strategy_map.train_time();
  }
  return SUCCESS;
}

Status StrategyCheckpoint::Save(const StrategyMap &strategy_map) {
  straspb::ParallelStrategyMap parallel_strategy_map;
  parallel_strategy_map.set_train_time(IntToUint(++current_train_time_));
  for (auto &node_stra : strategy_map) {
    straspb::ParallelStrategyItem *parallel_strategy_item = parallel_strategy_map.add_parallel_strategy_item();
    MS_EXCEPTION_IF_NULL(parallel_strategy_item);
    parallel_strategy_item->set_node_name(node_stra.first);
    straspb::ParallelStrategys *parallel_strategys = parallel_strategy_item->mutable_parallel_strategys();
    MS_EXCEPTION_IF_NULL(parallel_strategys);
    parallel_strategys->set_stage(IntToUint(node_stra.second->GetInputStage()));
    for (auto &dims : node_stra.second->GetInputDim()) {
      straspb::ParallelStrategy *parallel_strategy = parallel_strategys->add_parallel_strategy();
      MS_EXCEPTION_IF_NULL(parallel_strategy);
      for (auto dim : dims) {
        parallel_strategy->add_dim(IntToUint(dim));
      }
    }
  }
  std::fstream output(path_, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!parallel_strategy_map.SerializeToOstream(&output)) {
    MS_LOG(ERROR) << "Save strategy file failed";
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
