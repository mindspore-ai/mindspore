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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_CHEKCPOINT_STRATEGY_CHECKPOINT_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_CHEKCPOINT_STRATEGY_CHECKPOINT_INFO_H_

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "nlohmann/json.hpp"
#include "utils/hash_map.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_layout.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"
#include "proto/node_strategy.pb.h"

namespace mindspore {
namespace parallel {
using StrategyMap = mindspore::HashMap<std::string, StrategyPtr>;
using TensorLayoutPtr = std::shared_ptr<TensorLayout>;
using TensorInfoMap = mindspore::HashMap<std::string, TensorLayoutPtr>;
using ParameterMap = std::vector<std::pair<std::string, ParameterPtr>>;
using ManualShapeMap = mindspore::HashMap<std::string, std::vector<std::pair<int64_t, int64_t>>>;
using GroupInfoMap = std::vector<std::pair<std::string, std::vector<uint32_t>>>;

class StrategyCheckpointInfo {
 public:
  StrategyCheckpointInfo() : current_stage_(0) {}
  ~StrategyCheckpointInfo() = default;
  void Init(const StrategyMap &strategy_map, const TensorInfoMap &tensor_info_map,
            const ManualShapeMap &manual_shape_map, int64_t current_stage) {
    strategy_map_ = strategy_map;
    tensor_info_map_ = tensor_info_map;
    manual_shape_map_ = manual_shape_map;
    current_stage_ = current_stage;
  }
  StrategyMap strategy_map() const { return strategy_map_; }
  void set_strategy_map(const StrategyMap &strategy_map);
  TensorInfoMap tensor_info_map() const { return tensor_info_map_; }
  void set_tensor_info_map(const TensorInfoMap &tensor_info_map);
  ManualShapeMap manual_shape_map() const { return manual_shape_map_; }
  void set_manual_shape_map(const ManualShapeMap &manual_shape_map);
  int64_t current_stage() const { return current_stage_; }

  void from_json(const nlohmann::json &stra_ckpt_info_j);
  nlohmann::json to_json() const;

  void from_protobuf(const straspb::ParallelStrategyMap &parallel_strategy_map);
  straspb::ParallelStrategyMap to_protobuf() const;

 private:
  int64_t current_stage_;
  StrategyMap strategy_map_;
  TensorInfoMap tensor_info_map_;
  ManualShapeMap manual_shape_map_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_CHEKCPOINT_STRATEGY_CHECKPOINT_INFO_H_
