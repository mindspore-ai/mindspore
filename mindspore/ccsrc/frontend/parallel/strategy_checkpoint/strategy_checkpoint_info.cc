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

#include "frontend/parallel/strategy_checkpoint/strategy_checkpoint_info.h"

#include <fstream>
#include <vector>
#include <utility>

namespace mindspore {
namespace parallel {
void StrategyCheckpointInfo::set_strategy_map(const StrategyMap &strategy_map) { strategy_map_ = strategy_map; }

void StrategyCheckpointInfo::set_tensor_info_map(const TensorInfoMap &tensor_info_map) {
  tensor_info_map_ = tensor_info_map;
}

void StrategyCheckpointInfo::set_manual_shape_map(const ManualShapeMap &manual_shape_map) {
  manual_shape_map_ = manual_shape_map;
}

void StrategyCheckpointInfo::from_json(const nlohmann::json &stra_ckpt_info_j) {
  current_stage_ = stra_ckpt_info_j.at("current_stage").get<int64_t>();
  for (const auto &stra_j : stra_ckpt_info_j.at("parallel_strategy_item").items()) {
    auto node_name = stra_j.key();
    auto stage = stra_j.value().at("stage").get<int64_t>();
    auto stra = stra_j.value().at("parallel_strategy").get<std::vector<std::vector<int64_t>>>();
    strategy_map_[node_name] = std::make_shared<Strategy>(stage, stra);
  }
  for (const auto &layout_j : stra_ckpt_info_j.at("parallel_layout_item").items()) {
    auto parameter_name = layout_j.key();
    auto dev_matrix = layout_j.value().at("dev_matrix").get<std::vector<int64_t>>();
    auto tensor_map = layout_j.value().at("tensor_map").get<std::vector<int64_t>>();
    auto tensor_shape = layout_j.value().at("tensor_shape").get<std::vector<int64_t>>();
    auto field = layout_j.value().at("field").get<int64_t>();
    auto opt_weight_shard_step = layout_j.value().at("opt_weight_shard_step").get<int64_t>();
    auto opt_weight_shard_size = layout_j.value().at("opt_weight_shard_size").get<int64_t>();
    if (layout_j.value().contains("param_split_shape") && layout_j.value().contains("indices_offset")) {
      auto param_split_shape = layout_j.value().at("param_split_shape").get<std::vector<int64_t>>();
      auto indices_offset = layout_j.value().at("indices_offset").get<std::vector<int64_t>>();
      if (param_split_shape.size() != indices_offset.size()) {
        MS_LOG(EXCEPTION) << "For field_split strategy, the size of param_split_shape " << param_split_shape.size()
                          << " is not equal to the size of indices_offset " << indices_offset.size();
      }
      for (size_t i = 0; i < param_split_shape.size(); ++i) {
        manual_shape_map_[parameter_name].push_back({param_split_shape[i], indices_offset[i]});
      }
    }
    tensor_info_map_[parameter_name] = std::make_shared<TensorLayout>();
    (void)tensor_info_map_[parameter_name]->InitFromVector(dev_matrix, tensor_map, tensor_shape);
    tensor_info_map_[parameter_name]->set_opt_weight_shard_size(opt_weight_shard_size);
    tensor_info_map_[parameter_name]->set_opt_weight_shard_step(opt_weight_shard_step);
    tensor_info_map_[parameter_name]->set_field_size(field);
  }
}

nlohmann::json StrategyCheckpointInfo::to_json() const {
  nlohmann::json stra_ckpt_info_j;
  stra_ckpt_info_j["current_stage"] = current_stage_;
  for (const auto &stra_pair : strategy_map_) {
    auto node_name = stra_pair.first;
    auto node_stra = stra_pair.second;
    nlohmann::json stra_j;
    stra_j["stage"] = node_stra->GetInputStage();
    stra_j["parallel_strategy"] = node_stra->GetInputDim();
    stra_ckpt_info_j["parallel_strategy_item"][node_name] = stra_j;
  }
  for (const auto &layout_pair : tensor_info_map_) {
    auto parameter_name = layout_pair.first;
    auto layout = layout_pair.second;
    nlohmann::json layout_j;
    layout_j["dev_matrix"] = layout->device_arrangement().array();
    layout_j["tensor_map"] = layout->tensor_map().array();
    layout_j["tensor_shape"] = layout->tensor_shape().array();
    layout_j["field"] = layout->get_field_size();
    layout_j["opt_weight_shard_step"] = layout->opt_weight_shard_step();
    layout_j["opt_weight_shard_size"] = layout->opt_weight_shard_size();
    if (manual_shape_map_.find(parameter_name) != manual_shape_map_.end()) {
      auto manual_shape = manual_shape_map_.at(parameter_name);
      for (auto dim_pair : manual_shape) {
        layout_j["param_split_shape"].push_back(dim_pair.first);
        layout_j["indices_offset"].push_back(dim_pair.second);
      }
    }
    stra_ckpt_info_j["parallel_layout_item"][parameter_name] = layout_j;
  }
  return stra_ckpt_info_j;
}

void StrategyCheckpointInfo::from_protobuf(const straspb::ParallelStrategyMap &parallel_strategy_map) {
  size_t node_num = LongToSize(parallel_strategy_map.parallel_strategy_item_size());
  for (size_t i = 0; i < node_num; i++) {
    straspb::ParallelStrategyItem parallel_strategy_item = parallel_strategy_map.parallel_strategy_item(SizeToInt(i));
    std::string node_name = parallel_strategy_item.node_name();
    straspb::ParallelStrategys parallel_strategys = parallel_strategy_item.parallel_strategys();
    int64_t stage = SizeToLong(parallel_strategys.stage());
    size_t strategys_num = LongToSize(parallel_strategys.parallel_strategy_size());
    Strategies strategy_inputs;
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
    strategy_map_[node_name] = strategy;
    current_stage_ = SizeToLong(parallel_strategy_map.current_stage());
  }
}

straspb::ParallelStrategyMap StrategyCheckpointInfo::to_protobuf() const {
  straspb::ParallelStrategyMap parallel_strategy_map;
  parallel_strategy_map.set_current_stage(UlongToUint(LongToUlong(current_stage_)));
  for (auto &node_stra : strategy_map_) {
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
  for (auto &node_tensor_info : tensor_info_map_) {
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
    parallel_layouts->set_field(LongToInt(tensor_layout->get_field_size()));
    parallel_layouts->set_opt_weight_shard_step(tensor_layout->opt_weight_shard_step());
    parallel_layouts->set_opt_weight_shard_size(tensor_layout->opt_weight_shard_size());
    if (manual_shape_map_.find(node_tensor_info.first) != manual_shape_map_.end()) {
      auto manual_shape = manual_shape_map_.at(node_tensor_info.first);
      for (auto dim_pair : manual_shape) {
        param_split_shape->add_dim(dim_pair.first);
        indices_offset->add_dim(dim_pair.second);
      }
    }
  }
  return parallel_strategy_map;
}
}  // namespace parallel
}  // namespace mindspore
