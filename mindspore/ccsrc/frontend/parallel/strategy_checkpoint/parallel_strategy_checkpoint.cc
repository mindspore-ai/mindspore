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
#include <memory>
#include <vector>

#include "utils/ms_utils.h"
#include "utils/convert_utils.h"
#include "utils/log_adapter.h"
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
  }
  return instance;
}

bool StrategyCheckpoint::CheckPointExit(const std::string path) const {
  std::ifstream fin(path);
  if (fin) {
    return true;
  }
  return false;
}

Status StrategyCheckpoint::Load(StrategyMap *strategy_map) {
  if (strategy_map == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:strategy_map is nullptr";
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
  size_t node_num = IntToSize(parallel_strategy_map.parallel_strategy_item_size());
  for (size_t i = 0; i < node_num; i++) {
    straspb::ParallelStrategyItem parallel_strategy_item = parallel_strategy_map.parallel_strategy_item(SizeToInt(i));
    std::string node_name = parallel_strategy_item.node_name();
    straspb::ParallelStrategys parallel_strategys = parallel_strategy_item.parallel_strategys();
    auto stage = (int32_t)parallel_strategys.stage();
    size_t strategys_num = IntToSize(parallel_strategys.parallel_strategy_size());
    Strategys strategy_inputs;
    for (size_t j = 0; j < strategys_num; j++) {
      straspb::ParallelStrategy parallel_strategy = parallel_strategys.parallel_strategy(SizeToInt(j));
      Dimensions dimension;
      size_t dim_num = IntToSize(parallel_strategy.dim_size());
      for (size_t k = 0; k < dim_num; k++) {
        dimension.push_back(parallel_strategy.dim(SizeToInt(k)));
      }
      strategy_inputs.push_back(dimension);
    }

    StrategyPtr strategy = NewStrategy(stage, strategy_inputs);
    (*strategy_map)[node_name] = strategy;
    current_stage_ = (int32_t)parallel_strategy_map.current_stage();
  }
  return SUCCESS;
}

Status StrategyCheckpoint::Save(const StrategyMap &strategy_map, const TensorInfoMap &tensor_info_map,
                                ManualShapeMap *manual_shape_map) {
  straspb::ParallelStrategyMap parallel_strategy_map;
  parallel_strategy_map.set_current_stage(IntToUint(++current_stage_));
  for (auto &node_stra : strategy_map) {
    straspb::ParallelStrategyItem *parallel_strategy_item = parallel_strategy_map.add_parallel_strategy_item();
    MS_EXCEPTION_IF_NULL(parallel_strategy_item);
    parallel_strategy_item->set_node_name(node_stra.first);
    straspb::ParallelStrategys *parallel_strategys = parallel_strategy_item->mutable_parallel_strategys();
    MS_EXCEPTION_IF_NULL(parallel_strategys);
    MS_EXCEPTION_IF_NULL(node_stra.second);
    parallel_strategys->set_stage(IntToUint(node_stra.second->GetInputStage()));
    for (auto &dims : node_stra.second->GetInputDim()) {
      straspb::ParallelStrategy *parallel_strategy = parallel_strategys->add_parallel_strategy();
      MS_EXCEPTION_IF_NULL(parallel_strategy);
      for (auto dim : dims) {
        parallel_strategy->add_dim(IntToUint(dim));
      }
    }
  }
  for (auto &node_tensor_info : tensor_info_map) {
    TensorInfo tensor_info = node_tensor_info.second;
    TensorLayout tensor_layout = tensor_info.tensor_layout();
    straspb::ParallelLayoutItem *parallel_layout_item = parallel_strategy_map.add_parallel_layout_item();
    MS_EXCEPTION_IF_NULL(parallel_layout_item);
    parallel_layout_item->set_param_name(node_tensor_info.first);
    straspb::ParallelLayouts *parallel_layouts = parallel_layout_item->mutable_parallel_layouts();
    straspb::DevMatrix *dev_matrix = parallel_layouts->add_dev_matrix();
    MS_EXCEPTION_IF_NULL(dev_matrix);
    for (auto dim : tensor_layout.device_arrangement().array()) {
      dev_matrix->add_dim(IntToUint(dim));
    }
    straspb::TensorMap *tensor_map = parallel_layouts->add_tensor_map();
    MS_EXCEPTION_IF_NULL(tensor_map);
    for (auto dim : tensor_layout.tensor_map().array()) {
      tensor_map->add_dim(dim);
    }
    straspb::ParamSplitShape *param_split_shape = parallel_layouts->add_param_split_shape();
    straspb::IndicesOffset *indices_offset = parallel_layouts->add_indices_offset();
    MS_EXCEPTION_IF_NULL(manual_shape_map);
    auto manual_shape = (*manual_shape_map)[node_tensor_info.first];
    for (auto dim_pair : manual_shape) {
      param_split_shape->add_dim(dim_pair.first);
      indices_offset->add_dim(dim_pair.second);
    }
  }

  std::fstream output(save_file_, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!parallel_strategy_map.SerializeToOstream(&output)) {
    MS_LOG(ERROR) << "Save strategy file failed";
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
