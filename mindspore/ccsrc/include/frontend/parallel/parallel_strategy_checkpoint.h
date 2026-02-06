/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_PARALLEL_PARALLEL_STRATEGY_CHECKPOINT_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_PARALLEL_PARALLEL_STRATEGY_CHECKPOINT_H_

#include <pybind11/operators.h>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <memory>
#include <shared_mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "include/utils/parallel_context.h"
#include "include/utils/visible.h"
#include "ir/dtype/type.h"
#include "utils/hash_map.h"

#include "frontend/parallel/strategy.h"
#include "frontend/parallel/strategy_checkpoint/strategy_checkpoint_info.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"
#include "frontend/parallel/tensor_layout/tensor_layout.h"

namespace py = pybind11;
namespace mindspore {
namespace parallel {

class FRONTEND_EXPORT StrategyInfo {
 public:
  StrategyInfo() = default;
  ~StrategyInfo() = default;

  void set_dev_matrix(const int64_t val) { dev_matrix_.push_back(val); }
  const std::vector<int64_t> &dev_matrix() const { return dev_matrix_; }

  void set_tensor_map(const std::vector<int64_t> &val) { tensor_map_.push_back(val); }
  const std::vector<std::vector<int64_t>> &tensor_map() const { return tensor_map_; }

  void set_tensor_shape(const std::vector<int64_t> &val) { tensor_shape_ = val; }
  const std::vector<int64_t> &tensor_shape() const { return tensor_shape_; }

  void set_tensor_type(const std::string &val) { tensor_type_ = val; }
  const std::string &tensor_type() const { return tensor_type_; }

  void set_field(int64_t val) { field_ = val; }
  int64_t field() const { return field_; }

  void set_opt_weight_shard_step(int64_t val) { opt_weight_shard_step_ = val; }
  int64_t opt_weight_shard_step() const { return opt_weight_shard_step_; }

  void set_opt_weight_shard_size(int64_t val) { opt_weight_shard_size_ = val; }
  int64_t opt_weight_shard_size() const { return opt_weight_shard_size_; }

  void set_param_split_shape(const int64_t val) { param_split_shape_.push_back(val); }
  const std::vector<int64_t> &param_split_shape() const { return param_split_shape_; }

  void set_indices_offset(const int64_t &val) { indices_offset_.push_back(val); }
  const std::vector<int64_t> &indices_offset() const { return indices_offset_; }

  void set_stage_id(int64_t val) { stage_id_ = val; }
  int64_t stage_id() const { return stage_id_; }

  void set_pipeline_stages(int64_t val) { pipeline_stages_ = val; }
  int64_t pipeline_stages() const { return pipeline_stages_; }

  void set_rank_list(const std::vector<int64_t> &val) { rank_list_ = val; }
  const std::vector<int64_t> &rank_list() const { return rank_list_; }

  std::string ToString() const;

 private:
  std::vector<int64_t> dev_matrix_;
  std::vector<std::vector<int64_t>> tensor_map_;
  std::vector<int64_t> tensor_shape_;
  std::string tensor_type_;
  int64_t field_ = 0;
  int64_t opt_weight_shard_step_ = 0;
  int64_t opt_weight_shard_size_ = 0;
  std::vector<int64_t> param_split_shape_;
  std::vector<int64_t> indices_offset_;
  int64_t stage_id_ = 0;
  int64_t pipeline_stages_ = 1;
  std::vector<int64_t> rank_list_;
};

using ParamStrategyMap = mindspore::HashMap<std::string, StrategyInfo>;
using RankParamStrategy = mindspore::HashMap<uint32_t, ParamStrategyMap>;
using SortedParamVec = std::vector<std::pair<std::string, StrategyInfo>>;
using SortedRankParamVec = std::vector<SortedParamVec>;
using SortedNetRankParamVec = std::vector<std::pair<std::string, SortedRankParamVec>>;

class FRONTEND_EXPORT StrategyLayout {
 public:
  StrategyLayout() = default;
  ~StrategyLayout() = default;
  StrategyLayout(const StrategyLayout &) = delete;
  StrategyLayout &operator=(const StrategyLayout &) = delete;
  static std::shared_ptr<StrategyLayout> GetInstance();

  void enable_save_strategy_online() { save_strategy_online_ = true; }
  bool save_strategy_online() const { return save_strategy_online_; }

  void SetParamStageIdRanks(const std::string &param_name, const int64_t &stage_id,
                            const std::vector<int64_t> &rank_list);
  mindspore::HashMap<std::string, std::pair<int64_t, std::vector<int64_t>>> ParamStageIdRankId() const;

  void SetParamGlobalShape(const AnfNodePtr &parameter);
  std::vector<int64_t> ParamGlobalShape(const std::string &param_name) const;

  void SetParamType(const AnfNodePtr &parameter);
  std::string ParamType(const std::string &param_name) const;

  void SetCellPhase(const std::string &value) { compile_phase_ = value; }
  std::string CellPhase() const { return compile_phase_; }

  void SetNetworkLayoutSaved(const std::string &name);
  bool NetworkLayoutSaved(const std::string &name) const;

  void SaveParamStraInfo(uint32_t rank_id, const ParamStrategyMap &param_map);
  void SaveNetworkGlobalLayout();

  std::string CurNetGlobalStraInfo() const;
  std::string CurNetLocalStraInfo() const;

  py::dict global_network_layout() const;
  py::dict local_network_layout() const;

  void clear_strategy_metadata();

 private:
  SortedParamVec SortParamStrategy(const ParamStrategyMap &param_stra_map) const;
  void SortCurNetGlobalLayout();
  void EnsureSorted() const;
  std::string DebugString(const SortedRankParamVec &layout) const;
  py::dict ConvertNetStraToPyDict(const SortedNetRankParamVec &layout) const;
  void ClearCurNet();

  std::string compile_phase_;
  mindspore::HashMap<std::string, std::pair<int64_t, std::vector<int64_t>>> parameter_stage_ranks_map_;
  mindspore::HashMap<std::string, std::vector<int64_t>> param_shape_map_;
  mindspore::HashMap<std::string, std::string> param_type_map_;
  RankParamStrategy global_rank_stra_map_;
  SortedRankParamVec global_layout_list_;
  SortedRankParamVec local_layout_list_;
  std::atomic<bool> layout_sorted_{false};

  SortedNetRankParamVec global_network_rank_stra_list_;
  SortedNetRankParamVec local_network_rank_stra_list_;
  mindspore::HashMap<std::string, bool> network_state_map_;
  bool save_strategy_online_ = false;
  inline static std::shared_ptr<StrategyLayout> layout_instance_{nullptr};
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_PARALLEL_PARALLEL_STRATEGY_CHECKPOINT_H_
