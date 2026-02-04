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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_CHEKCPOINT_PARALLEL_STRATEGY_CHECKPOINT_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_CHEKCPOINT_PARALLEL_STRATEGY_CHECKPOINT_H_

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
#include "ir/dtype/type.h"
#include "utils/hash_map.h"

#include "frontend/parallel/strategy.h"
#include "frontend/parallel/strategy_checkpoint/strategy_checkpoint_info.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"
#include "frontend/parallel/tensor_layout/tensor_layout.h"
#include "include/frontend/parallel/parallel_strategy_checkpoint.h"

namespace py = pybind11;
namespace mindspore {
namespace parallel {
class VectorUtils {
 public:
  template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
  static std::string PrintVector(const std::vector<T> &vec) {
    const int MAX_PRINT_NUM = 100;
    std::stringstream ss;
    ss << "[";
    int size = std::min(static_cast<int>(vec.size()), MAX_PRINT_NUM);
    for (int i = 0; i < size; ++i) {
      ss << std::to_string(vec[i]);
      if (i != size - 1) {
        ss << ", ";
      }
    }
    if (vec.size() > MAX_PRINT_NUM) {
      ss << ", ... to be continue";
    }
    ss << "]";
    return ss.str();
  }

  template <typename T, typename Dummy = std::enable_if_t<std::is_arithmetic<T>::value>>
  static std::string PrintNestedVector(const std::vector<std::vector<T>> &double_vec) {
    std::stringstream ss;
    ss << '[';
    for (size_t i = 0; i < double_vec.size(); ++i) {
      ss << VectorUtils::PrintVector(double_vec[i]);
      if (i + 1 < double_vec.size()) ss << ", ";
    }
    ss << ']';
    return ss.str();
  }
};

class StrategyCheckpoint {
 public:
  StrategyCheckpoint() {
    load_file_ = "";
    save_file_ = "";
    group_info_save_file_ = "";
    auto_op_strategy_file_ = "";
  }
  ~StrategyCheckpoint() = default;

  Status Load(StrategyMap *strategy_map);
  Status LoadGroupInfo(const std::string &file, GroupInfoMap *group_info_map) const;
  Status Save(const StrategyMap &strategy_map, const TensorInfoMap &tensor_info_map,
              const ManualShapeMap &manual_shape_map);
  Status SaveOnline(const StrategyMap &strategy_map, const TensorInfoMap &tensor_info_map,
                    const ManualShapeMap &manual_shape_map);
  Status SaveGroupInfo(const GroupInfoMap &group_info_map, const RankList &restore_rank_list);
  bool group_info_save_on() const { return group_info_save_on_; }

  static StrategyCheckpoint &GetInstance();
  bool LoadCheckPointOn() const { return load_checkpoint_on_; }
  bool SaveCheckPointOn() const { return save_checkpoint_on_; }

  void set_common_mirror_group(const RankList &comm_group) { common_mirror_group_ = comm_group; }
  RankList common_mirror_group() const { return common_mirror_group_; }

  bool LoadAutoOpStrategyOn() const { return load_auto_op_strategy_on_; }
  bool SaveAutoOpStrategyOn() const { return save_auto_op_strategy_on_; }
  Status LoadAutoOpStrategy(StrategyMap *strategy_map, StrategyMap *out_strategy_map,
                            TensorLayoutValueMap *tensor_layout_map, TensorLayoutValueMap *out_tensor_layout_map,
                            TensorLayoutValueMap *tensor_layout_newshape_map,
                            TensorLayoutValueMap *out_tensor_layout_newshape_map);
  Status SaveAutoOpStrategy(const StrategyMap &strategy_map, const StrategyMap &out_strategy_map,
                            const TensorLayoutValueMap &tensor_layout_map,
                            const TensorLayoutValueMap &out_tensor_layout_map,
                            const TensorLayoutValueMap &tensor_layout_newshape_map,
                            const TensorLayoutValueMap &out_tensor_layout_newshape_map);
  void SaveStrategyParamLayout();

 private:
  std::string auto_op_strategy_file_;
  std::string auto_op_strategy_file_type_;
  bool load_auto_op_strategy_on_ = false;
  bool save_auto_op_strategy_on_ = false;
  StrategyJsonInfo strategy_json_info_;

  std::string load_file_;
  std::string save_file_;
  bool load_checkpoint_on_ = false;
  bool save_checkpoint_on_ = false;
  bool CheckPointExit(const std::string path) const;
  bool CheckPath(const std::string path) const;
  void HandleEmptyParallelLayout();
  bool PipelineNotSupported();
  StrategyInfo BuildStrategyInfo(const std::string &param_name, const straspb::ParallelLayouts &layouts) const;
  int64_t current_stage_ = 0;
  std::string group_info_save_file_;
  bool group_info_save_on_ = false;
  bool load_format_json_ = true;
  bool save_format_json_ = true;
  StrategyCheckpointInfo strategy_checkpoint_info_;
  RankList common_mirror_group_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_CHEKCPOINT_PARALLEL_STRATEGY_CHECKPOINT_H_
