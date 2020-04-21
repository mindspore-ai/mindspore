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

#ifndef MINDSPORE_CCSRC_PARALLEL_CONTEXT_H_
#define MINDSPORE_CCSRC_PARALLEL_CONTEXT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "parallel/ops_info/ops_utils.h"
#include "parallel/status.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace parallel {
constexpr char STAND_ALONE[] = "stand_alone";
constexpr char DATA_PARALLEL[] = "data_parallel";
constexpr char HYBRID_PARALLEL[] = "hybrid_parallel";
constexpr char AUTO_PARALLEL[] = "auto_parallel";
constexpr char SEMI_AUTO_PARALLEL[] = "semi_auto_parallel";

constexpr char DYNAMIC_PROGRAMMING[] = "dynamic_programming";
constexpr char RECURSIVE_PROGRAMMING[] = "recursive_programming";

class ParallelContext {
 public:
  ~ParallelContext() = default;
  ParallelContext(const ParallelContext &) = delete;
  ParallelContext &operator=(const ParallelContext &) = delete;

  static std::shared_ptr<ParallelContext> GetInstance();

  void set_mirror_mean(bool mirror_mean);
  bool mirror_mean() const { return mirror_mean_; }

  void set_cast_before_mirror(bool cast_before_mirror);
  bool cast_before_mirror() const { return cast_before_mirror_; }

  void set_loss_repeated_mean(bool loss_repeated_mean);
  bool loss_repeated_mean() const { return loss_repeated_mean_; }

  void set_device_num(int32_t device_num);
  int32_t device_num() const { return device_num_; }

  void set_global_rank(int32_t global_rank);
  int32_t global_rank() const { return global_rank_; }

  void set_communication_backend(const std::string &communication_backend);
  std::string communication_backend() const { return communication_backend_; }

  bool set_parallel_mode(const std::string &parallel_mode);
  std::string parallel_mode() const { return parallel_mode_; }

  bool set_strategy_search_mode(const std::string &strategy_search_mode);
  std::string strategy_search_mode() const { return strategy_search_mode_; }

  void set_parameter_broadcast(bool parameter_broadcast);
  bool parameter_broadcast() const { return parameter_broadcast_; }

  bool device_num_is_set() const { return device_num_is_set_; }
  bool global_rank_is_set() const { return global_rank_is_set_; }
  bool parameter_broadcast_is_set() const { return parameter_broadcast_is_set_; }

  void set_all_reduce_fusion_split_indices(const std::vector<uint32_t> indices);
  const std::vector<uint32_t> all_reduce_fusion_split_indices() const;
  void set_all_reduce_fusion_split_sizes(const std::vector<uint32_t> sizes);
  const std::vector<uint32_t> all_reduce_fusion_split_sizes() const;

  void Reset();

 private:
  ParallelContext();
  static std::shared_ptr<ParallelContext> inst_context_;
  bool mirror_mean_;
  bool cast_before_mirror_;
  bool loss_repeated_mean_;
  int32_t device_num_;
  int32_t global_rank_;
  std::string communication_backend_;
  std::string parallel_mode_;
  std::string strategy_search_mode_;
  bool parameter_broadcast_;
  bool device_num_is_set_;
  bool global_rank_is_set_;
  bool parameter_broadcast_is_set_;
  std::vector<uint32_t> all_reduce_fusion_split_indices_;
  std::vector<uint32_t> all_reduce_fusion_split_sizes_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_CONTEXT_H_
